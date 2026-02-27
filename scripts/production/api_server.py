import os
import sys
import time
from typing import List, Literal, Optional

import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src import config
from src.cagr_metrics import compute_cagr_summary
from src.portfolio import get_target_weights
from src.production_db import (
    RUN_SUMMARY_FIELDS as DB_RUN_SUMMARY_FIELDS,
    append_pending_adjustments as db_append_pending_adjustments,
    count_broker_orders as db_count_broker_orders,
    count_latest_broker_orders as db_count_latest_broker_orders,
    clear_pending_adjustments as db_clear_pending_adjustments,
    db_enabled,
    delete_run_calendar_override as db_delete_run_calendar_override,
    init_db as db_init,
    list_excluded_tickers as db_list_excluded_tickers,
    list_latest_broker_orders as db_list_latest_broker_orders,
    list_latest_broker_positions as db_list_latest_broker_positions,
    list_latest_trades as db_list_latest_trades,
    list_pending_adjustments as db_list_pending_adjustments,
    list_run_calendar_overrides_paginated as db_list_run_calendar_overrides_paginated,
    list_run_summaries_paginated as db_list_run_summaries_paginated,
    list_universe_map as db_list_universe_map,
    latest_run_date as db_latest_run_date,
    latest_prices as db_latest_prices,
    latest_price_run_date as db_latest_price_run_date,
    latest_summary as db_latest_summary,
    latest_broker_account as db_latest_broker_account,
    latest_universe_monitor_summary as db_latest_universe_monitor_summary,
    list_run_summaries as db_list_run_summaries,
    list_broker_orders as db_list_broker_orders,
    list_trades as db_list_trades,
    list_universe_monitor_candidates as db_list_universe_monitor_candidates,
    load_excluded_tickers as db_load_excluded_tickers,
    load_run_calendar_override as db_load_run_calendar_override,
    load_universe_map as db_load_universe_map,
    price_tickers_for_date as db_price_tickers_for_date,
    replace_excluded_tickers as db_replace_excluded_tickers,
    load_state as db_load_state,
    reset_production_data as db_reset_production_data,
    upsert_run_calendar_override as db_upsert_run_calendar_override,
)
from src.regime import compute_market_regime_table, get_regime_state, regime_top_k
from src.run_calendar import (
    RUN_CALENDAR_ACTION_FORCE_RUN,
    RUN_CALENDAR_ACTION_SKIP,
    evaluate_run_day,
    list_us_federal_holidays,
)
from src.strategy import load_strategies
from src.universe import NASDAQ100_TICKERS
from src.universe_quality import apply_quality_filter

API_KEY = os.getenv("API_KEY", "").strip()
FX_PROVIDER_URL = os.getenv("FX_PROVIDER_URL", "https://api.frankfurter.app/latest")
FX_TIMEOUT_SECONDS = float(os.getenv("FX_TIMEOUT_SECONDS", "4.0"))
FX_CACHE_TTL_SECONDS = int(os.getenv("FX_CACHE_TTL_SECONDS", "3600"))
SUPPORTED_FX_CURRENCIES = {"USD", "GBP"}
_fx_cache: dict[tuple[str, str], dict[str, object]] = {}
DEFAULT_PAGE_LIMIT = int(os.getenv("API_DEFAULT_PAGE_LIMIT", "200"))
MAX_PAGE_LIMIT = int(os.getenv("API_MAX_PAGE_LIMIT", "1000"))

if not db_enabled():
    raise RuntimeError(
        "Database is required for the production API. Set DATABASE_URL or POSTGRES_URL."
    )
db_init()


def _normalize_currency(code: str | None, default: str = "USD") -> str:
    value = str(code or default).strip().upper()
    if value not in SUPPORTED_FX_CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported currency '{value}'. Supported: {sorted(SUPPORTED_FX_CURRENCIES)}",
        )
    return value


def _to_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(parsed):
        return None
    return parsed


def _derive_usd_to_gbp_from_db() -> float | None:
    rows = db_list_run_summaries()
    if not rows:
        return None
    candidate_pairs = (
        ("broker_net_worth", "broker_net_worth_usd"),
        ("broker_execution_cost", "broker_execution_cost_usd"),
        ("broker_total_execution_cost", "broker_total_execution_cost_usd"),
    )
    for row in reversed(rows):
        currency = str(row.get("broker_currency") or "").strip().upper()
        if currency == "USD":
            return 1.0
        if currency != "GBP":
            continue
        for native_key, usd_key in candidate_pairs:
            native = _to_float(row.get(native_key))
            usd = _to_float(row.get(usd_key))
            if native is None or usd is None or abs(usd) <= 1e-12:
                continue
            ratio = native / usd
            if 0.1 <= ratio <= 10.0:
                return ratio
    return None


def _fetch_usd_to_gbp_from_provider() -> float | None:
    try:
        response = requests.get(
            FX_PROVIDER_URL,
            params={"from": "USD", "to": "GBP"},
            timeout=FX_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None
    if response.status_code < 200 or response.status_code >= 300:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    rates = payload.get("rates")
    if not isinstance(rates, dict):
        return None
    rate = _to_float(rates.get("GBP"))
    if rate is None or rate <= 0:
        return None
    return rate


def _get_usd_to_gbp_rate() -> tuple[float | None, str]:
    cache_key = ("USD", "GBP")
    now = time.time()
    cached = _fx_cache.get(cache_key)
    if cached:
        fetched_at = _to_float(cached.get("fetched_at_epoch"))
        rate = _to_float(cached.get("rate"))
        if (
            fetched_at is not None
            and rate is not None
            and (now - fetched_at) <= FX_CACHE_TTL_SECONDS
        ):
            return rate, "cache"

    provider_rate = _fetch_usd_to_gbp_from_provider()
    if provider_rate is not None:
        _fx_cache[cache_key] = {
            "rate": provider_rate,
            "fetched_at_epoch": now,
        }
        return provider_rate, "provider"

    db_rate = _derive_usd_to_gbp_from_db()
    if db_rate is not None:
        return db_rate, "db_fallback"

    return None, "unavailable"


def _load_effective_excluded_tickers() -> tuple[set[str], str]:
    return db_load_excluded_tickers(), "db"


class AdjustmentRequest(BaseModel):
    cash_amount: float = 0.0
    cash_note: str = "app"
    tickers: Optional[List[str]] = None
    ticker_exchanges: Optional[dict[str, str]] = None
    source: str = "app"


class ExcludeTickersRequest(BaseModel):
    tickers: List[str]


class ResetRequest(BaseModel):
    confirm: str
    preserve_universe_monitor: bool = True


class RunCalendarOverrideRequest(BaseModel):
    date: str
    action: Literal["skip", "force_run"] = RUN_CALENDAR_ACTION_SKIP
    reason: str = ""
    source: str = "app"


def _normalize_iso_date(value: str, field_name: str = "date") -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    try:
        return str(pd.to_datetime(text).date())
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} format. Use YYYY-MM-DD.",
        )


def _normalize_pagination(
    limit: int,
    offset: int,
    *,
    default_limit: int = DEFAULT_PAGE_LIMIT,
    max_limit: int = MAX_PAGE_LIMIT,
) -> tuple[int, int]:
    normalized_limit = default_limit if limit is None else int(limit)
    normalized_offset = 0 if offset is None else int(offset)
    if normalized_limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")
    if normalized_offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if normalized_limit > max_limit:
        raise HTTPException(status_code=400, detail=f"limit must be <= {max_limit}")
    return normalized_limit, normalized_offset


def _parse_csv_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [value.strip() for value in str(raw).split(",") if value.strip()]


BROKER_SUMMARY_FIELDS: tuple[str, ...] = (
    "run_date",
    "broker",
    "currency",
    "cash",
    "investments",
    "net_worth",
    "payload",
)
BROKER_POSITION_FIELDS: tuple[str, ...] = (
    "run_date",
    "broker",
    "ticker",
    "quantity",
    "average_price",
    "current_price",
    "instrument_currency",
    "account_currency",
    "wallet_current_value",
    "payload",
)
BROKER_ORDER_FIELDS: tuple[str, ...] = (
    "run_date",
    "broker",
    "ticker",
    "action",
    "quantity",
    "filled_quantity",
    "exec_price",
    "currency",
    "status",
    "order_id",
    "payload",
    "created_at",
)
SUMMARY_FIELDS: tuple[str, ...] = DB_RUN_SUMMARY_FIELDS


def _normalize_broker_fields(
    raw_fields: str | None,
    *,
    allowed: tuple[str, ...],
    include_payload: bool,
) -> list[str] | None:
    requested = _parse_csv_values(raw_fields)
    if not requested:
        return None
    allowed_set = set(allowed)
    unknown = sorted({field for field in requested if field not in allowed_set})
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fields: {', '.join(unknown)}",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for field in requested:
        if not include_payload and field == "payload":
            continue
        if field in seen:
            continue
        seen.add(field)
        normalized.append(field)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="No valid fields requested after applying include_payload=false.",
        )
    return normalized


def _normalize_summary_fields(raw_fields: str | None) -> list[str] | None:
    requested = _parse_csv_values(raw_fields)
    if not requested:
        return None
    allowed_set = set(SUMMARY_FIELDS)
    unknown = sorted({field for field in requested if field not in allowed_set})
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fields: {', '.join(unknown)}",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for field in requested:
        if field in seen:
            continue
        seen.add(field)
        normalized.append(field)
    return normalized


def _parse_strategy_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    text = str(value).strip()
    if not text:
        return []
    return sorted({part.strip() for part in text.split(",") if part.strip()})


def _filter_tickers_by_universe_filter(tickers: set[str]) -> set[str]:
    universe_filter = (config.UNIVERSE_FILTER or "").strip().lower()
    if not universe_filter or universe_filter in ("all", "none"):
        return set(tickers)
    if universe_filter == "nasdaq100":
        return set(tickers).intersection(set(NASDAQ100_TICKERS))
    requested = {item.strip().upper() for item in universe_filter.split(",") if item.strip()}
    return set(tickers).intersection(requested)


def _safe_close_for_ticker(day_data: pd.DataFrame, ticker: str) -> float | None:
    if ticker not in day_data.index:
        return None
    row = day_data.loc[ticker]
    if isinstance(row, pd.DataFrame):
        close_val = row.iloc[0].get("Close")
    else:
        close_val = row.get("Close")
    return _to_float(close_val)


def _compute_selection_snapshot(
    filtered_df: pd.DataFrame,
    target_date: pd.Timestamp,
    strategy_names: list[str],
    strategy_roots: list[str],
    regime_scope: str,
) -> tuple[dict[str, float], set[str], dict]:
    if filtered_df.empty:
        return {}, set(), {"computed": False, "detail": "No rows left after pre-signal filters."}
    if not strategy_names:
        return {}, set(), {"computed": False, "detail": "No strategies provided."}

    strategies = load_strategies(strategy_names, strategy_roots)
    if not strategies:
        return {}, set(), {"computed": False, "detail": "No valid strategies loaded."}

    regime_source_df = filtered_df
    if regime_scope == "global":
        data_path = config.resolve_path(config.DATA_FILE)
        global_df = pd.read_parquet(data_path)
        regime_source_df = global_df

    regime_table = compute_market_regime_table(regime_source_df)
    state = get_regime_state(regime_table, target_date)
    regime_label = state.get("regime_label")
    active = [
        strategy
        for strategy in strategies
        if not strategy.regime_tags or regime_label in strategy.regime_tags
    ]
    active = active if active else strategies

    universe = filtered_df.index.get_level_values("ticker").unique().tolist()
    strategy_scores: dict[str, pd.Series] = {}
    for strategy in active:
        series = strategy.score_func(filtered_df)
        if not isinstance(series, pd.Series):
            raise HTTPException(
                status_code=500,
                detail=f"Strategy '{strategy.name}' returned non-Series scores.",
            )
        if len(series) != len(filtered_df):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Strategy '{strategy.name}' score length mismatch: "
                    f"{len(series)} != {len(filtered_df)}."
                ),
            )
        if not series.index.equals(filtered_df.index):
            series = series.reindex(filtered_df.index)
        strategy_scores[strategy.name] = series.astype(float).fillna(0.0)

    combined: np.ndarray | None = None
    for strategy in active:
        series = strategy_scores[strategy.name]
        try:
            day_scores = series.xs(target_date, level="date")
        except KeyError:
            continue
        arr = day_scores.reindex(universe).to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) == 0:
            continue
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std > 1e-9:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
        combined = arr if combined is None else combined + arr
    if combined is None:
        combined = np.zeros(len(universe), dtype=np.float32)

    try:
        day_slice = filtered_df.xs(target_date, level="date")
    except KeyError:
        return {}, set(), {"computed": False, "detail": f"No filtered data on {target_date.date()}."}
    day_data = day_slice.reindex(universe)
    prices = day_data["Close"].to_numpy(dtype=float)
    mask = np.isfinite(prices) & (prices > 0)
    vol = day_data.get("vol_21")
    vol = vol.to_numpy(dtype=float) if vol is not None else np.ones_like(prices)

    dynamic_top_k = config.TOP_K
    if config.USE_REGIME_SYSTEM:
        dynamic_top_k = regime_top_k(state, config.TOP_K)

    weights = get_target_weights(combined, vol, mask.astype(float), top_k=dynamic_top_k)
    weights = weights * mask
    total = np.sum(weights)
    if total > 1e-9:
        weights = weights / total
    else:
        weights = np.zeros_like(weights)

    scores_by_ticker = {ticker: float(score) for ticker, score in zip(universe, combined)}
    selected = {
        ticker for ticker, weight in zip(universe, weights) if float(weight) > 1e-12
    }
    metadata = {
        "computed": True,
        "active_strategies": [strategy.name for strategy in active],
        "regime_label": str(regime_label or "unknown"),
        "top_k": int(dynamic_top_k),
        "selection_note": (
            "Top-k score snapshot ignores runtime weight smoothing/state transitions. "
            "Use as diagnostic ranking, not exact replay."
        ),
    }
    return scores_by_ticker, selected, metadata


def _require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    if not API_KEY:
        return
    if x_api_key and x_api_key == API_KEY:
        return
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        if token == API_KEY:
            return
    raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title="Trading Production API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fx-rate", dependencies=[Depends(_require_api_key)])
def fx_rate(base: str = "USD", quote: str = "GBP"):
    base_code = _normalize_currency(base, default="USD")
    quote_code = _normalize_currency(quote, default="GBP")
    if base_code == quote_code:
        return {
            "base": base_code,
            "quote": quote_code,
            "rate": 1.0,
            "source": "identity",
        }

    if {base_code, quote_code} != {"USD", "GBP"}:
        raise HTTPException(
            status_code=400,
            detail="Only USD/GBP exchange rates are currently supported.",
        )

    usd_to_gbp, source = _get_usd_to_gbp_rate()
    if usd_to_gbp is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to resolve USD/GBP FX rate from provider or database fallback. "
                "Retry shortly."
            ),
        )

    rate = usd_to_gbp if (base_code, quote_code) == ("USD", "GBP") else 1.0 / usd_to_gbp
    return {
        "base": base_code,
        "quote": quote_code,
        "rate": rate,
        "source": source,
    }


@app.get("/latest-run", dependencies=[Depends(_require_api_key)])
def latest_run():
    latest = db_latest_run_date()
    if not latest:
        raise HTTPException(status_code=404, detail="No production runs found.")
    return {"latest_run": latest}


@app.get("/latest-summary", dependencies=[Depends(_require_api_key)])
def latest_summary():
    payload = db_latest_summary()
    if not payload:
        raise HTTPException(status_code=404, detail="Latest summary not found.")
    return payload


@app.get("/summaries", dependencies=[Depends(_require_api_key)])
def summaries(
    start: str | None = None,
    end: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    fields: str | None = None,
):
    limit, offset = _normalize_pagination(limit, offset)
    start_date = _normalize_iso_date(start, field_name="start") if start else None
    end_date = _normalize_iso_date(end, field_name="end") if end else None
    requested_fields = _normalize_summary_fields(fields)
    total, filtered = db_list_run_summaries_paginated(
        start=start_date,
        end=end_date,
        limit=limit,
        offset=offset,
        fields=requested_fields,
    )

    return {
        "total": total,
        "count": len(filtered),
        "limit": limit,
        "offset": offset,
        "summaries": filtered,
    }


@app.get("/latest-trades", dependencies=[Depends(_require_api_key)])
def latest_trades(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, rows = db_list_latest_trades(limit=limit, offset=offset)
    if total <= 0:
        raise HTTPException(status_code=404, detail="Latest trades not found.")
    if not rows:
        return {"total": total, "count": 0, "limit": limit, "offset": offset, "trades": []}
    return {
        "total": total,
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "trades": rows,
    }


@app.get("/trades", dependencies=[Depends(_require_api_key)])
def trades(limit: int = 200, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)

    total, records = db_list_trades(limit, offset)
    return {
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "trades": records,
    }


@app.get("/cagr", dependencies=[Depends(_require_api_key)])
def cagr_summary():
    latest = db_latest_summary()
    if latest:
        payload = latest.get("cagr_payload")
        if isinstance(payload, dict):
            return payload

    summaries = db_list_run_summaries()
    return compute_cagr_summary(summaries)


@app.get("/portfolio", dependencies=[Depends(_require_api_key)])
def portfolio_snapshot():
    state = None
    db_state = db_load_state()
    if db_state:
        state = {
            "last_date": db_state.last_date,
            "cash": db_state.cash,
            "positions": db_state.positions,
            "prev_weights": db_state.prev_weights,
            "total_costs_usd": db_state.total_costs_usd,
        }
    if not state:
        raise HTTPException(status_code=404, detail="Portfolio state not found.")

    cash = float(state.get("cash", 0.0))
    positions = state.get("positions", {}) or {}
    if not isinstance(positions, dict):
        positions = {}

    as_of, price_map = db_latest_prices(list(positions.keys()))
    if as_of is None:
        raise HTTPException(status_code=404, detail="No price snapshot found in database.")

    portfolio_rows = []
    missing_prices = []
    portfolio_value = 0.0

    for ticker, shares in positions.items():
        try:
            shares_val = int(shares)
        except (TypeError, ValueError):
            shares_val = 0
        ticker_key = str(ticker).strip().upper()
        price = price_map.get(ticker_key)
        value = None
        if price is not None:
            value = price * shares_val
            portfolio_value += value
        else:
            missing_prices.append(str(ticker))
        portfolio_rows.append(
            {
                "ticker": str(ticker),
                "shares": shares_val,
                "quantity": shares_val,
                "price_usd": price,
                "value_usd": value,
            }
        )

    return {
        "as_of": as_of,
        "cash_usd": cash,
        "portfolio_value_usd": portfolio_value,
        "net_worth_usd": cash + portfolio_value,
        "positions": portfolio_rows,
        "missing_prices": missing_prices,
    }


@app.get("/universe", dependencies=[Depends(_require_api_key)])
def universe_listing(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, universe = db_list_universe_map(limit=limit, offset=offset)
    if total <= 0:
        raise HTTPException(status_code=404, detail="Universe mapping not found.")
    return {
        "total": total,
        "count": len(universe),
        "limit": limit,
        "offset": offset,
        "universe": universe,
    }


@app.get("/universe/selection-diagnostics", dependencies=[Depends(_require_api_key)])
def universe_selection_diagnostics(
    sector: str = "Technology",
    date: str | None = None,
    regime_scope: Literal["global", "sector"] = "global",
    strategies: str | None = None,
    strategy_roots: str = "alphas",
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
):
    limit, offset = _normalize_pagination(limit, offset)
    data_path = config.resolve_path(config.DATA_FILE)
    try:
        df = pd.read_parquet(data_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read data file: {exc}")

    if "sector" not in df.columns or "Close" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Data file must include 'sector' and 'Close' columns.",
        )

    sector_name = str(sector or "").strip()
    if not sector_name:
        raise HTTPException(status_code=400, detail="sector is required.")

    sector_df = df[df["sector"].astype(str).str.lower() == sector_name.lower()]
    if sector_df.empty:
        available = sorted({str(s) for s in df["sector"].dropna().unique().tolist()})
        raise HTTPException(
            status_code=404,
            detail=f"No data for sector '{sector_name}'. Available sectors: {available}",
        )

    target_date = (
        pd.to_datetime(date).tz_localize(None)
        if date
        else pd.to_datetime(sector_df.index.get_level_values("date").max()).tz_localize(None)
    )
    target_date_str = str(target_date.date())
    all_dates = sector_df.index.get_level_values("date")
    if target_date not in all_dates:
        raise HTTPException(
            status_code=400,
            detail=f"Target date {target_date_str} is not present in sector data.",
        )

    sector_tickers = sorted(
        {
            str(ticker).strip().upper()
            for ticker in sector_df.index.get_level_values("ticker").unique()
            if str(ticker).strip()
        }
    )
    sector_set = set(sector_tickers)
    allowed = _filter_tickers_by_universe_filter(sector_set)
    excluded_set = db_load_excluded_tickers()
    removed_by_excluded = allowed.intersection(excluded_set)
    after_excluded = allowed - removed_by_excluded

    quality_input = sector_df[sector_df.index.get_level_values("ticker").isin(after_excluded)]
    quality_filtered, removed_by_quality = apply_quality_filter(quality_input)
    removed_by_quality = set(removed_by_quality)
    after_quality = {
        str(ticker).strip().upper()
        for ticker in quality_filtered.index.get_level_values("ticker").unique()
        if str(ticker).strip()
    }

    try:
        day_data = quality_filtered.xs(target_date, level="date")
    except KeyError:
        day_data = pd.DataFrame(columns=quality_filtered.columns)
    removed_invalid_close: set[str] = set()
    for ticker in sorted(after_quality):
        close_val = _safe_close_for_ticker(day_data, ticker)
        if close_val is None or not np.isfinite(close_val) or close_val <= 0:
            removed_invalid_close.add(ticker)
    candidates = after_quality - removed_invalid_close

    summary_by_date = {}
    _, summaries_for_date = db_list_run_summaries_paginated(
        start=target_date_str,
        end=target_date_str,
        limit=1,
        offset=0,
    )
    if summaries_for_date:
        summary_by_date = summaries_for_date[0] or {}

    strategy_names = _parse_csv_values(strategies)
    strategy_source = "query"
    if not strategy_names:
        strategy_names = _parse_strategy_names(summary_by_date.get("strategies"))
        strategy_source = "summary"

    roots = _parse_csv_values(strategy_roots) or ["alphas"]
    scores_by_ticker: dict[str, float] = {}
    selected_tickers: set[str] = set()
    selection_info: dict = {
        "computed": False,
        "detail": "Selection snapshot skipped (no strategy names provided).",
        "strategy_source": strategy_source,
        "strategy_names": strategy_names,
        "strategy_roots": roots,
    }
    if strategy_names:
        scores_by_ticker, selected_tickers, selection_meta = _compute_selection_snapshot(
            filtered_df=quality_filtered,
            target_date=target_date,
            strategy_names=strategy_names,
            strategy_roots=roots,
            regime_scope=regime_scope,
        )
        selection_info = {
            **selection_meta,
            "strategy_source": strategy_source,
            "strategy_names": strategy_names,
            "strategy_roots": roots,
        }

    sorted_score_tickers = sorted(
        scores_by_ticker.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    rank_by_ticker = {ticker: idx + 1 for idx, (ticker, _) in enumerate(sorted_score_tickers)}

    records = []
    for ticker in sector_tickers:
        if ticker not in allowed:
            stage = "filtered_universe_filter"
            reason = f"Ticker excluded by UNIVERSE_FILTER={config.UNIVERSE_FILTER!r}."
        elif ticker in removed_by_excluded:
            stage = "filtered_excluded_ticker"
            reason = "Ticker present in excluded tickers list."
        elif ticker in removed_by_quality:
            stage = "filtered_quality"
            reason = "Ticker failed universe quality thresholds."
        elif ticker in removed_invalid_close:
            stage = "filtered_invalid_close"
            reason = f"Ticker has missing/non-positive Close on {target_date_str}."
        else:
            if selection_info.get("computed"):
                if ticker in selected_tickers:
                    stage = "selected_top_k"
                    reason = "Ticker is in the top-k score snapshot for this date."
                else:
                    stage = "not_selected_top_k"
                    reason = "Ticker passed pre-filters but is outside top-k score snapshot."
            else:
                stage = "candidate_pool"
                reason = "Ticker passed all pre-signal filters."
        records.append(
            {
                "ticker": ticker,
                "stage": stage,
                "reason": reason,
                "combined_score": scores_by_ticker.get(ticker),
                "score_rank": rank_by_ticker.get(ticker),
            }
        )

    stage_counts: dict[str, int] = {}
    for row in records:
        stage = str(row.get("stage"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    paged = records[offset : offset + limit]
    return {
        "date": target_date_str,
        "sector": sector_name,
        "regime_scope": regime_scope,
        "total": len(records),
        "count": len(paged),
        "limit": limit,
        "offset": offset,
        "filters": {
            "universe_filter": config.UNIVERSE_FILTER,
            "excluded_tickers_count": len(excluded_set),
            "quality_filter_enabled": bool(config.ENABLE_UNIVERSE_QUALITY_FILTER),
        },
        "selection": selection_info,
        "counts": {
            "sector_universe": len(sector_tickers),
            "after_universe_filter": len(allowed),
            "removed_by_excluded_tickers": len(removed_by_excluded),
            "removed_by_quality": len(removed_by_quality),
            "removed_by_invalid_close": len(removed_invalid_close),
            "candidate_pool": len(candidates),
            "stage_counts": stage_counts,
        },
        "records": paged,
    }


@app.get("/universe-monitor/summary", dependencies=[Depends(_require_api_key)])
def universe_monitor_summary():
    payload = db_latest_universe_monitor_summary()
    if not payload:
        raise HTTPException(status_code=404, detail="Universe monitor summary not found.")
    full_payload = payload.get("payload")
    if isinstance(full_payload, dict):
        return full_payload
    return payload


@app.get("/universe-monitor/candidates", dependencies=[Depends(_require_api_key)])
def universe_monitor_candidates(
    limit: int = 200,
    offset: int = 0,
    watchlist: bool = False,
    potential: bool = False,
):
    limit, offset = _normalize_pagination(limit, offset)

    total, records = db_list_universe_monitor_candidates(
        limit=limit,
        offset=offset,
        watchlist=watchlist,
        potential=potential,
    )
    return {
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "watchlist": watchlist,
        "potential": potential,
        "candidates": records,
    }


@app.get("/universe-monitor/potential", dependencies=[Depends(_require_api_key)])
def universe_monitor_potential(limit: int = 200, offset: int = 0):
    return universe_monitor_candidates(
        limit=limit,
        offset=offset,
        watchlist=False,
        potential=True,
    )


@app.get("/broker-summary", dependencies=[Depends(_require_api_key)])
def broker_summary(
    broker: str = "trading212",
    include_payload: bool = False,
    fields: str | None = None,
):
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_SUMMARY_FIELDS,
        include_payload=include_payload,
    )
    summary = db_latest_broker_account(
        broker,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if summary:
        return summary
    raise HTTPException(status_code=404, detail="Broker summary not found.")


@app.get("/broker-positions", dependencies=[Depends(_require_api_key)])
def broker_positions(
    broker: str = "trading212",
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    include_payload: bool = False,
    fields: str | None = None,
):
    limit, offset = _normalize_pagination(limit, offset)
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_POSITION_FIELDS,
        include_payload=include_payload,
    )
    total, positions = db_list_latest_broker_positions(
        broker=broker,
        limit=limit,
        offset=offset,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if total > 0:
        return {
            "total": total,
            "count": len(positions),
            "limit": limit,
            "offset": offset,
            "positions": positions,
        }
    raise HTTPException(status_code=404, detail="Broker positions not found.")


@app.get("/broker-orders", dependencies=[Depends(_require_api_key)])
def broker_orders(
    broker: str = "trading212",
    limit: int = 200,
    offset: int = 0,
    include_payload: bool = False,
    fields: str | None = None,
    meta_only: bool = False,
):
    limit, offset = _normalize_pagination(limit, offset)
    if meta_only:
        total = db_count_broker_orders(broker)
        return {
            "total": total,
            "count": 0,
            "limit": limit,
            "offset": offset,
            "orders": [],
            "meta_only": True,
        }
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_ORDER_FIELDS,
        include_payload=include_payload,
    )
    total, records = db_list_broker_orders(
        broker,
        limit,
        offset,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if total > 0:
        return {
            "total": total,
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "orders": records,
        }
    return {"total": 0, "count": 0, "limit": limit, "offset": offset, "orders": []}


@app.get("/latest-broker-orders", dependencies=[Depends(_require_api_key)])
def latest_broker_orders(
    broker: str = "trading212",
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    include_payload: bool = False,
    fields: str | None = None,
    meta_only: bool = False,
):
    limit, offset = _normalize_pagination(limit, offset)
    if meta_only:
        run_date, total = db_count_latest_broker_orders(broker)
        if total > 0:
            return {
                "broker": broker,
                "run_date": str(run_date),
                "total": total,
                "count": 0,
                "limit": limit,
                "offset": offset,
                "orders": [],
                "meta_only": True,
            }
        raise HTTPException(status_code=404, detail="Broker orders not found.")
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_ORDER_FIELDS,
        include_payload=include_payload,
    )
    total, records = db_list_latest_broker_orders(
        broker=broker,
        limit=limit,
        offset=offset,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if total > 0:
        return {
            "total": total,
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "orders": records,
        }
    raise HTTPException(status_code=404, detail="Broker orders not found.")


@app.get("/latest-broker-orders/count", dependencies=[Depends(_require_api_key)])
def latest_broker_orders_count(broker: str = "trading212"):
    run_date, total = db_count_latest_broker_orders(broker)
    if total <= 0:
        raise HTTPException(status_code=404, detail="Broker orders not found.")
    return {
        "broker": broker,
        "run_date": str(run_date),
        "total": total,
    }


@app.get("/excluded-tickers", dependencies=[Depends(_require_api_key)])
def list_excluded_tickers(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, excluded = db_list_excluded_tickers(limit=limit, offset=offset)
    return {
        "total": total,
        "count": len(excluded),
        "limit": limit,
        "offset": offset,
        "excluded_tickers": excluded,
        "source": "db",
    }


@app.post("/exclude-tickers", dependencies=[Depends(_require_api_key)])
def exclude_tickers(payload: ExcludeTickersRequest):
    tickers = [str(t).strip().upper() for t in (payload.tickers or []) if str(t).strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided.")
    excluded, source = _load_effective_excluded_tickers()
    excluded.update(tickers)
    db_replace_excluded_tickers(excluded)
    return {"count": len(excluded), "excluded_tickers": sorted(excluded), "source": source}


@app.get("/run-calendar/decision", dependencies=[Depends(_require_api_key)])
def run_calendar_decision(date: str):
    run_date = _normalize_iso_date(date, field_name="date")
    override = db_load_run_calendar_override(run_date) or {}
    decision = evaluate_run_day(
        run_date,
        override_action=str(override.get("action") or "").strip().lower() or None,
        override_reason=str(override.get("reason") or "").strip() or None,
        skip_weekends=config.RUN_CALENDAR_SKIP_WEEKENDS,
        skip_us_federal_holidays=config.RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS,
    )
    return {
        "date": run_date,
        "should_run": bool(decision.get("should_run")),
        "reason_code": decision.get("reason_code"),
        "reason": decision.get("reason"),
        "source": decision.get("source"),
        "override": override or None,
        "settings": {
            "skip_weekends": bool(config.RUN_CALENDAR_SKIP_WEEKENDS),
            "skip_us_federal_holidays": bool(config.RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS),
        },
    }


@app.get("/run-calendar/overrides", dependencies=[Depends(_require_api_key)])
def run_calendar_overrides(
    start: str | None = None,
    end: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
):
    limit, offset = _normalize_pagination(limit, offset)
    start_date = _normalize_iso_date(start, field_name="start") if start else None
    end_date = _normalize_iso_date(end, field_name="end") if end else None
    total, rows = db_list_run_calendar_overrides_paginated(
        start=start_date,
        end=end_date,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "overrides": rows,
    }


@app.post("/run-calendar/overrides", dependencies=[Depends(_require_api_key)])
def upsert_run_calendar_override(payload: RunCalendarOverrideRequest):
    run_date = _normalize_iso_date(payload.date, field_name="date")
    action = str(payload.action).strip().lower()
    if action not in (RUN_CALENDAR_ACTION_SKIP, RUN_CALENDAR_ACTION_FORCE_RUN):
        raise HTTPException(status_code=400, detail="Invalid action. Use 'skip' or 'force_run'.")
    reason = str(payload.reason or "").strip()
    source = str(payload.source or "app").strip() or "app"
    db_upsert_run_calendar_override(
        run_date=run_date,
        action=action,
        reason=reason or None,
        source=source,
    )
    decision = evaluate_run_day(
        run_date,
        override_action=action,
        override_reason=reason or None,
        skip_weekends=config.RUN_CALENDAR_SKIP_WEEKENDS,
        skip_us_federal_holidays=config.RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS,
    )
    return {
        "upserted": True,
        "date": run_date,
        "action": action,
        "reason": reason,
        "source": source,
        "decision": decision,
    }


@app.delete("/run-calendar/overrides/{run_date}", dependencies=[Depends(_require_api_key)])
def delete_run_calendar_override(run_date: str):
    date_value = _normalize_iso_date(run_date, field_name="run_date")
    deleted = db_delete_run_calendar_override(date_value)
    return {"deleted": bool(deleted), "date": date_value}


@app.get("/run-calendar/us-federal-holidays", dependencies=[Depends(_require_api_key)])
def run_calendar_us_federal_holidays(year: int):
    if year < 1970 or year > 2100:
        raise HTTPException(status_code=400, detail="year must be between 1970 and 2100.")
    holidays = list_us_federal_holidays(year)
    return {"year": year, "count": len(holidays), "holidays": holidays}


@app.get("/stale-tickers", dependencies=[Depends(_require_api_key)])
def stale_tickers(
    date: Optional[str] = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
):
    limit, offset = _normalize_pagination(limit, offset)
    mapping = db_load_universe_map()
    if not mapping:
        raise HTTPException(status_code=404, detail="Universe mapping not found.")

    if date:
        try:
            target_date = pd.to_datetime(date).tz_localize(None)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid date format.")
    else:
        latest = db_latest_price_run_date()
        if not latest:
            raise HTTPException(status_code=404, detail="No production price snapshots found.")
        target_date = pd.to_datetime(latest).tz_localize(None)

    target_run_date = str(pd.to_datetime(target_date).date())
    tickers = set(mapping.keys())
    excluded, _ = _load_effective_excluded_tickers()
    if excluded:
        tickers -= excluded
    observed = db_price_tickers_for_date(target_run_date)
    if observed:
        tickers -= observed
    stale_sorted = sorted(tickers)
    paged = stale_sorted[offset : offset + limit]

    return {
        "as_of": target_run_date,
        "total": len(stale_sorted),
        "count": len(paged),
        "limit": limit,
        "offset": offset,
        "stale_tickers": paged,
        "source": "db",
    }


@app.post("/queue-adjustments", dependencies=[Depends(_require_api_key)])
def queue_adjustments(payload: AdjustmentRequest):
    entries: list[dict] = []
    timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if payload.cash_amount:
        entries.append(
            {
                "type": "cash",
                "amount": float(payload.cash_amount),
                "note": payload.cash_note,
                "source": payload.source,
                "created_at": timestamp,
            }
        )
    tickers = [t.strip().upper() for t in (payload.tickers or []) if t and t.strip()]
    exchange_map: dict[str, str] = {}
    if payload.ticker_exchanges:
        for ticker, exchange in payload.ticker_exchanges.items():
            t = str(ticker).strip().upper()
            ex = str(exchange).strip().upper()
            if t:
                exchange_map[t] = ex or "UNKNOWN"
    if exchange_map:
        tickers.extend(exchange_map.keys())
        tickers = [t for t in tickers if t and t.strip()]
        tickers = sorted(set(tickers))
    if tickers:
        entry = {
            "type": "tickers",
            "tickers": sorted(set(tickers)),
            "source": payload.source,
            "created_at": timestamp,
        }
        if exchange_map:
            entry["exchanges"] = exchange_map
        entries.append(entry)
    if not entries:
        raise HTTPException(status_code=400, detail="No adjustments provided.")

    db_append_pending_adjustments(entries)

    return {"queued": len(entries)}


@app.get("/pending-adjustments", dependencies=[Depends(_require_api_key)])
def list_pending_adjustments(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, entries = db_list_pending_adjustments(limit=limit, offset=offset)
    return {
        "total": total,
        "count": len(entries),
        "limit": limit,
        "offset": offset,
        "pending": entries,
    }


@app.post("/clear-pending", dependencies=[Depends(_require_api_key)])
def clear_pending_adjustments():
    db_clear_pending_adjustments()
    return {"cleared": True}


@app.delete("/pending-adjustments", dependencies=[Depends(_require_api_key)])
def delete_pending_adjustments():
    db_clear_pending_adjustments()
    return {"cleared": True}


@app.post("/reset-production", dependencies=[Depends(_require_api_key)])
def reset_production(payload: ResetRequest):
    preserve_monitor = bool(payload.preserve_universe_monitor)
    if preserve_monitor:
        if payload.confirm != "RESET_PRODUCTION_DATA":
            raise HTTPException(status_code=400, detail="Confirmation token missing or invalid.")
    else:
        if payload.confirm != "RESET_PRODUCTION_AND_MONITOR_DATA":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid confirmation token for full reset. "
                    "Use RESET_PRODUCTION_AND_MONITOR_DATA."
                ),
            )
    db_reset_production_data(preserve_universe_monitor=preserve_monitor)
    return {
        "reset": True,
        "preserve_universe_monitor": preserve_monitor,
    }
