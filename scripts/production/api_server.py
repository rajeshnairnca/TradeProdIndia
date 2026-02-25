import os
import sys
import time
from typing import List, Literal, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src import config
from src.production_db import (
    append_pending_adjustments as db_append_pending_adjustments,
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
from src.run_calendar import (
    RUN_CALENDAR_ACTION_FORCE_RUN,
    RUN_CALENDAR_ACTION_SKIP,
    evaluate_run_day,
    list_us_federal_holidays,
)

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


def _xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    if len(cashflows) < 2:
        return None
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]

    def _year_frac(ts: pd.Timestamp) -> float:
        return (ts - t0).days / 365.25

    def npv(rate: float) -> float:
        total = 0.0
        for dt, amount in cashflows:
            total += amount / ((1.0 + rate) ** _year_frac(dt))
        return total

    def d_npv(rate: float) -> float:
        total = 0.0
        for dt, amount in cashflows:
            yf = _year_frac(dt)
            total += -yf * amount / ((1.0 + rate) ** (yf + 1.0))
        return total

    rate = 0.1
    for _ in range(50):
        f_val = npv(rate)
        if abs(f_val) < 1e-7:
            return rate
        deriv = d_npv(rate)
        if abs(deriv) < 1e-12:
            break
        rate -= f_val / deriv
        if rate <= -0.9999:
            rate = -0.9999

    low, high = -0.9999, 10.0
    f_low, f_high = npv(low), npv(high)
    if f_low * f_high > 0:
        return None
    mid = 0.0
    for _ in range(100):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return mid


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
    total, filtered = db_list_run_summaries_paginated(
        start=start_date,
        end=end_date,
        limit=limit,
        offset=offset,
    )

    field_list: list[str] | None = None
    if fields:
        field_list = [f.strip() for f in fields.split(",") if f.strip()]

    if field_list:
        trimmed = []
        for row in filtered:
            trimmed.append({key: row.get(key) for key in field_list})
        filtered = trimmed

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
    summaries = db_list_run_summaries()

    if len(summaries) < 2:
        return {
            "cagr": None,
            "cagr_adjusted": None,
            "detail": "Not enough history to compute CAGR.",
        }

    summaries = sorted(summaries, key=lambda s: s["date"])
    start = summaries[0]
    end = summaries[-1]
    start_date = pd.to_datetime(start["date"])
    end_date = pd.to_datetime(end["date"])
    # Match backtester CAGR annualization: treat each summary row as one trading day.
    years = len(summaries) / 252.0
    start_value = float(start.get("net_worth_usd", 0.0))
    end_value = float(end.get("net_worth_usd", 0.0))
    if years <= 0 or start_value <= 0:
        return {
            "cagr": None,
            "cagr_adjusted": None,
            "detail": "Invalid dates or starting value for CAGR calculation.",
        }

    cagr = (end_value / start_value) ** (1.0 / years) - 1.0
    cash_adjustments = sum(float(s.get("cash_adjustment", 0.0)) for s in summaries)

    twr_growth = 1.0
    for i in range(1, len(summaries)):
        start_val = float(summaries[i - 1].get("net_worth_usd", 0.0))
        end_val = float(summaries[i].get("net_worth_usd", 0.0))
        flow = float(summaries[i].get("cash_adjustment", 0.0))
        denom = start_val + flow
        if denom <= 0:
            twr_growth = None
            break
        period_return = (end_val / denom) - 1.0
        twr_growth *= 1.0 + period_return

    cagr_adjusted = None
    if twr_growth is not None:
        cagr_adjusted = twr_growth ** (1.0 / years) - 1.0

    cashflows = [(start_date, -start_value)]
    for item in summaries[1:]:
        flow = float(item.get("cash_adjustment", 0.0))
        if flow != 0.0:
            cashflows.append((pd.to_datetime(item["date"]), -flow))
    cashflows.append((end_date, end_value))
    irr = _xirr(cashflows)

    broker_payload = None
    broker_summaries = [
        s for s in summaries if s.get("broker_net_worth") is not None
    ]
    if len(broker_summaries) >= 2:
        broker_summaries = sorted(broker_summaries, key=lambda s: s["date"])
        b_start = broker_summaries[0]
        b_end = broker_summaries[-1]
        b_start_date = pd.to_datetime(b_start["date"])
        b_end_date = pd.to_datetime(b_end["date"])
        # Use the same 252-trading-day basis for broker-side CAGR values.
        b_years = len(broker_summaries) / 252.0
        b_start_val = float(b_start.get("broker_net_worth", 0.0))
        b_end_val = float(b_end.get("broker_net_worth", 0.0))
        if b_years > 0 and b_start_val > 0:
            b_cagr = (b_end_val / b_start_val) ** (1.0 / b_years) - 1.0
        else:
            b_cagr = None

        b_twr_growth = 1.0
        for i in range(1, len(broker_summaries)):
            start_val = float(broker_summaries[i - 1].get("broker_net_worth", 0.0))
            end_val = float(broker_summaries[i].get("broker_net_worth", 0.0))
            fx_rate = float(broker_summaries[i].get("broker_fx_rate_gbp_per_usd") or 0.0)
            flow_usd = float(broker_summaries[i].get("cash_adjustment", 0.0))
            flow = flow_usd * fx_rate if fx_rate else flow_usd
            denom = start_val + flow
            if denom <= 0:
                b_twr_growth = None
                break
            period_return = (end_val / denom) - 1.0
            b_twr_growth *= 1.0 + period_return

        b_cagr_adjusted = None
        if b_twr_growth is not None and b_years > 0:
            b_cagr_adjusted = b_twr_growth ** (1.0 / b_years) - 1.0

        b_cashflows = [(b_start_date, -b_start_val)]
        for item in broker_summaries[1:]:
            fx_rate = float(item.get("broker_fx_rate_gbp_per_usd") or 0.0)
            flow_usd = float(item.get("cash_adjustment", 0.0))
            flow = flow_usd * fx_rate if fx_rate else flow_usd
            if flow != 0.0:
                b_cashflows.append((pd.to_datetime(item["date"]), -flow))
        b_cashflows.append((b_end_date, b_end_val))
        b_irr = _xirr(b_cashflows)
        broker_payload = {
            "currency": b_start.get("broker_currency"),
            "cagr": b_cagr,
            "cagr_adjusted": b_cagr_adjusted,
            "irr": b_irr,
        }

    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "years": years,
        "start_value_usd": start_value,
        "end_value_usd": end_value,
        "cagr": cagr,
        "cagr_adjusted": cagr_adjusted,
        "irr": irr,
        "cash_adjustments_total_usd": cash_adjustments,
        "broker": broker_payload,
        "note": "CAGR and cagr_adjusted are annualized on a 252-trading-day basis from summary rows; cagr_adjusted uses time-weighted returns with cash flows applied at period start; irr is money-weighted using net worth and cash adjustments.",
    }


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
def broker_summary(broker: str = "trading212"):
    summary = db_latest_broker_account(broker)
    if summary:
        return summary
    raise HTTPException(status_code=404, detail="Broker summary not found.")


@app.get("/broker-positions", dependencies=[Depends(_require_api_key)])
def broker_positions(
    broker: str = "trading212",
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
):
    limit, offset = _normalize_pagination(limit, offset)
    total, positions = db_list_latest_broker_positions(
        broker=broker,
        limit=limit,
        offset=offset,
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
def broker_orders(broker: str = "trading212", limit: int = 200, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, records = db_list_broker_orders(broker, limit, offset)
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
):
    limit, offset = _normalize_pagination(limit, offset)
    total, records = db_list_latest_broker_orders(
        broker=broker,
        limit=limit,
        offset=offset,
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
