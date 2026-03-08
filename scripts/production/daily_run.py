import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.cagr_metrics import compute_cagr_summary
from src.market_data_validation import validate_market_data_frame
from src.production_market_data import add_universe_tickers, update_market_data
from src.selection_diagnostics import build_selection_diagnostics
from src.production import (
    generate_trades_for_date,
    ProductionState,
)
from src.production_db import (
    append_pending_adjustments as db_append_pending_adjustments,
    clear_pending_adjustments as db_clear_pending_adjustments,
    db_enabled,
    init_db as db_init,
    list_run_summaries as db_list_run_summaries,
    load_excluded_tickers as db_load_excluded_tickers,
    load_run_calendar_override as db_load_run_calendar_override,
    load_pending_adjustments as db_load_pending_adjustments,
    load_state as db_load_state,
    load_universe_map as db_load_universe_map,
    replace_universe_map as db_replace_universe_map,
    replace_broker_orders as db_replace_broker_orders,
    replace_broker_positions as db_replace_broker_positions,
    replace_trades as db_replace_trades,
    replace_prices as db_replace_prices,
    replace_universe_selection_diagnostics_snapshot as db_replace_universe_selection_diagnostics_snapshot,
    upsert_broker_account as db_upsert_broker_account,
    upsert_run_summary as db_upsert_run_summary,
    upsert_state as db_upsert_state,
)
from src.run_calendar import evaluate_run_day, resolve_schedule_date
from src.regime import compute_market_regime_table
from src.strategy import list_strategy_names, load_strategies
from src.universe_quality import apply_quality_filter
from src.kite import (
    KiteApiError,
    KiteClient,
    account_cash_available as kite_account_cash_available,
    account_net_worth as kite_account_net_worth,
    build_instrument_index as build_kite_instrument_index,
    kite_enabled,
    load_instruments_cache as load_kite_instruments_cache,
    load_ticker_overrides as load_kite_ticker_overrides,
    positions_to_internal_positions as kite_positions_to_internal_positions,
    resolve_kite_instrument,
)
from src.trading212 import (
    Trading212ApiError,
    Trading212Client,
    account_cash_available as t212_account_cash_available,
    account_net_worth as t212_account_net_worth,
    build_instrument_index as build_t212_instrument_index,
    compare_positions,
    extract_fx_rates,
    load_instruments_cache as load_t212_instruments_cache,
    load_ticker_overrides as load_t212_ticker_overrides,
    positions_to_internal_positions as t212_positions_to_internal_positions,
    resolve_t212_ticker,
    trading212_enabled,
)

TRADE_COLUMNS = [
    "date",
    "ticker",
    "action",
    "shares",
    "price_usd",
    "value_usd",
    "net_worth_usd",
    "cash_usd",
    "portfolio_value_usd",
    "cash_weight",
    "regime",
    "strategies",
]

DEFAULT_REGIME_MAPPING_INDIA = {
    "bear_high_vol": "india_rule_crash_resilient_slow",
    "bear_low_vol": "india_rule_quality_defensive_slow",
    "bull_high_vol": "india_rule_pullback_reentry_slow",
    "bull_low_vol": "india_rule_trend_carry_slow",
    "sideways_high_vol": "india_rule_range_stability_slow",
    "sideways_low_vol": "india_rule_liquidity_momentum_core",
}

DEFAULT_REGIME_MAPPING = DEFAULT_REGIME_MAPPING_INDIA


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str, **fields) -> None:
    payload = f"[{_utc_now()}] {message}"
    if fields:
        payload = f"{payload} | {json.dumps(fields, default=str, sort_keys=True)}"
    print(payload, flush=True)


def _apply_confirmed_switch(
    pred_labels: pd.Series,
    confirm_days: int,
    confirm_days_sideways: int | None = None,
) -> pd.Series:
    confirm_days = max(1, int(confirm_days))
    confirm_days_sideways = (
        max(1, int(confirm_days_sideways))
        if confirm_days_sideways is not None
        else confirm_days
    )
    if pred_labels.empty or (confirm_days <= 1 and confirm_days_sideways <= 1):
        return pred_labels
    held = pd.Series(index=pred_labels.index, dtype=object)
    current = None
    candidate = None
    streak = 0
    for i, value in enumerate(pred_labels.to_numpy()):
        if current is None:
            current = value
            held.iloc[i] = current
            continue
        if value == current:
            candidate = None
            streak = 0
            held.iloc[i] = current
            continue
        if candidate == value:
            streak += 1
        else:
            candidate = value
            streak = 1
        threshold = confirm_days_sideways if str(value).startswith("sideways_") else confirm_days
        if streak >= threshold:
            current = candidate
            candidate = None
            streak = 0
        held.iloc[i] = current
    return held


def _apply_regime_confirmation(
    regime_table: pd.DataFrame,
    *,
    confirm_days: int,
    confirm_days_sideways: int,
) -> pd.DataFrame:
    if regime_table is None or regime_table.empty or "regime_label" not in regime_table.columns:
        return regime_table
    out = regime_table.copy()
    label_series = out["regime_label"].astype(str)
    confirmed = _apply_confirmed_switch(
        label_series,
        confirm_days=confirm_days,
        confirm_days_sideways=confirm_days_sideways,
    )
    out["regime_label"] = confirmed.astype(str)

    trend_state_map = {
        "bull_low_vol": "bull",
        "bull_high_vol": "bull",
        "bear_low_vol": "bear",
        "bear_high_vol": "bear",
        "sideways_low_vol": "sideways",
        "sideways_high_vol": "sideways",
    }
    trend_up_map = {
        "bull_low_vol": True,
        "bull_high_vol": True,
        "bear_low_vol": False,
        "bear_high_vol": False,
        "sideways_low_vol": False,
        "sideways_high_vol": False,
    }
    vol_high_map = {
        "bull_low_vol": False,
        "bull_high_vol": True,
        "bear_low_vol": False,
        "bear_high_vol": True,
        "sideways_low_vol": False,
        "sideways_high_vol": True,
    }
    trend_state_new = out["regime_label"].map(trend_state_map)
    trend_up_new = out["regime_label"].map(trend_up_map)
    vol_high_new = out["regime_label"].map(vol_high_map)

    if "trend_state" in out.columns:
        out["trend_state"] = trend_state_new.fillna(out["trend_state"])
    else:
        out["trend_state"] = trend_state_new.fillna("unknown")
    if "trend_up" in out.columns:
        out["trend_up"] = trend_up_new.fillna(out["trend_up"])
    else:
        out["trend_up"] = trend_up_new.fillna(False)
    if "vol_high" in out.columns:
        out["vol_high"] = vol_high_new.fillna(out["vol_high"])
    else:
        out["vol_high"] = vol_high_new.fillna(False)
    return out


def _apply_runtime_trading_overrides(args: argparse.Namespace) -> None:
    config.CONFIRM_DAYS = max(1, int(args.confirm_days))
    config.CONFIRM_DAYS_SIDEWAYS = max(1, int(args.confirm_days_sideways))
    args.confirm_days = config.CONFIRM_DAYS
    args.confirm_days_sideways = config.CONFIRM_DAYS_SIDEWAYS
    config.REBALANCE_EVERY = max(1, int(args.rebalance_every))
    config.REBALANCE_EVERY_N_DAYS = config.REBALANCE_EVERY
    args.rebalance_every = config.REBALANCE_EVERY
    config.MIN_WEIGHT_CHANGE_TO_TRADE = max(0.0, float(args.min_weight_change))
    args.min_weight_change = config.MIN_WEIGHT_CHANGE_TO_TRADE
    config.MIN_TRADE_DOLLARS = max(0.0, float(args.min_trade_dollars))
    args.min_trade_dollars = config.MIN_TRADE_DOLLARS
    max_turnover = None if args.max_daily_turnover is None else float(args.max_daily_turnover)
    if max_turnover is not None and max_turnover <= 0:
        max_turnover = None
    config.MAX_DAILY_TURNOVER = max_turnover
    args.max_daily_turnover = max_turnover
    config.WEIGHT_SMOOTHING = max(0.0, float(args.weight_smoothing))
    args.weight_smoothing = config.WEIGHT_SMOOTHING


def _attach_cagr_payload(summary: dict) -> None:
    run_date = str(summary.get("date") or "").strip()
    if not run_date:
        return
    history = db_list_run_summaries()
    by_date: dict[str, dict] = {}
    for item in history:
        date_key = str(item.get("date") or "").strip()
        if not date_key:
            continue
        by_date[date_key] = item
    by_date[run_date] = summary
    summary["cagr_payload"] = compute_cagr_summary(by_date.values())


def _persist_selection_diagnostics_snapshot(
    *,
    full_df: pd.DataFrame,
    summary: dict,
    sector: str,
    target_date: pd.Timestamp,
    regime_scope: str,
    strategy_roots: list[str],
    excluded_tickers: set[str],
) -> None:
    try:
        snapshot = build_selection_diagnostics(
            full_df=full_df,
            sector=sector,
            target_date=target_date,
            regime_scope=regime_scope,
            excluded_tickers=excluded_tickers,
            strategies=summary.get("strategies"),
            strategy_roots=strategy_roots,
        )
        records = snapshot.get("records") if isinstance(snapshot.get("records"), list) else []
        payload = dict(snapshot)
        payload.pop("records", None)
        db_replace_universe_selection_diagnostics_snapshot(
            run_date=str(summary["date"]),
            sector=sector,
            regime_scope=regime_scope,
            payload=payload,
            records=records,
        )
        _log(
            "selection_diagnostics_snapshot_persisted",
            run_date=summary["date"],
            sector=sector,
            regime_scope=regime_scope,
            records=len(records),
        )
    except Exception as exc:
        _log(
            "selection_diagnostics_snapshot_failed",
            run_date=summary.get("date"),
            sector=sector,
            regime_scope=regime_scope,
            error=str(exc),
            error_type=type(exc).__name__,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Daily production run: update data + generate trades.")
    parser.add_argument("--strategies", nargs="+", help="List of strategy names to include.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root directory containing strategies.")
    parser.add_argument(
        "--regime-mapping",
        type=str,
        help="JSON mapping of regime_label -> strategy name (defaults to the tech mapping).",
    )
    parser.add_argument("--data-file", default=None, help="Override data file path.")
    parser.add_argument("--date", type=str, help="Override target date (YYYY-MM-DD). Defaults to last date in data.")
    parser.add_argument("--skip-update", action="store_true", help="Skip market data update.")
    parser.add_argument(
        "--market-data-source",
        choices=("auto", "tradingview", "kite_ohlc"),
        default=(
            config.MARKET_DATA_SOURCE
            if config.MARKET_DATA_SOURCE in {"auto", "tradingview", "kite_ohlc"}
            else "auto"
        ),
        help="Market data provider for incremental updates (default: auto).",
    )
    parser.add_argument("--lookback-days", type=int, default=420, help="Days of lookback for incremental update.")
    parser.add_argument("--rolling-window", type=int, default=None, help="Rolling window for ADV/vol/VIX z.")
    parser.add_argument(
        "--interval",
        default="1d",
        help="TradingView interval (used when --market-data-source=tradingview).",
    )
    parser.add_argument(
        "--vix-ticker",
        default=config.DEFAULT_VIX_TICKER,
        help="VIX symbol (TradingView format for tradingview source, exchange:symbol for Kite).",
    )
    parser.add_argument("--tv-screener", default="india", help="TradingView screener (default: india).")
    parser.add_argument(
        "--tv-exchanges",
        default="NSE,BSE",
        help="Comma-separated exchange fallback list (default: NSE,BSE).",
    )
    parser.add_argument("--tv-timeout", type=float, default=None, help="TradingView request timeout in seconds.")
    parser.add_argument(
        "--kite-quote-batch-size",
        type=int,
        default=config.KITE_QUOTE_BATCH_SIZE,
        help="Kite /quote/ohlc batch size (default: 1000).",
    )
    parser.add_argument(
        "--kite-quote-max-batches",
        type=int,
        default=config.KITE_QUOTE_MAX_BATCHES,
        help="Optional cap on Kite /quote/ohlc batches; 0 disables the cap.",
    )
    parser.add_argument("--add-cash", type=float, default=0.0, help="Add cash to the portfolio before trading.")
    parser.add_argument("--cash-note", default="manual", help="Optional note for cash injections.")
    parser.add_argument(
        "--add-tickers",
        default=None,
        help="Comma-separated tickers to add via yfinance history.",
    )
    parser.add_argument(
        "--add-tickers-file",
        default=None,
        help="Path to newline-delimited tickers to add via yfinance history.",
    )
    parser.add_argument("--history-period", default="20y", help="yfinance period for new tickers (default: 20y).")
    parser.add_argument("--history-interval", default="1d", help="yfinance interval for new tickers (default: 1d).")
    parser.add_argument(
        "--history-vix-ticker",
        default=config.DEFAULT_HISTORY_VIX_TICKER,
        help="yfinance VIX symbol used for history backfill.",
    )
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=50,
        help="Minimum history days required for new tickers.",
    )
    parser.add_argument(
        "--confirm-days",
        type=int,
        default=config.CONFIRM_DAYS,
        help="Required consecutive days before switching non-sideways regimes.",
    )
    parser.add_argument(
        "--confirm-days-sideways",
        type=int,
        default=config.CONFIRM_DAYS_SIDEWAYS,
        help="Required consecutive days before switching into sideways regimes.",
    )
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=config.REBALANCE_EVERY,
        help="Rebalance interval in trading days.",
    )
    parser.add_argument(
        "--min-weight-change",
        type=float,
        default=config.MIN_WEIGHT_CHANGE_TO_TRADE,
        help="Minimum per-asset absolute target weight change required to trade.",
    )
    parser.add_argument(
        "--min-trade-dollars",
        type=float,
        default=config.MIN_TRADE_DOLLARS,
        help="Minimum absolute trade notional required to execute a trade.",
    )
    parser.add_argument(
        "--max-daily-turnover",
        type=float,
        default=config.MAX_DAILY_TURNOVER,
        help="Optional cap on daily turnover as sum(abs(weight changes)); <=0 disables.",
    )
    parser.add_argument(
        "--weight-smoothing",
        type=float,
        default=config.WEIGHT_SMOOTHING,
        help="Blend factor for previous weights vs fresh weights (0 = none).",
    )
    parser.add_argument(
        "--skip-pending",
        action="store_true",
        help="Ignore pending adjustments table for this run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate trades without updating state.")
    parser.add_argument("--force", action="store_true", help="Run even if state already has the target date.")
    parser.add_argument("--print-trades", action="store_true", help="Print trades to stdout.")
    parser.add_argument("--max-print", type=int, default=100, help="Max trades to print.")
    parser.add_argument("--sector", default="Technology", help="Sector filter (default: Technology).")
    parser.add_argument(
        "--regime-scope",
        choices=("global", "sector"),
        default="global",
        help="Regime table scope for sector runs (default: global).",
    )
    parser.add_argument(
        "--allow-partial-updates",
        action="store_true",
        help="Allow data updates if some tickers fail source fetch.",
    )
    return parser.parse_args()


def _resolve_path(path: str) -> str:
    resolved = config.resolve_path(path)
    if os.path.isabs(resolved):
        return resolved
    return os.path.join(PROJECT_ROOT, resolved)


def _normalize_exchange_map(mapping: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for ticker, exchange in mapping.items():
        t = str(ticker).strip().upper()
        ex = str(exchange).strip().upper()
        if not t:
            continue
        normalized[t] = ex or "UNKNOWN"
    return normalized


def _build_retry_pending_ticker_entries(
    pending_entries: list[dict],
    retry_tickers: set[str],
) -> list[dict]:
    if not retry_tickers:
        return []
    retry_entries: list[dict] = []
    for entry in pending_entries:
        if str(entry.get("type", "")).strip().lower() != "tickers":
            continue
        tickers = entry.get("tickers", [])
        if isinstance(tickers, str):
            tickers = [tickers]
        selected_tickers = sorted(
            {
                str(ticker).strip().upper()
                for ticker in (tickers or [])
                if str(ticker).strip().upper() in retry_tickers
            }
        )
        if not selected_tickers:
            continue
        retry_entry = dict(entry)
        retry_entry["tickers"] = selected_tickers
        exchanges_raw = entry.get("exchanges") or entry.get("ticker_exchanges") or {}
        filtered_exchanges: dict[str, str] = {}
        if isinstance(exchanges_raw, dict):
            for ticker, exchange in exchanges_raw.items():
                ticker_key = str(ticker).strip().upper()
                if ticker_key not in selected_tickers:
                    continue
                exchange_key = str(exchange).strip().upper()
                filtered_exchanges[ticker_key] = exchange_key or "UNKNOWN"
        elif isinstance(exchanges_raw, list):
            for item in exchanges_raw:
                if not isinstance(item, dict):
                    continue
                ticker_key = str(item.get("ticker", "")).strip().upper()
                if ticker_key not in selected_tickers:
                    continue
                exchange_key = str(item.get("exchange", "")).strip().upper()
                filtered_exchanges[ticker_key] = exchange_key or "UNKNOWN"
        retry_entry.pop("ticker_exchanges", None)
        if filtered_exchanges:
            retry_entry["exchanges"] = filtered_exchanges
        else:
            retry_entry.pop("exchanges", None)
        retry_entries.append(retry_entry)
    return retry_entries


def _apply_universe_filter_with_exclusions(
    df: pd.DataFrame,
    excluded: set[str],
) -> pd.DataFrame:
    universe_filter = config.UNIVERSE_FILTER
    if not universe_filter or universe_filter in ("all", "none"):
        filtered = df
    else:
        allowed = {t.strip().upper() for t in universe_filter.split(",") if t.strip()}
        filtered = df[df.index.get_level_values("ticker").isin(allowed)]
    if excluded:
        filtered = filtered[~filtered.index.get_level_values("ticker").isin(excluded)]
    filtered, _ = apply_quality_filter(filtered)
    return filtered


def _build_price_snapshot(df: pd.DataFrame, target_date: pd.Timestamp) -> list[dict[str, float]]:
    if df is None or df.empty or "Close" not in df.columns:
        return []
    date_key = pd.to_datetime(target_date).tz_localize(None)
    try:
        day_data = df.xs(date_key, level="date")
    except KeyError:
        return []
    if day_data.empty:
        return []
    close = pd.to_numeric(day_data["Close"], errors="coerce")
    tickers = day_data.index.astype(str).str.strip().str.upper()
    valid_mask = close.notna() & (tickers != "")
    if not valid_mask.any():
        return []
    valid_tickers = tickers[valid_mask].to_numpy()
    valid_prices = close[valid_mask].astype(float).to_numpy()
    return [
        {"ticker": ticker, "close_price": float(price)}
        for ticker, price in zip(valid_tickers, valid_prices)
    ]


def _price_lookup_from_snapshot(snapshot: list[dict[str, float]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for item in snapshot:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        price = _safe_float(item.get("close_price"), default=0.0)
        if price <= 0:
            continue
        lookup[ticker] = price
    return lookup


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _extract_order_id(order_response: dict) -> int | str | None:
    direct = order_response.get("id")
    if direct is not None and str(direct).strip():
        return direct
    for key in ("orderId", "order_id"):
        value = order_response.get(key)
        if value is not None and str(value).strip():
            return value
    nested = order_response.get("order")
    if isinstance(nested, dict):
        nested_id = nested.get("id")
        if nested_id is not None and str(nested_id).strip():
            return nested_id
    return None


def _position_quantity_for_ticker(positions: list[dict], ticker: str) -> float:
    target = str(ticker or "").strip().upper()
    if not target:
        return 0.0
    total = 0.0
    for pos in positions:
        instrument = pos.get("instrument") if isinstance(pos, dict) else {}
        instrument = instrument if isinstance(instrument, dict) else {}
        pos_ticker = str(pos.get("ticker") or instrument.get("ticker") or "").strip().upper()
        if pos_ticker != target:
            continue
        total += _safe_float(pos.get("quantity"))
    return total


def _history_order_id(item: dict) -> str | None:
    order = item.get("order")
    if isinstance(order, dict):
        order_id = order.get("id")
        if order_id is not None and str(order_id).strip():
            return str(order_id).strip()
    for key in ("orderId", "order_id", "id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _history_status(item: dict) -> str:
    order = item.get("order")
    if isinstance(order, dict):
        status = str(order.get("status") or "").strip().upper()
        if status:
            return status
    status = str(item.get("status") or "").strip().upper()
    if status:
        return status
    fill = item.get("fill")
    if isinstance(fill, dict) and _safe_float(fill.get("quantity")) > 0:
        return "FILLED"
    return "UNKNOWN"


def _history_fill_snapshot(item: dict) -> dict:
    fill = item.get("fill")
    fill = fill if isinstance(fill, dict) else {}
    order = item.get("order")
    order = order if isinstance(order, dict) else {}
    filled_qty = abs(_safe_float(fill.get("quantity")))
    if filled_qty <= 0:
        filled_qty = abs(_safe_float(item.get("filledQuantity")))
    if filled_qty <= 0:
        filled_qty = abs(_safe_float(order.get("filledQuantity")))

    fill_price = abs(_safe_float(fill.get("price")))
    if fill_price <= 0:
        fill_price = abs(_safe_float(item.get("fillPrice")))
    if fill_price <= 0:
        fill_price = abs(_safe_float(item.get("executedPrice")))
    if fill_price <= 0:
        fill_price = abs(_safe_float(order.get("price")))

    fill_value = _safe_float(fill.get("value"))
    if fill_value <= 0:
        fill_value = _safe_float(item.get("filledValue"))
    if fill_value <= 0:
        fill_value = _safe_float(order.get("filledValue"))
    fill_value = abs(fill_value)
    if fill_value <= 0 and filled_qty > 0 and fill_price > 0:
        fill_value = filled_qty * fill_price
    payload: dict = {
        "status": _history_status(item),
        "historyStatusSource": "history.orders",
    }
    if filled_qty > 0:
        payload["filledQuantity"] = filled_qty
    if fill_value > 0:
        payload["filledValue"] = fill_value
    if fill_price > 0:
        payload["fillPrice"] = fill_price
        payload["resolutionPriceSource"] = "history.orders.fill.price"
    currency = (
        fill.get("currencyCode")
        or fill.get("currency")
        or item.get("currencyCode")
        or item.get("currency")
        or order.get("currencyCode")
        or order.get("currency")
    )
    if currency:
        currency_code = str(currency).strip().upper()
        if currency_code:
            payload["currency"] = currency_code
            payload["filledValueCurrency"] = currency_code
            payload["fillPriceCurrency"] = currency_code
    return payload


def _next_history_cursor(page: dict) -> str | None:
    direct_keys = ("nextCursor", "nextPageCursor")
    for key in direct_keys:
        value = page.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    pagination = page.get("pagination")
    if isinstance(pagination, dict):
        for key in direct_keys:
            value = pagination.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    for key in ("nextPagePath", "next"):
        path = page.get(key)
        if not path:
            continue
        parsed = urlparse(str(path))
        query = parse_qs(parsed.query)
        for query_key in ("cursor", "nextCursor", "pageToken"):
            values = query.get(query_key)
            if values and str(values[0]).strip():
                return str(values[0]).strip()
    return None


def _load_history_order_updates(
    client: Trading212Client,
    order_ids: list[str],
    page_limit: int = 50,
    max_pages: int = 1,
) -> dict[str, dict]:
    target_ids = {str(order_id).strip() for order_id in order_ids if str(order_id).strip()}
    if not target_ids:
        return {}
    updates: dict[str, dict] = {}
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for page_idx in range(1, max_pages + 1):
        if len(updates) >= len(target_ids):
            break
        page = client.get_historical_orders(limit=page_limit, cursor=cursor)
        items = page.get("items")
        if not isinstance(items, list):
            items = []
        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            order_id = _history_order_id(raw_item)
            if not order_id or order_id not in target_ids:
                continue
            updates[order_id] = _history_fill_snapshot(raw_item)
        _log(
            "broker_history_orders_page",
            page=page_idx,
            requested=len(target_ids),
            found=len(updates),
            items=len(items),
        )
        cursor = _next_history_cursor(page)
        if not cursor:
            break
        if cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
    return updates


def _monitor_orders_with_positions_and_history(
    client: Trading212Client,
    placed: list[dict],
    phase: str,
    phase_start_qty_by_ticker: dict[str, float],
) -> dict[str, dict]:
    tracked: dict[str, dict] = {
        str(item["order_id"]): item for item in placed if str(item.get("order_id") or "").strip()
    }
    snapshots: dict[str, dict] = {
        order_id: {"id": order_id, "status": "PENDING"} for order_id in tracked
    }
    unresolved: set[str] = set(tracked.keys())
    if not unresolved:
        return snapshots

    timeout_sec = max(1.0, float(config.TRADING212_ORDER_TIMEOUT))
    positions_poll_sec = max(1.0, float(getattr(config, "TRADING212_POSITIONS_POLL_SEC", 1.0)))
    history_poll_sec = max(10.0, float(getattr(config, "TRADING212_HISTORY_POLL_SEC", 10.0)))
    history_page_limit = max(1, int(getattr(config, "TRADING212_HISTORY_PAGE_LIMIT", 50)))
    history_max_pages = max(1, int(getattr(config, "TRADING212_HISTORY_MAX_PAGES", 1)))

    by_t212_ticker: dict[str, list[dict]] = {}
    for item in placed:
        t212_ticker = str(item.get("t212_ticker") or "").strip().upper()
        if t212_ticker:
            by_t212_ticker.setdefault(t212_ticker, []).append(item)
    expected_post_qty: dict[str, float] = {}
    for t212_ticker, ticker_items in by_t212_ticker.items():
        expected_delta = sum(
            _expected_position_delta(str(item.get("action") or ""), _safe_float(item.get("shares")))
            for item in ticker_items
        )
        expected_post_qty[t212_ticker] = _safe_float(phase_start_qty_by_ticker.get(t212_ticker)) + expected_delta

    _log(
        "broker_phase_monitor_mode",
        phase=phase,
        mode="positions_plus_history",
        positions_poll_sec=positions_poll_sec,
        history_poll_sec=history_poll_sec,
        history_page_limit=history_page_limit,
    )
    deadline = time.time() + timeout_sec
    next_positions_at = 0.0
    next_history_at = 0.0

    while time.time() <= deadline and unresolved:
        now = time.time()
        did_work = False

        if now >= next_positions_at:
            positions = client.get_positions()
            quantities = _position_quantities_by_ticker(positions)
            for t212_ticker, ticker_items in by_t212_ticker.items():
                remaining_items = [
                    item for item in ticker_items if str(item.get("order_id") or "").strip() in unresolved
                ]
                if not remaining_items:
                    continue
                pre_qty = _safe_float(phase_start_qty_by_ticker.get(t212_ticker))
                post_qty = _safe_float(quantities.get(t212_ticker))
                expected_qty = _safe_float(expected_post_qty.get(t212_ticker))
                _log(
                    "broker_position_reconcile_result",
                    phase=phase,
                    t212_ticker=t212_ticker,
                    post_qty=post_qty,
                    expected_post_qty=expected_qty,
                    unresolved_orders=len(remaining_items),
                )
                if not _float_close(post_qty, expected_qty, tol=1e-4):
                    continue
                for item in remaining_items:
                    order_id = str(item["order_id"])
                    shares = _safe_float(item.get("shares"))
                    payload = dict(snapshots.get(order_id) or {"id": order_id})
                    payload["status"] = "FILLED"
                    payload["filledQuantity"] = abs(shares)
                    payload["resolution"] = "reconciled_via_positions_endpoint"
                    payload["positionQuantityBefore"] = pre_qty
                    payload["positionQuantityAfter"] = post_qty
                    payload["positionQuantityExpected"] = expected_qty
                    snapshots[order_id] = payload
                    unresolved.discard(order_id)
                    _log(
                        "broker_position_reconcile_applied",
                        phase=phase,
                        ticker=item["ticker"],
                        order_id=order_id,
                        t212_ticker=t212_ticker,
                    )
            next_positions_at = now + positions_poll_sec
            did_work = True

        if unresolved and now >= next_history_at:
            updates = _load_history_order_updates(
                client=client,
                order_ids=sorted(unresolved),
                page_limit=history_page_limit,
                max_pages=history_max_pages,
            )
            for order_id, patch in updates.items():
                current = dict(snapshots.get(order_id) or {"id": order_id})
                for key, value in patch.items():
                    if key == "status":
                        current_status = str(current.get("status") or "").strip().upper()
                        if current_status == "FILLED" and value != "FILLED":
                            continue
                    current[key] = value
                if "resolution" not in current:
                    current["resolution"] = "resolved_via_history_orders_endpoint"
                snapshots[order_id] = current
                status = str(current.get("status") or "").upper()
                if status in {"FILLED", "REJECTED", "CANCELLED"}:
                    unresolved.discard(order_id)
                    _log(
                        "broker_history_order_resolved",
                        phase=phase,
                        order_id=order_id,
                        status=status,
                    )
            next_history_at = now + history_poll_sec
            did_work = True

        if unresolved:
            if not did_work:
                next_tick = min(next_positions_at, next_history_at)
                sleep_for = max(0.05, min(0.5, next_tick - time.time()))
                time.sleep(sleep_for)
            continue

    history_price_targets = [
        order_id
        for order_id, payload in snapshots.items()
        if str(payload.get("status") or "").upper() == "FILLED"
        and _safe_float(payload.get("filledQuantity")) > 0
        and _safe_float(payload.get("filledValue")) <= 0
    ]
    if history_price_targets:
        updates = _load_history_order_updates(
            client=client,
            order_ids=history_price_targets,
            page_limit=history_page_limit,
            max_pages=history_max_pages,
        )
        for order_id, patch in updates.items():
            current = dict(snapshots.get(order_id) or {"id": order_id})
            current_filled_value = abs(_safe_float(current.get("filledValue")))
            current_fill_price = abs(_safe_float(current.get("fillPrice")))
            patch_filled_value = abs(_safe_float(patch.get("filledValue")))
            patch_fill_price = abs(_safe_float(patch.get("fillPrice")))
            if current_filled_value > 0 and current_fill_price > 0:
                continue
            if patch_filled_value <= 0 and patch_fill_price <= 0:
                continue
            if patch_filled_value > 0 and current_filled_value <= 0:
                current["filledValue"] = patch["filledValue"]
            if patch_fill_price > 0 and current_fill_price <= 0:
                current["fillPrice"] = patch["fillPrice"]
            for key in ("resolutionPriceSource", "currency", "filledValueCurrency", "fillPriceCurrency"):
                if patch.get(key):
                    current[key] = patch[key]
            snapshots[order_id] = current
            _log(
                "broker_history_fill_price_applied",
                phase=phase,
                order_id=order_id,
                filled_qty=_safe_float(current.get("filledQuantity")),
                filled_value=_safe_float(current.get("filledValue")),
                fill_price=_safe_float(current.get("fillPrice")),
            )

    return snapshots


def _position_quantities_by_ticker(positions: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        instrument = pos.get("instrument")
        instrument = instrument if isinstance(instrument, dict) else {}
        ticker = str(pos.get("ticker") or instrument.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        totals[ticker] = totals.get(ticker, 0.0) + _safe_float(pos.get("quantity"))
    return totals


def _expected_position_delta(action: str, shares: float) -> float:
    action_upper = str(action or "").strip().upper()
    if action_upper == "SELL" or shares < 0:
        return -abs(_safe_float(shares))
    return abs(_safe_float(shares))


def _active_broker_name() -> str:
    kite_on = kite_enabled()
    t212_on = trading212_enabled()
    if kite_on and t212_on:
        raise ValueError(
            "Both Kite and Trading212 broker integrations are enabled. "
            "Enable only one broker (USE_KITE or USE_TRADING212)."
        )
    if kite_on:
        return "kite"
    if t212_on:
        return "trading212"
    return ""


def _build_trading212_context(state: ProductionState) -> dict:
    _log("broker_context_build_start", broker="trading212")
    client = Trading212Client()
    instruments = load_t212_instruments_cache(client)
    overrides = load_t212_ticker_overrides()
    by_ticker, by_symbol = build_t212_instrument_index(instruments)
    _log(
        "broker_instruments_loaded",
        instruments=len(instruments),
        overrides=len(overrides),
        symbols=len(by_symbol),
    )

    summary_raw = client.get_account_summary()
    positions_raw = client.get_positions()
    _log("broker_snapshot_loaded", positions=len(positions_raw))

    account_currency = str(
        summary_raw.get("currencyCode") or summary_raw.get("currency") or "GBP"
    ).upper()
    broker_cash_account = t212_account_cash_available(summary_raw)
    broker_net_worth_account = t212_account_net_worth(summary_raw)
    broker_positions = t212_positions_to_internal_positions(positions_raw, overrides)
    fx_rates = extract_fx_rates(positions_raw, account_currency=account_currency)
    fx_rate = fx_rates.get("USD") or config.TRADING212_FX_RATE_USD_GBP
    if account_currency == "USD":
        fx_rate = 1.0
    if account_currency not in {"USD", "GBP"}:
        raise ValueError(
            f"Unsupported Trading212 account currency '{account_currency}'. Only USD/GBP are supported."
        )
    if not fx_rate:
        raise ValueError(
            "Unable to derive GBP/USD FX rate for Trading212. "
            "Set TRADING212_FX_RATE_USD_GBP to proceed."
        )
    broker_cash_usd = _convert_between_usd_gbp(
        amount=broker_cash_account,
        source_currency=account_currency,
        target_currency="USD",
        fx_rate_gbp_per_usd=fx_rate,
    )
    if broker_cash_usd is None:
        raise ValueError(
            "Unable to convert Trading212 cash balance to USD. "
            f"account_currency={account_currency} fx_rate_gbp_per_usd={fx_rate}"
        )
    broker_net_worth_usd = _convert_between_usd_gbp(
        amount=broker_net_worth_account,
        source_currency=account_currency,
        target_currency="USD",
        fx_rate_gbp_per_usd=fx_rate,
    )
    discrepancies = compare_positions(
        state.positions,
        broker_positions,
        state.cash,
        float(broker_cash_usd),
    )
    _log(
        "broker_context_build_complete",
        account_currency=account_currency,
        broker_positions=len(broker_positions),
        fx_rate_gbp_per_usd=fx_rate,
    )
    return {
        "broker": "trading212",
        "client": client,
        "instruments": instruments,
        "overrides": overrides,
        "by_ticker": by_ticker,
        "by_symbol": by_symbol,
        "summary_raw": summary_raw,
        "positions_raw": positions_raw,
        "account_currency": account_currency,
        "broker_cash": broker_cash_account,
        "broker_cash_gbp": broker_cash_account,
        "broker_cash_usd": float(broker_cash_usd),
        "broker_net_worth": broker_net_worth_account,
        "broker_net_worth_gbp": broker_net_worth_account,
        "broker_net_worth_usd": float(broker_net_worth_usd) if broker_net_worth_usd is not None else None,
        "broker_positions": broker_positions,
        "fx_rates": fx_rates,
        "fx_rate_gbp_per_usd": fx_rate,
        "discrepancies": discrepancies,
    }


def _build_kite_context(state: ProductionState) -> dict:
    _log("broker_context_build_start", broker="kite")
    client = KiteClient()
    try:
        instruments = load_kite_instruments_cache(client)
    except Exception as exc:
        instruments = []
        _log(
            "broker_instruments_load_failed",
            broker="kite",
            error=str(exc),
            error_type=type(exc).__name__,
        )
    overrides = load_kite_ticker_overrides()
    by_ticker, by_symbol = build_kite_instrument_index(instruments)
    _log(
        "broker_instruments_loaded",
        broker="kite",
        instruments=len(instruments),
        overrides=len(overrides),
        symbols=len(by_symbol),
    )

    margins_raw = client.get_margins()
    holdings_raw = client.get_holdings()
    positions_payload = client.get_positions()
    _log("broker_snapshot_loaded", broker="kite", positions=len(holdings_raw))

    account_currency = "INR"
    broker_cash_account = kite_account_cash_available(margins_raw)
    broker_net_worth_account = kite_account_net_worth(margins_raw, holdings_raw)
    broker_positions = kite_positions_to_internal_positions(holdings_raw, overrides)
    discrepancies = compare_positions(
        state.positions,
        broker_positions,
        state.cash,
        float(broker_cash_account),
    )
    _log(
        "broker_context_build_complete",
        broker="kite",
        account_currency=account_currency,
        broker_positions=len(broker_positions),
    )
    return {
        "broker": "kite",
        "client": client,
        "instruments": instruments,
        "overrides": overrides,
        "by_ticker": by_ticker,
        "by_symbol": by_symbol,
        "summary_raw": margins_raw,
        "positions_raw": holdings_raw,
        "positions_payload_raw": positions_payload,
        "account_currency": account_currency,
        "broker_cash": broker_cash_account,
        "broker_cash_usd": float(broker_cash_account),
        "broker_net_worth": broker_net_worth_account,
        "broker_net_worth_usd": None,
        "broker_positions": broker_positions,
        "fx_rates": {},
        "fx_rate_gbp_per_usd": None,
        "discrepancies": discrepancies,
    }


def _state_from_broker_context(
    state: ProductionState,
    context: dict,
    price_lookup: dict[str, float] | None = None,
) -> ProductionState:
    broker_positions_raw = context.get("broker_positions")
    broker_positions: dict[str, int | float] = {}
    fractional_tickers: list[str] = []
    if isinstance(broker_positions_raw, dict):
        for ticker, quantity in broker_positions_raw.items():
            ticker_key = str(ticker or "").strip().upper()
            if not ticker_key:
                continue
            qty = _safe_float(quantity)
            if abs(qty) <= 1e-9:
                continue
            if abs(qty - round(qty)) > 1e-6:
                fractional_tickers.append(ticker_key)
            broker_positions[ticker_key] = qty
    if fractional_tickers:
        preview = ", ".join(sorted(set(fractional_tickers))[:10])
        _log(
            "broker_fractional_positions_detected",
            count=len(set(fractional_tickers)),
            preview=preview,
        )
    broker_cash_usd = _safe_float(context.get("broker_cash_usd"), default=state.cash)
    prev_weights: dict[str, float] = {}
    if price_lookup:
        priced_values: dict[str, float] = {}
        missing_price_tickers: list[str] = []
        for ticker, quantity in broker_positions.items():
            price = _safe_float(price_lookup.get(ticker), default=0.0)
            if price <= 0:
                missing_price_tickers.append(ticker)
                continue
            priced_values[ticker] = _safe_float(quantity) * price
        if missing_price_tickers:
            preview = ", ".join(sorted(set(missing_price_tickers))[:10])
            _log(
                "broker_prev_weights_missing_prices",
                count=len(set(missing_price_tickers)),
                preview=preview,
            )
        net_worth = broker_cash_usd + float(sum(priced_values.values()))
        if net_worth > 1e-9:
            prev_weights = {
                ticker: float(value / net_worth)
                for ticker, value in priced_values.items()
                if abs(value / net_worth) > 1e-9
            }
            _log(
                "broker_prev_weights_computed",
                weights=len(prev_weights),
                net_worth_usd=net_worth,
            )
        else:
            _log("broker_prev_weights_skipped", reason="non_positive_net_worth")
    else:
        _log("broker_prev_weights_skipped", reason="missing_price_lookup")
    return ProductionState(
        last_date=state.last_date,
        cash=broker_cash_usd,
        positions=broker_positions,
        prev_weights=prev_weights,
        total_costs_usd=state.total_costs_usd,
        rebalance_day_index=state.rebalance_day_index,
        initial_deploy_completed=state.initial_deploy_completed,
    )


def _state_from_broker_snapshot(
    new_state: ProductionState,
    summary: dict,
    context: dict,
    post_positions: list[dict],
) -> ProductionState:
    broker = str(context.get("broker") or "").strip().lower()
    if broker == "kite":
        broker_positions_internal = kite_positions_to_internal_positions(
            post_positions,
            context.get("overrides") or {},
        )
    else:
        broker_positions_internal = t212_positions_to_internal_positions(
            post_positions,
            context.get("overrides") or {},
        )
    broker_currency = str(summary.get("broker_currency") or "").strip().upper()
    fx_rate = _safe_float(summary.get("broker_fx_rate_gbp_per_usd"), default=0.0)
    broker_cash_after = _safe_float(summary.get("broker_cash_after"), default=new_state.cash)

    synced_cash_usd = new_state.cash
    if broker_currency == "USD":
        synced_cash_usd = broker_cash_after
    elif broker_currency == "GBP" and fx_rate > 0:
        synced_cash_usd = broker_cash_after / fx_rate
    else:
        synced_cash_usd = broker_cash_after

    return ProductionState(
        last_date=new_state.last_date,
        cash=synced_cash_usd,
        positions=broker_positions_internal,
        prev_weights=dict(new_state.prev_weights),
        total_costs_usd=new_state.total_costs_usd,
        rebalance_day_index=new_state.rebalance_day_index,
        initial_deploy_completed=new_state.initial_deploy_completed,
    )


def _ensure_trading212_universe(
    df: pd.DataFrame,
    by_ticker: dict[str, dict],
    by_symbol: dict[str, list[dict]],
    overrides: dict[str, str],
    preferred_currency: str | None = None,
) -> dict[str, str]:
    tickers = sorted(set(df.index.get_level_values("ticker").unique()))
    _log("broker_universe_validation_start", candidate_tickers=len(tickers))
    missing: list[str] = []
    mapping: dict[str, str] = {}
    for ticker in tickers:
        mapped = resolve_t212_ticker(
            ticker,
            by_symbol,
            overrides,
            preferred_currency=preferred_currency,
            by_ticker=by_ticker,
        )
        if not mapped or mapped not in by_ticker:
            missing.append(ticker)
        else:
            mapping[ticker] = mapped
    if missing:
        preview = ", ".join(missing[:10])
        _log("broker_universe_validation_failed", missing=len(missing), preview=preview)
        raise ValueError(
            "Missing Trading212 mappings for tickers. "
            f"Missing {len(missing)} (first 10): {preview}"
        )
    _log("broker_universe_validation_complete", mapped=len(mapping))
    return mapping


def _ensure_kite_universe(
    df: pd.DataFrame,
    by_ticker: dict[str, dict],
    by_symbol: dict[str, list[dict]],
    overrides: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    tickers = sorted(set(df.index.get_level_values("ticker").unique()))
    _log("broker_universe_validation_start", broker="kite", candidate_tickers=len(tickers))
    missing: list[str] = []
    mapping: dict[str, tuple[str, str]] = {}
    index_by_ticker = by_ticker if by_ticker else None
    index_by_symbol = by_symbol if by_symbol else None
    for ticker in tickers:
        mapped = resolve_kite_instrument(
            ticker,
            overrides,
            by_key=index_by_ticker,
            by_symbol=index_by_symbol,
        )
        if not mapped:
            missing.append(ticker)
            continue
        mapping[ticker] = mapped
    if missing:
        preview = ", ".join(missing[:10])
        _log(
            "broker_universe_validation_failed",
            broker="kite",
            missing=len(missing),
            preview=preview,
        )
        raise ValueError(
            "Missing Kite mappings for tickers. "
            f"Missing {len(missing)} (first 10): {preview}"
        )
    _log("broker_universe_validation_complete", broker="kite", mapped=len(mapping))
    return mapping


def _execute_trading212_orders(
    trades: list[dict],
    context: dict,
    dry_run: bool,
) -> tuple[list[dict], list[str], list[dict]]:
    client: Trading212Client = context["client"]
    by_ticker = context["by_ticker"]
    by_symbol = context["by_symbol"]
    overrides = context["overrides"]
    missing: list[str] = []
    orders: list[dict] = []
    issues: list[dict] = []
    mapped_trades: list[dict] = []
    _log("broker_order_execution_start", trades=len(trades), dry_run=bool(dry_run), phases=["SELL", "BUY"])
    for trade in trades:
        shares = _safe_float(trade.get("shares"))
        if abs(shares) <= 1e-6:
            continue
        ticker = str(trade.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        action = str(trade.get("action") or "").strip().upper()
        phase = "SELL" if action == "SELL" or shares < 0 else "BUY"
        t212_ticker = resolve_t212_ticker(ticker, by_symbol, overrides, by_ticker=by_ticker)
        if not t212_ticker:
            missing.append(ticker)
            _log("broker_order_mapping_missing", ticker=ticker)
            continue
        mapped_trades.append(
            {
                "ticker": ticker,
                "action": action or ("SELL" if shares < 0 else "BUY"),
                "shares": shares,
                "phase": phase,
                "t212_ticker": t212_ticker,
            }
        )
        if dry_run:
            _log("broker_order_dry_run_skip", ticker=ticker, quantity=shares, t212_ticker=t212_ticker)
    if dry_run:
        _log("broker_order_execution_complete", orders=0, missing_mappings=len(missing), issues=0)
        return orders, missing, issues

    halt_execution = False
    position_qty_by_ticker = _position_quantities_by_ticker(context.get("positions_raw") or [])
    for phase in ("SELL", "BUY"):
        if halt_execution:
            break
        phase_trades = [item for item in mapped_trades if item["phase"] == phase]
        if not phase_trades:
            continue
        phase_start_qty_by_ticker = dict(position_qty_by_ticker)
        _log("broker_phase_place_start", phase=phase, orders=len(phase_trades))
        placed: list[dict] = []
        for item in phase_trades:
            ticker = item["ticker"]
            shares = _safe_float(item["shares"])
            t212_ticker = str(item["t212_ticker"])
            try:
                _log("broker_order_place_start", phase=phase, ticker=ticker, quantity=shares, t212_ticker=t212_ticker)
                order_resp = client.place_market_order(
                    ticker=t212_ticker,
                    quantity=shares,
                )
                order_id = _extract_order_id(order_resp)
                _log(
                    "broker_order_place_response",
                    phase=phase,
                    ticker=ticker,
                    order_id=order_id,
                    response_status=order_resp.get("status") if isinstance(order_resp, dict) else None,
                    response_keys=sorted(order_resp.keys()) if isinstance(order_resp, dict) else [],
                )
                if order_id is None:
                    issue = {
                        "phase": phase,
                        "ticker": ticker,
                        "error": "missing_order_id",
                        "details": order_resp,
                    }
                    issues.append(issue)
                    orders.append(
                        {
                            "ticker": ticker,
                            "action": item["action"],
                            "quantity": shares,
                            "filled_quantity": 0.0,
                            "exec_price": None,
                            "currency": context.get("account_currency"),
                            "status": "FAILED",
                            "order_id": None,
                            "payload": {"issue": issue, "response": order_resp},
                        }
                    )
                    _log("broker_order_issue", phase=phase, ticker=ticker, error="missing_order_id")
                    halt_execution = True
                    break
                placed.append(
                    {
                        "phase": phase,
                        "ticker": ticker,
                        "action": item["action"],
                        "shares": shares,
                        "t212_ticker": t212_ticker,
                        "order_id": str(order_id),
                    }
                )
            except Exception as exc:
                issue = {
                    "phase": phase,
                    "ticker": ticker,
                    "error": "order_execution_error",
                    "details": str(exc),
                }
                issues.append(issue)
                _log(
                    "broker_order_exception",
                    phase=phase,
                    ticker=ticker,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                orders.append(
                    {
                        "ticker": ticker,
                        "action": item["action"],
                        "quantity": shares,
                        "filled_quantity": 0.0,
                        "exec_price": None,
                        "currency": context.get("account_currency"),
                        "status": "FAILED",
                        "order_id": None,
                        "payload": {"issue": issue},
                    }
                )
                halt_execution = True
                break
        if halt_execution:
            break
        if not placed:
            continue

        order_ids = [item["order_id"] for item in placed]
        _log("broker_phase_monitor_start", phase=phase, orders=len(order_ids), order_ids=order_ids)
        try:
            bulk_status = _monitor_orders_with_positions_and_history(
                client=client,
                placed=placed,
                phase=phase,
                phase_start_qty_by_ticker=phase_start_qty_by_ticker,
            )
        except Exception as exc:
            issue = {
                "phase": phase,
                "error": "order_monitor_error",
                "details": str(exc),
                "order_ids": order_ids,
            }
            issues.append(issue)
            _log(
                "broker_phase_monitor_exception",
                phase=phase,
                error=str(exc),
                error_type=type(exc).__name__,
                orders=len(order_ids),
            )
            for item in placed:
                orders.append(
                    {
                        "ticker": item["ticker"],
                        "action": item["action"],
                        "quantity": item["shares"],
                        "filled_quantity": 0.0,
                        "exec_price": None,
                        "currency": context.get("account_currency"),
                        "status": "UNKNOWN",
                        "order_id": item["order_id"],
                        "payload": {"issue": issue},
                    }
                )
            halt_execution = True
            break

        phase_has_unfilled = False
        for item in placed:
            order_id = item["order_id"]
            filled = bulk_status.get(order_id) or {"id": order_id, "status": "PENDING"}
            status = str(filled.get("status", "")).upper()
            filled_qty = abs(_safe_float(filled.get("filledQuantity")))
            filled_value = abs(_safe_float(filled.get("filledValue")))
            fill_price = abs(_safe_float(filled.get("fillPrice")))
            exec_price = None
            if fill_price > 0:
                exec_price = fill_price
            elif filled_qty > 0 and filled_value > 0:
                exec_price = filled_value / filled_qty
            currency_code = (
                str(
                    filled.get("fillPriceCurrency")
                    or filled.get("currency")
                    or context.get("account_currency")
                    or ""
                )
                .strip()
                .upper()
            )
            orders.append(
                {
                    "ticker": item["ticker"],
                    "action": item["action"],
                    "quantity": item["shares"],
                    "filled_quantity": filled_qty,
                    "exec_price": exec_price,
                    "currency": currency_code or None,
                    "status": status,
                    "order_id": order_id,
                    "payload": filled,
                }
            )
            _log(
                "broker_order_wait_result",
                phase=phase,
                ticker=item["ticker"],
                order_id=order_id,
                status=status,
                filled_qty=filled_qty,
                expected_qty=abs(_safe_float(item["shares"])),
            )
            if status != "FILLED" or not _float_close(filled_qty, abs(_safe_float(item["shares"]))):
                issues.append(
                    {
                        "phase": phase,
                        "ticker": item["ticker"],
                        "order_id": order_id,
                        "status": status,
                        "filled_quantity": filled_qty,
                        "expected_quantity": abs(_safe_float(item["shares"])),
                        "details": "Order not fully filled; halting further order placement for this run.",
                    }
                )
                phase_has_unfilled = True
            else:
                t212_ticker = str(item.get("t212_ticker") or "").strip().upper()
                if t212_ticker:
                    delta = _expected_position_delta(
                        str(item.get("action") or ""),
                        _safe_float(item.get("shares")),
                    )
                    position_qty_by_ticker[t212_ticker] = (
                        _safe_float(position_qty_by_ticker.get(t212_ticker)) + delta
                    )
        if phase_has_unfilled:
            _log("broker_phase_unfilled", phase=phase, details="Halting execution after this phase.")
            halt_execution = True

    _log(
        "broker_order_execution_complete",
        orders=len(orders),
        missing_mappings=len(missing),
        issues=len(issues),
    )
    return orders, missing, issues


def _normalized_kite_quantity(shares: float) -> int:
    return max(0, int(round(abs(_safe_float(shares)))))


def _execute_kite_orders(
    trades: list[dict],
    context: dict,
    dry_run: bool,
) -> tuple[list[dict], list[str], list[dict]]:
    client: KiteClient = context["client"]
    by_ticker = context.get("by_ticker") or {}
    by_symbol = context.get("by_symbol") or {}
    overrides = context.get("overrides") or {}
    missing: list[str] = []
    orders: list[dict] = []
    issues: list[dict] = []
    mapped_trades: list[dict] = []
    _log(
        "broker_order_execution_start",
        broker="kite",
        trades=len(trades),
        dry_run=bool(dry_run),
        phases=["SELL", "BUY"],
    )
    for trade in trades:
        shares = _safe_float(trade.get("shares"))
        if abs(shares) <= 1e-6:
            continue
        ticker = str(trade.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        action = str(trade.get("action") or "").strip().upper()
        phase = "SELL" if action == "SELL" or shares < 0 else "BUY"
        exchange_symbol = resolve_kite_instrument(
            ticker,
            overrides,
            by_key=by_ticker if by_ticker else None,
            by_symbol=by_symbol if by_symbol else None,
        )
        if not exchange_symbol:
            missing.append(ticker)
            _log("broker_order_mapping_missing", broker="kite", ticker=ticker)
            continue
        quantity = _normalized_kite_quantity(shares)
        if quantity <= 0:
            continue
        exchange, tradingsymbol = exchange_symbol
        mapped_trades.append(
            {
                "ticker": ticker,
                "action": action or ("SELL" if shares < 0 else "BUY"),
                "shares": shares,
                "quantity": quantity,
                "phase": phase,
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
            }
        )
        if dry_run:
            _log(
                "broker_order_dry_run_skip",
                broker="kite",
                ticker=ticker,
                quantity=quantity,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
            )
    if dry_run:
        _log(
            "broker_order_execution_complete",
            broker="kite",
            orders=0,
            missing_mappings=len(missing),
            issues=0,
        )
        return orders, missing, issues

    halt_execution = False
    for phase in ("SELL", "BUY"):
        if halt_execution:
            break
        phase_trades = [item for item in mapped_trades if item["phase"] == phase]
        if not phase_trades:
            continue
        _log("broker_phase_place_start", broker="kite", phase=phase, orders=len(phase_trades))
        placed: list[dict] = []
        for item in phase_trades:
            ticker = item["ticker"]
            quantity = int(item["quantity"])
            exchange = str(item["exchange"])
            tradingsymbol = str(item["tradingsymbol"])
            transaction_type = "SELL" if phase == "SELL" else "BUY"
            try:
                _log(
                    "broker_order_place_start",
                    broker="kite",
                    phase=phase,
                    ticker=ticker,
                    quantity=quantity,
                    exchange=exchange,
                    tradingsymbol=tradingsymbol,
                )
                order_resp = client.place_market_order(
                    exchange=exchange,
                    tradingsymbol=tradingsymbol,
                    transaction_type=transaction_type,
                    quantity=quantity,
                )
                order_id = _extract_order_id(order_resp)
                _log(
                    "broker_order_place_response",
                    broker="kite",
                    phase=phase,
                    ticker=ticker,
                    order_id=order_id,
                    response_keys=sorted(order_resp.keys()) if isinstance(order_resp, dict) else [],
                )
                if order_id is None:
                    issue = {
                        "phase": phase,
                        "ticker": ticker,
                        "error": "missing_order_id",
                        "details": order_resp,
                    }
                    issues.append(issue)
                    orders.append(
                        {
                            "ticker": ticker,
                            "action": transaction_type,
                            "quantity": item["shares"],
                            "filled_quantity": 0.0,
                            "exec_price": None,
                            "currency": "INR",
                            "status": "FAILED",
                            "order_id": None,
                            "payload": {"issue": issue, "response": order_resp},
                        }
                    )
                    _log("broker_order_issue", broker="kite", phase=phase, ticker=ticker, error="missing_order_id")
                    halt_execution = True
                    break
                placed.append(
                    {
                        "phase": phase,
                        "ticker": ticker,
                        "action": transaction_type,
                        "shares": item["shares"],
                        "quantity": quantity,
                        "exchange": exchange,
                        "tradingsymbol": tradingsymbol,
                        "order_id": str(order_id),
                    }
                )
            except Exception as exc:
                issue = {
                    "phase": phase,
                    "ticker": ticker,
                    "error": "order_execution_error",
                    "details": str(exc),
                }
                issues.append(issue)
                _log(
                    "broker_order_exception",
                    broker="kite",
                    phase=phase,
                    ticker=ticker,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                orders.append(
                    {
                        "ticker": ticker,
                        "action": transaction_type,
                        "quantity": item["shares"],
                        "filled_quantity": 0.0,
                        "exec_price": None,
                        "currency": "INR",
                        "status": "FAILED",
                        "order_id": None,
                        "payload": {"issue": issue},
                    }
                )
                halt_execution = True
                break
        if halt_execution:
            break
        if not placed:
            continue

        order_ids = [item["order_id"] for item in placed]
        _log("broker_phase_monitor_start", broker="kite", phase=phase, orders=len(order_ids), order_ids=order_ids)
        try:
            bulk_status = client.wait_for_orders(order_ids)
        except Exception as exc:
            issue = {
                "phase": phase,
                "error": "order_monitor_error",
                "details": str(exc),
                "order_ids": order_ids,
            }
            issues.append(issue)
            _log(
                "broker_phase_monitor_exception",
                broker="kite",
                phase=phase,
                error=str(exc),
                error_type=type(exc).__name__,
                orders=len(order_ids),
            )
            for item in placed:
                orders.append(
                    {
                        "ticker": item["ticker"],
                        "action": item["action"],
                        "quantity": item["shares"],
                        "filled_quantity": 0.0,
                        "exec_price": None,
                        "currency": "INR",
                        "status": "UNKNOWN",
                        "order_id": item["order_id"],
                        "payload": {"issue": issue},
                    }
                )
            halt_execution = True
            break

        phase_has_unfilled = False
        for item in placed:
            order_id = item["order_id"]
            filled = bulk_status.get(order_id) or {"id": order_id, "status": "PENDING"}
            status = str(filled.get("status", "")).upper()
            filled_qty = abs(_safe_float(filled.get("filledQuantity")))
            filled_value = abs(_safe_float(filled.get("filledValue")))
            fill_price = abs(_safe_float(filled.get("fillPrice")))
            exec_price = None
            if fill_price > 0:
                exec_price = fill_price
            elif filled_qty > 0 and filled_value > 0:
                exec_price = filled_value / filled_qty
            currency_code = str(filled.get("currency") or "INR").strip().upper() or "INR"
            orders.append(
                {
                    "ticker": item["ticker"],
                    "action": item["action"],
                    "quantity": item["shares"],
                    "filled_quantity": filled_qty,
                    "exec_price": exec_price,
                    "currency": currency_code,
                    "status": status,
                    "order_id": order_id,
                    "payload": filled,
                }
            )
            _log(
                "broker_order_wait_result",
                broker="kite",
                phase=phase,
                ticker=item["ticker"],
                order_id=order_id,
                status=status,
                filled_qty=filled_qty,
                expected_qty=abs(_safe_float(item["quantity"])),
            )
            if status != "FILLED" or not _float_close(filled_qty, abs(_safe_float(item["quantity"]))):
                issues.append(
                    {
                        "phase": phase,
                        "ticker": item["ticker"],
                        "order_id": order_id,
                        "status": status,
                        "filled_quantity": filled_qty,
                        "expected_quantity": abs(_safe_float(item["quantity"])),
                        "details": "Order not fully filled; halting further order placement for this run.",
                    }
                )
                phase_has_unfilled = True
        if phase_has_unfilled:
            _log("broker_phase_unfilled", broker="kite", phase=phase, details="Halting execution after this phase.")
            halt_execution = True

    _log(
        "broker_order_execution_complete",
        broker="kite",
        orders=len(orders),
        missing_mappings=len(missing),
        issues=len(issues),
    )
    return orders, missing, issues


def _build_broker_positions_rows(
    positions: list[dict],
    account_currency: str,
    broker: str = "trading212",
) -> list[dict]:
    rows: list[dict] = []
    broker_name = str(broker or "").strip().lower()
    for pos in positions:
        if broker_name == "kite":
            exchange = str(pos.get("exchange") or "").strip().upper()
            tradingsymbol = str(pos.get("tradingsymbol") or "").strip().upper()
            quantity = _safe_float(pos.get("quantity")) + _safe_float(pos.get("t1_quantity"))
            average_price = _safe_float(pos.get("average_price"))
            current_price = _safe_float(pos.get("last_price"))
            wallet_current_value = _safe_float(pos.get("value"))
            if wallet_current_value <= 0 and quantity > 0 and current_price > 0:
                wallet_current_value = quantity * current_price
            rows.append(
                {
                    "ticker": f"{exchange}:{tradingsymbol}" if exchange and tradingsymbol else tradingsymbol,
                    "quantity": quantity,
                    "average_price": average_price if average_price > 0 else None,
                    "current_price": current_price if current_price > 0 else None,
                    "instrument_currency": account_currency,
                    "account_currency": account_currency,
                    "wallet_current_value": wallet_current_value if wallet_current_value > 0 else None,
                    "payload": pos,
                }
            )
            continue

        instrument = pos.get("instrument") or {}
        wallet = pos.get("walletImpact") or {}
        rows.append(
            {
                "ticker": pos.get("ticker") or instrument.get("ticker"),
                "quantity": pos.get("quantity"),
                "average_price": pos.get("averagePricePaid"),
                "current_price": pos.get("currentPrice"),
                "instrument_currency": instrument.get("currencyCode"),
                "account_currency": wallet.get("currencyCode") or account_currency,
                "wallet_current_value": wallet.get("currentValue"),
                "payload": pos,
            }
        )
    return rows


def _build_broker_account_row(
    summary: dict,
    account_currency: str,
    broker: str = "trading212",
    positions: list[dict] | None = None,
) -> dict:
    broker_name = str(broker or "").strip().lower()
    if broker_name == "kite":
        holdings = positions or []
        cash = kite_account_cash_available(summary)
        net_worth = kite_account_net_worth(summary, holdings)
        investments = net_worth - cash
        return {
            "currency": account_currency,
            "cash": cash,
            "investments": investments,
            "net_worth": net_worth,
            "payload": summary,
        }

    cash = summary.get("cash") or {}
    investments = summary.get("investments") or {}
    return {
        "currency": account_currency,
        "cash": cash.get("availableToTrade"),
        "investments": investments.get("currentValue"),
        "net_worth": t212_account_net_worth(summary),
        "payload": summary,
    }


def _convert_between_usd_gbp(
    amount: float,
    source_currency: str,
    target_currency: str,
    fx_rate_gbp_per_usd: float,
) -> float | None:
    source = str(source_currency or "").strip().upper()
    target = str(target_currency or "").strip().upper()
    if not source:
        source = target
    if not source or not target:
        return None
    if source == target:
        return amount
    if fx_rate_gbp_per_usd <= 0:
        return None
    if source == "USD" and target == "GBP":
        return amount * fx_rate_gbp_per_usd
    if source == "GBP" and target == "USD":
        return amount / fx_rate_gbp_per_usd
    return None


def _broker_notionals(
    orders: list[dict],
    broker_currency: str,
    fx_rate_gbp_per_usd: float,
) -> tuple[float, float, int, int]:
    buy_notional = 0.0
    sell_notional = 0.0
    filled_orders = 0
    covered_orders = 0
    for order in orders:
        filled_qty = _safe_float(order.get("filled_quantity"))
        if filled_qty <= 0:
            filled_qty = abs(_safe_float(order.get("quantity")))
        if filled_qty <= 0:
            continue
        filled_orders += 1
        payload = order.get("payload")
        payload = payload if isinstance(payload, dict) else {}

        notional = _safe_float(payload.get("filledValue"))
        if notional <= 0:
            price = _safe_float(order.get("exec_price"))
            if price > 0:
                notional = abs(filled_qty * price)
        if notional <= 0:
            continue

        filled_value_currency = str(
            payload.get("filledValueCurrency")
            or payload.get("currency")
            or ""
        ).strip().upper()
        price_currency = str(
            payload.get("fillPriceCurrency")
            or order.get("currency")
            or ""
        ).strip().upper()
        order_currency = filled_value_currency or price_currency
        converted_notional = _convert_between_usd_gbp(
            amount=abs(notional),
            source_currency=order_currency,
            target_currency=broker_currency,
            fx_rate_gbp_per_usd=fx_rate_gbp_per_usd,
        )
        if converted_notional is None:
            continue
        covered_orders += 1

        action = str(order.get("action") or "").strip().upper()
        if action == "BUY":
            buy_notional += converted_notional
        elif action == "SELL":
            sell_notional += converted_notional
        elif _safe_float(order.get("quantity")) < 0:
            sell_notional += converted_notional
        else:
            buy_notional += converted_notional
    return buy_notional, sell_notional, filled_orders, covered_orders


def _prior_broker_cost_totals(target_date: str) -> tuple[float, float]:
    total_broker = 0.0
    total_usd = 0.0
    rows = db_list_run_summaries()
    for row in rows:
        row_date = str(row.get("date") or "")
        if not row_date or row_date >= target_date:
            continue
        daily_broker = _safe_float(row.get("broker_execution_cost"), default=0.0)
        daily_usd = row.get("broker_execution_cost_usd")
        if daily_usd is None:
            currency = str(row.get("broker_currency") or "").strip().upper()
            fx_rate = _safe_float(row.get("broker_fx_rate_gbp_per_usd"), default=0.0)
            if currency == "USD":
                daily_usd = daily_broker
            elif fx_rate > 0:
                daily_usd = daily_broker / fx_rate
            else:
                daily_usd = 0.0
        total_broker += daily_broker
        total_usd += _safe_float(daily_usd, default=0.0)
    return total_broker, total_usd


def main():
    args = parse_args()
    _apply_runtime_trading_overrides(args)
    _log(
        "daily_run_start",
        dry_run=bool(args.dry_run),
        skip_update=bool(args.skip_update),
        sector=args.sector,
        regime_scope=args.regime_scope,
        force=bool(args.force),
        confirm_days=int(args.confirm_days),
        confirm_days_sideways=int(args.confirm_days_sideways),
        rebalance_every=int(args.rebalance_every),
        min_weight_change=float(args.min_weight_change),
        min_trade_dollars=float(args.min_trade_dollars),
        max_daily_turnover=(
            None
            if args.max_daily_turnover is None or float(args.max_daily_turnover) <= 0
            else float(args.max_daily_turnover)
        ),
        weight_smoothing=float(args.weight_smoothing),
    )
    if not db_enabled():
        raise RuntimeError(
            "Database is required for production runs. Set DATABASE_URL or POSTGRES_URL."
        )
    _log("db_init_start")
    db_init()
    _log("db_init_complete")

    schedule_date = resolve_schedule_date(args.date, timezone_name=config.RUN_CALENDAR_TIMEZONE)
    schedule_date_str = schedule_date.strftime("%Y-%m-%d")
    schedule_override = db_load_run_calendar_override(schedule_date_str) or {}
    run_decision = evaluate_run_day(
        schedule_date,
        override_action=str(schedule_override.get("action") or "").strip().lower() or None,
        override_reason=str(schedule_override.get("reason") or "").strip() or None,
        skip_weekends=config.RUN_CALENDAR_SKIP_WEEKENDS,
        skip_us_federal_holidays=config.RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS,
    )
    if not run_decision["should_run"]:
        if args.force:
            _log(
                "daily_run_calendar_blocked_but_forced",
                schedule_date=schedule_date_str,
                reason_code=run_decision.get("reason_code"),
                reason=run_decision.get("reason"),
            )
        else:
            _log(
                "daily_run_skipped_by_calendar",
                schedule_date=schedule_date_str,
                reason_code=run_decision.get("reason_code"),
                reason=run_decision.get("reason"),
            )
            print(
                f"Skipping run for {schedule_date_str}: {run_decision.get('reason')}"
            )
            return

    strategy_roots = args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS)
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
    _log("strategy_discovery_complete", strategy_roots=strategy_roots, discovered=len(strategy_names))
    if not strategy_names:
        raise ValueError("No strategies found.")

    mapping_selector = None
    mapping_payload = args.regime_mapping or json.dumps(DEFAULT_REGIME_MAPPING)
    if mapping_payload:
        try:
            mapping = json.loads(mapping_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --regime-mapping: {exc}") from exc
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("--regime-mapping must be a non-empty JSON object.")
        mapping = {str(key): str(value) for key, value in mapping.items()}
        mapping_strategy_names = {value for value in mapping.values() if value}
        missing = sorted(name for name in mapping_strategy_names if name not in strategy_names)
        if missing:
            strategy_names.extend(missing)
        strategies = load_strategies(strategy_names, strategy_roots)
        strategy_lookup = {strategy.name: strategy for strategy in strategies}

        def mapping_selector(current_date, state, strategies):
            label = str(state.get("regime_label", "unknown"))
            chosen = mapping.get(label)
            if not chosen:
                return None
            selected = strategy_lookup.get(chosen)
            if selected is None:
                return None
            return [selected]
    else:
        strategies = load_strategies(strategy_names, strategy_roots)

    if not strategies:
        raise ValueError("No valid strategies loaded.")
    _log("strategy_load_complete", loaded=len(strategies))

    pending_entries: list[dict] = []
    pending_cash_entries: list[dict] = []
    pending_tickers: list[str] = []
    pending_exchange_map: dict[str, str] = {}
    failed_new_tickers: set[str] = set()
    pending_ticker_candidates: set[str] = set()
    if not args.skip_pending:
        _log("pending_adjustments_load_start")
        pending_entries = db_load_pending_adjustments()
        for entry in pending_entries:
            entry_type = str(entry.get("type", "")).lower()
            if entry_type == "cash":
                try:
                    amount = float(entry.get("amount", 0.0))
                except (TypeError, ValueError):
                    continue
                if amount != 0:
                    pending_cash_entries.append(entry)
            elif entry_type == "tickers":
                tickers = entry.get("tickers", [])
                if isinstance(tickers, str):
                    tickers = [tickers]
                if isinstance(tickers, list):
                    pending_tickers.extend([str(t).strip() for t in tickers if str(t).strip()])
                exchanges = entry.get("exchanges") or entry.get("ticker_exchanges") or {}
                if isinstance(exchanges, dict):
                    for ticker, exchange in exchanges.items():
                        t = str(ticker).strip().upper()
                        ex = str(exchange).strip().upper()
                        if t:
                            pending_exchange_map[t] = ex or "UNKNOWN"
                elif isinstance(exchanges, list):
                    for item in exchanges:
                        if not isinstance(item, dict):
                            continue
                        t = str(item.get("ticker", "")).strip().upper()
                        ex = str(item.get("exchange", "")).strip().upper()
                        if t:
                            pending_exchange_map[t] = ex or "UNKNOWN"
    _log(
        "pending_adjustments_load_complete",
        entries=len(pending_entries),
        cash_entries=len(pending_cash_entries),
        pending_tickers=len(pending_tickers),
        pending_exchange_overrides=len(pending_exchange_map),
    )

    universe_map = _normalize_exchange_map(db_load_universe_map())
    if pending_exchange_map:
        universe_map.update(_normalize_exchange_map(pending_exchange_map))
        db_replace_universe_map(universe_map)

    data_path = _resolve_path(args.data_file or config.DATA_FILE)
    _log("data_load_start", data_path=data_path, skip_update=bool(args.skip_update))
    df = pd.read_parquet(data_path) if args.skip_update else None
    if df is not None:
        validate_market_data_frame(
            df,
            source=data_path,
            required_columns=["Close", "Open", "High", "Low", "Volume"],
        )
        _log("data_load_complete", rows=len(df), columns=len(df.columns))

    new_tickers: list[str] = []
    if args.add_tickers:
        new_tickers.extend([t.strip() for t in args.add_tickers.split(",") if t.strip()])
    if args.add_tickers_file:
        with open(args.add_tickers_file, "r") as f:
            new_tickers.extend([line.strip() for line in f if line.strip()])
    if pending_tickers:
        new_tickers.extend(pending_tickers)
    if pending_exchange_map:
        new_tickers.extend(list(pending_exchange_map.keys()))
    if new_tickers:
        new_tickers = sorted(set(new_tickers))
    pending_ticker_candidates = {
        str(ticker).strip().upper()
        for ticker in pending_tickers
        if str(ticker).strip()
    } | {
        str(ticker).strip().upper()
        for ticker in pending_exchange_map.keys()
        if str(ticker).strip()
    }
    if new_tickers:
        _log("universe_add_tickers_start", requested=len(new_tickers))
        if df is None:
            df = pd.read_parquet(data_path)
            validate_market_data_frame(
                df,
                source=data_path,
                required_columns=["Close", "Open", "High", "Low", "Volume"],
            )
        existing_tickers = set(df.index.get_level_values("ticker").unique())
        df, failed_ticker_list = add_universe_tickers(
            data_path,
            new_tickers,
            period=args.history_period,
            interval=args.history_interval,
            min_trading_days=args.min_trading_days,
            rolling_window=args.rolling_window,
            vix_ticker=args.history_vix_ticker,
            fail_on_no_valid_tickers=False,
            return_failed_tickers=True,
        )
        failed_new_tickers = {
            str(ticker).strip().upper()
            for ticker in failed_ticker_list
            if str(ticker).strip()
        }
        added = sorted(set(df.index.get_level_values("ticker").unique()) - existing_tickers)
        _log(
            "universe_add_tickers_complete",
            added=len(added),
            failed=len(failed_new_tickers),
            failed_preview=sorted(failed_new_tickers)[:10],
        )
        if added:
            for ticker in added:
                universe_map.setdefault(str(ticker).strip().upper(), "UNKNOWN")
            db_replace_universe_map(universe_map)

    if not args.skip_update:
        update_start = time.monotonic()
        resolved_market_source = (
            args.market_data_source
            if args.market_data_source != "auto"
            else ("kite_ohlc" if config.USE_KITE else "tradingview")
        )
        _log(
            "market_update_start",
            lookback_days=args.lookback_days,
            interval=args.interval,
            source_requested=args.market_data_source,
            source_resolved=resolved_market_source,
        )
        df = update_market_data(
            data_path,
            lookback_days=args.lookback_days,
            interval=args.interval,
            rolling_window=args.rolling_window,
            vix_ticker=args.vix_ticker,
            screener=args.tv_screener,
            exchange_list=[ex.strip() for ex in args.tv_exchanges.split(",") if ex.strip()],
            timeout=args.tv_timeout,
            require_all_tickers=not args.allow_partial_updates,
            exchange_map=universe_map,
            market_data_source=args.market_data_source,
            kite_quote_batch_size=args.kite_quote_batch_size,
            kite_quote_max_batches=args.kite_quote_max_batches,
        )
        _log("market_update_complete", elapsed_sec=round(time.monotonic() - update_start, 2))
    if df is None:
        _log("data_reload_start", data_path=data_path)
        df = pd.read_parquet(data_path)
        _log("data_reload_complete", rows=len(df), columns=len(df.columns))
    validate_market_data_frame(
        df,
        source=data_path,
        required_columns=["Close", "Open", "High", "Low", "Volume", "sector"],
    )
    _log("data_validation_complete", rows=len(df), columns=len(df.columns))

    if "sector" not in df.columns:
        raise ValueError("Sector column missing in data; re-run data extraction with sector enabled.")

    all_tickers = sorted({str(t).strip().upper() for t in df.index.get_level_values("ticker").unique()})
    refreshed_universe_map = {
        ticker: universe_map.get(ticker, "UNKNOWN")
        for ticker in all_tickers
        if ticker
    }
    if refreshed_universe_map != universe_map:
        universe_map = refreshed_universe_map
        db_replace_universe_map(universe_map)

    sector = args.sector.strip()
    sector_df = df[df["sector"].astype(str).str.lower() == sector.lower()]
    if sector_df.empty:
        available = sorted({s for s in df["sector"].dropna().unique().tolist()})
        raise ValueError(f"No data for sector '{sector}'. Available sectors: {available}")

    target_date = (
        pd.to_datetime(args.date).tz_localize(None)
        if args.date
        else pd.to_datetime(sector_df.index.get_level_values("date").max()).tz_localize(None)
    )
    price_snapshot: list[dict[str, float]] = _build_price_snapshot(df, target_date)
    price_lookup = _price_lookup_from_snapshot(price_snapshot)
    _log(
        "target_selection_complete",
        target_date=target_date.strftime("%Y-%m-%d"),
        sector_rows=len(sector_df),
        snapshot_prices=len(price_snapshot),
    )

    state = db_load_state()
    if state is None:
        state = ProductionState(last_date=None, cash=config.INITIAL_CAPITAL, positions={}, prev_weights={})
    costs_backfilled = False
    if state.total_costs_usd == 0.0:
        backfill = 0.0
        summaries = db_list_run_summaries()
        if summaries:
            for item in summaries:
                if item.get("daily_costs_usd") is not None:
                    backfill += float(item.get("daily_costs_usd") or 0.0)
                else:
                    backfill += float(item.get("total_costs_usd") or 0.0)
        if backfill:
            state.total_costs_usd = backfill
            costs_backfilled = True
    _log(
        "state_load_complete",
        last_date=state.last_date,
        positions=len(state.positions),
        cash=state.cash,
        costs_backfilled=bool(costs_backfilled),
    )
    cash_entries: list[dict] = []
    if args.add_cash:
        cash_entries.append(
            {
                "amount": float(args.add_cash),
                "note": args.cash_note,
                "source": "manual",
                "created_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    cash_entries.extend(pending_cash_entries)
    total_cash = sum(float(entry.get("amount", 0.0)) for entry in cash_entries)
    if total_cash:
        state.cash += float(total_cash)
    if state.last_date == target_date.strftime("%Y-%m-%d") and not args.force:
        if not args.dry_run:
            if total_cash or costs_backfilled:
                db_upsert_state(state)
            if pending_entries:
                db_clear_pending_adjustments()
            if total_cash:
                print(f"Cash added and state updated for {state.last_date}.")
                return
        print(f"State already up to date for {state.last_date}. Use --force to rerun.")
        return

    broker_context = None
    broker_orders: list[dict] = []
    broker_missing: list[str] = []
    broker_order_issues: list[dict] = []
    broker_account_row: dict | None = None
    broker_positions_rows: list[dict] = []
    broker_name = _active_broker_name()
    excluded_tickers = db_load_excluded_tickers()
    state_for_trades = state
    if broker_name == "trading212":
        _log("broker_integration_enabled", broker=broker_name)
        broker_context = _build_trading212_context(state)
        tradable_df = _apply_universe_filter_with_exclusions(sector_df, excluded_tickers)
        _ensure_trading212_universe(
            tradable_df,
            broker_context.get("by_ticker") or {},
            broker_context.get("by_symbol") or {},
            broker_context.get("overrides") or {},
            preferred_currency=config.TRADING212_PREFERRED_CURRENCY,
        )
        if config.TRADING212_USE_BROKER_STATE_FOR_SIGNALS:
            state_for_trades = _state_from_broker_context(
                state,
                broker_context,
                price_lookup=price_lookup,
            )
            _log(
                "broker_state_for_signals_enabled",
                cash_usd=state_for_trades.cash,
                positions=len(state_for_trades.positions),
                prev_weights=len(state_for_trades.prev_weights),
            )
        else:
            _log("broker_state_for_signals_disabled")
    elif broker_name == "kite":
        _log("broker_integration_enabled", broker=broker_name)
        broker_context = _build_kite_context(state)
        tradable_df = _apply_universe_filter_with_exclusions(sector_df, excluded_tickers)
        _ensure_kite_universe(
            tradable_df,
            broker_context.get("by_ticker") or {},
            broker_context.get("by_symbol") or {},
            broker_context.get("overrides") or {},
        )
        if config.KITE_USE_BROKER_STATE_FOR_SIGNALS:
            state_for_trades = _state_from_broker_context(
                state,
                broker_context,
                price_lookup=price_lookup,
            )
            _log(
                "broker_state_for_signals_enabled",
                broker=broker_name,
                cash_usd=state_for_trades.cash,
                positions=len(state_for_trades.positions),
                prev_weights=len(state_for_trades.prev_weights),
            )
        else:
            _log("broker_state_for_signals_disabled", broker=broker_name)
    else:
        _log("broker_integration_disabled")

    trade_gen_start = time.monotonic()
    _log("trade_generation_start")
    regime_table = compute_market_regime_table(df if args.regime_scope == "global" else sector_df)
    regime_table = _apply_regime_confirmation(
        regime_table,
        confirm_days=args.confirm_days,
        confirm_days_sideways=args.confirm_days_sideways,
    )
    trades, new_state, summary = generate_trades_for_date(
        sector_df,
        strategies,
        target_date=target_date,
        state=state_for_trades,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
        excluded_tickers=excluded_tickers,
        rebalance_every=args.rebalance_every,
        min_weight_change=args.min_weight_change,
        min_trade_dollars=args.min_trade_dollars,
        max_daily_turnover=args.max_daily_turnover,
    )
    _log(
        "trade_generation_complete",
        elapsed_sec=round(time.monotonic() - trade_gen_start, 2),
        trades=len(trades),
        new_positions=len(new_state.positions),
    )

    summary["sector"] = sector
    summary["regime_scope"] = args.regime_scope
    summary["cash_adjustment"] = float(total_cash)
    summary["confirm_days"] = int(args.confirm_days)
    summary["confirm_days_sideways"] = int(args.confirm_days_sideways)
    summary["rebalance_every"] = int(args.rebalance_every)
    summary["min_weight_change"] = float(args.min_weight_change)
    summary["min_trade_dollars"] = float(args.min_trade_dollars)
    summary["max_daily_turnover"] = (
        None
        if args.max_daily_turnover is None or float(args.max_daily_turnover) <= 0
        else float(args.max_daily_turnover)
    )
    summary["weight_smoothing"] = float(args.weight_smoothing)
    if broker_context:
        summary["broker_name"] = broker_name
        summary["broker_currency"] = broker_context.get("account_currency")
        summary["broker_discrepancies"] = broker_context.get("discrepancies")
        summary["broker_fx_rates"] = broker_context.get("fx_rates")
        summary["broker_fx_rate_gbp_per_usd"] = broker_context.get("fx_rate_gbp_per_usd")
        broker_exec_start = time.monotonic()
        if broker_name == "kite":
            broker_orders, broker_missing, broker_order_issues = _execute_kite_orders(
                trades,
                broker_context,
                args.dry_run,
            )
        else:
            broker_orders, broker_missing, broker_order_issues = _execute_trading212_orders(
                trades,
                broker_context,
                args.dry_run,
            )
        _log(
            "broker_order_execution_summary",
            broker=broker_name,
            elapsed_sec=round(time.monotonic() - broker_exec_start, 2),
            orders=len(broker_orders),
            missing=len(broker_missing),
            issues=len(broker_order_issues),
        )
        if broker_missing:
            missing = sorted(set(broker_missing))
            summary["broker_missing_tickers"] = missing
            raise ValueError(
                f"{broker_name.capitalize()} mapping missing for generated trades. "
                f"Missing {len(missing)} (first 10): {', '.join(missing[:10])}"
            )
        if broker_order_issues:
            summary["broker_order_issues"] = broker_order_issues
            summary["broker_order_issue_count"] = len(broker_order_issues)
            _log("broker_order_issues_recorded", broker=broker_name, count=len(broker_order_issues))
        if args.dry_run:
            post_summary = broker_context.get("summary_raw", {})
            post_positions = broker_context.get("positions_raw", [])
        else:
            client = broker_context["client"]
            _log("broker_post_trade_snapshot_start")
            if broker_name == "kite":
                try:
                    post_summary = client.get_margins()
                except KiteApiError as exc:
                    post_summary = broker_context.get("summary_raw", {}) or {}
                    summary["broker_snapshot_warning"] = (
                        "Using pre-trade account summary due to post-trade snapshot rate limit/error."
                    )
                    _log(
                        "broker_post_trade_summary_failed",
                        broker=broker_name,
                        status_code=exc.status_code,
                        error=str(exc),
                    )
                try:
                    post_positions = client.get_holdings()
                except KiteApiError as exc:
                    post_positions = broker_context.get("positions_raw", []) or []
                    summary["broker_snapshot_warning"] = (
                        "Using pre-trade positions due to post-trade snapshot rate limit/error."
                    )
                    _log(
                        "broker_post_trade_positions_failed",
                        broker=broker_name,
                        status_code=exc.status_code,
                        error=str(exc),
                    )
            else:
                try:
                    post_summary = client.get_account_summary()
                except Trading212ApiError as exc:
                    post_summary = broker_context.get("summary_raw", {}) or {}
                    summary["broker_snapshot_warning"] = (
                        "Using pre-trade account summary due to post-trade snapshot rate limit/error."
                    )
                    _log(
                        "broker_post_trade_summary_failed",
                        broker=broker_name,
                        status_code=exc.status_code,
                        error=str(exc),
                    )
                try:
                    post_positions = client.get_positions()
                except Trading212ApiError as exc:
                    post_positions = broker_context.get("positions_raw", []) or []
                    summary["broker_snapshot_warning"] = (
                        "Using pre-trade positions due to post-trade snapshot rate limit/error."
                    )
                    _log(
                        "broker_post_trade_positions_failed",
                        broker=broker_name,
                        status_code=exc.status_code,
                        error=str(exc),
                    )
            _log("broker_post_trade_snapshot_complete", positions=len(post_positions))
        broker_context["post_summary"] = post_summary
        broker_context["post_positions"] = post_positions
        broker_account_row = _build_broker_account_row(
            post_summary,
            broker_context.get("account_currency") or "INR",
            broker=broker_name,
            positions=post_positions,
        )
        broker_positions_rows = _build_broker_positions_rows(
            post_positions,
            broker_context.get("account_currency") or "INR",
            broker=broker_name,
        )
        broker_cash_before = _safe_float(broker_context.get("broker_cash"))
        broker_cash_after = _safe_float(broker_account_row.get("cash"))
        broker_net_worth_gbp = _safe_float(broker_account_row.get("net_worth"))
        broker_portfolio_value = _safe_float(broker_account_row.get("investments"))
        summary["broker_cash"] = broker_cash_after
        summary["broker_cash_before"] = broker_cash_before
        summary["broker_cash_after"] = broker_cash_after
        summary["broker_portfolio_value"] = broker_portfolio_value
        summary["broker_net_worth"] = broker_net_worth_gbp
        summary["broker_cash_weight"] = (
            broker_cash_after / broker_net_worth_gbp if broker_net_worth_gbp > 0 else 0.0
        )
        buy_notional, sell_notional, filled_orders, covered_orders = _broker_notionals(
            broker_orders,
            broker_currency=str(summary.get("broker_currency") or "").strip().upper(),
            fx_rate_gbp_per_usd=_safe_float(broker_context.get("fx_rate_gbp_per_usd"), default=0.0),
        )
        summary["broker_buy_notional"] = buy_notional
        summary["broker_sell_notional"] = sell_notional

        fx_rate = _safe_float(broker_context.get("fx_rate_gbp_per_usd"), default=0.0)
        broker_currency = str(summary.get("broker_currency") or "").strip().upper()
        summary["broker_notional_filled_orders"] = filled_orders
        summary["broker_notional_covered_orders"] = covered_orders
        notionals_complete = covered_orders >= filled_orders
        can_convert_flow = abs(float(total_cash)) <= 1e-12 or broker_currency == "USD" or fx_rate > 0
        if args.dry_run:
            summary["broker_external_flow"] = None
            summary["broker_external_flow_usd"] = float(total_cash)
            summary["broker_execution_cost"] = None
            summary["broker_execution_cost_usd"] = None
            summary["broker_total_execution_cost"] = None
            summary["broker_total_execution_cost_usd"] = None
            summary["broker_cost_warning"] = (
                "Broker execution cost is not computed during dry-run."
            )
        elif not notionals_complete:
            summary["broker_external_flow"] = None
            summary["broker_external_flow_usd"] = float(total_cash)
            summary["broker_execution_cost"] = None
            summary["broker_execution_cost_usd"] = None
            summary["broker_total_execution_cost"] = None
            summary["broker_total_execution_cost_usd"] = None
            summary["broker_cost_warning"] = (
                "Broker execution cost skipped because one or more filled broker orders are "
                "missing fill notional or have unsupported currency conversion."
            )
        elif can_convert_flow:
            if broker_currency == "USD":
                external_flow_broker = float(total_cash)
            elif fx_rate > 0:
                external_flow_broker = float(total_cash) * fx_rate
            else:
                external_flow_broker = 0.0
            summary["broker_external_flow"] = external_flow_broker
            summary["broker_external_flow_usd"] = float(total_cash)

            # Reconcile broker cash movement to isolate execution costs:
            # cash_after ~= cash_before + sells - buys - costs + external_flow
            # => costs ~= cash_before + sells - buys + external_flow - cash_after
            broker_execution_cost = (
                broker_cash_before
                + sell_notional
                - buy_notional
                - broker_cash_after
                + external_flow_broker
            )
            if abs(broker_execution_cost) <= 1e-9:
                broker_execution_cost = 0.0
            summary["broker_execution_cost"] = broker_execution_cost
            if broker_currency == "USD":
                broker_execution_cost_usd: float | None = broker_execution_cost
            elif fx_rate > 0:
                broker_execution_cost_usd = broker_execution_cost / fx_rate
            else:
                broker_execution_cost_usd = None
            summary["broker_execution_cost_usd"] = broker_execution_cost_usd

            prior_broker_total, prior_broker_total_usd = _prior_broker_cost_totals(summary["date"])
            summary["broker_total_execution_cost"] = prior_broker_total + broker_execution_cost
            if broker_execution_cost_usd is None:
                summary["broker_total_execution_cost_usd"] = None
            else:
                summary["broker_total_execution_cost_usd"] = (
                    prior_broker_total_usd + broker_execution_cost_usd
                )
        else:
            summary["broker_external_flow"] = None
            summary["broker_external_flow_usd"] = float(total_cash)
            summary["broker_execution_cost"] = None
            summary["broker_execution_cost_usd"] = None
            summary["broker_total_execution_cost"] = None
            summary["broker_total_execution_cost_usd"] = None
            summary["broker_cost_warning"] = (
                "Unable to convert cash adjustments to broker currency for execution-cost reconciliation."
            )

        if fx_rate > 0:
            summary["broker_net_worth_usd"] = broker_net_worth_gbp / fx_rate

    _attach_cagr_payload(summary)

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    state_to_persist = new_state
    if broker_context and not args.dry_run:
        post_positions = broker_context.get("post_positions") or broker_context.get("positions_raw", [])
        state_to_persist = _state_from_broker_snapshot(
            new_state=new_state,
            summary=summary,
            context=broker_context,
            post_positions=post_positions,
        )
        if broker_order_issues:
            issue_preview = ", ".join(
                sorted(
                    {
                        str(item.get("ticker", "")).strip().upper()
                        for item in broker_order_issues
                        if item.get("ticker")
                    }
                )[:5]
            )
            print(
                f"Warning: One or more {broker_name} orders were not fully executed. "
                "Persisting state from broker snapshot to avoid drift."
                + (f" Affected tickers: {issue_preview}" if issue_preview else "")
            )

    if not args.dry_run:
        _log("db_persist_start")
        db_upsert_run_summary(summary)
        _persist_selection_diagnostics_snapshot(
            full_df=df,
            summary=summary,
            sector=sector,
            target_date=target_date,
            regime_scope=args.regime_scope,
            strategy_roots=strategy_roots,
            excluded_tickers=excluded_tickers,
        )
        db_replace_trades(summary["date"], trades)
        if price_snapshot:
            db_replace_prices(summary["date"], price_snapshot)
        if broker_context and broker_account_row is not None:
            db_upsert_broker_account(summary["date"], broker_name, broker_account_row)
            db_replace_broker_positions(summary["date"], broker_name, broker_positions_rows)
            db_replace_broker_orders(summary["date"], broker_name, broker_orders)
        db_upsert_state(state_to_persist)
        if pending_entries:
            retry_pending_tickers = failed_new_tickers & pending_ticker_candidates
            retry_entries = _build_retry_pending_ticker_entries(
                pending_entries,
                retry_tickers=retry_pending_tickers,
            )
            db_clear_pending_adjustments()
            if retry_entries:
                db_append_pending_adjustments(retry_entries)
                _log(
                    "pending_adjustments_requeued",
                    entries=len(retry_entries),
                    retry_tickers=len(retry_pending_tickers),
                    retry_preview=sorted(retry_pending_tickers)[:10],
                )
        _log("db_persist_complete")

    print(f"Run date: {summary['date']}")
    print(f"Trades: {summary['num_trades']} | Net Worth: ${summary['net_worth_usd']:,.2f}")
    print(f"Sector: {sector} | Regime scope: {args.regime_scope}")
    _log("daily_run_complete", run_date=summary["date"], trades=summary["num_trades"])

    if args.print_trades:
        if trades_df.empty:
            print("No trades to execute.")
        else:
            print(trades_df.head(args.max_print).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _log(
            "daily_run_fatal",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
