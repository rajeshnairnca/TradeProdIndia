import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.market_data_validation import validate_market_data_frame
from src.production_market_data import add_universe_tickers, update_market_data
from src.production import (
    generate_trades_for_date,
    ProductionState,
)
from src.production_db import (
    clear_pending_adjustments as db_clear_pending_adjustments,
    db_enabled,
    init_db as db_init,
    list_run_summaries as db_list_run_summaries,
    load_excluded_tickers as db_load_excluded_tickers,
    load_pending_adjustments as db_load_pending_adjustments,
    load_state as db_load_state,
    load_universe_map as db_load_universe_map,
    replace_universe_map as db_replace_universe_map,
    replace_broker_orders as db_replace_broker_orders,
    replace_broker_positions as db_replace_broker_positions,
    replace_trades as db_replace_trades,
    replace_prices as db_replace_prices,
    upsert_broker_account as db_upsert_broker_account,
    upsert_run_summary as db_upsert_run_summary,
    upsert_state as db_upsert_state,
)
from src.regime import compute_market_regime_table
from src.strategy import list_strategy_names, load_strategies
from src.universe import NASDAQ100_TICKERS
from src.universe_quality import apply_quality_filter
from src.trading212 import (
    Trading212ApiError,
    Trading212Client,
    account_cash_available,
    account_net_worth,
    build_instrument_index,
    compare_positions,
    extract_fx_rates,
    load_instruments_cache,
    load_ticker_overrides,
    positions_to_internal_positions,
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

DEFAULT_REGIME_MAPPING = {
    "bear_high_vol": "rule_mean_reversion",
    "bear_low_vol": "rule_low_vol_defensive",
    "bull_high_vol": "rule_quality_min_vol",
    "bull_low_vol": "rule_momentum_acceleration",
    "sideways_high_vol": "rule_range_reversion",
    "sideways_low_vol": "rule_trend_strength",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str, **fields) -> None:
    payload = f"[{_utc_now()}] {message}"
    if fields:
        payload = f"{payload} | {json.dumps(fields, default=str, sort_keys=True)}"
    print(payload, flush=True)


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
    parser.add_argument("--skip-update", action="store_true", help="Skip TradingView data update.")
    parser.add_argument("--lookback-days", type=int, default=420, help="Days of lookback for incremental update.")
    parser.add_argument("--rolling-window", type=int, default=None, help="Rolling window for ADV/vol/VIX z.")
    parser.add_argument("--interval", default="1d", help="TradingView interval (default: 1d).")
    parser.add_argument("--vix-ticker", default="CBOE:VIX", help="TradingView VIX symbol (default: CBOE:VIX).")
    parser.add_argument("--tv-screener", default="america", help="TradingView screener (default: america).")
    parser.add_argument(
        "--tv-exchanges",
        default="NASDAQ,NYSE,AMEX",
        help="Comma-separated exchange fallback list (default: NASDAQ,NYSE,AMEX).",
    )
    parser.add_argument("--tv-timeout", type=float, default=None, help="TradingView request timeout in seconds.")
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
    parser.add_argument("--history-vix-ticker", default="^VIX", help="yfinance VIX symbol (default: ^VIX).")
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=50,
        help="Minimum history days required for new tickers.",
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
        help="Allow data updates if some tickers fail TradingView fetch.",
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


def _apply_universe_filter_with_exclusions(
    df: pd.DataFrame,
    excluded: set[str],
) -> pd.DataFrame:
    universe_filter = config.UNIVERSE_FILTER
    if not universe_filter or universe_filter in ("all", "none"):
        filtered = df
    elif universe_filter == "nasdaq100":
        allowed = set(NASDAQ100_TICKERS)
        filtered = df[df.index.get_level_values("ticker").isin(allowed)]
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
    date_values = df.index.get_level_values("date")
    day_data = df.loc[date_values == date_key]
    if day_data.empty:
        return []
    day_data = day_data.reset_index()
    if "ticker" not in day_data.columns or "Close" not in day_data.columns:
        return []
    prices: list[dict[str, float]] = []
    for _, row in day_data[["ticker", "Close"]].iterrows():
        if pd.isna(row["Close"]):
            continue
        ticker = str(row["ticker"]).strip().upper()
        if not ticker:
            continue
        try:
            price = float(row["Close"])
        except (TypeError, ValueError):
            continue
        prices.append({"ticker": ticker, "close_price": price})
    return prices


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


def _build_trading212_context(state: ProductionState) -> dict:
    _log("broker_context_build_start", broker="trading212")
    client = Trading212Client()
    instruments = load_instruments_cache(client)
    overrides = load_ticker_overrides()
    by_ticker, by_symbol = build_instrument_index(instruments)
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
    broker_cash_gbp = account_cash_available(summary_raw)
    broker_net_worth_gbp = account_net_worth(summary_raw)
    broker_positions = positions_to_internal_positions(positions_raw, overrides)
    fx_rates = extract_fx_rates(positions_raw, account_currency=account_currency)
    fx_rate = fx_rates.get("USD") or config.TRADING212_FX_RATE_USD_GBP
    if not fx_rate:
        raise ValueError(
            "Unable to derive GBP/USD FX rate for Trading212. "
            "Set TRADING212_FX_RATE_USD_GBP to proceed."
        )
    broker_cash_usd = broker_cash_gbp / fx_rate
    discrepancies = compare_positions(
        state.positions,
        broker_positions,
        state.cash,
        broker_cash_usd,
    )
    _log(
        "broker_context_build_complete",
        account_currency=account_currency,
        broker_positions=len(broker_positions),
        fx_rate_gbp_per_usd=fx_rate,
    )
    return {
        "client": client,
        "instruments": instruments,
        "overrides": overrides,
        "by_ticker": by_ticker,
        "by_symbol": by_symbol,
        "summary_raw": summary_raw,
        "positions_raw": positions_raw,
        "account_currency": account_currency,
        "broker_cash_gbp": broker_cash_gbp,
        "broker_cash_usd": broker_cash_usd,
        "broker_net_worth_gbp": broker_net_worth_gbp,
        "broker_positions": broker_positions,
        "fx_rates": fx_rates,
        "fx_rate_gbp_per_usd": fx_rate,
        "discrepancies": discrepancies,
    }


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
            bulk_status = client.wait_for_orders(order_ids)
        except Exception as exc:
            issue = {
                "phase": phase,
                "error": "bulk_status_error",
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

        by_t212_ticker: dict[str, list[dict]] = {}
        unresolved_by_t212_ticker: dict[str, list[dict]] = {}
        for item in placed:
            order_id = item["order_id"]
            expected_qty = abs(_safe_float(item["shares"]))
            payload = bulk_status.get(order_id) or {"id": order_id, "status": "PENDING"}
            status = str(payload.get("status", "")).upper()
            filled_qty = _safe_float(payload.get("filledQuantity"))
            t212_ticker = str(item.get("t212_ticker") or "").strip().upper()
            if t212_ticker:
                by_t212_ticker.setdefault(t212_ticker, []).append(item)
                if status != "FILLED" or not _float_close(filled_qty, expected_qty):
                    unresolved_by_t212_ticker.setdefault(t212_ticker, []).append(item)

        for t212_ticker, unresolved_items in unresolved_by_t212_ticker.items():
            pre_qty = _safe_float(phase_start_qty_by_ticker.get(t212_ticker))
            phase_items = by_t212_ticker.get(t212_ticker, [])
            expected_delta = sum(
                _expected_position_delta(str(item.get("action") or ""), _safe_float(item.get("shares")))
                for item in phase_items
            )
            expected_post_qty = pre_qty + expected_delta
            _log(
                "broker_position_reconcile_start",
                phase=phase,
                t212_ticker=t212_ticker,
                unresolved_orders=len(unresolved_items),
                pre_qty=pre_qty,
                expected_post_qty=expected_post_qty,
            )
            try:
                positions = client.get_positions(ticker=t212_ticker)
            except Exception as exc:
                _log(
                    "broker_position_reconcile_exception",
                    phase=phase,
                    t212_ticker=t212_ticker,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                continue
            post_qty = _position_quantity_for_ticker(positions, t212_ticker)
            _log(
                "broker_position_reconcile_result",
                phase=phase,
                t212_ticker=t212_ticker,
                positions=len(positions),
                post_qty=post_qty,
                expected_post_qty=expected_post_qty,
            )
            if not _float_close(post_qty, expected_post_qty, tol=1e-4):
                continue
            for unresolved in unresolved_items:
                order_id = unresolved["order_id"]
                shares = _safe_float(unresolved.get("shares"))
                payload = dict(bulk_status.get(order_id) or {"id": order_id})
                payload["status"] = "FILLED"
                payload["filledQuantity"] = abs(shares)
                payload["resolution"] = "reconciled_via_positions_endpoint"
                payload["positionQuantityBefore"] = pre_qty
                payload["positionQuantityAfter"] = post_qty
                payload["positionQuantityExpected"] = expected_post_qty
                bulk_status[order_id] = payload
                _log(
                    "broker_position_reconcile_applied",
                    phase=phase,
                    ticker=unresolved["ticker"],
                    order_id=order_id,
                    t212_ticker=t212_ticker,
                )

        phase_has_unfilled = False
        for item in placed:
            order_id = item["order_id"]
            filled = bulk_status.get(order_id) or {"id": order_id, "status": "PENDING"}
            status = str(filled.get("status", "")).upper()
            filled_qty = _safe_float(filled.get("filledQuantity"))
            filled_value = _safe_float(filled.get("filledValue"))
            exec_price = None
            if filled_qty > 0 and filled_value > 0:
                exec_price = filled_value / filled_qty
            orders.append(
                {
                    "ticker": item["ticker"],
                    "action": item["action"],
                    "quantity": item["shares"],
                    "filled_quantity": filled_qty,
                    "exec_price": exec_price,
                    "currency": filled.get("currency") or context.get("account_currency"),
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


def _build_broker_positions_rows(
    positions: list[dict],
    account_currency: str,
) -> list[dict]:
    rows: list[dict] = []
    for pos in positions:
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


def _build_broker_account_row(summary: dict, account_currency: str) -> dict:
    cash = summary.get("cash") or {}
    investments = summary.get("investments") or {}
    return {
        "currency": account_currency,
        "cash": cash.get("availableToTrade"),
        "investments": investments.get("currentValue"),
        "net_worth": account_net_worth(summary),
        "payload": summary,
    }


def _broker_notionals(orders: list[dict]) -> tuple[float, float]:
    buy_notional = 0.0
    sell_notional = 0.0
    for order in orders:
        price = _safe_float(order.get("exec_price"))
        if price <= 0:
            continue
        filled_qty = _safe_float(order.get("filled_quantity"))
        if filled_qty <= 0:
            filled_qty = abs(_safe_float(order.get("quantity")))
        if filled_qty <= 0:
            continue
        notional = abs(filled_qty * price)
        action = str(order.get("action") or "").strip().upper()
        if action == "BUY":
            buy_notional += notional
        elif action == "SELL":
            sell_notional += notional
        elif _safe_float(order.get("quantity")) < 0:
            sell_notional += notional
        else:
            buy_notional += notional
    return buy_notional, sell_notional


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
    _log(
        "daily_run_start",
        dry_run=bool(args.dry_run),
        skip_update=bool(args.skip_update),
        sector=args.sector,
        regime_scope=args.regime_scope,
        force=bool(args.force),
    )
    if not db_enabled():
        raise RuntimeError(
            "Database is required for production runs. Set DATABASE_URL or POSTGRES_URL."
        )
    _log("db_init_start")
    db_init()
    _log("db_init_complete")

    strategy_roots = args.strategy_roots or ["alphas"]
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
        df = add_universe_tickers(
            data_path,
            new_tickers,
            period=args.history_period,
            interval=args.history_interval,
            min_trading_days=args.min_trading_days,
            rolling_window=args.rolling_window,
            vix_ticker=args.history_vix_ticker,
        )
        added = sorted(set(df.index.get_level_values("ticker").unique()) - existing_tickers)
        _log("universe_add_tickers_complete", added=len(added))
        if added:
            for ticker in added:
                universe_map.setdefault(str(ticker).strip().upper(), "UNKNOWN")
            db_replace_universe_map(universe_map)

    if not args.skip_update:
        update_start = time.monotonic()
        _log("market_update_start", lookback_days=args.lookback_days, interval=args.interval)
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
    excluded_tickers = db_load_excluded_tickers()
    state_for_trades = state
    if trading212_enabled():
        _log("broker_integration_enabled", broker="trading212")
        broker_context = _build_trading212_context(state)
        tradable_df = _apply_universe_filter_with_exclusions(sector_df, excluded_tickers)
        _ensure_trading212_universe(
            tradable_df,
            broker_context.get("by_ticker") or {},
            broker_context.get("by_symbol") or {},
            broker_context.get("overrides") or {},
            preferred_currency=config.TRADING212_PREFERRED_CURRENCY,
        )
    else:
        _log("broker_integration_disabled")

    trade_gen_start = time.monotonic()
    _log("trade_generation_start")
    regime_table = compute_market_regime_table(df if args.regime_scope == "global" else sector_df)
    trades, new_state, summary = generate_trades_for_date(
        sector_df,
        strategies,
        target_date=target_date,
        state=state_for_trades,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
        excluded_tickers=excluded_tickers,
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
    if broker_context:
        summary["broker_name"] = "trading212"
        summary["broker_currency"] = broker_context.get("account_currency")
        summary["broker_discrepancies"] = broker_context.get("discrepancies")
        summary["broker_fx_rates"] = broker_context.get("fx_rates")
        summary["broker_fx_rate_gbp_per_usd"] = broker_context.get("fx_rate_gbp_per_usd")
        broker_exec_start = time.monotonic()
        broker_orders, broker_missing, broker_order_issues = _execute_trading212_orders(
            trades,
            broker_context,
            args.dry_run,
        )
        _log(
            "broker_order_execution_summary",
            elapsed_sec=round(time.monotonic() - broker_exec_start, 2),
            orders=len(broker_orders),
            missing=len(broker_missing),
            issues=len(broker_order_issues),
        )
        if broker_missing:
            missing = sorted(set(broker_missing))
            summary["broker_missing_tickers"] = missing
            raise ValueError(
                "Trading212 mapping missing for generated trades. "
                f"Missing {len(missing)} (first 10): {', '.join(missing[:10])}"
            )
        if broker_order_issues:
            summary["broker_order_issues"] = broker_order_issues
            summary["broker_order_issue_count"] = len(broker_order_issues)
            _log("broker_order_issues_recorded", count=len(broker_order_issues))
        if args.dry_run:
            post_summary = broker_context.get("summary_raw", {})
            post_positions = broker_context.get("positions_raw", [])
        else:
            client: Trading212Client = broker_context["client"]
            _log("broker_post_trade_snapshot_start")
            try:
                post_summary = client.get_account_summary()
            except Trading212ApiError as exc:
                post_summary = broker_context.get("summary_raw", {}) or {}
                summary["broker_snapshot_warning"] = (
                    "Using pre-trade account summary due to post-trade snapshot rate limit/error."
                )
                _log(
                    "broker_post_trade_summary_failed",
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
                    status_code=exc.status_code,
                    error=str(exc),
                )
            _log("broker_post_trade_snapshot_complete", positions=len(post_positions))
        broker_context["post_summary"] = post_summary
        broker_context["post_positions"] = post_positions
        broker_account_row = _build_broker_account_row(
            post_summary,
            broker_context.get("account_currency") or "GBP",
        )
        broker_positions_rows = _build_broker_positions_rows(
            post_positions,
            broker_context.get("account_currency") or "GBP",
        )
        broker_cash_before = _safe_float(broker_context.get("broker_cash_gbp"))
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
        buy_notional, sell_notional = _broker_notionals(broker_orders)
        summary["broker_buy_notional"] = buy_notional
        summary["broker_sell_notional"] = sell_notional

        fx_rate = _safe_float(broker_context.get("fx_rate_gbp_per_usd"), default=0.0)
        broker_currency = str(summary.get("broker_currency") or "").strip().upper()
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
        elif can_convert_flow:
            if broker_currency == "USD":
                external_flow_broker = float(total_cash)
            elif fx_rate > 0:
                external_flow_broker = float(total_cash) * fx_rate
            else:
                external_flow_broker = 0.0
            summary["broker_external_flow"] = external_flow_broker
            summary["broker_external_flow_usd"] = float(total_cash)

            broker_execution_cost = (
                broker_cash_before
                + sell_notional
                - buy_notional
                - broker_cash_after
                - external_flow_broker
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

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    state_to_persist = new_state
    if broker_context and broker_order_issues and not args.dry_run:
        post_positions = broker_context.get("post_positions") or broker_context.get("positions_raw", [])
        broker_positions_internal = positions_to_internal_positions(
            post_positions,
            broker_context.get("overrides") or {},
        )
        broker_currency = str(summary.get("broker_currency") or "").strip().upper()
        fx_rate = _safe_float(summary.get("broker_fx_rate_gbp_per_usd"), default=0.0)
        broker_cash_after = _safe_float(summary.get("broker_cash_after"), default=new_state.cash)
        synced_cash_usd = new_state.cash
        if broker_currency == "USD":
            synced_cash_usd = broker_cash_after
        elif fx_rate > 0:
            synced_cash_usd = broker_cash_after / fx_rate
        state_to_persist = ProductionState(
            last_date=new_state.last_date,
            cash=synced_cash_usd,
            positions=broker_positions_internal,
            prev_weights={},
            total_costs_usd=new_state.total_costs_usd,
        )
        issue_preview = ", ".join(
            sorted({str(item.get("ticker", "")).strip().upper() for item in broker_order_issues if item.get("ticker")})[:5]
        )
        print(
            "Warning: One or more Trading212 orders were not fully executed. "
            "Persisting state from broker snapshot to avoid drift."
            + (f" Affected tickers: {issue_preview}" if issue_preview else "")
        )

    if not args.dry_run:
        _log("db_persist_start")
        db_upsert_run_summary(summary)
        db_replace_trades(summary["date"], trades)
        if price_snapshot:
            db_replace_prices(summary["date"], price_snapshot)
        if broker_context and broker_account_row is not None:
            db_upsert_broker_account(summary["date"], "trading212", broker_account_row)
            db_replace_broker_positions(summary["date"], "trading212", broker_positions_rows)
            db_replace_broker_orders(summary["date"], "trading212", broker_orders)
        db_upsert_state(state_to_persist)
        if pending_entries:
            db_clear_pending_adjustments()
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
