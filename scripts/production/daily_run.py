import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.market_data_validation import validate_market_data_frame
from src.production_market_data import add_universe_tickers, update_market_data
from src.production import (
    generate_trades_for_date,
    load_state,
    ProductionState,
    save_state,
    _apply_universe_filter,
)
from src.production_db import (
    clear_pending_adjustments as db_clear_pending_adjustments,
    db_enabled,
    init_db as db_init,
    list_run_summaries as db_list_run_summaries,
    load_excluded_tickers as db_load_excluded_tickers,
    load_pending_adjustments as db_load_pending_adjustments,
    load_state as db_load_state,
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
from src.trading212 import (
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

DEFAULT_UNIVERSE_REGISTRY = "data/universe_registry.csv"
DEFAULT_CASH_LOG = "runs/production/cash_injections.csv"
DEFAULT_PENDING_FILE = "runs/production/pending_adjustments.jsonl"
DEFAULT_PENDING_APPLIED_FILE = "runs/production/pending_adjustments_applied.jsonl"
DEFAULT_EXCHANGE_MAP_FILE = config.TRADINGVIEW_EXCHANGE_MAP_FILE
DEFAULT_EXCLUDED_TICKERS_FILE = config.EXCLUDED_TICKERS_FILE


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
    parser.add_argument("--state-file", default="runs/production/state.json", help="Path to state file.")
    parser.add_argument("--output-dir", default="runs/production", help="Root output directory for daily runs.")
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
    parser.add_argument(
        "--exchange-map-file",
        default=DEFAULT_EXCHANGE_MAP_FILE,
        help="Path to ticker->exchange map JSON for TradingView lookups.",
    )
    parser.add_argument("--tv-timeout", type=float, default=None, help="TradingView request timeout in seconds.")
    parser.add_argument("--add-cash", type=float, default=0.0, help="Add cash to the portfolio before trading.")
    parser.add_argument("--cash-note", default="manual", help="Optional note for cash injections.")
    parser.add_argument("--cash-log", default=DEFAULT_CASH_LOG, help="Path to cash injection log CSV.")
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
        "--universe-registry",
        default=DEFAULT_UNIVERSE_REGISTRY,
        help="CSV registry path for tracking added tickers.",
    )
    parser.add_argument(
        "--pending-file",
        default=DEFAULT_PENDING_FILE,
        help="Path to pending cash/ticker adjustments JSONL file.",
    )
    parser.add_argument(
        "--pending-applied-file",
        default=DEFAULT_PENDING_APPLIED_FILE,
        help="Path to append applied pending adjustments JSONL.",
    )
    parser.add_argument(
        "--skip-pending",
        action="store_true",
        help="Ignore pending adjustments file for this run.",
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
    parser.add_argument(
        "--update-diagnostics-file",
        default="market_data_update_diagnostics.json",
        help="Filename to store per-ticker market-data update diagnostics under each run directory.",
    )
    return parser.parse_args()


def _resolve_path(path: str) -> str:
    resolved = config.resolve_path(path)
    if os.path.isabs(resolved):
        return resolved
    return os.path.join(PROJECT_ROOT, resolved)


def _log_cash_injection(
    path: str,
    amount: float,
    note: str,
    state_file: str,
    source: str = "manual",
    created_at: str | None = None,
) -> None:
    if amount == 0:
        return
    log_path = Path(_resolve_path(path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = created_at or pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    row = f"{timestamp},{amount:.2f},{note},{state_file},{source}\n"
    if not log_path.exists():
        with log_path.open("w") as f:
            f.write("timestamp,amount_usd,note,state_file,source\n")
            f.write(row)
    else:
        with log_path.open("a") as f:
            f.write(row)


def _update_universe_registry(path: str, tickers: list[str], source: str) -> None:
    if not tickers:
        return
    reg_path = Path(_resolve_path(path))
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [f"{t},{timestamp},{source}\n" for t in sorted(set(tickers))]
    if not reg_path.exists():
        with reg_path.open("w") as f:
            f.write("ticker,added_at,source\n")
            f.writelines(rows)
    else:
        with reg_path.open("a") as f:
            f.writelines(rows)


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


def _load_pending_adjustments(path: str) -> list[dict]:
    pending_path = Path(_resolve_path(path))
    if not pending_path.exists():
        return []
    entries: list[dict] = []
    with pending_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _build_trading212_context(state: ProductionState) -> dict:
    client = Trading212Client()
    instruments = load_instruments_cache(client)
    overrides = load_ticker_overrides()
    by_ticker, by_symbol = build_instrument_index(instruments)

    summary_raw = client.get_account_summary()
    positions_raw = client.get_positions()

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
    missing: list[str] = []
    mapping: dict[str, str] = {}
    for ticker in tickers:
        mapped = resolve_t212_ticker(ticker, by_symbol, overrides, preferred_currency)
        if not mapped or mapped not in by_ticker:
            missing.append(ticker)
        else:
            mapping[ticker] = mapped
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Missing Trading212 mappings for tickers. "
            f"Missing {len(missing)} (first 10): {preview}"
        )
    return mapping


def _execute_trading212_orders(
    trades: list[dict],
    context: dict,
    dry_run: bool,
) -> tuple[list[dict], list[str]]:
    client: Trading212Client = context["client"]
    by_symbol = context["by_symbol"]
    overrides = context["overrides"]
    missing: list[str] = []
    orders: list[dict] = []
    for trade in trades:
        shares = _safe_float(trade.get("shares"))
        if abs(shares) <= 1e-6:
            continue
        ticker = str(trade.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        t212_ticker = resolve_t212_ticker(ticker, by_symbol, overrides)
        if not t212_ticker:
            missing.append(ticker)
            continue
        if dry_run:
            continue
        order_resp = client.place_market_order(
            ticker=t212_ticker,
            quantity=shares,
        )
        order_id = order_resp.get("id")
        if order_id is None:
            raise ValueError(f"Trading212 order failed for {ticker}: {order_resp}")
        filled = client.wait_for_fill(order_id, expected_qty=abs(shares))
        status = str(filled.get("status", "")).upper()
        filled_qty = _safe_float(filled.get("filledQuantity"))
        if status != "FILLED" or not _float_close(filled_qty, abs(shares)):
            raise ValueError(
                f"Trading212 order not fully filled for {ticker}: status={status}, "
                f"filled={filled_qty}, expected={abs(shares)}"
            )
        filled_value = _safe_float(filled.get("filledValue"))
        exec_price = None
        if filled_qty > 0 and filled_value > 0:
            exec_price = filled_value / filled_qty
        orders.append(
            {
                "ticker": ticker,
                "action": trade.get("action"),
                "quantity": shares,
                "filled_quantity": filled_qty,
                "exec_price": exec_price,
                "currency": filled.get("currency") or context.get("account_currency"),
                "status": status,
                "order_id": str(order_id),
                "payload": filled,
            }
        )
    return orders, missing


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


def _backfill_cumulative_costs(output_dir: str) -> float:
    base = Path(_resolve_path(output_dir))
    if not base.exists():
        return 0.0
    total = 0.0
    for run_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        if "daily_costs_usd" in payload:
            total += float(payload.get("daily_costs_usd", 0.0))
        else:
            total += float(payload.get("total_costs_usd", 0.0))
    return total


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


def _prior_broker_cost_totals(output_dir: str, target_date: str) -> tuple[float, float]:
    total_broker = 0.0
    total_usd = 0.0
    if db_enabled():
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

    base = Path(_resolve_path(output_dir))
    if not base.exists():
        return 0.0, 0.0
    for run_dir in sorted(d for d in base.iterdir() if d.is_dir() and d.name < target_date):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            row = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
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


def _mark_pending_applied(pending_path: str, applied_path: str, entries: list[dict]) -> None:
    if not entries:
        return
    applied = Path(_resolve_path(applied_path))
    applied.parent.mkdir(parents=True, exist_ok=True)
    with applied.open("a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    pending_file = Path(_resolve_path(pending_path))
    pending_file.write_text("")


def _update_exchange_map(path: str, updates: dict[str, str]) -> None:
    if not updates:
        return
    exchange_path = Path(_resolve_path(path))
    exchange_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, str] = {}
    if exchange_path.exists():
        try:
            payload = json.loads(exchange_path.read_text())
        except json.JSONDecodeError:
            payload = {}
    for ticker, exchange in updates.items():
        t = str(ticker).strip().upper()
        ex = str(exchange).strip().upper()
        if not t:
            continue
        payload[t] = ex or "UNKNOWN"
    exchange_path.write_text(
        "{\n" + ",\n".join([f'  \"{k}\": \"{v}\"' for k, v in sorted(payload.items())]) + "\n}\n"
    )


def _load_exchange_map(path: str) -> dict[str, str]:
    exchange_path = Path(_resolve_path(path))
    if not exchange_path.exists():
        return {}
    try:
        payload = json.loads(exchange_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for ticker, exchange in payload.items():
        t = str(ticker).strip().upper()
        ex = str(exchange).strip().upper()
        if not t:
            continue
        out[t] = ex or "UNKNOWN"
    return out


def _sync_excluded_tickers_from_db(path: str) -> None:
    if not db_enabled():
        return
    tickers = db_load_excluded_tickers()
    excluded_path = Path(_resolve_path(path))
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    if not tickers:
        excluded_path.write_text("")
        return
    excluded_path.write_text("\n".join(sorted(tickers)) + "\n")


def main():
    args = parse_args()
    strategy_roots = args.strategy_roots or ["alphas"]
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
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

    if db_enabled():
        db_init()
        _sync_excluded_tickers_from_db(DEFAULT_EXCLUDED_TICKERS_FILE)

    pending_entries: list[dict] = []
    pending_cash_entries: list[dict] = []
    pending_tickers: list[str] = []
    pending_exchange_map: dict[str, str] = {}
    if not args.skip_pending:
        if db_enabled():
            pending_entries = db_load_pending_adjustments()
        else:
            pending_entries = _load_pending_adjustments(args.pending_file)
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

    data_path = _resolve_path(args.data_file or config.DATA_FILE)
    df = pd.read_parquet(data_path) if args.skip_update else None
    update_diagnostics: dict | None = None
    if df is not None:
        validate_market_data_frame(
            df,
            source=data_path,
            required_columns=["Close", "Open", "High", "Low", "Volume"],
        )

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
        _update_universe_registry(args.universe_registry, added, source="yfinance")

    if pending_exchange_map:
        _update_exchange_map(args.exchange_map_file, pending_exchange_map)
    if db_enabled():
        exchange_map_payload = _load_exchange_map(args.exchange_map_file)
        if exchange_map_payload:
            db_replace_universe_map(exchange_map_payload)

    if not args.skip_update:
        updated_result = update_market_data(
            data_path,
            lookback_days=args.lookback_days,
            interval=args.interval,
            rolling_window=args.rolling_window,
            vix_ticker=args.vix_ticker,
            screener=args.tv_screener,
            exchange_list=[ex.strip() for ex in args.tv_exchanges.split(",") if ex.strip()],
            timeout=args.tv_timeout,
            require_all_tickers=not args.allow_partial_updates,
            exchange_map_path=args.exchange_map_file,
            return_diagnostics=True,
        )
        df, update_diagnostics = updated_result
    if df is None:
        df = pd.read_parquet(data_path)
    validate_market_data_frame(
        df,
        source=data_path,
        required_columns=["Close", "Open", "High", "Low", "Volume", "sector"],
    )

    if "sector" not in df.columns:
        raise ValueError("Sector column missing in data; re-run data extraction with sector enabled.")

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
    price_snapshot: list[dict[str, float]] = []
    if db_enabled():
        price_snapshot = _build_price_snapshot(df, target_date)

    state_path = _resolve_path(args.state_file)
    state = db_load_state() if db_enabled() else None
    if state is None:
        state = load_state(state_path, config.INITIAL_CAPITAL)
    costs_backfilled = False
    if state.total_costs_usd == 0.0:
        backfill = 0.0
        if db_enabled():
            summaries = db_list_run_summaries()
            if summaries:
                for item in summaries:
                    if item.get("daily_costs_usd") is not None:
                        backfill += float(item.get("daily_costs_usd") or 0.0)
                    else:
                        backfill += float(item.get("total_costs_usd") or 0.0)
        if not backfill:
            backfill = _backfill_cumulative_costs(args.output_dir)
        if backfill:
            state.total_costs_usd = backfill
            costs_backfilled = True
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
                save_state(state_path, state)
                if db_enabled():
                    db_upsert_state(state)
                for entry in cash_entries:
                    _log_cash_injection(
                        args.cash_log,
                        float(entry.get("amount", 0.0)),
                        str(entry.get("note", "pending")),
                        args.state_file,
                        source=str(entry.get("source", "pending")),
                        created_at=str(entry.get("created_at")) if entry.get("created_at") else None,
                    )
            if pending_entries:
                if db_enabled():
                    db_clear_pending_adjustments()
                else:
                    _mark_pending_applied(args.pending_file, args.pending_applied_file, pending_entries)
            if total_cash:
                print(f"Cash added and state updated for {state.last_date}.")
                return
        print(f"State already up to date for {state.last_date}. Use --force to rerun.")
        return

    broker_context = None
    broker_orders: list[dict] = []
    broker_missing: list[str] = []
    broker_account_row: dict | None = None
    broker_positions_rows: list[dict] = []
    state_for_trades = state
    if trading212_enabled():
        broker_context = _build_trading212_context(state)
        tradable_df = _apply_universe_filter(sector_df)
        _ensure_trading212_universe(
            tradable_df,
            broker_context.get("by_ticker") or {},
            broker_context.get("by_symbol") or {},
            broker_context.get("overrides") or {},
            preferred_currency=config.TRADING212_PREFERRED_CURRENCY,
        )

    regime_table = compute_market_regime_table(df if args.regime_scope == "global" else sector_df)
    trades, new_state, summary = generate_trades_for_date(
        sector_df,
        strategies,
        target_date=target_date,
        state=state_for_trades,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
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
        broker_orders, broker_missing = _execute_trading212_orders(
            trades,
            broker_context,
            args.dry_run,
        )
        if broker_missing:
            missing = sorted(set(broker_missing))
            summary["broker_missing_tickers"] = missing
            raise ValueError(
                "Trading212 mapping missing for generated trades. "
                f"Missing {len(missing)} (first 10): {', '.join(missing[:10])}"
            )
        if args.dry_run:
            post_summary = broker_context.get("summary_raw", {})
            post_positions = broker_context.get("positions_raw", [])
        else:
            client: Trading212Client = broker_context["client"]
            post_summary = client.get_account_summary()
            post_positions = client.get_positions()
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

            prior_broker_total, prior_broker_total_usd = _prior_broker_cost_totals(
                args.output_dir,
                summary["date"],
            )
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

    run_dir = Path(_resolve_path(args.output_dir)) / summary["date"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if update_diagnostics is not None:
        diagnostics_path = run_dir / args.update_diagnostics_file
        with diagnostics_path.open("w") as f:
            json.dump(update_diagnostics, f, indent=2)

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    trades_path = run_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    if broker_context:
        broker_orders_path = run_dir / "broker_orders.json"
        with broker_orders_path.open("w") as f:
            json.dump(
                {
                    "orders": broker_orders,
                    "missing_tickers": summary.get("broker_missing_tickers", []),
                },
                f,
                indent=2,
            )

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if not args.dry_run:
        if db_enabled():
            db_upsert_run_summary(summary)
            db_replace_trades(summary["date"], trades)
            if price_snapshot:
                db_replace_prices(summary["date"], price_snapshot)
            if broker_context and broker_account_row is not None:
                db_upsert_broker_account(summary["date"], "trading212", broker_account_row)
                db_replace_broker_positions(summary["date"], "trading212", broker_positions_rows)
                db_replace_broker_orders(summary["date"], "trading212", broker_orders)
        save_state(state_path, new_state)
        if db_enabled():
            db_upsert_state(new_state)
        for entry in cash_entries:
            _log_cash_injection(
                args.cash_log,
                float(entry.get("amount", 0.0)),
                str(entry.get("note", "pending")),
                args.state_file,
                source=str(entry.get("source", "pending")),
                created_at=str(entry.get("created_at")) if entry.get("created_at") else None,
            )
        if pending_entries:
            if db_enabled():
                db_clear_pending_adjustments()
            else:
                _mark_pending_applied(args.pending_file, args.pending_applied_file, pending_entries)

    print(f"Trades saved to {trades_path}")
    print(f"Summary saved to {summary_path}")
    if update_diagnostics is not None:
        print(f"Market update diagnostics saved to {run_dir / args.update_diagnostics_file}")
    print(f"Trades: {summary['num_trades']} | Net Worth: ${summary['net_worth_usd']:,.2f}")
    print(f"Sector: {sector} | Regime scope: {args.regime_scope}")

    if args.print_trades:
        if trades_df.empty:
            print("No trades to execute.")
        else:
            print(trades_df.head(args.max_print).to_string(index=False))


if __name__ == "__main__":
    main()
