import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

os.environ.setdefault("TRADING_REGION", "us")
os.environ.setdefault("REGIME_MODE", "heuristic")

from src import config
from src.production import (
    add_universe_tickers,
    generate_trades_for_date,
    load_state,
    save_state,
    update_market_data,
)
from src.production_db import (
    clear_pending_adjustments as db_clear_pending_adjustments,
    db_enabled,
    init_db as db_init,
    list_run_summaries as db_list_run_summaries,
    load_pending_adjustments as db_load_pending_adjustments,
    load_state as db_load_state,
    replace_trades as db_replace_trades,
    upsert_run_summary as db_upsert_run_summary,
    upsert_state as db_upsert_state,
)
from src.regime import compute_market_regime_table
from src.strategy import list_strategy_names, load_strategies

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

    if not args.skip_update:
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
            exchange_map_path=args.exchange_map_file,
        )
    if df is None:
        df = pd.read_parquet(data_path)

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

    regime_table = compute_market_regime_table(df if args.regime_scope == "global" else sector_df)
    trades, new_state, summary = generate_trades_for_date(
        sector_df,
        strategies,
        target_date=target_date,
        state=state,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
    )

    summary["sector"] = sector
    summary["regime_scope"] = args.regime_scope
    summary["cash_adjustment"] = float(total_cash)

    run_dir = Path(_resolve_path(args.output_dir)) / summary["date"]
    run_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    trades_path = run_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if not args.dry_run:
        if db_enabled():
            db_upsert_run_summary(summary)
            db_replace_trades(summary["date"], trades)
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
    print(f"Trades: {summary['num_trades']} | Net Worth: ${summary['net_worth_usd']:,.2f}")
    print(f"Sector: {sector} | Regime scope: {args.regime_scope}")

    if args.print_trades:
        if trades_df.empty:
            print("No trades to execute.")
        else:
            print(trades_df.head(args.max_print).to_string(index=False))


if __name__ == "__main__":
    main()
