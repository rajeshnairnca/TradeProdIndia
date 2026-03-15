import os
import sys
import json
import argparse
import hashlib
import shlex
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.emerging_universe import EmergingUniverseParams, build_emerging_universe_schedule
from src.market_data_validation import validate_market_data_frame
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import list_strategy_names, load_strategies


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
    confirm_days: int,
    confirm_days_sideways: int,
) -> pd.DataFrame:
    if regime_table is None or regime_table.empty or "regime_label" not in regime_table.columns:
        return regime_table
    label_series = regime_table["regime_label"].astype(str)
    confirmed = _apply_confirmed_switch(
        label_series,
        confirm_days=confirm_days,
        confirm_days_sideways=confirm_days_sideways,
    )
    out = regime_table.copy()
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


def make_ensemble_dirname(strategy_names: list[str]) -> str:
    """Generate a short, stable ensemble folder name with a hash suffix."""
    sorted_names = sorted(strategy_names)
    if len(sorted_names) == 1:
        return sorted_names[0]

    digest = hashlib.sha1("::".join(sorted_names).encode("utf-8")).hexdigest()[:8]
    preview_parts = [name[:12] for name in sorted_names[:3]]
    preview = "_vs_".join(preview_parts)
    if len(sorted_names) > 3:
        preview += f"_plus{len(sorted_names) - 3}"
    return f"{preview}__{digest}"


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest rule-based alpha strategies.")
    parser.add_argument("--strategies", nargs="+", help="List of strategy names to include.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root directory containing strategies.")
    parser.add_argument("--output-root", default="runs", help="Root directory to store ensemble results.")
    parser.add_argument("--use-full-history", action="store_true", help="Backtest on full history instead of holdout.")
    parser.add_argument("--start-date", type=str, help="Optional YYYY-MM-DD to start backtest.")
    parser.add_argument("--end-date", type=str, help="Optional YYYY-MM-DD to end backtest (exclusive).")
    parser.add_argument(
        "--confirm-days",
        type=int,
        default=config.CONFIRM_DAYS,
        help="Required consecutive days before switching (non-sideways regimes).",
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
        "--regime-mapping",
        type=str,
        help="JSON mapping of regime_label -> strategy name for per-regime strategy selection.",
    )
    parser.add_argument(
        "--enable-emerging-universe",
        action="store_true",
        help="Enable monthly emerging-universe ticker schedule overlay.",
    )
    parser.add_argument("--emerging-top-n", type=int, default=150)
    parser.add_argument("--emerging-min-history-days", type=int, default=252)
    parser.add_argument("--emerging-min-price", type=float, default=20.0)
    parser.add_argument("--emerging-min-adv-rank", type=float, default=0.05)
    parser.add_argument("--emerging-max-adv-rank", type=float, default=0.85)
    parser.add_argument("--emerging-min-ret-6m", type=float, default=0.10)
    parser.add_argument("--emerging-min-ret-12m", type=float, default=-0.10)
    parser.add_argument("--emerging-min-adv-growth-6m", type=float, default=1.05)
    parser.add_argument("--emerging-rebalance-frequency", type=str, default="MS")
    return parser.parse_args()


def _build_invocation_command() -> str:
    parts = [sys.executable, *sys.argv]
    return " ".join(shlex.quote(str(part)) for part in parts)


def _append_backtest_run_log(entry: dict) -> str:
    run_log_dir = os.path.join(PROJECT_ROOT, "runs", "backtesting")
    os.makedirs(run_log_dir, exist_ok=True)
    run_log_path = os.path.join(run_log_dir, "backtester_runs.jsonl")
    payload = dict(entry)
    payload["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")
    return run_log_path


def main():
    args = parse_args()
    if args.rebalance_every < 1:
        raise ValueError("--rebalance-every must be >= 1.")
    strategy_roots = args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS)
    mapping = None
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
    if args.regime_mapping:
        try:
            mapping = json.loads(args.regime_mapping)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --regime-mapping: {exc}") from exc
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("--regime-mapping must be a non-empty JSON object.")
        mapping = {str(key): str(value) for key, value in mapping.items()}
        mapping_strategy_names = {value for value in mapping.values() if value}
        missing = sorted(name for name in mapping_strategy_names if name not in strategy_names)
        if missing:
            strategy_names.extend(missing)
    if not strategy_names:
        raise ValueError("No strategies found to backtest.")

    strategies = load_strategies(strategy_names, strategy_roots)
    if not strategies:
        raise ValueError("No valid strategies loaded.")
    mapping_selector = None
    if mapping:
        strategy_lookup = {strategy.name: strategy for strategy in strategies}
        for name in sorted(set(mapping.values())):
            if name and name not in strategy_lookup:
                raise ValueError(f"Unknown strategy in --regime-mapping: {name}")

        def mapping_selector(current_date, state, strategies):
            label = str(state.get("regime_label", "unknown"))
            chosen = mapping.get(label)
            if not chosen:
                return None
            selected = strategy_lookup.get(chosen)
            if selected is None:
                return None
            return [selected]

    data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    df = pd.read_parquet(data_path)
    validate_market_data_frame(
        df,
        source=data_path,
        required_columns=["Close"],
    )
    regime_table = compute_market_regime_table(df)
    regime_table = _apply_regime_confirmation(
        regime_table,
        confirm_days=args.confirm_days,
        confirm_days_sideways=args.confirm_days_sideways,
    )

    start_date = None
    end_date = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)

    if start_date is None and not (args.use_full_history or config.BACKTEST_USE_FULL_HISTORY):
        all_dates = df.index.get_level_values("date").unique().sort_values()
        split_index = int(len(all_dates) * config.TRAIN_RATIO)
        if split_index < len(all_dates):
            start_date = pd.to_datetime(all_dates[split_index])

    emerging_params = None
    allowed_tickers_by_date = None
    schedule_diagnostics = None
    if args.enable_emerging_universe:
        emerging_params = EmergingUniverseParams(
            enabled=True,
            rebalance_frequency=args.emerging_rebalance_frequency,
            top_n=max(1, int(args.emerging_top_n)),
            min_history_days=max(1, int(args.emerging_min_history_days)),
            min_price=float(args.emerging_min_price),
            min_adv_rank=float(args.emerging_min_adv_rank),
            max_adv_rank=float(args.emerging_max_adv_rank),
            min_ret_6m=float(args.emerging_min_ret_6m),
            min_ret_12m=float(args.emerging_min_ret_12m),
            min_adv_growth_6m=float(args.emerging_min_adv_growth_6m),
        )
        schedule = build_emerging_universe_schedule(
            df,
            params=emerging_params,
            start_date=start_date,
            end_date=end_date,
        )
        allowed_tickers_by_date = schedule.members_by_date
        schedule_diagnostics = schedule.diagnostics

    backtester = RuleBasedBacktester(
        df,
        strategies,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
        allowed_tickers_by_date=allowed_tickers_by_date,
    )
    result = backtester.run(
        start_date=start_date,
        end_date=end_date,
        rebalance_every=args.rebalance_every,
        min_weight_change=args.min_weight_change,
        min_trade_dollars=args.min_trade_dollars,
        max_daily_turnover=args.max_daily_turnover,
    )

    ensemble_dirname = make_ensemble_dirname([s.name for s in strategies])
    region = config.TRADING_REGION
    output_dir = os.path.join(PROJECT_ROOT, args.output_root, "_ensembles", region, ensemble_dirname)
    os.makedirs(output_dir, exist_ok=True)
    transactions_path = None
    plot_path = None

    if result.transactions:
        transactions_path = os.path.join(output_dir, "transactions.csv")
        pd.DataFrame(result.transactions).to_csv(transactions_path, index=False)
        print(f"Transaction log saved to {transactions_path}")

    if result.equity_curve and result.dates:
        plt.figure(figsize=(12, 6))
        plt.plot(result.dates, result.equity_curve)
        plt.title(f"Backtest Performance ({len(strategies)} Strategies)")
        plt.xlabel("Date")
        plt.ylabel("Net Worth ($)")
        plt.grid(True)
        plot_path = os.path.join(output_dir, "backtest_performance.png")
        plt.savefig(plot_path)
        print(f"Performance plot saved to {plot_path}")
    if schedule_diagnostics is not None and not schedule_diagnostics.empty:
        schedule_path = os.path.join(output_dir, "emerging_universe_schedule_diagnostics.csv")
        schedule_diagnostics.to_csv(schedule_path, index=False)
        print(f"Emerging universe schedule diagnostics saved to {schedule_path}")

    invocation_command = _build_invocation_command()
    results_payload = {
        "cagr": result.metrics.get("CAGR", 0.0) / 100.0,
        "final_net_worth": result.metrics.get("final_net_worth", config.INITIAL_CAPITAL),
        "metrics": result.metrics,
        "num_strategies": len(strategies),
        "strategies": [s.name for s in strategies],
        "ensemble_dir": ensemble_dirname,
        "rebalance_every": args.rebalance_every,
        "confirm_days": args.confirm_days,
        "confirm_days_sideways": args.confirm_days_sideways,
        "min_weight_change": args.min_weight_change,
        "min_trade_dollars": args.min_trade_dollars,
        "max_daily_turnover": args.max_daily_turnover,
        "rebalance_every_n_days": int(args.rebalance_every),
        "start_date": start_date.strftime("%Y-%m-%d") if isinstance(start_date, pd.Timestamp) else None,
        "end_date": end_date.strftime("%Y-%m-%d") if isinstance(end_date, pd.Timestamp) else None,
        "command": invocation_command,
        "argv": list(sys.argv[1:]),
        "emerging_universe_enabled": bool(args.enable_emerging_universe),
        "emerging_universe_params": emerging_params.__dict__ if emerging_params is not None else None,
        "emerging_universe_schedule_days": int(len(allowed_tickers_by_date or {})),
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    run_log_entry = {
        "command": invocation_command,
        "argv": list(sys.argv[1:]),
        "output_dir": output_dir,
        "results_path": results_path,
        "transactions_path": transactions_path,
        "plot_path": plot_path,
        "trading_region": config.TRADING_REGION,
        "data_file": config.DATA_FILE,
        "metrics": result.metrics,
        "final_net_worth": results_payload["final_net_worth"],
        "cagr": results_payload["cagr"],
        "strategies": results_payload["strategies"],
        "num_strategies": results_payload["num_strategies"],
        "rebalance_every": results_payload["rebalance_every"],
        "confirm_days": results_payload["confirm_days"],
        "confirm_days_sideways": results_payload["confirm_days_sideways"],
        "min_weight_change": results_payload["min_weight_change"],
        "min_trade_dollars": results_payload["min_trade_dollars"],
        "max_daily_turnover": results_payload["max_daily_turnover"],
        "start_date": results_payload["start_date"],
        "end_date": results_payload["end_date"],
    }
    try:
        run_log_path = _append_backtest_run_log(run_log_entry)
        print(f"Run log appended to {run_log_path}")
    except OSError as exc:
        print(f"Warning: failed to append run log ({exc})")

    print("\n--- Backtest Results ---")
    print(f"Final Net Worth: ${results_payload['final_net_worth']:,.2f} | CAGR: {results_payload['cagr']:.2%}")


if __name__ == "__main__":
    main()
