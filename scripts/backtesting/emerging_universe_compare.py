import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

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
    out["trend_state"] = trend_state_new.fillna(out.get("trend_state", "unknown"))
    out["trend_up"] = trend_up_new.fillna(out.get("trend_up", False))
    out["vol_high"] = vol_high_new.fillna(out.get("vol_high", False))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline backtest versus emerging-universe monthly overlay."
    )
    parser.add_argument("--strategies", nargs="+", help="List of strategy names to include.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root directory containing strategies.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD (exclusive).")
    parser.add_argument("--confirm-days", type=int, default=config.CONFIRM_DAYS)
    parser.add_argument("--confirm-days-sideways", type=int, default=config.CONFIRM_DAYS_SIDEWAYS)
    parser.add_argument("--rebalance-every", type=int, default=config.REBALANCE_EVERY)
    parser.add_argument("--min-weight-change", type=float, default=config.MIN_WEIGHT_CHANGE_TO_TRADE)
    parser.add_argument("--min-trade-dollars", type=float, default=config.MIN_TRADE_DOLLARS)
    parser.add_argument("--max-daily-turnover", type=float, default=config.MAX_DAILY_TURNOVER)
    parser.add_argument("--regime-mapping", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="runs/backtesting")
    parser.add_argument("--show-progress", action="store_true", help="Show backtest progress bars.")

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


def _to_ts(value: str) -> pd.Timestamp:
    return pd.to_datetime(value).tz_localize(None)


def _selector_from_mapping(mapping: dict[str, str], strategies):
    strategy_lookup = {strategy.name: strategy for strategy in strategies}

    def _selector(current_date, state, _available):
        label = str(state.get("regime_label", "unknown"))
        chosen = mapping.get(label)
        selected = strategy_lookup.get(chosen)
        return [selected] if selected is not None else None

    return _selector


def _result_payload(result) -> dict:
    metrics = result.metrics or {}
    return {
        "CAGR": float(metrics.get("CAGR", 0.0)),
        "Sharpe Ratio": float(metrics.get("Sharpe Ratio", 0.0)),
        "Max Drawdown": float(metrics.get("Max Drawdown", 0.0)),
        "final_net_worth": float(metrics.get("final_net_worth", config.INITIAL_CAPITAL)),
    }


def main() -> None:
    args = parse_args()
    strategy_roots = args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS)
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
    if not strategy_names:
        raise ValueError("No strategies found.")

    mapping_selector = None
    mapping_payload = None
    if args.regime_mapping:
        mapping_payload = json.loads(args.regime_mapping)
        if not isinstance(mapping_payload, dict) or not mapping_payload:
            raise ValueError("--regime-mapping must be a non-empty JSON object.")
        mapping_payload = {str(k): str(v) for k, v in mapping_payload.items()}
        needed = sorted({v for v in mapping_payload.values() if v})
        missing = sorted(name for name in needed if name not in strategy_names)
        if missing:
            strategy_names.extend(missing)

    strategies = load_strategies(strategy_names, strategy_roots)
    if not strategies:
        raise ValueError("No valid strategies loaded.")
    if mapping_payload:
        mapping_selector = _selector_from_mapping(mapping_payload, strategies)

    data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    df = pd.read_parquet(data_path)
    validate_market_data_frame(df, source=data_path, required_columns=["Close", "adv_21"])

    regime_table = compute_market_regime_table(df)
    regime_table = _apply_regime_confirmation(
        regime_table,
        confirm_days=args.confirm_days,
        confirm_days_sideways=args.confirm_days_sideways,
    )

    start_date = _to_ts(args.start_date)
    end_date = _to_ts(args.end_date)
    if end_date <= start_date:
        raise ValueError("--end-date must be after --start-date.")

    baseline = RuleBasedBacktester(
        df,
        strategies,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
    ).run(
        start_date=start_date,
        end_date=end_date,
        show_progress=bool(args.show_progress),
        rebalance_every=args.rebalance_every,
        min_weight_change=args.min_weight_change,
        min_trade_dollars=args.min_trade_dollars,
        max_daily_turnover=args.max_daily_turnover,
    )

    params = EmergingUniverseParams(
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
        params=params,
        start_date=start_date,
        end_date=end_date,
    )

    emerging = RuleBasedBacktester(
        df,
        strategies,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
        allowed_tickers_by_date=schedule.members_by_date,
    ).run(
        start_date=start_date,
        end_date=end_date,
        show_progress=bool(args.show_progress),
        rebalance_every=args.rebalance_every,
        min_weight_change=args.min_weight_change,
        min_trade_dollars=args.min_trade_dollars,
        max_daily_turnover=args.max_daily_turnover,
    )

    baseline_payload = _result_payload(baseline)
    emerging_payload = _result_payload(emerging)
    delta_payload = {
        key: float(emerging_payload.get(key, 0.0) - baseline_payload.get(key, 0.0))
        for key in baseline_payload.keys()
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, args.output_root, f"emerging_universe_compare_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "strategies": [s.name for s in strategies],
        "rebalance_every": int(args.rebalance_every),
        "confirm_days": int(args.confirm_days),
        "confirm_days_sideways": int(args.confirm_days_sideways),
        "min_weight_change": float(args.min_weight_change),
        "min_trade_dollars": float(args.min_trade_dollars),
        "max_daily_turnover": None if args.max_daily_turnover is None or float(args.max_daily_turnover) <= 0 else float(args.max_daily_turnover),
        "emerging_params": params.__dict__,
        "baseline": baseline_payload,
        "emerging_overlay": emerging_payload,
        "delta_emerging_minus_baseline": delta_payload,
        "schedule_rebalance_dates": len(schedule.rebalance_dates),
        "schedule_days_covered": len(schedule.members_by_date),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not schedule.diagnostics.empty:
        schedule.diagnostics.to_csv(os.path.join(out_dir, "schedule_diagnostics.csv"), index=False)

    print(f"Output directory: {out_dir}")
    print("--- Baseline ---")
    print(
        f"CAGR: {baseline_payload['CAGR']:.4f}% | Sharpe: {baseline_payload['Sharpe Ratio']:.4f} "
        f"| Max DD: {baseline_payload['Max Drawdown']:.4f}% | Final: {baseline_payload['final_net_worth']:.2f}"
    )
    print("--- Emerging Overlay ---")
    print(
        f"CAGR: {emerging_payload['CAGR']:.4f}% | Sharpe: {emerging_payload['Sharpe Ratio']:.4f} "
        f"| Max DD: {emerging_payload['Max Drawdown']:.4f}% | Final: {emerging_payload['final_net_worth']:.2f}"
    )
    print("--- Delta (Emerging - Baseline) ---")
    print(
        f"CAGR: {delta_payload['CAGR']:.4f} pp | Sharpe: {delta_payload['Sharpe Ratio']:.4f} "
        f"| Max DD: {delta_payload['Max Drawdown']:.4f} pp | Final: {delta_payload['final_net_worth']:.2f}"
    )
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
