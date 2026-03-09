import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.market_data_validation import validate_market_data_frame
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import list_strategy_names, load_strategies
from src.utils import calculate_performance_metrics


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
    out = regime_table.copy()
    labels = out["regime_label"].astype(str)
    confirmed = _apply_confirmed_switch(
        labels,
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


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _selector_from_mapping(mapping: dict[str, str], strategies):
    strategy_lookup = {strategy.name: strategy for strategy in strategies}

    def _selector(current_date, state, _available):
        label = str(state.get("regime_label", "unknown"))
        chosen = mapping.get(label)
        selected = strategy_lookup.get(chosen)
        return [selected] if selected is not None else None

    return _selector


def _series_from_backtest_result(result) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(result.dates))
    values = pd.Series(np.asarray(result.equity_curve, dtype=float), index=idx)
    values = values[~values.index.duplicated(keep="last")]
    return values.sort_index()


def _build_phase_averaged_portfolio(series_list: list[pd.Series], initial_capital: float) -> pd.Series | None:
    if not series_list:
        return None
    common_index = series_list[0].index
    for series in series_list[1:]:
        common_index = common_index.intersection(series.index)
    common_index = common_index.sort_values()
    if len(common_index) < 2:
        return None

    returns = []
    for series in series_list:
        aligned = series.reindex(common_index)
        returns.append(aligned.pct_change().fillna(0.0))
    returns_df = pd.concat(returns, axis=1)
    mean_returns = returns_df.mean(axis=1)
    equity = float(initial_capital) * (1.0 + mean_returns).cumprod()
    return equity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rebalance-phase robustness via start-date offsets and "
            "a staggered-sleeves aggregate portfolio."
        )
    )
    parser.add_argument("--strategies", nargs="+", help="List of strategy names to include.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root directory containing strategies.")
    parser.add_argument("--start-date", type=str, required=True, help="Anchor start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, required=True, help="End date (exclusive, YYYY-MM-DD).")
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
        "--offset-count",
        type=int,
        default=None,
        help="Number of consecutive trading-day offsets to evaluate (default: rebalance-every).",
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
        default=None,
        help="JSON mapping of regime_label -> strategy name.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs/backtesting",
        help="Output root directory for robustness reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebalance_every < 1:
        raise ValueError("--rebalance-every must be >= 1.")
    offset_count = int(args.offset_count) if args.offset_count is not None else int(args.rebalance_every)
    if offset_count < 1:
        raise ValueError("--offset-count must be >= 1.")

    strategy_roots = args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS)
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
    if not strategy_names:
        raise ValueError("No strategies found.")

    mapping = None
    if args.regime_mapping:
        try:
            mapping = json.loads(args.regime_mapping)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --regime-mapping: {exc}") from exc
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("--regime-mapping must be a non-empty JSON object.")
        mapping = {str(k): str(v) for k, v in mapping.items()}
        needed = sorted(set(mapping.values()))
        missing = sorted(name for name in needed if name not in strategy_names)
        if missing:
            strategy_names.extend(missing)

    strategies = load_strategies(strategy_names, strategy_roots)
    if not strategies:
        raise ValueError("No valid strategies loaded.")

    data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    df = pd.read_parquet(data_path)
    validate_market_data_frame(df, source=data_path, required_columns=["Close"])

    regime_table = compute_market_regime_table(df)
    regime_table = _apply_regime_confirmation(
        regime_table,
        confirm_days=int(args.confirm_days),
        confirm_days_sideways=int(args.confirm_days_sideways),
    )

    selector = _selector_from_mapping(mapping, strategies) if mapping else None
    backtester = RuleBasedBacktester(
        df=df,
        strategies=strategies,
        regime_table=regime_table,
        strategy_selector=selector,
    )

    start_anchor = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    candidate_dates = [d for d in backtester.dates if d >= start_anchor and d < end_date]
    if not candidate_dates:
        raise ValueError("No trading dates available in the provided date range.")
    start_dates = candidate_dates[:offset_count]
    if len(start_dates) < offset_count:
        print(f"Only {len(start_dates)} offsets available in range; evaluating all available offsets.")

    max_daily_turnover = float(args.max_daily_turnover) if args.max_daily_turnover is not None else None
    if max_daily_turnover is not None and max_daily_turnover <= 0:
        max_daily_turnover = None

    rows: list[dict] = []
    offset_series: list[pd.Series] = []
    for i, current_start in enumerate(start_dates, start=1):
        result = backtester.run(
            start_date=current_start,
            end_date=end_date,
            show_progress=False,
            rebalance_every=int(args.rebalance_every),
            min_weight_change=float(args.min_weight_change),
            min_trade_dollars=float(args.min_trade_dollars),
            max_daily_turnover=max_daily_turnover,
        )
        metrics = result.metrics
        rows.append(
            {
                "offset_index": i - 1,
                "start_date": pd.Timestamp(current_start).strftime("%Y-%m-%d"),
                "cagr_pct": _to_float(metrics.get("CAGR")),
                "sharpe": _to_float(metrics.get("Sharpe Ratio")),
                "max_drawdown": _to_float(metrics.get("Max Drawdown")),
                "final_net_worth": _to_float(metrics.get("final_net_worth")),
            }
        )
        offset_series.append(_series_from_backtest_result(result))
        print(f"completed {i}/{len(start_dates)} offsets ({rows[-1]['start_date']})")

    detail_df = pd.DataFrame(rows).sort_values("offset_index")
    baseline = detail_df.iloc[0]
    detail_df["delta_cagr_vs_baseline"] = detail_df["cagr_pct"] - _to_float(baseline["cagr_pct"])
    detail_df["delta_final_vs_baseline"] = detail_df["final_net_worth"] - _to_float(baseline["final_net_worth"])

    cagr_values = detail_df["cagr_pct"].to_numpy(dtype=float)
    final_values = detail_df["final_net_worth"].to_numpy(dtype=float)
    summary = {
        "offset_count": int(len(detail_df)),
        "baseline_start_date": str(baseline["start_date"]),
        "baseline_cagr_pct": _to_float(baseline["cagr_pct"]),
        "baseline_final_net_worth": _to_float(baseline["final_net_worth"]),
        "median_cagr_pct": float(np.median(cagr_values)),
        "p10_cagr_pct": float(np.percentile(cagr_values, 10)),
        "p90_cagr_pct": float(np.percentile(cagr_values, 90)),
        "min_cagr_pct": float(np.min(cagr_values)),
        "max_cagr_pct": float(np.max(cagr_values)),
        "cagr_range_pct_points": float(np.max(cagr_values) - np.min(cagr_values)),
        "median_final_net_worth": float(np.median(final_values)),
        "p10_final_net_worth": float(np.percentile(final_values, 10)),
        "p90_final_net_worth": float(np.percentile(final_values, 90)),
        "min_final_net_worth": float(np.min(final_values)),
        "max_final_net_worth": float(np.max(final_values)),
        "final_net_worth_range": float(np.max(final_values) - np.min(final_values)),
    }

    staggered_equity = _build_phase_averaged_portfolio(offset_series, initial_capital=float(config.INITIAL_CAPITAL))
    staggered_summary = None
    if staggered_equity is not None:
        metrics = calculate_performance_metrics(staggered_equity, len(staggered_equity))
        staggered_summary = {
            "num_common_days": int(len(staggered_equity)),
            "start_date": str(staggered_equity.index[0].date()),
            "end_date": str(staggered_equity.index[-1].date()),
            "final_net_worth": float(staggered_equity.iloc[-1]),
            "cagr_pct": _to_float(metrics.get("CAGR")),
            "sharpe": _to_float(metrics.get("Sharpe Ratio")),
            "max_drawdown": _to_float(metrics.get("Max Drawdown")),
        }

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, args.output_root, f"phase_robustness_{stamp}")
    os.makedirs(output_dir, exist_ok=True)

    detail_path = os.path.join(output_dir, "offset_detail.csv")
    detail_df.to_csv(detail_path, index=False)

    if staggered_equity is not None:
        staggered_path = os.path.join(output_dir, "staggered_equity.csv")
        staggered_equity.rename("net_worth").to_csv(staggered_path, index_label="date")
    else:
        staggered_path = None

    payload = {
        "config": {
            "data_file": config.DATA_FILE,
            "strategy_roots": strategy_roots,
            "strategy_names": [s.name for s in strategies],
            "start_date": str(start_anchor.date()),
            "end_date": str(end_date.date()),
            "confirm_days": int(args.confirm_days),
            "confirm_days_sideways": int(args.confirm_days_sideways),
            "rebalance_every": int(args.rebalance_every),
            "offset_count": int(len(detail_df)),
            "min_weight_change": float(args.min_weight_change),
            "min_trade_dollars": float(args.min_trade_dollars),
            "max_daily_turnover": max_daily_turnover,
            "weight_smoothing": float(config.WEIGHT_SMOOTHING),
            "regime_mapping": mapping,
        },
        "offset_summary": summary,
        "staggered_sleeves_summary": staggered_summary,
        "artifacts": {
            "offset_detail_csv": detail_path,
            "staggered_equity_csv": staggered_path,
        },
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nOffset Robustness Summary")
    print(json.dumps(summary, indent=2))
    if staggered_summary:
        print("\nStaggered Sleeves Summary")
        print(json.dumps(staggered_summary, indent=2))
    print(f"\nArtifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
