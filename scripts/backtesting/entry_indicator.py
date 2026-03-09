import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.market_data_validation import validate_market_data_frame
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import load_strategies


DEFAULT_BASE_STRATEGIES = [
    "india_rule_crash_resilient_slow",
    "india_rule_liquidity_momentum_core",
    "india_rule_pullback_reentry_slow",
]

DEFAULT_REGIME_MAPPING = {
    "bear_high_vol": "india_rule_range_stability_slow",
    "bear_low_vol": "india_rule_trend_carry_slow",
    "bull_high_vol": "india_rule_pullback_reentry_slow",
    "bull_low_vol": "india_rule_pullback_reentry_slow",
    "sideways_high_vol": "india_rule_range_stability_slow",
    "sideways_low_vol": "india_rule_liquidity_momentum_core",
}


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
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _empirical_cdf(sorted_values: np.ndarray, value: float) -> float:
    if sorted_values.size == 0:
        return 0.5
    pos = np.searchsorted(sorted_values, value, side="right")
    return float(pos) / float(sorted_values.size)


def _annual_metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {"CAGR": 0.0, "Max Drawdown": 0.0, "Sharpe Ratio": 0.0}
    eq = (1.0 + returns).cumprod()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    drawdown = float((eq / eq.cummax() - 1.0).min())
    vol = float(returns.std())
    sharpe = float(np.sqrt(252.0) * returns.mean() / vol) if vol > 1e-12 else 0.0
    return {"CAGR": cagr, "Max Drawdown": drawdown, "Sharpe Ratio": sharpe}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a data-driven market-entry indicator using historical forward returns "
            "of the configured strategy stack."
        )
    )
    parser.add_argument(
        "--strategy-roots",
        action="append",
        default=[],
        help="Root directory containing strategies.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_BASE_STRATEGIES),
        help="Base strategies to include (regime mapping strategies are auto-added).",
    )
    parser.add_argument(
        "--regime-mapping",
        type=str,
        default=json.dumps(DEFAULT_REGIME_MAPPING),
        help="JSON mapping of regime_label -> strategy name.",
    )
    parser.add_argument("--start-date", type=str, default="2013-01-01", help="Backtest start date.")
    parser.add_argument("--end-date", type=str, help="Optional backtest end date (exclusive).")
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Date to score (default: latest available backtest date).",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=126,
        help="Forward horizon in trading days used for entry-quality calibration.",
    )
    parser.add_argument(
        "--confirm-days",
        type=int,
        default=config.CONFIRM_DAYS,
        help="Regime confirmation days for non-sideways regimes.",
    )
    parser.add_argument(
        "--confirm-days-sideways",
        type=int,
        default=config.CONFIRM_DAYS_SIDEWAYS,
        help="Regime confirmation days for sideways regimes.",
    )
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=config.REBALANCE_EVERY,
        help="Rebalance cadence in trading days.",
    )
    parser.add_argument(
        "--ignore-stock-filters",
        action="store_true",
        help="Ignore excluded ticker file and quality-filter gating while building indicator history.",
    )
    parser.add_argument("--output-json", type=str, help="Optional output file for JSON payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lookahead_days < 20:
        raise ValueError("--lookahead-days must be >= 20.")

    try:
        mapping = json.loads(args.regime_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --regime-mapping JSON: {exc}") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("--regime-mapping must be a non-empty JSON object.")
    mapping = {str(k): str(v) for k, v in mapping.items()}

    orig_excluded = config.EXCLUDED_TICKERS_FILE
    orig_quality = config.ENABLE_UNIVERSE_QUALITY_FILTER
    if args.ignore_stock_filters:
        config.EXCLUDED_TICKERS_FILE = ""
        config.ENABLE_UNIVERSE_QUALITY_FILTER = False

    try:
        strategy_roots = args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS)
        strategy_names = list(args.strategies or [])
        for name in mapping.values():
            if name and name not in strategy_names:
                strategy_names.append(name)
        if not strategy_names:
            raise ValueError("No strategies resolved.")

        strategies = load_strategies(strategy_names, strategy_roots)
        if not strategies:
            raise ValueError("No valid strategies loaded.")
        strategy_lookup = {s.name: s for s in strategies}

        data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
        df = pd.read_parquet(data_path)
        validate_market_data_frame(df, source=data_path, required_columns=["Close"])

        regime_table = compute_market_regime_table(df)
        regime_table = _apply_regime_confirmation(
            regime_table,
            confirm_days=args.confirm_days,
            confirm_days_sideways=args.confirm_days_sideways,
        )

        def selector(current_date, state, _available):
            label = str(state.get("regime_label", "unknown"))
            chosen = mapping.get(label)
            selected = strategy_lookup.get(chosen)
            return [selected] if selected is not None else None

        backtester = RuleBasedBacktester(
            df,
            strategies,
            regime_table=regime_table,
            strategy_selector=selector,
        )
        result = backtester.run(
            start_date=pd.to_datetime(args.start_date),
            end_date=pd.to_datetime(args.end_date) if args.end_date else None,
            show_progress=False,
            rebalance_every=args.rebalance_every,
            min_weight_change=config.MIN_WEIGHT_CHANGE_TO_TRADE,
            min_trade_dollars=config.MIN_TRADE_DOLLARS,
            max_daily_turnover=config.MAX_DAILY_TURNOVER,
        )

        equity = pd.Series(
            np.asarray(result.equity_curve, dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(result.dates)),
            name="equity",
        ).sort_index()
        if equity.empty:
            raise ValueError("Backtest returned empty equity curve.")

        market_close = df.groupby(level="date")["Close"].mean().reindex(equity.index)
        states = regime_table.reindex(equity.index)
        feat = pd.DataFrame(index=equity.index)
        feat["equity"] = equity
        feat["trail_20"] = feat["equity"] / feat["equity"].shift(20) - 1.0
        feat["trail_60"] = feat["equity"] / feat["equity"].shift(60) - 1.0
        feat["trail_126"] = feat["equity"] / feat["equity"].shift(126) - 1.0
        feat["drawdown"] = feat["equity"] / feat["equity"].cummax() - 1.0
        feat["mkt_60"] = market_close / market_close.shift(60) - 1.0
        feat["breadth"] = states["breadth"]
        feat["regime_label"] = states["regime_label"].astype(str)
        feat["forward"] = feat["equity"].shift(-args.lookahead_days) / feat["equity"] - 1.0

        cols = ["trail_126", "trail_60", "mkt_60", "breadth", "drawdown", "forward"]
        cal = feat.dropna(subset=cols).copy()
        if len(cal) < 200:
            raise ValueError(
                f"Not enough calibration rows ({len(cal)}). Adjust start/end dates or lookahead horizon."
            )

        # Convert each factor to empirical percentile ranks, then combine.
        cal["p_trail_126"] = cal["trail_126"].rank(pct=True)
        cal["p_trail_60"] = cal["trail_60"].rank(pct=True)
        cal["p_mkt_60"] = cal["mkt_60"].rank(pct=True)
        cal["p_breadth"] = cal["breadth"].rank(pct=True)
        cal["p_drawdown"] = cal["drawdown"].rank(pct=True)  # less negative drawdown -> higher percentile

        weights = {
            "p_trail_126": 0.35,
            "p_trail_60": 0.20,
            "p_mkt_60": 0.20,
            "p_breadth": 0.15,
            "p_drawdown": 0.10,
        }
        cal["entry_score"] = 100.0 * sum(cal[k] * w for k, w in weights.items())

        decile_edges = np.quantile(cal["entry_score"], np.linspace(0.0, 1.0, 11))
        decile_edges = np.unique(decile_edges)
        if len(decile_edges) < 3:
            raise ValueError("Entry-score distribution collapsed; cannot build deciles.")
        cal["score_bucket"] = np.searchsorted(decile_edges, cal["entry_score"], side="right") - 1
        cal["score_bucket"] = cal["score_bucket"].clip(0, len(decile_edges) - 2)

        bucket_stats = cal.groupby("score_bucket")["forward"].agg(
            count="size",
            forward_mean="mean",
            forward_median="median",
            forward_p10=lambda s: s.quantile(0.10),
            prob_positive=lambda s: (s > 0).mean(),
        )

        as_of = pd.to_datetime(args.as_of_date) if args.as_of_date else feat.index.max()
        eligible = feat.index[feat.index <= as_of]
        if eligible.empty:
            raise ValueError(f"No backtest rows at or before as-of date: {as_of.date()}")
        as_of_idx = eligible.max()
        point = feat.loc[as_of_idx]
        if point[["trail_126", "trail_60", "mkt_60", "breadth", "drawdown"]].isna().any():
            raise ValueError(
                f"Insufficient feature history at as-of date {as_of_idx.date()} "
                "(need at least 126 trading days of history)."
            )

        sorted_map = {
            "trail_126": np.sort(cal["trail_126"].to_numpy(dtype=float)),
            "trail_60": np.sort(cal["trail_60"].to_numpy(dtype=float)),
            "mkt_60": np.sort(cal["mkt_60"].to_numpy(dtype=float)),
            "breadth": np.sort(cal["breadth"].to_numpy(dtype=float)),
            "drawdown": np.sort(cal["drawdown"].to_numpy(dtype=float)),
        }
        pvals = {
            "p_trail_126": _empirical_cdf(sorted_map["trail_126"], float(point["trail_126"])),
            "p_trail_60": _empirical_cdf(sorted_map["trail_60"], float(point["trail_60"])),
            "p_mkt_60": _empirical_cdf(sorted_map["mkt_60"], float(point["mkt_60"])),
            "p_breadth": _empirical_cdf(sorted_map["breadth"], float(point["breadth"])),
            "p_drawdown": _empirical_cdf(sorted_map["drawdown"], float(point["drawdown"])),
        }
        score = 100.0 * sum(pvals[k] * w for k, w in weights.items())
        bucket = int(np.searchsorted(decile_edges, score, side="right") - 1)
        bucket = int(min(max(bucket, 0), len(decile_edges) - 2))
        bstats = bucket_stats.loc[bucket]

        prob_positive = _safe_float(bstats["prob_positive"])
        median_forward = _safe_float(bstats["forward_median"])
        p10_forward = _safe_float(bstats["forward_p10"])
        if prob_positive >= 0.70 and p10_forward > -0.10 and median_forward > 0.10:
            signal = "GREEN"
            deploy_fraction = 1.00
        elif prob_positive >= 0.55 and median_forward > 0.03:
            signal = "AMBER"
            deploy_fraction = 0.50
        else:
            signal = "RED"
            deploy_fraction = 0.00

        backtest_returns = equity.pct_change().fillna(0.0)
        perf = _annual_metrics_from_returns(backtest_returns)

        payload = {
            "as_of_date": as_of_idx.strftime("%Y-%m-%d"),
            "signal": signal,
            "entry_score": round(score, 4),
            "score_bucket": int(bucket),
            "score_bucket_bounds": [
                round(float(decile_edges[bucket]), 4),
                round(float(decile_edges[bucket + 1]), 4),
            ],
            "recommended_deploy_fraction": deploy_fraction,
            "expected_forward_stats": {
                "lookahead_days": int(args.lookahead_days),
                "prob_positive": round(prob_positive, 4),
                "median_return": round(median_forward, 4),
                "mean_return": round(_safe_float(bstats["forward_mean"]), 4),
                "p10_return": round(p10_forward, 4),
                "sample_size": int(_safe_float(bstats["count"], 0)),
            },
            "features": {
                "regime_label": str(point["regime_label"]),
                "strategy_trail_20": round(float(point["trail_20"]), 6),
                "strategy_trail_60": round(float(point["trail_60"]), 6),
                "strategy_trail_126": round(float(point["trail_126"]), 6),
                "strategy_drawdown": round(float(point["drawdown"]), 6),
                "market_trail_60": round(float(point["mkt_60"]), 6),
                "breadth": round(float(point["breadth"]), 6),
            },
            "factor_percentiles": {k: round(float(v), 6) for k, v in pvals.items()},
            "calibration": {
                "start_date": cal.index.min().strftime("%Y-%m-%d"),
                "end_date": cal.index.max().strftime("%Y-%m-%d"),
                "rows": int(len(cal)),
            },
            "backtest_context": {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "rebalance_every": int(args.rebalance_every),
                "confirm_days": int(args.confirm_days),
                "confirm_days_sideways": int(args.confirm_days_sideways),
                "ignore_stock_filters": bool(args.ignore_stock_filters),
                "cagr": round(_safe_float(perf["CAGR"]), 6),
                "max_drawdown": round(_safe_float(perf["Max Drawdown"]), 6),
                "sharpe": round(_safe_float(perf["Sharpe Ratio"]), 6),
            },
        }

        out = json.dumps(payload, indent=2)
        print(out)
        if args.output_json:
            out_path = args.output_json
            if not os.path.isabs(out_path):
                out_path = os.path.join(PROJECT_ROOT, out_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out)
                f.write("\n")
            print(f"\nSaved indicator payload to {out_path}")
    finally:
        config.EXCLUDED_TICKERS_FILE = orig_excluded
        config.ENABLE_UNIVERSE_QUALITY_FILTER = orig_quality


if __name__ == "__main__":
    main()
