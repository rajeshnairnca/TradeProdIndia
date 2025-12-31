import os
import sys
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import load_strategies
from src.utils import calculate_performance_metrics


def _make_ensemble_dirname(strategy_names: list[str]) -> str:
    if len(strategy_names) == 1:
        return strategy_names[0]
    import hashlib

    sorted_names = sorted(strategy_names)
    digest = hashlib.sha1("::".join(sorted_names).encode("utf-8")).hexdigest()[:8]
    preview = "_vs_".join(name[:12] for name in sorted_names[:3])
    if len(sorted_names) > 3:
        preview += f"_plus{len(sorted_names) - 3}"
    return f"{preview}__{digest}"


def run_walk_forward(strategy_names: list[str], strategy_roots: list[str] | None = None):
    """
    Performs walk-forward validation for one or more rule-based strategies.
    """
    roots = strategy_roots or ["alphas"]
    print(f"--- Starting Walk-Forward Validation for: {strategy_names} ---")

    df = pd.read_parquet(os.path.join(PROJECT_ROOT, config.DATA_FILE))
    strategies = load_strategies(strategy_names, roots)
    if not strategies:
        raise ValueError("No strategies found for walk-forward validation.")

    all_dates = df.index.get_level_values("date").unique().sort_values()
    trading_days_per_year = 252
    train_window = config.TRAIN_YEARS * trading_days_per_year
    validation_window = config.VALIDATION_YEARS * trading_days_per_year

    overall_net_worths = []
    all_fold_results = {}

    for i in range(config.N_SPLITS):
        print(f"--- Running Fold {i+1}/{config.N_SPLITS} ---")

        train_start_idx = i * validation_window
        train_end_idx = train_start_idx + train_window
        validation_start_idx = train_end_idx
        validation_end_idx = validation_start_idx + validation_window

        if validation_end_idx > len(all_dates):
            print("Not enough data for this fold. Stopping walk-forward validation.")
            break

        train_start_date, train_end_date = all_dates[train_start_idx], all_dates[train_end_idx]
        validation_start_date, validation_end_date = all_dates[validation_start_idx], all_dates[validation_end_idx]

        print(f"Train Period: {train_start_date.date()} to {train_end_date.date()}")
        print(f"Validation Period: {validation_start_date.date()} to {validation_end_date.date()}")

        fold_df = df[
            (df.index.get_level_values("date") >= train_start_date)
            & (df.index.get_level_values("date") < validation_end_date)
        ]
        fold_regime_table = compute_market_regime_table(fold_df)

        backtester = RuleBasedBacktester(fold_df, strategies, regime_table=fold_regime_table)
        result = backtester.run(start_date=validation_start_date, end_date=validation_end_date)

        fold_net_worths = result.equity_curve
        if not fold_net_worths:
            print(f"No results for fold {i+1}; skipping.")
            continue

        fold_min = min(fold_net_worths)
        fold_max = max(fold_net_worths)
        fold_final = fold_net_worths[-1]
        print(f"Fold {i+1} diagnostics: min_worth={fold_min:,.2f}, max_worth={fold_max:,.2f}, final_worth={fold_final:,.2f}")

        output_root = roots[0]
        ensemble_dir = _make_ensemble_dirname(strategy_names)
        if len(strategy_names) == 1:
            diag_path = os.path.join(PROJECT_ROOT, output_root, strategy_names[0], f"fold_{i+1}_diagnostics.json")
        else:
            diag_path = os.path.join(
                PROJECT_ROOT,
                output_root,
                "_ensembles",
                config.TRADING_REGION,
                ensemble_dir,
                f"fold_{i+1}_diagnostics.json",
            )
        os.makedirs(os.path.dirname(diag_path), exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump(
                {
                    "fold": i + 1,
                    "min_worth": float(fold_min),
                    "max_worth": float(fold_max),
                    "final_worth": float(fold_final),
                },
                f,
                indent=2,
            )

        fold_days = len(fold_net_worths)
        fold_metrics = calculate_performance_metrics(pd.Series(fold_net_worths), fold_days)
        print(f"Fold {i+1} CAGR: {fold_metrics['CAGR']:.2f}%")
        all_fold_results[f"Fold_{i+1}"] = fold_metrics

        if not overall_net_worths:
            overall_net_worths.extend(fold_net_worths)
        else:
            last_overall = overall_net_worths[-1]
            scaled_fold = [w * (last_overall / fold_net_worths[0]) for w in fold_net_worths[1:]]
            overall_net_worths.extend(scaled_fold)

    print("\n--- Walk-Forward Validation Summary ---")
    if not overall_net_worths:
        print("Validation could not be completed. No results to show.")
        return None

    total_days = len(overall_net_worths)
    final_metrics = calculate_performance_metrics(pd.Series(overall_net_worths), total_days)

    print(f"Overall CAGR: {final_metrics['CAGR']:.2f}%")
    print(f"Overall Sharpe Ratio: {final_metrics['Sharpe Ratio']:.2f}")
    print(f"Overall Max Drawdown: {final_metrics['Max Drawdown']:.2f}%")

    output_root = roots[0]
    ensemble_dir = _make_ensemble_dirname(strategy_names)
    if len(strategy_names) == 1:
        plot_dir = os.path.join(PROJECT_ROOT, output_root, strategy_names[0])
    else:
        plot_dir = os.path.join(PROJECT_ROOT, output_root, "_ensembles", config.TRADING_REGION, ensemble_dir)
    os.makedirs(plot_dir, exist_ok=True)
    label = strategy_names[0] if len(strategy_names) == 1 else ensemble_dir
    plot_path = os.path.join(plot_dir, f"{label}_walkforward_performance.png")
    plt.figure(figsize=(12, 6))
    plt.plot(overall_net_worths)
    plt.title(f'Walk-Forward Validation Equity Curve for "{label}"')
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Saved walk-forward performance chart to {plot_path}")

    output_path = os.path.join(plot_dir, "walk_forward_results.json")
    print(f"Saving walk-forward validation results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run walk-forward validation for one or more strategies.")
    parser.add_argument("--alpha-name", type=str, help="Name of a single strategy to validate.")
    parser.add_argument("--strategies", nargs="+", help="List of strategies to validate together.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root folder containing strategies.")
    args = parser.parse_args()

    roots = args.strategy_roots or ["alphas"]
    if args.strategies:
        names = args.strategies
    elif args.alpha_name:
        names = [args.alpha_name]
    else:
        raise SystemExit("Provide --alpha-name or --strategies.")

    run_walk_forward(names, strategy_roots=roots)
