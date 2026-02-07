import os
import sys
import json
import argparse
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.market_data_validation import validate_market_data_frame
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import list_strategy_names, load_strategies


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
    parser.add_argument("--output-root", default="alphas", help="Root directory to store ensemble results.")
    parser.add_argument("--use-full-history", action="store_true", help="Backtest on full history instead of holdout.")
    parser.add_argument("--start-date", type=str, help="Optional YYYY-MM-DD to start backtest.")
    parser.add_argument("--end-date", type=str, help="Optional YYYY-MM-DD to end backtest (exclusive).")
    parser.add_argument(
        "--regime-mapping",
        type=str,
        help="JSON mapping of regime_label -> strategy name for per-regime strategy selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_roots = args.strategy_roots or ["alphas"]
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

    backtester = RuleBasedBacktester(
        df,
        strategies,
        regime_table=regime_table,
        strategy_selector=mapping_selector,
    )
    result = backtester.run(start_date=start_date, end_date=end_date)

    ensemble_dirname = make_ensemble_dirname([s.name for s in strategies])
    region = config.TRADING_REGION
    output_dir = os.path.join(PROJECT_ROOT, args.output_root, "_ensembles", region, ensemble_dirname)
    os.makedirs(output_dir, exist_ok=True)

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

    results_payload = {
        "cagr": result.metrics.get("CAGR", 0.0) / 100.0,
        "final_net_worth": result.metrics.get("final_net_worth", config.INITIAL_CAPITAL),
        "metrics": result.metrics,
        "num_strategies": len(strategies),
        "strategies": [s.name for s in strategies],
        "ensemble_dir": ensemble_dirname,
        "start_date": start_date.strftime("%Y-%m-%d") if isinstance(start_date, pd.Timestamp) else None,
        "end_date": end_date.strftime("%Y-%m-%d") if isinstance(end_date, pd.Timestamp) else None,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_payload, f, indent=2)

    print("\n--- Backtest Results ---")
    print(f"Final Net Worth: ${results_payload['final_net_worth']:,.2f} | CAGR: {results_payload['cagr']:.2%}")


if __name__ == "__main__":
    main()
