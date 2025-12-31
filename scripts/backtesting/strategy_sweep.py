import os
import sys
import json
import argparse
from datetime import datetime

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.strategy import list_strategy_names, load_strategies
from src.strategy_sweep import METRIC_KEYS, sweep_strategy_combinations


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep strategy combinations and report the best result.")
    parser.add_argument("--strategies", nargs="+", help="List of strategy names to consider.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root directory containing strategies.")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum number of strategies in a combo.")
    parser.add_argument("--max-size", type=int, default=None, help="Maximum number of strategies in a combo.")
    parser.add_argument("--max-combos", type=int, default=None, help="Cap the number of combos to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling combos.")
    parser.add_argument("--metric", choices=sorted(METRIC_KEYS.keys()), default="cagr", help="Metric to optimize.")
    parser.add_argument("--start-date", type=str, help="Optional YYYY-MM-DD to start backtest.")
    parser.add_argument("--end-date", type=str, help="Optional YYYY-MM-DD to end backtest (exclusive).")
    parser.add_argument("--output-root", default="alphas", help="Root directory to store sweep results.")
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_roots = args.strategy_roots or ["alphas"]
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
    if not strategy_names:
        raise ValueError("No strategies found to sweep.")

    strategies = load_strategies(strategy_names, strategy_roots)
    strategies_by_name = {s.name: s for s in strategies}
    if not strategies_by_name:
        raise ValueError("No valid strategies loaded.")

    data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    df = pd.read_parquet(data_path)

    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    result = sweep_strategy_combinations(
        df=df,
        strategies_by_name=strategies_by_name,
        min_size=args.min_size,
        max_size=args.max_size,
        max_combos=args.max_combos,
        seed=args.seed,
        metric=args.metric,
        start_date=start_date,
        end_date=end_date,
    )

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    region = config.TRADING_REGION
    output_dir = os.path.join(PROJECT_ROOT, args.output_root, "_ensembles", region, "_sweeps", stamp)
    os.makedirs(output_dir, exist_ok=True)

    results_payload = {
        "best_combo": list(result.best_combo),
        "best_metrics": result.best_metrics,
        "metric": args.metric,
        "rows": result.rows,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
        json.dump(results_payload, f, indent=2)

    print("\n--- Sweep Results ---")
    print(f"Metric: {args.metric}")
    print(f"Best combo: {list(result.best_combo)}")
    best_cagr = result.best_metrics.get("CAGR", 0.0)
    print(f"Best CAGR: {best_cagr:.2f}%")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
