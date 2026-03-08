import os
import sys
import json
import argparse
import random
from datetime import datetime
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.market_data_validation import validate_market_data_frame
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import list_strategy_names, load_strategies
from src.strategy_sweep import METRIC_KEYS, SweepResult, sweep_strategy_combinations

_WORK_DF = None
_WORK_REGIME_TABLE = None
_WORK_STRATEGIES_BY_NAME = None
_WORK_START_DATE = None
_WORK_END_DATE = None
_WORK_METRIC_KEY = None
_WORK_REBALANCE_EVERY_DAYS = None


def _generate_combos(names: list[str], min_size: int, max_size: int):
    for size in range(min_size, max_size + 1):
        yield from combinations(names, size)


def _sample_combos(combos: list[tuple[str, ...]], max_combos: int | None, seed: int):
    if not max_combos or len(combos) <= max_combos:
        return combos
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:max_combos]


def _init_worker(
    data_path: str,
    strategy_names: list[str],
    strategy_roots: list[str],
    start_date,
    end_date,
    metric_key: str,
    rebalance_every_days: int,
):
    global _WORK_DF
    global _WORK_REGIME_TABLE
    global _WORK_STRATEGIES_BY_NAME
    global _WORK_START_DATE
    global _WORK_END_DATE
    global _WORK_METRIC_KEY
    global _WORK_REBALANCE_EVERY_DAYS

    df = pd.read_parquet(data_path)
    validate_market_data_frame(df, source=data_path, required_columns=["Close"])
    regime_table = compute_market_regime_table(df)
    strategies = load_strategies(strategy_names, strategy_roots)
    strategies_by_name = {s.name: s for s in strategies}

    _WORK_DF = df
    _WORK_REGIME_TABLE = regime_table
    _WORK_STRATEGIES_BY_NAME = strategies_by_name
    _WORK_START_DATE = start_date
    _WORK_END_DATE = end_date
    _WORK_METRIC_KEY = metric_key
    _WORK_REBALANCE_EVERY_DAYS = rebalance_every_days


def _evaluate_combo(combo: tuple[str, ...]):
    strategies = [_WORK_STRATEGIES_BY_NAME[name] for name in combo if name in _WORK_STRATEGIES_BY_NAME]
    backtester = RuleBasedBacktester(
        _WORK_DF,
        strategies,
        regime_table=_WORK_REGIME_TABLE,
        rebalance_every_n_days=_WORK_REBALANCE_EVERY_DAYS,
    )
    result = backtester.run(start_date=_WORK_START_DATE, end_date=_WORK_END_DATE)
    metrics = result.metrics
    score = float(metrics.get(_WORK_METRIC_KEY, 0.0))
    return combo, score, metrics


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
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=config.REBALANCE_EVERY_N_DAYS,
        help="Rebalance cadence in trading days (default from REBALANCE_EVERY_N_DAYS).",
    )
    parser.add_argument("--output-root", default="alphas", help="Root directory to store sweep results.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for sweeps (1 = serial).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rebalance_every < 1:
        raise ValueError("--rebalance-every must be >= 1.")
    strategy_roots = args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS)
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
    if not strategy_names:
        raise ValueError("No strategies found to sweep.")

    strategies = load_strategies(strategy_names, strategy_roots)
    strategies_by_name = {s.name: s for s in strategies}
    if not strategies_by_name:
        raise ValueError("No valid strategies loaded.")

    data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    df = pd.read_parquet(data_path)
    validate_market_data_frame(df, source=data_path, required_columns=["Close"])

    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    if args.jobs <= 1:
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
            rebalance_every_n_days=args.rebalance_every,
        )
    else:
        metric_key = METRIC_KEYS.get(args.metric, "CAGR")
        names = sorted(strategies_by_name.keys())
        max_size = args.max_size or len(names)
        combos = list(_generate_combos(names, min_size=args.min_size, max_size=max_size))
        combos = _sample_combos(combos, args.max_combos, args.seed)

        best_combo = None
        best_score = None
        best_metrics = None
        rows = []
        total = len(combos)
        progress_interval = max(1, min(500, total // 20)) if total else 1

        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(
                data_path,
                names,
                strategy_roots,
                start_date,
                end_date,
                metric_key,
                args.rebalance_every,
            ),
        ) as executor:
            for combo, score, metrics in executor.map(_evaluate_combo, combos, chunksize=1):
                row = {"combo": list(combo), "score": float(score), "metrics": metrics}
                rows.append(row)

                if best_score is None or score > best_score:
                    best_score = score
                    best_combo = combo
                    best_metrics = metrics

                if len(rows) % progress_interval == 0 or len(rows) == total:
                    print(f"Progress: {len(rows)}/{total} combos evaluated")

        if best_combo is None or best_metrics is None:
            raise RuntimeError("Sweep did not produce any results.")

        result = SweepResult(best_combo=best_combo, best_metrics=best_metrics, rows=rows)

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
        "rebalance_every_n_days": int(args.rebalance_every),
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
