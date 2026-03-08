from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence
import random

import pandas as pd

from .regime import compute_market_regime_table
from .rule_backtester import RuleBasedBacktester
from .strategy import StrategySpec

METRIC_KEYS = {
    "cagr": "CAGR",
    "sharpe": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown",
}


@dataclass
class SweepResult:
    best_combo: tuple[str, ...]
    best_metrics: dict
    rows: list[dict]


def _generate_combos(names: Sequence[str], min_size: int, max_size: int) -> Iterable[tuple[str, ...]]:
    for size in range(min_size, max_size + 1):
        yield from combinations(names, size)


def _sample_combos(combos: list[tuple[str, ...]], max_combos: int | None, seed: int) -> list[tuple[str, ...]]:
    if not max_combos or len(combos) <= max_combos:
        return combos
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:max_combos]


def sweep_strategy_combinations(
    df: pd.DataFrame,
    strategies_by_name: dict[str, StrategySpec],
    min_size: int = 1,
    max_size: int | None = None,
    max_combos: int | None = None,
    seed: int = 42,
    metric: str = "cagr",
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rebalance_every_n_days: int | None = None,
) -> SweepResult:
    if not strategies_by_name:
        raise ValueError("No strategies provided for sweep.")

    max_size = max_size or len(strategies_by_name)
    metric_key = METRIC_KEYS.get(metric, "CAGR")

    names = sorted(strategies_by_name.keys())
    combos = list(_generate_combos(names, min_size=min_size, max_size=max_size))
    combos = _sample_combos(combos, max_combos, seed)

    regime_table = compute_market_regime_table(df)
    best_combo: tuple[str, ...] | None = None
    best_score: float | None = None
    best_metrics: dict | None = None
    rows: list[dict] = []

    for combo in combos:
        strategies = [strategies_by_name[name] for name in combo]
        backtester = RuleBasedBacktester(
            df,
            strategies,
            regime_table=regime_table,
            rebalance_every_n_days=rebalance_every_n_days,
        )
        result = backtester.run(start_date=start_date, end_date=end_date)
        metrics = result.metrics

        score = metrics.get(metric_key, 0.0)
        row = {
            "combo": list(combo),
            "score": float(score),
            "metrics": metrics,
        }
        rows.append(row)

        if best_score is None or score > best_score:
            best_score = score
            best_combo = combo
            best_metrics = metrics

    if best_combo is None or best_metrics is None:
        raise RuntimeError("Sweep did not produce any results.")

    return SweepResult(best_combo=best_combo, best_metrics=best_metrics, rows=rows)
