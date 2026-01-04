from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

REGIME_LABELS = (
    "bull_low_vol",
    "bull_high_vol",
    "bear_low_vol",
    "bear_high_vol",
    "sideways_low_vol",
    "sideways_high_vol",
)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    regime_tags: tuple[str, ...]
    score_func: Callable[[pd.DataFrame], pd.Series]


def load_strategy_from_file(path: str | Path, name_override: str | None = None) -> StrategySpec:
    strategy_path = Path(path)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    module_name = f"strategy_{strategy_path.stem}_{abs(hash(strategy_path))}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import strategy module from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    score_func = getattr(module, "generate_scores", None)
    if score_func is None or not callable(score_func):
        raise ValueError(f"Strategy {strategy_path} must define a callable generate_scores(df) function.")

    description = getattr(module, "DESCRIPTION", "").strip()
    regime_tags = getattr(module, "REGIME_TAGS", ())
    if regime_tags is None:
        regime_tags = ()
    if isinstance(regime_tags, str):
        regime_tags = (regime_tags,)
    regime_tags = tuple(tag for tag in regime_tags if tag)

    name = name_override or getattr(module, "NAME", "") or strategy_path.parent.name
    return StrategySpec(
        name=name,
        description=description,
        regime_tags=regime_tags,
        score_func=score_func,
    )


def load_strategies(strategy_names: Iterable[str], strategy_roots: Iterable[str | Path]) -> list[StrategySpec]:
    roots = [Path(root) for root in strategy_roots]
    strategies: list[StrategySpec] = []
    for name in strategy_names:
        strategy = None
        for root in roots:
            candidate = root / name / "strategy.py"
            if candidate.exists():
                strategy = load_strategy_from_file(candidate, name_override=name)
                break
        if strategy is None:
            print(f"Warning: strategy '{name}' not found in roots {roots}")
            continue
        strategies.append(strategy)
    return strategies


def list_strategy_names(strategy_roots: Iterable[str | Path]) -> list[str]:
    roots = [Path(root) for root in strategy_roots]
    names: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            if path.name.startswith("_"):
                continue
            if (path / "strategy.py").exists():
                names.append(path.name)
    return sorted(set(names))


def validate_strategy(strategy: StrategySpec, df: pd.DataFrame) -> None:
    scores = strategy.score_func(df)
    if not isinstance(scores, pd.Series):
        raise ValueError(f"Strategy {strategy.name} must return a pandas Series from generate_scores.")
    if len(scores) != len(df):
        raise ValueError(
            f"Strategy {strategy.name} returned {len(scores)} scores, expected {len(df)}."
        )
    if not scores.index.equals(df.index):
        raise ValueError(f"Strategy {strategy.name} must return scores indexed to the input DataFrame.")
