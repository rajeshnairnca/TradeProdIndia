from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config
from .portfolio import get_target_weights
from .regime import compute_market_regime_table, get_regime_state, regime_top_k
from .strategy import load_strategies
from .universe import NASDAQ100_TICKERS
from .universe_quality import apply_quality_filter


def _safe_close_for_ticker(day_data: pd.DataFrame, ticker: str) -> float | None:
    if ticker not in day_data.index:
        return None
    row = day_data.loc[ticker]
    if isinstance(row, pd.DataFrame):
        close_val = row.iloc[0].get("Close")
    else:
        close_val = row.get("Close")
    try:
        parsed = float(close_val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _filter_tickers_by_universe_filter(tickers: set[str]) -> set[str]:
    universe_filter = (config.UNIVERSE_FILTER or "").strip().lower()
    if not universe_filter or universe_filter in {"all", "none"}:
        return set(tickers)
    if universe_filter == "nasdaq100":
        return set(tickers).intersection(set(NASDAQ100_TICKERS))
    requested = {item.strip().upper() for item in universe_filter.split(",") if item.strip()}
    return set(tickers).intersection(requested)


def _parse_csv_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def _parse_strategy_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    text = str(value).strip()
    if not text:
        return []
    return sorted({part.strip() for part in text.split(",") if part.strip()})


def _compute_selection_snapshot(
    *,
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    target_date: pd.Timestamp,
    strategy_names: list[str],
    strategy_roots: list[str],
    regime_scope: str,
) -> tuple[dict[str, float], set[str], dict[str, Any]]:
    if filtered_df.empty:
        return {}, set(), {"computed": False, "detail": "No rows left after pre-signal filters."}
    if not strategy_names:
        return {}, set(), {"computed": False, "detail": "No strategies provided."}

    strategies = load_strategies(strategy_names, strategy_roots)
    if not strategies:
        return {}, set(), {"computed": False, "detail": "No valid strategies loaded."}

    regime_source_df = full_df if regime_scope == "global" else filtered_df
    regime_table = compute_market_regime_table(regime_source_df)
    state = get_regime_state(regime_table, target_date)
    regime_label = state.get("regime_label")
    active = [
        strategy
        for strategy in strategies
        if not strategy.regime_tags or regime_label in strategy.regime_tags
    ]
    active = active if active else strategies

    universe = filtered_df.index.get_level_values("ticker").unique().tolist()
    strategy_scores: dict[str, pd.Series] = {}
    for strategy in active:
        series = strategy.score_func(filtered_df)
        if not isinstance(series, pd.Series):
            raise ValueError(f"Strategy '{strategy.name}' returned non-Series scores.")
        if len(series) != len(filtered_df):
            raise ValueError(
                f"Strategy '{strategy.name}' score length mismatch: {len(series)} != {len(filtered_df)}."
            )
        if not series.index.equals(filtered_df.index):
            series = series.reindex(filtered_df.index)
        strategy_scores[strategy.name] = series.astype(float).fillna(0.0)

    combined: np.ndarray | None = None
    for strategy in active:
        series = strategy_scores[strategy.name]
        try:
            day_scores = series.xs(target_date, level="date")
        except KeyError:
            continue
        arr = day_scores.reindex(universe).to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) == 0:
            continue
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std > 1e-9:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
        combined = arr if combined is None else combined + arr
    if combined is None:
        combined = np.zeros(len(universe), dtype=np.float32)

    try:
        day_slice = filtered_df.xs(target_date, level="date")
    except KeyError:
        return {}, set(), {"computed": False, "detail": f"No filtered data on {target_date.date()}."}

    day_data = day_slice.reindex(universe)
    prices = day_data["Close"].to_numpy(dtype=float)
    mask = np.isfinite(prices) & (prices > 0)
    vol = day_data.get("vol_21")
    vol = vol.to_numpy(dtype=float) if vol is not None else np.ones_like(prices)

    dynamic_top_k = config.TOP_K
    if config.USE_REGIME_SYSTEM:
        dynamic_top_k = regime_top_k(state, config.TOP_K)

    weights = get_target_weights(combined, vol, mask.astype(float), top_k=dynamic_top_k)
    weights = weights * mask
    total = np.sum(weights)
    if total > 1e-9:
        weights = weights / total
    else:
        weights = np.zeros_like(weights)

    scores_by_ticker = {ticker: float(score) for ticker, score in zip(universe, combined)}
    selected = {ticker for ticker, weight in zip(universe, weights) if float(weight) > 1e-12}
    metadata = {
        "computed": True,
        "active_strategies": [strategy.name for strategy in active],
        "regime_label": str(regime_label or "unknown"),
        "top_k": int(dynamic_top_k),
        "selection_note": (
            "Top-k score snapshot ignores runtime weight smoothing/state transitions. "
            "Use as diagnostic ranking, not exact replay."
        ),
    }
    return scores_by_ticker, selected, metadata


def build_selection_diagnostics(
    *,
    full_df: pd.DataFrame,
    sector: str,
    target_date: pd.Timestamp,
    regime_scope: str,
    excluded_tickers: set[str] | None = None,
    strategies: str | list[str] | None = None,
    strategy_roots: list[str] | None = None,
) -> dict[str, Any]:
    if "sector" not in full_df.columns or "Close" not in full_df.columns:
        raise ValueError("Data file must include 'sector' and 'Close' columns.")

    sector_name = str(sector or "").strip()
    if not sector_name:
        raise ValueError("sector is required.")

    sector_df = full_df[full_df["sector"].astype(str).str.lower() == sector_name.lower()]
    if sector_df.empty:
        raise ValueError(f"No data for sector '{sector_name}'.")

    normalized_target_date = pd.to_datetime(target_date).tz_localize(None)
    target_date_str = str(normalized_target_date.date())
    all_dates = sector_df.index.get_level_values("date")
    if normalized_target_date not in all_dates:
        raise ValueError(f"Target date {target_date_str} is not present in sector data.")

    sector_tickers = sorted(
        {
            str(ticker).strip().upper()
            for ticker in sector_df.index.get_level_values("ticker").unique()
            if str(ticker).strip()
        }
    )
    sector_set = set(sector_tickers)
    allowed = _filter_tickers_by_universe_filter(sector_set)
    excluded_set = {str(t).strip().upper() for t in (excluded_tickers or set()) if str(t).strip()}
    removed_by_excluded = allowed.intersection(excluded_set)
    after_excluded = allowed - removed_by_excluded

    quality_input = sector_df[sector_df.index.get_level_values("ticker").isin(after_excluded)]
    quality_filtered, removed_by_quality = apply_quality_filter(quality_input)
    removed_by_quality = set(removed_by_quality)
    after_quality = {
        str(ticker).strip().upper()
        for ticker in quality_filtered.index.get_level_values("ticker").unique()
        if str(ticker).strip()
    }

    try:
        day_data = quality_filtered.xs(normalized_target_date, level="date")
    except KeyError:
        day_data = pd.DataFrame(columns=quality_filtered.columns)
    removed_invalid_close: set[str] = set()
    for ticker in sorted(after_quality):
        close_val = _safe_close_for_ticker(day_data, ticker)
        if close_val is None or not np.isfinite(close_val) or close_val <= 0:
            removed_invalid_close.add(ticker)

    strategy_names = _parse_strategy_names(strategies)
    roots = strategy_roots or ["alphas"]
    scores_by_ticker: dict[str, float] = {}
    selected_tickers: set[str] = set()
    selection_info: dict[str, Any] = {
        "computed": False,
        "detail": "Selection snapshot skipped (no strategy names provided).",
        "strategy_source": "provided",
        "strategy_names": strategy_names,
        "strategy_roots": roots,
    }

    if strategy_names:
        scores_by_ticker, selected_tickers, selection_meta = _compute_selection_snapshot(
            full_df=full_df,
            filtered_df=quality_filtered,
            target_date=normalized_target_date,
            strategy_names=strategy_names,
            strategy_roots=roots,
            regime_scope=regime_scope,
        )
        selection_info = {
            **selection_meta,
            "strategy_source": "provided",
            "strategy_names": strategy_names,
            "strategy_roots": roots,
        }

    sorted_score_tickers = sorted(
        scores_by_ticker.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    rank_by_ticker = {ticker: idx + 1 for idx, (ticker, _) in enumerate(sorted_score_tickers)}

    records: list[dict[str, Any]] = []
    for ticker in sector_tickers:
        if ticker not in allowed:
            stage = "filtered_universe_filter"
            reason = f"Ticker excluded by UNIVERSE_FILTER={config.UNIVERSE_FILTER!r}."
        elif ticker in removed_by_excluded:
            stage = "filtered_excluded_ticker"
            reason = "Ticker present in excluded tickers list."
        elif ticker in removed_by_quality:
            stage = "filtered_quality"
            reason = "Ticker failed universe quality thresholds."
        elif ticker in removed_invalid_close:
            stage = "filtered_invalid_close"
            reason = f"Ticker has missing/non-positive Close on {target_date_str}."
        else:
            if selection_info.get("computed"):
                if ticker in selected_tickers:
                    stage = "selected_top_k"
                    reason = "Ticker is in the top-k score snapshot for this date."
                else:
                    stage = "not_selected_top_k"
                    reason = "Ticker passed pre-filters but is outside top-k score snapshot."
            else:
                stage = "candidate_pool"
                reason = "Ticker passed all pre-signal filters."
        records.append(
            {
                "ticker": ticker,
                "stage": stage,
                "reason": reason,
                "combined_score": scores_by_ticker.get(ticker),
                "score_rank": rank_by_ticker.get(ticker),
            }
        )

    stage_counts: dict[str, int] = {}
    for row in records:
        stage = str(row.get("stage"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    return {
        "date": target_date_str,
        "sector": sector_name,
        "regime_scope": regime_scope,
        "total": len(records),
        "filters": {
            "universe_filter": config.UNIVERSE_FILTER,
            "excluded_tickers_count": len(excluded_set),
            "quality_filter_enabled": bool(config.ENABLE_UNIVERSE_QUALITY_FILTER),
        },
        "selection": selection_info,
        "counts": {
            "sector_universe": len(sector_tickers),
            "after_universe_filter": len(allowed),
            "removed_by_excluded_tickers": len(removed_by_excluded),
            "removed_by_quality": len(removed_by_quality),
            "removed_by_invalid_close": len(removed_invalid_close),
            "candidate_pool": len(after_quality - removed_invalid_close),
            "stage_counts": stage_counts,
        },
        "records": records,
    }
