from __future__ import annotations

import pandas as pd

from . import config

_QUALITY_CACHE: dict[tuple, set[str]] = {}


def _cache_key(df: pd.DataFrame) -> tuple:
    idx = df.index
    first = idx[0] if len(idx) else None
    last = idx[-1] if len(idx) else None
    return (
        id(df),
        len(df),
        first,
        last,
        config.ENABLE_UNIVERSE_QUALITY_FILTER,
        config.UNIVERSE_MIN_HISTORY_ROWS,
        config.UNIVERSE_MIN_MEDIAN_ADV_DOLLARS,
        config.UNIVERSE_MIN_P20_ADV_DOLLARS,
        config.UNIVERSE_MIN_P05_PRICE,
        config.UNIVERSE_MAX_MEDIAN_VOL21,
        config.UNIVERSE_MAX_P95_ABS_LOG_RETURN,
        config.UNIVERSE_QUALITY_START_DATE,
        config.UNIVERSE_QUALITY_END_DATE,
    )


def _to_timestamp(value: str) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value)
    except (TypeError, ValueError):
        return None


def _apply_date_window(df: pd.DataFrame) -> pd.DataFrame:
    start_date = _to_timestamp(config.UNIVERSE_QUALITY_START_DATE)
    end_date = _to_timestamp(config.UNIVERSE_QUALITY_END_DATE)
    if start_date is None and end_date is None:
        return df

    dates = df.index.get_level_values("date")
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= dates >= start_date
    if end_date is not None:
        mask &= dates < end_date
    return df[mask.to_numpy()]


def _series_by_ticker(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None
    return df[column].groupby(level="ticker")


def compute_quality_exclusions(df: pd.DataFrame) -> set[str]:
    if not config.ENABLE_UNIVERSE_QUALITY_FILTER:
        return set()
    if df.empty:
        return set()

    key = _cache_key(df)
    cached = _QUALITY_CACHE.get(key)
    if cached is not None:
        return set(cached)

    work = _apply_date_window(df)
    if work.empty:
        return set()

    rows = work.groupby(level="ticker").size()
    metrics = pd.DataFrame(index=rows.index)
    metrics["rows"] = rows
    fail = pd.Series(False, index=metrics.index)

    fail |= metrics["rows"] < config.UNIVERSE_MIN_HISTORY_ROWS

    if {"adv_21", "Close"}.issubset(work.columns):
        adv_dollars = work["adv_21"] * work["Close"]
        g_adv = adv_dollars.groupby(level="ticker")
        metrics["median_adv_dollars"] = g_adv.median()
        metrics["p20_adv_dollars"] = g_adv.quantile(0.20)
        fail |= metrics["median_adv_dollars"] < config.UNIVERSE_MIN_MEDIAN_ADV_DOLLARS
        fail |= metrics["p20_adv_dollars"] < config.UNIVERSE_MIN_P20_ADV_DOLLARS

    g_close = _series_by_ticker(work, "Close")
    if g_close is not None:
        metrics["p05_price"] = g_close.quantile(0.05)
        fail |= metrics["p05_price"] < config.UNIVERSE_MIN_P05_PRICE

    g_vol = _series_by_ticker(work, "vol_21")
    if g_vol is not None:
        metrics["median_vol21"] = g_vol.median()
        fail |= metrics["median_vol21"] > config.UNIVERSE_MAX_MEDIAN_VOL21

    g_ret = _series_by_ticker(work, "log_return")
    if g_ret is not None:
        metrics["p95_abs_log_return"] = g_ret.apply(lambda s: s.abs().quantile(0.95))
        fail |= metrics["p95_abs_log_return"] > config.UNIVERSE_MAX_P95_ABS_LOG_RETURN

    excluded = set(metrics.index[fail].tolist())
    _QUALITY_CACHE[key] = set(excluded)
    return excluded


def apply_quality_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    excluded = compute_quality_exclusions(df)
    if not excluded:
        return df, set()
    filtered = df[~df.index.get_level_values("ticker").isin(excluded)]
    return filtered, excluded
