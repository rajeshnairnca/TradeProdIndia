import numpy as np
import pandas as pd

NAME = "india_rule_quality_defensive_slow"
DESCRIPTION = "Defensive quality blend for lower-vol regimes, with low-turnover smoothing."
REGIME_TAGS = ["bear_low_vol", "sideways_low_vol", "bull_low_vol"]


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _cs_zscore(series: pd.Series) -> pd.Series:
    return series.groupby(level="date").transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )


def _smooth_by_ticker(series: pd.Series, span: int) -> pd.Series:
    min_periods = max(5, span // 4)
    return series.groupby(level="ticker").transform(
        lambda x: x.ewm(span=span, adjust=False, min_periods=min_periods).mean()
    )


def generate_scores(df: pd.DataFrame) -> pd.Series:
    close = _col(df, "Close")
    sma_250 = _col(df, "SMA_250")
    trend = (close / (sma_250.abs() + 1e-9)) - 1.0
    momentum = _col(df, "ROC_50_z")
    rvol = _col(df, "rvol_20_z")

    vol_21 = _col(df, "vol_21")
    vol_z = _cs_zscore(vol_21)
    liquidity_z = _cs_zscore(np.log1p(_col(df, "adv_21").clip(lower=0.0)))

    raw = (-1.10 * vol_z) + (0.50 * trend) + (0.35 * momentum) - (0.35 * rvol) + (0.35 * liquidity_z)
    scores = _smooth_by_ticker(raw, span=45)
    return pd.Series(scores, index=df.index)

