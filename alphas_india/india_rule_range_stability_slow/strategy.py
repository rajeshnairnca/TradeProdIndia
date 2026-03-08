import numpy as np
import pandas as pd

NAME = "india_rule_range_stability_slow"
DESCRIPTION = "Sideways-market stability model that favors controlled mean reversion and low turbulence."
REGIME_TAGS = ["sideways_low_vol", "sideways_high_vol", "bear_low_vol"]


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
    dist_sma20 = _col(df, "dist_sma20_z")
    rsi = _col(df, "RSI_14_z")
    day_noise = _col(df, "log_return_z").abs()
    vol_z = _cs_zscore(_col(df, "vol_21"))
    liquidity_z = _cs_zscore(np.log1p(_col(df, "adv_21").clip(lower=0.0)))

    raw = (-0.80 * dist_sma20) - (0.50 * rsi) - (0.40 * day_noise) - (0.30 * vol_z) + (0.20 * liquidity_z)
    scores = _smooth_by_ticker(raw, span=30)
    return pd.Series(scores, index=df.index)

