import numpy as np
import pandas as pd

NAME = "india_rule_crash_resilient_slow"
DESCRIPTION = "Drawdown-aware scoring for stressed regimes using volatility penalties and liquidity preference."
REGIME_TAGS = ["bear_high_vol", "bear_low_vol", "sideways_high_vol"]


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
    atr = _col(df, "ATRr_14_z")
    rvol = _col(df, "rvol_20_z")
    vol_z = _cs_zscore(_col(df, "vol_21"))
    liquidity_z = _cs_zscore(np.log1p(_col(df, "adv_21").clip(lower=0.0)))

    raw = (-1.00 * vol_z) - (0.70 * atr) - (0.50 * rvol) + (0.35 * momentum) + (0.25 * trend) + (0.30 * liquidity_z)
    scores = _smooth_by_ticker(raw, span=40)
    return pd.Series(scores, index=df.index)

