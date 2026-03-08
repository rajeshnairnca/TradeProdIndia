import numpy as np
import pandas as pd

NAME = "india_rule_trend_carry_slow"
DESCRIPTION = "Long-horizon trend and momentum with ticker-level smoothing to reduce churn."
REGIME_TAGS = ["bull_low_vol", "bull_high_vol"]


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
    adx = _col(df, "ADX_14_z")
    atr = _col(df, "ATRr_14_z")

    liquidity = np.log1p(_col(df, "adv_21").clip(lower=0.0))
    liquidity_z = _cs_zscore(liquidity)

    raw = (0.70 * trend) + (0.60 * momentum) + (0.25 * adx) + (0.20 * liquidity_z) - (0.25 * atr)
    scores = _smooth_by_ticker(raw, span=35)
    return pd.Series(scores, index=df.index)

