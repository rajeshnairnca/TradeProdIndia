import numpy as np
import pandas as pd

NAME = "india_rule_liquidity_momentum_core"
DESCRIPTION = "Core India model with long-horizon momentum and explicit churn penalty for slower turnover."
REGIME_TAGS = [
    "bull_low_vol",
    "bull_high_vol",
    "bear_low_vol",
    "bear_high_vol",
    "sideways_low_vol",
    "sideways_high_vol",
]


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

    liquidity_z = _cs_zscore(np.log1p(_col(df, "adv_21").clip(lower=0.0)))
    vol_z = _cs_zscore(_col(df, "vol_21"))

    # Penalize fast day-to-day momentum shifts to keep rankings more stable.
    momentum_churn = momentum.groupby(level="ticker").transform(
        lambda x: x.diff().abs().ewm(span=20, adjust=False, min_periods=5).mean()
    )

    raw = (0.75 * momentum) + (0.45 * trend) + (0.40 * liquidity_z) - (0.70 * momentum_churn) - (0.25 * vol_z)
    scores = _smooth_by_ticker(raw, span=50)
    return pd.Series(scores, index=df.index)

