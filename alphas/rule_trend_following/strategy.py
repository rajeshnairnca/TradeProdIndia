

import pandas as pd

NAME = "rule_trend_following"
DESCRIPTION = "Trend-following on long-term price strength with momentum confirmation."
REGIME_TAGS = ["bull_low_vol", "bull_high_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    sma_250 = df["SMA_250"] if "SMA_250" in df.columns else close
    trend = (close / (sma_250 + 1e-9)) - 1.0
    momentum = df.get("ROC_50_z", 0.0)
    adx = df.get("ADX_14_z", 0.0)
    scores = trend + 0.5 * momentum + 0.2 * adx
    return pd.Series(scores, index=df.index)
