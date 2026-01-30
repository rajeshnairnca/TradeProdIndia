import pandas as pd

NAME = "rule_mean_reversion"
DESCRIPTION = "Mean reversion using oversold RSI and distance below short-term trend."
REGIME_TAGS = ["bear_low_vol", "bear_high_vol", "sideways_low_vol", "sideways_high_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    rsi = df.get("RSI_14_z", 0.0)
    dist = df.get("dist_sma20_z", 0.0)
    macd = df.get("MACD_12_26_9_z", 0.0)
    scores = (-1.0 * rsi) + (-0.6 * dist) + 0.2 * macd
    return pd.Series(scores, index=df.index)
