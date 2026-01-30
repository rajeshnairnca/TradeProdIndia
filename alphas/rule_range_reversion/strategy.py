import pandas as pd

NAME = "rule_range_reversion"
DESCRIPTION = "Range-bound mean reversion favoring oversold RSI, below short SMA, and low trend strength."
REGIME_TAGS = ["sideways_low_vol", "sideways_high_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    rsi = df.get("RSI_14_z", 0.0)
    dist = df.get("dist_sma20_z", 0.0)
    adx = df.get("ADX_14_z", 0.0)
    scores = (-1.0 * rsi) + (-0.7 * dist) + (-0.3 * adx)
    return pd.Series(scores, index=df.index)
