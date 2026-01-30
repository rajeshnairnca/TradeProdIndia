import pandas as pd

NAME = "rule_trend_strength"
DESCRIPTION = "Trend strength based on directional movement and ADX."
REGIME_TAGS = ["bull_low_vol", "bull_high_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    dmp = df.get("DMP_14", 0.0)
    dmn = df.get("DMN_14", 0.0)
    adx = df.get("ADX_14_z", 0.0)
    scores = (dmp - dmn) + 0.3 * adx
    return pd.Series(scores, index=df.index)
