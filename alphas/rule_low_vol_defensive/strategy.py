import pandas as pd

NAME = "rule_low_vol_defensive"
DESCRIPTION = "Defensive selection favoring low volatility with mild positive drift."
REGIME_TAGS = ["bear_low_vol", "bull_low_vol", "sideways_low_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    vol = df.get("vol_21", 0.0)
    roc = df.get("ROC_50_z", 0.0)
    scores = (-1.0 * vol) + 0.4 * roc
    return pd.Series(scores, index=df.index)
