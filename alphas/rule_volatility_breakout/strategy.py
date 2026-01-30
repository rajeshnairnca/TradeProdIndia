import pandas as pd

NAME = "rule_volatility_breakout"
DESCRIPTION = "Volatility breakout using ATR expansion, momentum, and volume."
REGIME_TAGS = ["bull_high_vol", "bear_high_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    atr = df.get("ATRr_14_z", 0.0)
    roc = df.get("ROC_10_z", 0.0)
    rvol = df.get("rvol_20_z", 0.0)
    scores = 0.5 * atr + 0.4 * roc + 0.2 * rvol
    return pd.Series(scores, index=df.index)
