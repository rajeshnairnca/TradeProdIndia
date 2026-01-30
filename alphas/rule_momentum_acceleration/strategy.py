import pandas as pd

NAME = "rule_momentum_acceleration"
DESCRIPTION = "Short-term momentum acceleration with volume confirmation."
REGIME_TAGS = ["bull_high_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    roc_fast = df.get("ROC_10_z", 0.0)
    roc_slow = df.get("ROC_50_z", 0.0)
    vol_confirm = df.get("SMA20_Volume_z", 0.0)
    scores = 0.6 * roc_fast + 0.4 * roc_slow + 0.2 * vol_confirm
    return pd.Series(scores, index=df.index)
