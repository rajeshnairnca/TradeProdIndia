import pandas as pd

NAME = "rule_momentum_breakout"
DESCRIPTION = "Aggressive momentum strategy focusing on strong ROC and price strength."
REGIME_TAGS = ["bull_low_vol", "bull_high_vol"]


def generate_scores(df: pd.DataFrame) -> pd.Series:
    # Combined Momentum Score
    mom_50 = df.get("ROC_50_z", 0.0)
    mom_10 = df.get("ROC_10_z", 0.0)
    trend = df.get("dist_sma50_z", 0.0)
    
    # Bonus for RSI not being too high (not extremely overbought)
    rsi = df.get("RSI_14", 50.0)
    rsi_factor = pd.Series(0.0, index=df.index)
    rsi_factor[rsi > 50] = 1.0
    rsi_factor[rsi > 80] = 0.5 # Penalty for extreme overbought
    
    scores = mom_50 + 0.5 * mom_10 + 0.5 * trend + 0.5 * rsi_factor
    return pd.Series(scores, index=df.index)
