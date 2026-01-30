import pandas as pd

NAME = "rule_quality_min_vol"
DESCRIPTION = "Quality selection favoring low volatility and steady momentum for high-vol regimes."
REGIME_TAGS = ["bear_high_vol", "bull_high_vol", "bear_low_vol", "sideways_low_vol"]

def generate_scores(df: pd.DataFrame) -> pd.Series:
    # 1. Low Volatility (Safety) - Heavy weight
    vol = df.get("vol_21", 0.0)
    
    # 2. Momentum (Quality/Strength) - Moderate weight to avoid falling knives
    roc = df.get("ROC_126_z", 0.0) # 6-month momentum (steadier than 1-month)
    
    # 3. Relative Volatility (vs 3-month avg) to ensure it's not currently blowing up
    rvol = df.get("rvol_63_z", 0.0)

    # Score: Penalize High Vol, Reward Long-Term Momentum, Penalize Vol Expansion
    scores = (-2.0 * vol) + (1.0 * roc) - (0.5 * rvol)
    
    return pd.Series(scores, index=df.index)
