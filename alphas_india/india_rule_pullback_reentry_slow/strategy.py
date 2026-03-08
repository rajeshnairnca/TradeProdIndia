import pandas as pd

NAME = "india_rule_pullback_reentry_slow"
DESCRIPTION = "Buy pullbacks within broader uptrends, with smoothed signals to avoid rapid flips."
REGIME_TAGS = ["bull_high_vol", "sideways_high_vol"]


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _smooth_by_ticker(series: pd.Series, span: int) -> pd.Series:
    min_periods = max(5, span // 4)
    return series.groupby(level="ticker").transform(
        lambda x: x.ewm(span=span, adjust=False, min_periods=min_periods).mean()
    )


def generate_scores(df: pd.DataFrame) -> pd.Series:
    close = _col(df, "Close")
    sma_250 = _col(df, "SMA_250")
    trend_long = (close / (sma_250.abs() + 1e-9)) - 1.0

    pullback = -1.0 * _col(df, "dist_sma20_z")
    rsi_revert = -1.0 * _col(df, "RSI_14_z")
    momentum = _col(df, "ROC_50_z")
    atr = _col(df, "ATRr_14_z")

    raw = (0.70 * trend_long) + (0.65 * pullback) + (0.35 * rsi_revert) + (0.25 * momentum) - (0.25 * atr)
    scores = _smooth_by_ticker(raw, span=28)
    return pd.Series(scores, index=df.index)

