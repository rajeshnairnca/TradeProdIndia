import pandas as pd
import numpy as np

from . import config


def compute_market_regime_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-date regime table using:
      - Trend: market SMA50 vs SMA200 (market = average close across universe)
      - Volatility: rolling std of market returns vs 75th percentile
      - Breadth: % of stocks with Close > SMA_50
      - Dispersion: cross-sectional std of ROC_10_z (as a proxy for momentum dispersion)
    Returns a DataFrame indexed by date with booleans and helper labels.
    """
    market_proxy = df.groupby(level="date")["Close"].mean().to_frame("market_close")
    market_proxy["market_return"] = market_proxy["market_close"].pct_change()
    market_proxy["sma_50"] = market_proxy["market_close"].rolling(window=50, min_periods=50).mean()
    market_proxy["sma_200"] = market_proxy["market_close"].rolling(window=200, min_periods=200).mean()
    market_proxy["trend_up"] = (market_proxy["sma_50"] > market_proxy["sma_200"])

    vol = market_proxy["market_return"].rolling(
        window=config.ROLLING_WINDOW_FOR_VOL, min_periods=config.ROLLING_WINDOW_FOR_VOL
    ).std()
    vol_threshold = vol.shift(1).expanding(min_periods=config.ROLLING_WINDOW_FOR_VOL).quantile(0.75)
    market_proxy["vol_high"] = vol > vol_threshold

    # Breadth: share of stocks above their SMA_50 (neutral 0.5 if SMA_50 missing)
    if "SMA_50" in df.columns:
        breadth = (df["Close"] > df["SMA_50"]).groupby(level="date").mean()
        breadth = breadth.reindex(market_proxy.index)
    else:
        breadth = pd.Series(0.5, index=market_proxy.index)
    market_proxy["breadth"] = breadth
    market_proxy["breadth_low"] = market_proxy["breadth"] < 0.45
    market_proxy["breadth_high"] = market_proxy["breadth"] > 0.55

    # Dispersion: cross-sectional std of ROC_10_z (neutral 0 if missing)
    if "ROC_10_z" in df.columns:
        dispersion = df.groupby(level="date")["ROC_10_z"].std()
        dispersion = dispersion.reindex(market_proxy.index)
    else:
        dispersion = pd.Series(0.0, index=market_proxy.index)
    dispersion_high_q = dispersion.shift(1).expanding(min_periods=30).quantile(0.75)
    dispersion_low_q = dispersion.shift(1).expanding(min_periods=30).quantile(0.25)
    market_proxy["dispersion"] = dispersion
    market_proxy["dispersion_high"] = market_proxy["dispersion"] > dispersion_high_q
    market_proxy["dispersion_low"] = market_proxy["dispersion"] < dispersion_low_q

    # Combined regime label (trend x vol)
    def _label(row):
        if row["trend_up"] and not row["vol_high"]:
            return "bull_low_vol"
        if row["trend_up"] and row["vol_high"]:
            return "bull_high_vol"
        if (not row["trend_up"]) and row["vol_high"]:
            return "bear_high_vol"
        return "bear_low_vol"

    market_proxy["regime_label"] = market_proxy.apply(_label, axis=1)
    market_proxy["combined_regime"] = 2 * market_proxy["trend_up"].astype(float) + market_proxy["vol_high"].astype(float)
    return market_proxy[
        [
            "trend_up",
            "vol_high",
            "breadth",
            "breadth_low",
            "breadth_high",
            "dispersion",
            "dispersion_high",
            "dispersion_low",
            "regime_label",
            "combined_regime",
        ]
    ]


def get_regime_state(regime_table: pd.DataFrame | None, current_date) -> dict:
    if regime_table is None or current_date not in regime_table.index:
        return {
            "trend_up": False,
            "vol_high": False,
            "dispersion_high": False,
            "dispersion_low": False,
            "breadth_low": False,
            "breadth_high": False,
            "regime_label": "unknown",
        }
    row = regime_table.loc[current_date]
    return {
        "trend_up": bool(row.get("trend_up", False)),
        "vol_high": bool(row.get("vol_high", False)),
        "dispersion_high": bool(row.get("dispersion_high", False)),
        "dispersion_low": bool(row.get("dispersion_low", False)),
        "breadth_low": bool(row.get("breadth_low", False)),
        "breadth_high": bool(row.get("breadth_high", False)),
        "regime_label": str(row.get("regime_label", "unknown")),
    }


def regime_top_k(state: dict, default_top_k: int) -> int:
    if state.get("vol_high") and state.get("dispersion_low") and state.get("breadth_low"):
        return 5
    if state.get("vol_high") and state.get("dispersion_high"):
        return 7
    if (not state.get("vol_high")) and state.get("dispersion_high") and state.get("breadth_high"):
        return 12
    if (not state.get("vol_high")) and state.get("dispersion_high"):
        return 10
    if state.get("vol_high"):
        return 6
    return default_top_k


def regime_gross_target(state: dict) -> float:
    trend_up = bool(state.get("trend_up", False))
    vol_high = bool(state.get("vol_high", False))
    dispersion_high = bool(state.get("dispersion_high", False))
    dispersion_low = bool(state.get("dispersion_low", False))
    breadth_low = bool(state.get("breadth_low", False))
    breadth_high = bool(state.get("breadth_high", False))

    gross = 0.85
    if (not trend_up) and vol_high:
        gross = 0.6
    elif (not trend_up) and (not vol_high):
        gross = 0.75
    elif vol_high and dispersion_low and breadth_low:
        gross = 0.6
    elif vol_high and dispersion_high:
        gross = 0.8
    elif (not vol_high) and dispersion_high and breadth_high:
        gross = 1.05
    elif (not vol_high) and dispersion_high:
        gross = 0.95
    elif (not vol_high) and dispersion_low:
        gross = 0.85
    return max(0.0, min(1.1, gross))
