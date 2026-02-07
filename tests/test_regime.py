from __future__ import annotations

import pandas as pd

from src.regime import compute_market_regime_table


def _build_regime_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=260, freq="D")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rows: list[tuple[pd.Timestamp, str, float, float, float]] = []
    for i, dt in enumerate(dates):
        rows.append((dt, "AAA", 100.0 + (0.8 * i), 98.0 + (0.78 * i), 0.01 + (i % 6) * 0.001))
        rows.append((dt, "BBB", 110.0 + (0.2 * i), 108.0 + (0.19 * i), -0.01 + (i % 5) * 0.001))
        rows.append((dt, "CCC", 90.0 - (0.1 * i), 91.0 - (0.1 * i), 0.02 - (i % 7) * 0.001))
        rows.append((dt, "DDD", 95.0 + (0.05 * i), 95.0 + (0.04 * i), 0.00 + (i % 4) * 0.001))
    df = pd.DataFrame(rows, columns=["date", "ticker", "Close", "SMA_50", "ROC_50"])
    return df.set_index(["date", "ticker"]).sort_index()


def test_compute_market_regime_table_heuristic():
    df = _build_regime_frame()
    table = compute_market_regime_table(df, mode="heuristic", dispersion_col="ROC_50")

    expected_cols = {
        "trend_up",
        "trend_state",
        "vol_high",
        "hmm_state",
        "breadth",
        "breadth_low",
        "breadth_high",
        "dispersion",
        "dispersion_high",
        "dispersion_low",
        "regime_label",
        "combined_regime",
    }
    assert set(table.columns) == expected_cols
    assert len(table) == len(df.index.get_level_values("date").unique())
    assert table.index.is_monotonic_increasing

    valid_labels = {
        "bull_low_vol",
        "bull_high_vol",
        "bear_low_vol",
        "bear_high_vol",
        "sideways_low_vol",
        "sideways_high_vol",
    }
    observed = set(table["regime_label"].dropna().astype(str).unique())
    assert observed <= valid_labels
