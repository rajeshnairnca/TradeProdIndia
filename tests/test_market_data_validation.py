from __future__ import annotations

import pandas as pd
import pytest

from src.market_data_validation import validate_market_data_frame


def test_validate_market_data_frame_accepts_expected_shape():
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.1, 11.2],
            "Volume": [1000, 1200],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-01"), "AAA"), (pd.Timestamp("2024-01-02"), "AAA")],
            names=["date", "ticker"],
        ),
    )
    validate_market_data_frame(df, source="unit", required_columns=["Close", "Volume"])


def test_validate_market_data_frame_rejects_bad_index():
    df = pd.DataFrame(
        {"Close": [10.0], "Volume": [1000]},
        index=pd.Index([pd.Timestamp("2024-01-01")], name="date"),
    )
    with pytest.raises(ValueError):
        validate_market_data_frame(df, source="unit", required_columns=["Close"])
