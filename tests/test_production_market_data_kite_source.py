from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import src.kite as kite_module
import src.production_market_data as production_market_data


def _build_existing_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "RELIANCE.NS")],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1_000_000.0],
            "sector": ["Energy"],
        },
        index=index,
    )


def test_update_market_data_uses_kite_ohlc_source(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_file = tmp_path / "daily_data.parquet"
    _build_existing_frame().to_parquet(data_file, engine="pyarrow")
    monkeypatch.setitem(sys.modules, "pandas_ta_classic", types.ModuleType("pandas_ta_classic"))

    class _FakeKiteClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(kite_module, "KiteClient", _FakeKiteClient)
    monkeypatch.setattr(
        production_market_data,
        "_build_kite_symbol_map",
        lambda tickers, client: {"RELIANCE.NS": "NSE:RELIANCE"},
    )
    monkeypatch.setattr(
        production_market_data,
        "_fetch_kite_ohlc_batch",
        lambda **kwargs: {
            "RELIANCE.NS": {
                "ohlc": {
                    "open": 102.0,
                    "high": 104.0,
                    "low": 101.0,
                    "close": 103.0,
                },
                "volume": 1_500_000.0,
            }
        },
    )
    monkeypatch.setattr(
        production_market_data,
        "_resolve_kite_bar_date",
        lambda quote: pd.Timestamp("2025-01-03"),
    )
    monkeypatch.setattr(production_market_data, "_fetch_vix_close_kite", lambda **kwargs: None)

    def _fake_build_updated_row(**kwargs):
        return {
            "date": kwargs["bar_date"],
            "ticker": kwargs["ticker"],
            "Open": kwargs["open_price"],
            "High": kwargs["high_price"],
            "Low": kwargs["low_price"],
            "Close": kwargs["close_price"],
            "Volume": kwargs["volume"],
            "sector": kwargs["sector"],
            "log_return": 0.01,
            "adv_21": 1_200_000.0,
            "vol_21": 0.02,
        }

    monkeypatch.setattr(production_market_data, "_build_updated_row", _fake_build_updated_row)

    out = production_market_data.update_market_data(
        data_file,
        market_data_source="kite_ohlc",
        require_all_tickers=True,
    )

    assert (pd.Timestamp("2025-01-03"), "RELIANCE.NS") in out.index
    assert out.loc[(pd.Timestamp("2025-01-03"), "RELIANCE.NS"), "Close"] == 103.0
