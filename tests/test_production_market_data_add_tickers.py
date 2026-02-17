from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import src.production_market_data as production_market_data


def _build_existing_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "AAPL")],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "Open": [190.0],
            "High": [191.0],
            "Low": [189.0],
            "Close": [190.5],
            "Volume": [1_000_000.0],
        },
        index=index,
    )


def _install_fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def history(self, period: str, interval: str, auto_adjust: bool = True) -> pd.DataFrame:
            idx = pd.date_range("2025-01-01", periods=10, freq="D")
            return pd.DataFrame(
                {
                    "Open": [10.0] * len(idx),
                    "High": [11.0] * len(idx),
                    "Low": [9.0] * len(idx),
                    "Close": [10.5] * len(idx),
                    "Volume": [100_000.0] * len(idx),
                },
                index=idx,
            )

        def get_info(self) -> dict:
            return {}

    fake_yfinance = types.ModuleType("yfinance")
    fake_yfinance.Ticker = _FakeTicker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)
    monkeypatch.setitem(sys.modules, "pandas_ta_classic", types.ModuleType("pandas_ta_classic"))


def _install_sparse_indicator_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in production_market_data.REQUIRED_INDICATOR_COLS:
            out[col] = pd.NA
        # Simulate a freshly listed ticker where only 4 rows survive indicator warmup.
        out.loc[out.index[-4:], production_market_data.REQUIRED_INDICATOR_COLS] = 1.0
        return out

    monkeypatch.setattr(production_market_data, "_calculate_indicators", _fake_calculate_indicators)


def test_add_universe_tickers_raises_when_no_valid_ticker_data(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_file = tmp_path / "daily_data.parquet"
    _build_existing_frame().to_parquet(data_file, engine="pyarrow")
    _install_fake_yfinance(monkeypatch)
    _install_sparse_indicator_stub(monkeypatch)

    with pytest.raises(ValueError, match="No new tickers had sufficient data to add"):
        production_market_data.add_universe_tickers(
            data_file,
            tickers=["SNDK"],
            min_trading_days=50,
        )


def test_add_universe_tickers_can_skip_invalid_new_tickers_without_failing_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_file = tmp_path / "daily_data.parquet"
    existing = _build_existing_frame()
    existing.to_parquet(data_file, engine="pyarrow")
    _install_fake_yfinance(monkeypatch)
    _install_sparse_indicator_stub(monkeypatch)

    out = production_market_data.add_universe_tickers(
        data_file,
        tickers=["SNDK"],
        min_trading_days=50,
        fail_on_no_valid_tickers=False,
    )

    assert set(out.index.get_level_values("ticker")) == {"AAPL"}
    assert len(out) == len(existing)
