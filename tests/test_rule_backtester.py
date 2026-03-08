from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.rule_backtester import RuleBasedBacktester
from src.strategy import StrategySpec


def _build_test_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    for i, dt in enumerate(dates):
        rows.append((dt, "AAA", 100.0 + (i * 2.0), 0.02, 2_000_000.0))
        rows.append((dt, "BBB", 100.0 + (i * 0.3), 0.03, 2_000_000.0))
        rows.append((dt, "CCC", 100.0 - (i * 1.5), 0.05, 2_000_000.0))
    df = pd.DataFrame(rows, columns=["date", "ticker", "Close", "vol_21", "adv_21"])
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def test_rule_backtester_runs_and_generates_trades(monkeypatch):
    df = _build_test_frame()

    def score_func(frame: pd.DataFrame) -> pd.Series:
        rank_map = {"AAA": 3.0, "BBB": 1.5, "CCC": 0.0}
        values = frame.index.get_level_values("ticker").map(rank_map).to_numpy(dtype=float)
        return pd.Series(values, index=frame.index, dtype=float)

    strategy = StrategySpec(
        name="unit_ranked",
        description="deterministic test strategy",
        regime_tags=(),
        score_func=score_func,
    )

    monkeypatch.setattr(config, "TOP_K", 2)
    monkeypatch.setattr(config, "WEIGHT_SMOOTHING", 0.0)
    monkeypatch.setattr(config, "CASH_RESERVE", 0.0)
    monkeypatch.setattr(config, "USE_REGIME_SYSTEM", False)
    monkeypatch.setattr(config, "UNIVERSE_FILTER", "all")
    monkeypatch.setattr(config, "ENABLE_UNIVERSE_QUALITY_FILTER", False)
    monkeypatch.setattr(config, "TRADING_REGION", "us")
    monkeypatch.setattr(config, "SLIPPAGE_COEFF", 0.0)
    monkeypatch.setattr(config, "US_COMMISSION_RATE", 0.0)
    monkeypatch.setattr(config, "US_FINRA_FEE_PER_SHARE", 0.0)
    monkeypatch.setattr(config, "US_SEC_FEE_RATE", 0.0)
    monkeypatch.setattr(config, "MIN_WEIGHT_CHANGE_TO_TRADE", 0.0)
    monkeypatch.setattr(config, "MIN_TRADE_DOLLARS", 0.0)
    monkeypatch.setattr(config, "MAX_DAILY_TURNOVER", None)
    monkeypatch.setattr(config, "BACKTEST_ENFORCE_CASH_BALANCE", False)

    backtester = RuleBasedBacktester(
        df=df,
        strategies=[strategy],
        regime_table=None,
        initial_capital=10_000.0,
        apply_regime_overlays=False,
    )
    result = backtester.run()

    assert len(result.dates) == len(df.index.get_level_values("date").unique())
    assert len(result.equity_curve) == len(result.dates)
    assert result.metrics["final_net_worth"] > 0
    assert len(result.transactions) > 0
    assert any(txn["ticker"] == "AAA" for txn in result.transactions)
    assert np.isfinite(result.metrics.get("CAGR", 0.0))


def test_rule_backtester_rebalance_cadence_scales_trade_days(monkeypatch):
    df = _build_test_frame()

    def rotating_score_func(frame: pd.DataFrame) -> pd.Series:
        dates = frame.index.get_level_values("date")
        tickers = frame.index.get_level_values("ticker")
        parity = (dates.day.to_numpy() % 2) == 0
        prefers_aaa = parity
        values = np.where(
            tickers.to_numpy() == "AAA",
            np.where(prefers_aaa, 2.0, 0.0),
            np.where(tickers.to_numpy() == "BBB", np.where(prefers_aaa, 0.0, 2.0), 0.0),
        )
        return pd.Series(values.astype(float), index=frame.index, dtype=float)

    strategy = StrategySpec(
        name="unit_rotating",
        description="rotates preferred ticker each day",
        regime_tags=(),
        score_func=rotating_score_func,
    )

    monkeypatch.setattr(config, "TOP_K", 1)
    monkeypatch.setattr(config, "WEIGHT_SMOOTHING", 0.0)
    monkeypatch.setattr(config, "CASH_RESERVE", 0.0)
    monkeypatch.setattr(config, "USE_REGIME_SYSTEM", False)
    monkeypatch.setattr(config, "UNIVERSE_FILTER", "all")
    monkeypatch.setattr(config, "ENABLE_UNIVERSE_QUALITY_FILTER", False)
    monkeypatch.setattr(config, "TRADING_REGION", "us")
    monkeypatch.setattr(config, "SLIPPAGE_COEFF", 0.0)
    monkeypatch.setattr(config, "US_COMMISSION_RATE", 0.0)
    monkeypatch.setattr(config, "US_FINRA_FEE_PER_SHARE", 0.0)
    monkeypatch.setattr(config, "US_SEC_FEE_RATE", 0.0)
    monkeypatch.setattr(config, "MIN_WEIGHT_CHANGE_TO_TRADE", 0.0)
    monkeypatch.setattr(config, "MIN_TRADE_DOLLARS", 0.0)
    monkeypatch.setattr(config, "MAX_DAILY_TURNOVER", None)
    monkeypatch.setattr(config, "BACKTEST_ENFORCE_CASH_BALANCE", False)

    daily = RuleBasedBacktester(
        df=df,
        strategies=[strategy],
        regime_table=None,
        initial_capital=10_000.0,
        apply_regime_overlays=False,
        rebalance_every_n_days=1,
    ).run()
    every_3 = RuleBasedBacktester(
        df=df,
        strategies=[strategy],
        regime_table=None,
        initial_capital=10_000.0,
        apply_regime_overlays=False,
        rebalance_every_n_days=3,
    ).run()

    daily_trade_dates = {txn["date"] for txn in daily.transactions}
    every_3_trade_dates = {txn["date"] for txn in every_3.transactions}
    assert len(every_3_trade_dates) <= 4
    assert len(every_3_trade_dates) < len(daily_trade_dates)


def test_rule_backtester_rebalance_cadence_must_be_positive(monkeypatch):
    df = _build_test_frame()
    monkeypatch.setattr(config, "UNIVERSE_FILTER", "all")
    monkeypatch.setattr(config, "ENABLE_UNIVERSE_QUALITY_FILTER", False)
    strategy = StrategySpec(
        name="unit_zero",
        description="test strategy",
        regime_tags=(),
        score_func=lambda frame: pd.Series(np.ones(len(frame)), index=frame.index, dtype=float),
    )
    with pytest.raises(ValueError, match="rebalance_every_n_days"):
        RuleBasedBacktester(
            df=df,
            strategies=[strategy],
            rebalance_every_n_days=0,
        )
