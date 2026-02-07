from __future__ import annotations

import numpy as np
import pandas as pd

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
    monkeypatch.setattr(config, "TRADING_REGION", "us")
    monkeypatch.setattr(config, "SLIPPAGE_COEFF", 0.0)
    monkeypatch.setattr(config, "US_COMMISSION_RATE", 0.0)
    monkeypatch.setattr(config, "US_FINRA_FEE_PER_SHARE", 0.0)
    monkeypatch.setattr(config, "US_SEC_FEE_RATE", 0.0)

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
