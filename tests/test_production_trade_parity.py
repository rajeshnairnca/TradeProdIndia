import numpy as np
import pandas as pd

from src.production import ProductionState, generate_trades_for_date
from src.rule_backtester import RuleBasedBacktester
from src.strategy import StrategySpec


def _build_df() -> pd.DataFrame:
    date = pd.Timestamp("2025-01-02")
    idx = pd.MultiIndex.from_product([[date], ["AAA", "BBB"]], names=["date", "ticker"])
    return pd.DataFrame(
        {
            "Close": [100.0, 100.0],
            "vol_21": [0.2, 0.2],
            "adv_21": [1_000_000.0, 1_000_000.0],
        },
        index=idx,
    )


def _make_strategy(preferred: str) -> StrategySpec:
    def _scores(df: pd.DataFrame) -> pd.Series:
        tickers = df.index.get_level_values("ticker")
        values = np.where(tickers == preferred, 1.0, 0.0)
        return pd.Series(values, index=df.index, dtype=float)

    return StrategySpec(
        name=f"pref_{preferred}",
        description="test",
        regime_tags=(),
        score_func=_scores,
    )


def test_generate_trades_respects_rebalance_cadence(monkeypatch) -> None:
    df = _build_df()
    state = ProductionState(
        last_date="2025-01-01",
        cash=0.0,
        positions={"AAA": 10},
        prev_weights={"AAA": 1.0},
        rebalance_day_index=1,
        initial_deploy_completed=True,
    )
    monkeypatch.setattr("src.config.CASH_RESERVE", 0.0)

    trades, new_state, summary = generate_trades_for_date(
        df=df,
        strategies=[_make_strategy("BBB")],
        target_date=pd.Timestamp("2025-01-02"),
        state=state,
        apply_regime_overlays=False,
        rebalance_every=2,
        min_weight_change=0.0,
        min_trade_dollars=0.0,
    )

    assert summary["should_rebalance"] is False
    assert trades == []
    assert new_state.positions == {"AAA": 10}
    assert new_state.rebalance_day_index == 2


def test_generate_trades_applies_min_weight_change_on_non_initial_deploy(monkeypatch) -> None:
    df = _build_df()
    state = ProductionState(
        last_date="2025-01-01",
        cash=0.0,
        positions={"AAA": 10},
        prev_weights={"AAA": 1.0},
        rebalance_day_index=4,
        initial_deploy_completed=True,
    )
    monkeypatch.setattr("src.config.CASH_RESERVE", 0.0)

    trades, new_state, summary = generate_trades_for_date(
        df=df,
        strategies=[_make_strategy("BBB")],
        target_date=pd.Timestamp("2025-01-02"),
        state=state,
        apply_regime_overlays=False,
        rebalance_every=1,
        min_weight_change=2.0,
        min_trade_dollars=0.0,
    )

    assert summary["should_rebalance"] is True
    assert trades == []
    assert new_state.positions == {"AAA": 10}


def test_generate_trades_bypasses_trade_thresholds_on_initial_deploy(monkeypatch) -> None:
    df = _build_df()
    state = ProductionState(
        last_date=None,
        cash=1_000.0,
        positions={},
        prev_weights={},
        rebalance_day_index=0,
        initial_deploy_completed=False,
    )
    monkeypatch.setattr("src.config.CASH_RESERVE", 0.0)

    trades, new_state, _ = generate_trades_for_date(
        df=df,
        strategies=[_make_strategy("AAA")],
        target_date=pd.Timestamp("2025-01-02"),
        state=state,
        apply_regime_overlays=False,
        rebalance_every=1,
        min_weight_change=2.0,
        min_trade_dollars=20_000.0,
    )

    assert len(trades) > 0
    assert new_state.initial_deploy_completed is True
    assert any(abs(quantity) > 0 for quantity in new_state.positions.values())


def test_generate_trades_matches_backtester_for_daily_replay(monkeypatch) -> None:
    dates = [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-06")]
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    df = pd.DataFrame(
        {
            "Close": [100.0, 100.0, 110.0, 95.0, 105.0, 100.0],
            "vol_21": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "adv_21": [1_000_000.0] * 6,
        },
        index=idx,
    )

    def _scores(data: pd.DataFrame) -> pd.Series:
        d = data.index.get_level_values("date")
        t = data.index.get_level_values("ticker")
        prefer_aaa = d.isin([dates[0], dates[2]])
        values = np.where((prefer_aaa & (t == "AAA")) | ((~prefer_aaa) & (t == "BBB")), 1.0, 0.0)
        return pd.Series(values, index=data.index, dtype=float)

    strategy = StrategySpec(name="flip", description="test", regime_tags=(), score_func=_scores)

    monkeypatch.setattr("src.config.CASH_RESERVE", 0.0)
    monkeypatch.setattr("src.config.USE_REGIME_SYSTEM", False)
    monkeypatch.setattr("src.config.WEIGHT_SMOOTHING", 0.0)
    monkeypatch.setattr("src.config.ADAPTIVE_TURNOVER_ENABLED", False)

    params = {
        "rebalance_every": 2,
        "min_weight_change": 0.01,
        "min_trade_dollars": 0.0,
        "max_daily_turnover": 0.35,
    }

    backtester = RuleBasedBacktester(
        df=df,
        strategies=[strategy],
        regime_table=None,
        initial_capital=1_000.0,
        apply_regime_overlays=False,
    )
    backtest_result = backtester.run(show_progress=False, **params)

    state = ProductionState(last_date=None, cash=1_000.0, positions={}, prev_weights={})
    production_trades: list[dict] = []
    for current_date in dates:
        trades, state, _ = generate_trades_for_date(
            df=df,
            strategies=[strategy],
            target_date=current_date,
            state=state,
            regime_table=None,
            apply_regime_overlays=False,
            **params,
        )
        production_trades.extend(trades)

    def _norm(rows: list[dict]) -> list[tuple]:
        return [
            (
                str(item["date"]),
                str(item["ticker"]),
                str(item["action"]),
                round(float(item["shares"]), 6),
                round(float(item["value_usd"]), 6),
            )
            for item in rows
        ]

    assert _norm(production_trades) == _norm(backtest_result.transactions)
