from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .costs import vectorized_brokerage_calculator
from .portfolio import get_target_weights
from .regime import get_regime_state, regime_gross_target, regime_top_k
from .strategy import StrategySpec
from .universe import NASDAQ100_TICKERS
from .utils import calculate_performance_metrics


@dataclass
class BacktestResult:
    equity_curve: list[float]
    dates: list[pd.Timestamp]
    transactions: list[dict]
    metrics: dict


class RuleBasedBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        strategies: Iterable[StrategySpec],
        regime_table: pd.DataFrame | None = None,
        initial_capital: float | None = None,
        apply_regime_overlays: bool = True,
        strategy_selector: Callable[[pd.Timestamp, dict, list[StrategySpec]], list[StrategySpec] | None]
        | None = None,
    ):
        self.df = df
        universe_filter = config.UNIVERSE_FILTER
        if universe_filter and universe_filter not in ("all", "none"):
            if universe_filter == "nasdaq100":
                allowed = set(NASDAQ100_TICKERS)
            else:
                allowed = {t.strip().upper() for t in universe_filter.split(",") if t.strip()}
            self.df = self.df[self.df.index.get_level_values("ticker").isin(allowed)]
        excluded = _load_excluded_tickers()
        if excluded:
            self.df = self.df[~self.df.index.get_level_values("ticker").isin(excluded)]
        if self.df.empty:
            raise ValueError("No data left after applying universe filter.")
        self.regime_table = regime_table
        self.initial_capital = initial_capital if initial_capital is not None else config.INITIAL_CAPITAL
        self.apply_regime_overlays = apply_regime_overlays and config.USE_REGIME_SYSTEM
        self.strategy_selector = strategy_selector

        self.strategies = list(strategies)
        if not self.strategies:
            raise ValueError("No strategies provided for backtest.")

        self.universe = self.df.index.get_level_values("ticker").unique().tolist()
        self.dates = self.df.index.get_level_values("date").unique().sort_values()
        self.strategy_scores = self._precompute_scores()

    def _precompute_scores(self) -> dict[str, pd.Series]:
        scores: dict[str, pd.Series] = {}
        for strategy in self.strategies:
            series = strategy.score_func(self.df)
            if not isinstance(series, pd.Series):
                raise ValueError(f"Strategy {strategy.name} must return a pandas Series.")
            if len(series) != len(self.df):
                raise ValueError(
                    f"Strategy {strategy.name} returned {len(series)} scores, expected {len(self.df)}."
                )
            if not series.index.equals(self.df.index):
                series = series.reindex(self.df.index)
            scores[strategy.name] = series.astype(float).fillna(0.0)
        return scores

    def _select_strategies(self, regime_label: str) -> list[StrategySpec]:
        active = [
            strategy
            for strategy in self.strategies
            if not strategy.regime_tags or regime_label in strategy.regime_tags
        ]
        return active if active else self.strategies

    def _combine_scores(self, active: list[StrategySpec], current_date) -> np.ndarray:
        combined = None
        for strategy in active:
            series = self.strategy_scores[strategy.name]
            try:
                day_scores = series.xs(current_date, level="date")
            except KeyError:
                continue
            day_scores = day_scores.reindex(self.universe).to_numpy(dtype=float)
            day_scores = np.nan_to_num(day_scores, nan=0.0, posinf=0.0, neginf=0.0)
            if len(day_scores) == 0:
                continue
            mean = np.nanmean(day_scores)
            std = np.nanstd(day_scores)
            if std > 1e-9:
                day_scores = (day_scores - mean) / std
            else:
                day_scores = day_scores - mean
            combined = day_scores if combined is None else combined + day_scores
        if combined is None:
            combined = np.zeros(len(self.universe), dtype=np.float32)
        return combined

    def run(self, start_date=None, end_date=None) -> BacktestResult:
        cash = float(self.initial_capital)
        positions = np.zeros(len(self.universe), dtype=np.int64)
        prev_weights = np.zeros(len(self.universe), dtype=np.float32)

        equity_curve: list[float] = []
        date_history: list[pd.Timestamp] = []
        transactions: list[dict] = []

        for current_date in tqdm(self.dates, desc="Backtesting", unit="day"):
            if start_date and current_date < start_date:
                continue
            if end_date and current_date >= end_date:
                break

            try:
                day_data = self.df.loc[current_date]
            except KeyError:
                continue
            day_data = day_data.reindex(self.universe)

            prices = day_data["Close"].to_numpy(dtype=float)
            mask = np.isfinite(prices) & (prices > 0)
            prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)

            vol = day_data.get("vol_21")
            vol = vol.to_numpy(dtype=float) if vol is not None else np.ones_like(prices)

            adv = day_data.get("adv_21")
            if adv is None:
                adv = np.zeros_like(prices)
            else:
                adv = adv.to_numpy(dtype=float)
            adv = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)

            state = get_regime_state(self.regime_table, current_date)
            selected = None
            if self.strategy_selector is not None:
                selected = self.strategy_selector(current_date, state, self.strategies)
            if selected is None:
                active_strategies = self._select_strategies(state["regime_label"])
            else:
                active_strategies = list(selected)
            active_strategy_names = [strategy.name for strategy in active_strategies]
            combined_scores = self._combine_scores(active_strategies, current_date)

            dynamic_top_k = config.TOP_K
            if self.apply_regime_overlays:
                dynamic_top_k = regime_top_k(state, config.TOP_K)

            weights = get_target_weights(combined_scores, vol, mask.astype(float), top_k=dynamic_top_k)

            if config.WEIGHT_SMOOTHING > 0 and np.any(prev_weights):
                weights = (1.0 - config.WEIGHT_SMOOTHING) * weights + config.WEIGHT_SMOOTHING * prev_weights
                total = np.sum(weights)
                if total > 1e-9:
                    weights = weights / total
            # Ensure non-tradable assets stay at zero after smoothing.
            weights = weights * mask
            total = np.sum(weights)
            if total > 1e-9:
                weights = weights / total
            else:
                weights = np.zeros_like(weights)

            if self.apply_regime_overlays:
                gross_target = min(regime_gross_target(state), 1.0)
                weights = weights * gross_target

            # Enforce a cash reserve to cover costs; avoid negative cash.
            reserve = config.CASH_RESERVE
            total = np.sum(weights)
            if total > 1e-9:
                cap = max(0.0, 1.0 - reserve)
                if total > cap:
                    weights = weights * (cap / total)

            cash_weight = max(0.0, 1.0 - np.sum(weights))

            net_worth = cash + np.sum(positions * prices)
            if net_worth <= 0:
                equity_curve.append(float(net_worth))
                date_history.append(pd.Timestamp(current_date))
                break
            target_alloc_dollars = weights * net_worth
            desired_shares = np.round(target_alloc_dollars / (prices + 1e-9)).astype(np.int64)
            desired_alloc_dollars = desired_shares * prices
            current_holdings_dollars = positions * prices
            trade_dollars = desired_alloc_dollars - current_holdings_dollars
            trade_shares = desired_shares - positions

            safe_adv_dollars = np.nan_to_num(adv * prices, nan=0.0, posinf=0.0, neginf=0.0)
            safe_adv_dollars = np.maximum(safe_adv_dollars, config.MIN_ADV_DOLLARS_SLIPPAGE)
            trade_frac_adv = np.abs(trade_dollars) / (safe_adv_dollars + 1e-9)
            slippage_costs = np.sum(np.abs(trade_dollars) * (config.SLIPPAGE_COEFF * trade_frac_adv))
            brokerage_costs = np.sum(
                vectorized_brokerage_calculator(trade_dollars, trade_shares=trade_shares)
            )
            total_costs = slippage_costs + brokerage_costs

            cash_after = cash - np.sum(trade_dollars) - total_costs
            if cash_after < -1e-6 and np.any(trade_dollars > 0):
                buy_mask = trade_dollars > 0
                current_holdings_dollars = positions * prices

                def _apply_buy_scale(scale: float):
                    adj_trade_shares = trade_shares.copy()
                    adj_trade_shares[buy_mask] = np.floor(
                        adj_trade_shares[buy_mask] * scale
                    ).astype(np.int64)
                    adj_desired_shares = positions + adj_trade_shares
                    adj_trade_dollars = (adj_desired_shares * prices) - current_holdings_dollars
                    adj_trade_frac_adv = np.abs(adj_trade_dollars) / (safe_adv_dollars + 1e-9)
                    adj_slippage_costs = np.abs(adj_trade_dollars) * (
                        config.SLIPPAGE_COEFF * adj_trade_frac_adv
                    )
                    adj_brokerage_costs = vectorized_brokerage_calculator(
                        adj_trade_dollars, trade_shares=adj_trade_shares
                    )
                    adj_total_costs = np.sum(adj_slippage_costs + adj_brokerage_costs)
                    adj_cash = cash - np.sum(adj_trade_dollars) - adj_total_costs
                    return (
                        adj_cash,
                        adj_desired_shares,
                        adj_trade_dollars,
                        adj_trade_shares,
                        adj_total_costs,
                    )

                lo, hi = 0.0, 1.0
                best = None
                for _ in range(12):
                    mid = (lo + hi) / 2.0
                    adj_cash, adj_desired, adj_dollars, adj_shares, adj_costs = _apply_buy_scale(mid)
                    if adj_cash >= -1e-6:
                        best = (adj_desired, adj_dollars, adj_shares, adj_costs)
                        lo = mid
                    else:
                        hi = mid
                if best is None:
                    _, adj_desired, adj_dollars, adj_shares, adj_costs = _apply_buy_scale(0.0)
                else:
                    adj_desired, adj_dollars, adj_shares, adj_costs = best
                desired_shares = adj_desired
                trade_dollars = adj_dollars
                trade_shares = adj_shares
                total_costs = adj_costs

            cash -= np.sum(trade_dollars) + total_costs
            positions = desired_shares

            net_worth = cash + np.sum(positions * prices)
            cash_weight = max(0.0, cash / net_worth) if net_worth > 0 else 0.0
            equity_curve.append(float(net_worth))
            date_history.append(pd.Timestamp(current_date))
            if net_worth <= 0:
                break

            for i in range(len(trade_dollars)):
                if abs(trade_dollars[i]) <= 1.0:
                    continue
                transactions.append(
                    {
                        "date": pd.Timestamp(current_date).strftime("%Y-%m-%d"),
                        "ticker": self.universe[i],
                        "action": "BUY" if trade_dollars[i] > 0 else "SELL",
                        "shares": float(trade_shares[i]),
                        "price_usd": float(prices[i]),
                        "value_usd": float(trade_dollars[i]),
                        "net_worth_usd": float(net_worth),
                        "cash_usd": float(cash),
                        "portfolio_value_usd": float(np.sum(positions * prices)),
                        "cash_weight": float(cash_weight),
                        "regime": state.get("regime_label", "unknown"),
                        "strategies": ",".join(active_strategy_names),
                    }
                )

            prev_weights = weights.copy()

        num_days = len(date_history)
        metrics = calculate_performance_metrics(pd.Series(equity_curve), num_days)
        metrics["final_net_worth"] = equity_curve[-1] if equity_curve else self.initial_capital

        return BacktestResult(
            equity_curve=equity_curve,
            dates=date_history,
            transactions=transactions,
            metrics=metrics,
        )


def _load_excluded_tickers() -> set[str]:
    path = config.resolve_path(config.EXCLUDED_TICKERS_FILE)
    if not path:
        return set()
    excluded_path = Path(path)
    if not excluded_path.exists():
        return set()
    try:
        lines = excluded_path.read_text().splitlines()
    except OSError:
        return set()
    return {line.strip().upper() for line in lines if line.strip()}
