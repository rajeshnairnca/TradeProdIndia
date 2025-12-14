import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from . import config

def vectorized_brokerage_calculator(trade_dollars: np.ndarray) -> np.ndarray:
    trade_value = np.abs(trade_dollars)
    buy_mask = trade_dollars > 1.0
    sell_mask = trade_dollars < -1.0
    stt = trade_value * 0.001
    nse_charges = trade_value * 0.0000322
    sebi_charges = trade_value * 1e-6
    stamp_charges = np.where(buy_mask, trade_value * 0.00015, 0.0)
    dp_charges = np.where(sell_mask, 15.34, 0.0)
    gst = (nse_charges + sebi_charges) * 0.18
    return stt + stamp_charges + nse_charges + sebi_charges + gst + dp_charges

class DailyCrossSectionalEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, symbol_to_sector_id: dict, num_sectors: int, universe: list, feature_keys: list, is_backtest=False, regime_table: pd.DataFrame | None = None, apply_regime_top_k: bool = False):
        super().__init__()
        self.use_regime_system = config.USE_REGIME_SYSTEM and apply_regime_top_k
        
        # --- REGIME CALCULATION ---
        if self.use_regime_system:
            if regime_table is not None and "trend_up" in regime_table.columns and "vol_high" in regime_table.columns:
                rt = regime_table.copy()
                if "combined_regime" not in rt.columns:
                    rt["combined_regime"] = 2 * rt["trend_up"].astype(float) + rt["vol_high"].astype(float)
                market_proxy = rt[["combined_regime"]]
            else:
                market_proxy = df.groupby(level='date')['Close'].mean().to_frame(name='market_close')
                market_proxy['sma_50'] = market_proxy['market_close'].rolling(window=50, min_periods=50).mean()
                market_proxy['sma_200'] = market_proxy['market_close'].rolling(window=200, min_periods=200).mean()
                market_proxy['bull_regime'] = (market_proxy['sma_50'] > market_proxy['sma_200']).astype(float)
                market_proxy['returns'] = market_proxy['market_close'].pct_change()
                market_proxy['volatility'] = market_proxy['returns'].rolling(window=config.ROLLING_WINDOW_FOR_VOL, min_periods=config.ROLLING_WINDOW_FOR_VOL).std()
                vol_threshold = market_proxy['volatility'].expanding(min_periods=config.ROLLING_WINDOW_FOR_VOL).quantile(0.75)
                market_proxy['vol_regime'] = (market_proxy['volatility'] > vol_threshold).astype(float)
                market_proxy['combined_regime'] = 2 * market_proxy['bull_regime'] + market_proxy['vol_regime']
            self.data = df.join(market_proxy[['combined_regime']])
            self.data['combined_regime'] = self.data['combined_regime'].bfill()
            self.data['combined_regime'] = self.data['combined_regime'].ffill()
        else:
            self.data = df.copy()
            self.data['combined_regime'] = 0.0

        self.dates = self.data.index.get_level_values('date').unique().sort_values().to_pydatetime()
        self.universe = universe
        if len(self.dates) < 200: raise RuntimeError("Not enough dates for 200-day moving average.")
        
        self.n_stocks = len(self.universe)
        self.symbol_to_sector_id = symbol_to_sector_id
        self.num_sectors = num_sectors
        self.is_backtest = is_backtest
        self.top_k = config.TOP_K # Added for dynamic portfolio sizing
        self.regime_table = regime_table
        self.apply_regime_top_k = apply_regime_top_k
        
        provided_feature_keys = feature_keys if feature_keys is not None else []
        self._feature_keys = sorted(provided_feature_keys) if provided_feature_keys else sorted([c for c in self.data.columns if c.endswith('_z')])
        num_stock_features = len(self._feature_keys)
        num_global_features = 2  # For VIX and Market Regime

        # Action space: target weights for each stock plus a cash slot (last element)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_stocks + 1,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "stock_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_stocks, num_stock_features), dtype=np.float32),
            "sector_ids": spaces.Box(low=0, high=self.num_sectors - 1, shape=(self.n_stocks,), dtype=np.int64),
            "global_features": spaces.Box(low=-np.inf, high=np.inf, shape=(num_global_features,), dtype=np.float32),
            "mask": spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)
        })
        self._reset_account_state()

    def _reset_account_state(self):
        self.cash = float(config.INITIAL_CAPITAL)
        self.positions = np.zeros(self.n_stocks, dtype=np.float32)
        self.prev_target_weights = np.zeros(self.n_stocks, dtype=np.float32)
        self.net_worth = float(config.INITIAL_CAPITAL)
        self.prev_net_worth = float(config.INITIAL_CAPITAL)
        self.current_step = 0
        self.returns_history = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_account_state()
        min_start_idx = max(config.ADV_LOOKBACK, config.ROLLING_WINDOW_FOR_VOL)
        if self.is_backtest:
            self.current_idx = min_start_idx
        else:
            max_start_idx = len(self.dates) - 2
            if max_start_idx <= min_start_idx: raise RuntimeError("Dataset too small for lookback periods.")
            self.current_idx = self.np_random.integers(min_start_idx, max_start_idx + 1)
        
        obs, _ = self._assemble_observation(self.current_idx)
        self.prev_net_worth = self.net_worth
        return obs, {"date": self.dates[self.current_idx].strftime('%Y-%m-%d'), "net_worth": self.net_worth}
    
    def _assemble_observation(self, date_idx):
        current_date = self.dates[date_idx]
        try:
            day_data = self.data.loc[current_date]
        except KeyError:
            day_data = pd.DataFrame(columns=self.data.columns)

        day_data = day_data.reindex(self.universe)
        # Ensure all expected feature columns exist
        for col in self._feature_keys:
            if col not in day_data.columns:
                day_data[col] = 0.0
        stock_features = day_data[self._feature_keys].fillna(0).to_numpy(dtype=np.float32)
        mask = (~day_data['Close'].isna()).to_numpy(dtype=np.float32)
        sector_ids = np.array([self.symbol_to_sector_id.get(ticker, 0) for ticker in self.universe], dtype=np.int64)
        
        if 'VIX_z' not in day_data.columns:
            day_data['VIX_z'] = 0.0
        vix_feature = day_data['VIX_z'].fillna(0).iloc[0] if not day_data.empty else 0.0
        regime_feature = day_data['combined_regime'].iloc[0] if not day_data.empty and 'combined_regime' in day_data.columns else 0.0
        global_features = np.array([vix_feature, regime_feature], dtype=np.float32)

        obs = {"stock_features": stock_features, "sector_ids": sector_ids, "global_features": global_features, "mask": mask}
        prices, adv, vol = self._get_market_data(day_data)
        return obs, (prices, adv, vol)

    def _get_market_data(self, day_data):
        prices = day_data['Close'].to_numpy(dtype=np.float32)
        adv = day_data['adv_21'].to_numpy(dtype=np.float32)
        vol = day_data['vol_21'].to_numpy(dtype=np.float32)
        return prices, adv, vol

    def _get_regime_flags(self, current_date):
        """Lookup precomputed regime flags if available."""
        if not self.use_regime_system:
            return {"vol_high": False, "dispersion_high": False, "dispersion_low": False, "breadth_low": False, "breadth_high": False}
        if self.regime_table is None or current_date not in self.regime_table.index:
            return {"vol_high": False, "dispersion_high": False, "dispersion_low": False, "breadth_low": False, "breadth_high": False}
        row = self.regime_table.loc[current_date]
        return {
            "vol_high": bool(row.get("vol_high", False)),
            "dispersion_high": bool(row.get("dispersion_high", False)),
            "dispersion_low": bool(row.get("dispersion_low", False)),
            "breadth_low": bool(row.get("breadth_low", False)),
            "breadth_high": bool(row.get("breadth_high", False)),
        }

    def _regime_top_k(self, current_date):
        flags = self._get_regime_flags(current_date)
        # Concentrate more in choppy, weak-signal regimes; broaden when signals are clean.
        if flags["vol_high"] and flags["dispersion_low"] and flags["breadth_low"]:
            return 5
        if flags["vol_high"] and flags["dispersion_high"]:
            return 7
        if (not flags["vol_high"]) and flags["dispersion_high"] and flags["breadth_high"]:
            return 12
        if (not flags["vol_high"]) and flags["dispersion_high"]:
            return 10
        if flags["vol_high"]:
            return 6
        return self.top_k

    def _regime_gross_target(self, current_date, trend_up):
        if not self.use_regime_system:
            return 1.0
        flags = self._get_regime_flags(current_date)
        gross = 0.85
        # Bear + high vol
        if (not trend_up) and flags["vol_high"]:
            gross = 0.6
        # Bear + low vol
        elif (not trend_up) and (not flags["vol_high"]):
            gross = 0.75
        # High vol + low dispersion + low breadth (choppy)
        elif flags["vol_high"] and flags["dispersion_low"] and flags["breadth_low"]:
            gross = 0.6
        # High vol + high dispersion (volatile but some winners)
        elif flags["vol_high"] and flags["dispersion_high"]:
            gross = 0.8
        # Low vol + high dispersion + high breadth (clean trend/momentum)
        elif (not flags["vol_high"]) and flags["dispersion_high"] and flags["breadth_high"]:
            gross = 1.05  # slightly more aggressive in the best regimes
        # Low vol + high dispersion (but breadth not strong)
        elif (not flags["vol_high"]) and flags["dispersion_high"]:
            gross = 0.95
        # Low vol + low dispersion (quiet, low signal differentiation)
        elif (not flags["vol_high"]) and flags["dispersion_low"]:
            gross = 0.85
        return max(0.0, min(1.1, gross))

    def step(self, target_weights: np.ndarray):
        # The action is the target portfolio weights (per stock) plus cash weight in the last slot.
        target_weights = np.asarray(target_weights, dtype=np.float32).reshape(-1)
        obs, (prices, adv, vol) = self._assemble_observation(self.current_idx)

        # Apply the mask from the observation to prevent trading unavailable assets.
        mask = obs['mask']
        stock_weights = target_weights[:-1] * mask
        cash_weight = float(target_weights[-1])

        cash_weight = min(max(cash_weight, 0.0), 1.0)

        # Normalize stock weights to sum to (1 - cash_weight)
        stock_sum = np.sum(stock_weights)
        if stock_sum > 1e-9:
            stock_weights = stock_weights / stock_sum
            stock_weights = stock_weights * max(0.0, 1.0 - cash_weight)
        else:
            stock_weights = np.zeros_like(stock_weights)

        # Smooth weights to encourage holding periods
        if np.any(self.prev_target_weights):
            stock_weights = (1.0 - config.WEIGHT_SMOOTHING) * stock_weights + config.WEIGHT_SMOOTHING * self.prev_target_weights
            # Renormalize after smoothing
            stock_sum = np.sum(stock_weights)
            if stock_sum > 1e-9:
                stock_weights = stock_weights / stock_sum
                stock_weights = stock_weights * max(0.0, 1.0 - cash_weight)
            else:
                stock_weights = np.zeros_like(stock_weights)
        self.prev_target_weights = stock_weights.copy()

        # Dynamically adjust portfolio size based on market regime
        # Use a more concentrated portfolio in high-volatility markets to focus on best ideas.
        regime_feature = obs["global_features"][1]  # [VIX_z, combined_regime]
        # Regimes 1 (Bear, High Vol) and 3 (Bull, High Vol) are high volatility
        if regime_feature in [1.0, 3.0]:
            dynamic_top_k = 5  # Be more selective and concentrated in volatile markets
        else:
            dynamic_top_k = self.top_k  # Use the default in normal markets

        # Apply optional regime-table-based top_k
        if self.use_regime_system and self.apply_regime_top_k:
            current_date = self.dates[self.current_idx]
            dynamic_top_k = min(dynamic_top_k, self._regime_top_k(current_date))

        # Apply concentration: keep only dynamic_top_k largest weights, set others to zero
        if dynamic_top_k < self.n_stocks:
            top_k_indices = np.argpartition(stock_weights, -dynamic_top_k)[-dynamic_top_k:]
            concentrated_weights = np.zeros_like(stock_weights)
            concentrated_weights[top_k_indices] = stock_weights[top_k_indices]
            if np.sum(concentrated_weights) > 1e-9:
                stock_weights = concentrated_weights / np.sum(concentrated_weights)
                stock_weights = stock_weights * max(0.0, 1.0 - cash_weight)
            else:
                stock_weights = np.zeros_like(stock_weights)
        # Regime-aware gross exposure
        if self.use_regime_system and self.apply_regime_top_k:
            current_date = self.dates[self.current_idx]
            trend_flag = regime_feature in [2.0, 3.0]  # combined_regime 2/3 are bull; 0/1 bear
            gross_target = self._regime_gross_target(current_date, trend_flag)
            stock_weights = stock_weights * gross_target
            cash_weight = 1.0 - np.sum(stock_weights)
        else:
            cash_weight = max(0.0, 1.0 - np.sum(stock_weights))

        prev_positions = self.positions.copy()
        target_alloc_dollars = stock_weights * self.net_worth
        current_prices = np.nan_to_num(prices)
        # Round target allocation to whole shares
        desired_shares = np.round(target_alloc_dollars / (current_prices + 1e-9))
        desired_shares = np.nan_to_num(desired_shares, nan=0.0, posinf=0.0, neginf=0.0)
        desired_shares = desired_shares.astype(np.int64)
        desired_alloc_dollars = desired_shares * current_prices
        current_holdings_dollars = self.positions * current_prices
        trade_dollars = desired_alloc_dollars - current_holdings_dollars
        trade_shares = desired_shares - prev_positions
        
        safe_adv_dollars = np.nan_to_num(adv * current_prices, nan=1e9)
        trade_frac_adv = np.abs(trade_dollars) / (safe_adv_dollars + 1e-9)
        slippage_costs = np.sum(np.abs(trade_dollars) * (config.SLIPPAGE_COEFF * trade_frac_adv))
        brokerage_costs = np.sum(vectorized_brokerage_calculator(trade_dollars))
        total_costs = slippage_costs + brokerage_costs
        
        self.cash -= np.sum(trade_dollars) + total_costs
        self.positions = desired_shares
        
        portfolio_value = np.sum(self.positions * current_prices)
        self.net_worth = self.cash + portfolio_value
        
        epsilon = 1e-8
        log_return = math.log(max(self.net_worth, epsilon) / max(self.prev_net_worth, epsilon))
        
        turnover = np.sum(np.abs(trade_dollars)) / self.prev_net_worth if self.prev_net_worth > 0 else 0
        turnover_pen = config.TURNOVER_PENALTY * turnover
        # Penalize weight changes to reduce churn (swing-friendly)
        weight_change = np.sum(np.abs(stock_weights - self.prev_target_weights))
        weight_change_pen = config.WEIGHT_CHANGE_PENALTY * weight_change
        
        leverage = portfolio_value / self.net_worth if self.net_worth > 0 else 1.0
        leverage_pen = config.LEVERAGE_PENALTY * max(0, leverage - 1.0)
        
        self.returns_history.append(log_return)
        
        # --- Regime-Adaptive Risk Penalty ---
        current_regime = obs['global_features'][1] 
        if current_regime == 0: # Bear, Low-Vol
            risk_coeff = 0.7
        elif current_regime == 1: # Bear, High-Vol
            risk_coeff = 0.9
        elif current_regime == 2: # Bull, Low-Vol
            risk_coeff = 0.3
        else: # Bull, High-Vol (current_regime == 3)
            risk_coeff = 0.5
        
        risk_penalty = 0.0
        if len(self.returns_history) > config.ROLLING_WINDOW_FOR_VOL:
            recent_returns = np.array(self.returns_history[-config.ROLLING_WINDOW_FOR_VOL:])
            downside_returns = recent_returns[recent_returns < 0]
            if len(downside_returns) > 1:
                downside_deviation = np.std(downside_returns)
                risk_penalty = risk_coeff * downside_deviation
        
        # Apply cash drag only when gross target is reasonably high; avoid penalizing defensive stances
        if 'gross_exposure' in locals():
            drag_active = gross_exposure >= 0.75
        else:
            drag_active = True
        cash_drag_pen = config.CASH_DRAG_COEFF * cash_weight if drag_active else 0.0
        reward = log_return - turnover_pen - leverage_pen - risk_penalty - cash_drag_pen - weight_change_pen
        
        done = False
        if self.net_worth <= 0:
            reward -= 1.0
            done = True
        if self.current_idx >= len(self.dates) - 1:
            done = True
            
        self.prev_net_worth = self.net_worth
        self.current_idx += 1
        
        next_obs, _ = self._assemble_observation(self.current_idx if not done else self.current_idx - 1)
        gross_exposure = np.sum(np.abs(target_alloc_dollars)) / (self.net_worth + epsilon)
        info = {
            "date": self.dates[self.current_idx - 1].strftime('%Y-%m-%d'),
            "net_worth": self.net_worth,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "gross_exposure": gross_exposure,
            "cash_weight": cash_weight,
            "leverage": leverage,
            "turnover": turnover,
            "reward": reward,
            "trade_dollars": trade_dollars,
            "trade_shares": trade_shares,
            "prices": current_prices
        }
        
        return next_obs, float(reward), bool(done), False, info
