# Agent Change Log

This file tracks code changes made by the assistant so they can be reviewed or reverted.

## 2025-12-31
- Removed liquidity filter gating in `src/rule_backtester.py` (ADV now used only for slippage scaling).
- Added region-aware data file selection in `src/config.py` (`TRADING_REGION` -> `data/daily_data_us.parquet` or `data/daily_data_india.parquet`), with `DATA_FILE` override.
- Defaulted `UNIVERSE_FILTER` to `all` in `src/config.py`.
- Routed ensemble/sweep outputs by region:
  - `scripts/backtesting/backtester.py` -> `alphas/_ensembles/<region>/...`
  - `scripts/backtesting/strategy_sweep.py` -> `alphas/_ensembles/<region>/_sweeps/...`
  - `scripts/xgboost/backtest_xgb_alpha.py` -> `alphas/_ensembles/<region>/...`
  - `src/walkforward.py` -> `alphas/_ensembles/<region>/...` for multi-strategy runs
  - `scripts/orchestration/orchestrator.py` now reads ensemble results from `alphas/_ensembles/<region>/...`
- Scoped orchestration history to region in `scripts/orchestration/orchestrator.py` (`ensemble_history_<region>.json`).
- Added `regime` and `strategies` columns to each transaction row in `src/rule_backtester.py`.
- Changed strategy selection in `src/rule_backtester.py` to apply only one strategy per day (first matching tag); if no tags match, no strategy is applied and the day stays in cash.
- Added EMA smoothing option for regime calculation in `src/regime.py` with `REGIME_EMA_SPAN` in `src/config.py` (defaults to 10).
- Increased regime smoothing windows and made them configurable in `src/config.py`:
  - `REGIME_EMA_SPAN = 20`
  - `ROLLING_WINDOW_FOR_VOL = 42`
  - `REGIME_TREND_SHORT_WINDOW = 100`, `REGIME_TREND_LONG_WINDOW = 200`
  - `REGIME_DISPERSION_MIN_PERIODS = 60`
  - `src/regime.py` now uses these settings.
- Reverted the single-strategy-per-day selection and regime smoothing/window changes, restoring the prior behavior and defaults.
- Added `scripts/ml_regime/ml_strategy_selector.py` to train an ML selector that predicts the best strategy per day and reports a stitched backtest (no switching costs).
- Updated `scripts/ml_regime/ml_strategy_selector.py` to label by forward-window returns, add `--label-horizon`, and report majority/random/top-k accuracy.
- Added ML-driven hybrid backtest support:
  - `src/rule_backtester.py` accepts an optional `strategy_selector` hook.
  - `scripts/ml_regime/ml_strategy_selector.py` now runs a hybrid backtest and writes `transactions.csv` and `results.json`.
- Updated ML selector to enforce a minimum hold period (`--hold-days`, defaults to `--label-horizon`) and evaluate decision accuracy at switch points.
- Replaced fixed hold with dynamic switching: the ML selector now switches only after `--confirm-days` consecutive predictions for a new strategy.
- Reworked `scripts/ml_regime/ml_strategy_selector.py` into a supervised regime classifier and hybrid backtest using predicted regimes to choose strategies.
- Added optional HMM-based regime labeling in `src/regime.py` with `REGIME_MODE` and `HMM_N_COMPONENTS` in `src/config.py` (requires `hmmlearn`).
- Added walk-forward HMM regime mode (`REGIME_MODE=hmm_rolling`) with `HMM_WARMUP_PERIOD` and `HMM_STEP_SIZE` to reduce lookahead bias.
- Added `--jobs` to `scripts/backtesting/strategy_sweep.py` to parallelize sweep backtests.
- Added `--jobs` to `scripts/ml_regime/ml_strategy_selector.py` to use multiple CPU threads for XGBoost training.
- Added exhaustive per-regime strategy mapping search to `scripts/ml_regime/ml_strategy_selector.py` (`--mapping-search`).
- Added `--mapping-jobs` to parallelize mapping search in `scripts/ml_regime/ml_strategy_selector.py`.
- Added a guard in `scripts/ml_regime/ml_strategy_selector.py` for single-class training splits to avoid XGBoost errors.
- Added optional HMM state-based regime labels in `src/regime.py` with `HMM_STATE_LABELS` in `src/config.py` to expose N unnamed regimes for HMM modes.
- Added `scripts/visualize_regimes.py` CLI options to render 4-state regime comparisons and optional HMM state plots.
- Fixed `scripts/visualize_regimes.py` to add project root to `sys.path` so `src` imports work when executed directly.
- Added HMM fit fallback controls in `src/config.py` (`HMM_COVARIANCE_TYPE`, `HMM_MIN_COVAR`) and a retry path in `src/regime.py` to avoid non‑positive‑definite covariance failures.
- Added a command/flag reference section to `README.md` covering scripts and environment flags.
- Fixed `scripts/visualize_regimes.py` to avoid Matplotlib deprecation warnings and handle missing HMM state values in plots.
- Added `--eval-full-period` to `scripts/ml_regime/ml_strategy_selector.py` so hybrid evaluation can run on the full filtered period and included extra period metadata in outputs.
- Added `performance.png` output to `scripts/ml_regime/ml_strategy_selector.py` for the hybrid backtest.
- Added regime overlays to the ML selector performance plot in `scripts/ml_regime/ml_strategy_selector.py`.
- Fixed colormap handling in `scripts/ml_regime/ml_strategy_selector.py` and `scripts/visualize_regimes.py` for older Matplotlib versions.
- Added `--skip-ml` to `scripts/ml_regime/ml_strategy_selector.py` to bypass the classifier and use regime labels directly.
- Made `--skip-ml` skip the train/test split so evaluation runs on the full filtered period.
- Added automatic run summaries to `runs/ml_regime/summary.csv` from `scripts/ml_regime/ml_strategy_selector.py` with detailed metadata and metrics.
- Expanded ML regime summary rows with eval duration and regime switch stats.

## 2025-01-05
- Added a change-log requirement in `AGENTS.md`.
- Added a TODO note in `src/regime.py` to consider ADX/slope sideways overrides and rolling vol quantile thresholds.
