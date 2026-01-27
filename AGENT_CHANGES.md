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

## 2025-01-06
- Added a raw SQLite store for market data in `src/data_store.py` and configuration in `src/config.py` (`RAW_DATA_DB_FILE`).
- Added `scripts/data_extraction/ingest_yfinance_db.py` to append yfinance data into the raw DB with duplicate protection and optional metadata.
- Added `src/data_prep.py` and `scripts/data_extraction/prepare_dataset.py` to build lookahead-safe, normalized datasets from the raw DB.
- Added universe CSVs in `data/universe_us.csv` and `data/universe_india.csv`, plus `.gitignore` allowlisting for `data/universe_*.csv`.
- Added per-ticker date bounds (`start_date`, `end_date`) and `--min-history-days` gating for newly listed tickers.
- Added flexible universe CSV parsing in `src/universe.py` and restored `NASDAQ100_TICKERS`.
- Added `--regime-mapping` support to `scripts/backtesting/backtester.py` for per-regime strategy selection.
- Added `--regime-mapping` support to `scripts/backtesting/sector_experiments.py` for fixed per-regime mappings.
- Documented the 31% CAGR sector experiment command in `README.md`.
- Set `WEIGHT_SMOOTHING` default to 0.85 in `src/config.py` for longer holds.

## 2026-01-24
- Added `--plot` support to `scripts/backtesting/sector_experiments.py` to save equity curve PNGs under the run output directory.
- Added regime-background shading to sector experiment plots using regime labels.
- Darkened regime background colors in sector experiment plots for better visibility.
- Added `BEAR_CASH_OUT` config flag to optionally zero exposure in bear regimes.
- Added `BEAR_GROSS_TARGET` to allow fixed bear-regime exposure overrides.
- Added `REGIME_GROSS_TARGETS` support for per-regime gross target overrides.

## 2026-01-25
- Added US sell-side FINRA per-share and SEC notional fees to the cost model, with env overrides and backtester wiring.
- Added production pipeline helpers in `src/production.py` to update market data and generate daily trades, now sourced from TradingView via `tradingview_ta`.
- Added `scripts/production/daily_run.py` for daily data refresh + trade generation output.
- Added `yfinance`, `tradingview_ta`, and `pandas_ta` to `requirements.txt` for market data/indicator dependencies.
- Added sector-filtered production runs with `--sector` and `--regime-scope` to mirror sector experiment behavior.
- Added a guard to require all tickers update from TradingView before writing the parquet (override with `--allow-partial-updates`).
- Defaulted production runs to the Technology sector with global heuristic regimes and the specified regime-to-strategy mapping in `scripts/production/daily_run.py`.
- Cleaned repository by removing non-production/backtesting scripts, logs, runs, and analysis artifacts (kept data files).
- Added production support for cash injections and adding new tickers with yfinance history before daily TradingView updates.
- Added a cash injection log and a universe registry CSV to track added tickers.
- Added pending adjustments queue support (cash/ticker) and a helper script `scripts/production/queue_adjustments.py` for app-driven updates.
- Added production data pruning to keep only the most recent trading days (configurable via `RETENTION_TRADING_DAYS`).
- Added `DATA_ROOT` path support and a minimal FastAPI server (`scripts/production/api_server.py`) with a POST endpoint to queue adjustments.
- Added API endpoints to list and clear pending adjustments.
- Added DELETE /pending-adjustments as an alias to clear the queue.
- Added API key protection for non-health endpoints, updated Dockerfile for Railway, and added a production dry-run script.
- Added /latest-run endpoint to report the most recent production run date.
- Added .dockerignore to keep container builds small and fixed pandas-ta dependency name for Docker installs.
- Switched pandas-ta dependency to pandas_ta for container compatibility.
- Switched to pandas-ta-classic and updated imports to pandas_ta_classic for container builds.
- Switched data extraction script to use `pandas_ta` for indicator generation on re-extraction runs.
- Defaulted data extraction to read tickers from `data/universe_us.txt` instead of the built-in list.
- Added `requirements.production.txt` with `pandas-ta-classic` + NumPy >=2 for containerized production.
- Updated Dockerfile to install production requirements; kept `requirements.txt` for pandas_ta-based extraction.
- Switched TradingView updates in `src/production.py` to batch calls via `get_multiple_analysis`.
- Added `data/universe_us_exchange_map.json` mapping tickers to NYSE for manual verification.
- Updated `data/universe_us_exchange_map.json` using TradingView scan data; 2 tickers (MMC, QUBTS) missing from scan.
- Updated `data/universe_us.txt` to remove MMC and replace QUBTS with QUBT; regenerated exchange map from TradingView data.
- Added TradingView exchange map support in production updates, with 200-symbol batches capped at 3 requests.
- Added exchange-aware ticker adjustments via API/queue scripts, and exchange map updates during daily runs.
- Added API endpoints for current portfolio snapshot and full universe exchange listing.
- Set initial portfolio cash to $27,000 in config.

## 2026-01-26
- Tightened .dockerignore to ship only US production data files in container builds.
- Defaulted Docker image to TRADING_REGION=us for production runs.
- Added cumulative cost tracking to production state and summaries (with per-day costs preserved).
- Added daily run backfill to seed cumulative costs from existing summaries.
- Added /trades endpoint with pagination and /cagr endpoint for overall performance.
- Added a Postgres-backed production layer (runs, trades, state, pending adjustments) with DATABASE_URL support.
- Updated production daily run and API server to read/write Postgres when configured.
- Added cash-flow-adjusted CAGR (time-weighted) alongside the standard CAGR and a money-weighted IRR.
- Added a backfill script to load existing production runs/trades/state into Postgres.
- Cleaned local caches/venv artifacts and expanded .gitignore to keep them out of GitHub.
- Removed local `runs/` directory per cleanup request.

## 2026-01-27
- Cleaned the repo for the GitHub move: removed local venv/caches/logs and selected scripts/utilities; updated .gitignore, Dockerfile, requirements, config, and the data extraction script.
