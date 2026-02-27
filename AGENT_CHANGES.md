# Agent Change Log

This file tracks code changes made by the assistant so they can be reviewed or reverted.

## 2026-02-27
- Optimized broker API payload control in `scripts/production/api_server.py`:
  - Added `include_payload` query param (default `false`) to `/broker-summary`, `/broker-positions`, `/broker-orders`, and `/latest-broker-orders`.
  - Added `fields` query param support for those endpoints with strict field validation so clients can request minimal column sets.
  - Added `meta_only=true` support to `/broker-orders` and `/latest-broker-orders` for lightweight metadata/count responses without row payloads.
  - Added `GET /latest-broker-orders/count` as a count-only endpoint for app polling.
- Extended broker DB read helpers in `src/production_db.py` to support SQL-level column projection (dynamic `SELECT` list) and payload omission:
  - `latest_broker_account(...)`
  - `list_latest_broker_positions(...)`
  - `list_broker_orders(...)`
  - `list_latest_broker_orders(...)`
  - Added `count_broker_orders(...)` and `count_latest_broker_orders(...)` for metadata-only API flows.
- Added shared CAGR computation utilities in `src/cagr_metrics.py` and moved `/cagr` (`scripts/production/api_server.py`) to DB-read-first behavior:
  - `/cagr` now returns the latest persisted `cagr_payload` when available and falls back to on-demand recomputation for older rows.
- Extended `production_runs` persistence in `src/production_db.py` with `cagr_payload jsonb` (schema migration + upsert/select wiring).
- Updated `scripts/production/daily_run.py` to compute and persist `cagr_payload` as part of each run so app reads no longer need full-history recomputation.
- Moved universe selection diagnostics to DB-backed snapshots:
  - Added `src/selection_diagnostics.py` to build per-ticker selection-stage diagnostics from production data.
  - Updated `scripts/production/daily_run.py` to compute and persist a diagnostics snapshot each non-dry run.
  - Added `production_universe_selection_diagnostics_state` and `production_universe_selection_diagnostics_records` tables plus DB helpers in `src/production_db.py`.
  - Updated `GET /universe/selection-diagnostics` in `scripts/production/api_server.py` to read persisted diagnostics from DB (no parquet/file dependency).
- Optimized `/summaries` field projection path:
  - Added summary field allowlist constant `RUN_SUMMARY_FIELDS` in `src/production_db.py` and reused it in API (`SUMMARY_FIELDS = DB_RUN_SUMMARY_FIELDS`) so API validation and DB projection stay synchronized.
  - Updated `scripts/production/api_server.py` to validate `fields` on `/summaries` and pass normalized field names to DB.
  - Updated `src/production_db.py::list_run_summaries_paginated(...)` to accept optional `fields` and build a safe dynamic SQL `SELECT` list from allowlisted columns (including `date -> run_date AS date` alias), so only requested columns are fetched.
  - Removed Python-side post-fetch trimming for `/summaries`; invalid requested fields now return HTTP 400 instead of silently returning null keys.

## 2026-02-25
- Updated `scripts/production/api_server.py` `/cagr` annualization to match backtester behavior by using a 252-trading-day basis from summary-row counts (including broker `cagr`/`cagr_adjusted`) instead of calendar-day `365.25` elapsed-time annualization.
- Fixed broker execution-cost reconciliation sign in `scripts/production/daily_run.py` so external cash adjustments (deposits/withdrawals) are added with the correct direction when computing `broker_execution_cost` and cumulative broker execution-cost totals.
- Added `/universe/selection-diagnostics` in `scripts/production/api_server.py` to expose per-ticker signal-selection stages/reasons (universe filter, excluded list, quality filter, invalid close, and top-k snapshot status) with summary counts and pagination.
- Converted `scripts/backtesting/tech_universe_monitor.py` streak/cache continuity to DB-backed loading (from latest universe-monitor snapshot rows + summary run date) and made DB required for monitor runs, removing file-state dependency that could reset `pass_streak` on non-persistent filesystems.

## 2026-02-20
- Added a shared pagination guard in `scripts/production/api_server.py` (`_normalize_pagination`) with configurable defaults (`API_DEFAULT_PAGE_LIMIT`, `API_MAX_PAGE_LIMIT`) and strict validation (`limit > 0`, `offset >= 0`, capped max limit) to prevent accidental unbounded reads.
- Converted list-style API endpoints to a consistent paginated response contract (`total`, `count`, `limit`, `offset`, data array) and removed unbounded defaults:
  - `/summaries`, `/latest-trades`, `/trades`
  - `/universe`
  - `/universe-monitor/candidates`, `/universe-monitor/potential`
  - `/broker-positions`, `/broker-orders`, `/latest-broker-orders`
  - `/excluded-tickers`
  - `/run-calendar/overrides`
  - `/stale-tickers`
  - `/pending-adjustments`
- Added SQL-level paginated helpers in `src/production_db.py` so pagination happens in Postgres (instead of loading full tables in memory first):
  - `list_universe_map(...)`
  - `list_excluded_tickers(...)`
  - `list_run_calendar_overrides_paginated(...)`
  - `list_run_summaries_paginated(...)`
  - `list_latest_trades(...)`
  - `list_latest_broker_positions(...)`
  - `list_latest_broker_orders(...)`
  - `list_pending_adjustments(...)`
- Kept existing full-load helper functions for non-API internal workflows (e.g., CAGR/existing production scripts) while routing API reads through the new paginated helpers.

## 2026-02-17
- Updated `src/production_market_data.py` `add_universe_tickers(...)` with a new `fail_on_no_valid_tickers` flag (default `True`) so callers can choose non-fatal behavior when all requested new tickers fail history sufficiency checks.
- Wired `scripts/production/daily_run.py` to call `add_universe_tickers(..., fail_on_no_valid_tickers=False)`, preventing production runs from aborting when queued tickers have insufficient post-indicator history (e.g., newly listed symbols).
- Added `tests/test_production_market_data_add_tickers.py` to cover both modes:
  - default strict mode still raises when no valid new ticker data is available,
  - non-fatal mode returns existing data and allows the run to continue.
- Extended `src/production_market_data.py` `add_universe_tickers(...)` with `return_failed_tickers`; `scripts/production/daily_run.py` now requeues only failed pending ticker adjustments (instead of dropping them), so newly listed symbols are retried automatically in subsequent production runs until they pass history checks.
- Added pending retry helper logic in `scripts/production/daily_run.py` to preserve only failed ticker entries (with filtered exchange mappings) when clearing/applying the pending queue.
- Hardened Trading212 history-fill parsing in `scripts/production/daily_run.py`:
  - normalize negative sell fill quantities/values to positive magnitudes,
  - capture explicit `fillPrice`,
  - prefer `fillPrice` (when present) over derived `filledValue/quantity` for `exec_price`,
  - propagate fill/value currency metadata for downstream notional conversion.
- Updated broker notional currency selection in `scripts/production/daily_run.py` to prefer filled-value currency metadata when converting order notional into broker currency.
- Expanded `tests/test_daily_run_trading212_execution.py` with regressions for:
  - sell-order fill normalization and `exec_price` population,
  - `exec_price` preference for explicit fill price,
  - pending ticker retry-entry filtering.
- Hardened production reset behavior to preserve universe-monitor pass-streak data by default:
  - `src/production_db.py` `reset_production_data(...)` now takes `preserve_universe_monitor` (default `True`) and only truncates universe-monitor tables when explicitly disabled.
  - `scripts/production/api_server.py` `POST /reset-production` now accepts `preserve_universe_monitor` (default `true`), returns the applied mode, and requires stronger confirmation token (`RESET_PRODUCTION_AND_MONITOR_DATA`) for full monitor-data wipes.

## 2026-02-16
- Added a run-calendar module at `src/run_calendar.py` with deterministic date decisions (`allow`/`skip`) using:
  - optional weekend blocking,
  - US federal holiday blocking,
  - explicit per-date overrides (`skip` / `force_run`).
- Added run-calendar config flags in `src/config.py`:
  - `RUN_CALENDAR_TIMEZONE` (default `America/New_York`)
  - `RUN_CALENDAR_SKIP_WEEKENDS` (default `False`, to align with weekday-only Railway cron)
  - `RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS` (default `True`)
- Added Postgres persistence for calendar overrides in `src/production_db.py`:
  - new table `production_run_calendar_overrides`
  - helpers to upsert/list/get/delete per-date overrides.
- Wired `scripts/production/daily_run.py` to evaluate run-calendar policy immediately after DB init and exit early on blocked dates (unless `--force` is provided).
- Added Flutter-friendly API endpoints in `scripts/production/api_server.py`:
  - `GET /run-calendar/decision?date=YYYY-MM-DD`
  - `GET /run-calendar/overrides`
  - `POST /run-calendar/overrides`
  - `DELETE /run-calendar/overrides/{run_date}`
  - `GET /run-calendar/us-federal-holidays?year=YYYY`
- Added unit tests in `tests/test_run_calendar.py` for default holiday blocking, optional weekend behavior, and override precedence.

## 2026-02-10
- Added `GET /fx-rate` in `scripts/production/api_server.py` to provide USD/GBP conversion rates for app-side currency normalization.
- Wired the FX endpoint to a free external provider (`frankfurter.app`) with in-process TTL caching and a DB-derived fallback from historical broker summary ratios when the provider is unavailable.
- Added strict currency validation and clear HTTP errors for unsupported/temporarily unavailable FX pairs in the API layer.
- Replaced the temporary positions-price fallback in `scripts/production/daily_run.py` with Trading212 historical-order fill price backfill (`GET /equity/history/orders`) keyed by `order_id`, so persisted `exec_price` reflects order-level fills instead of position averages.
- Removed `/equity/orders` and `/equity/orders/{id}` polling from production order monitoring; phase execution now resolves orders via `/equity/positions` (quantity reconciliation) + `/equity/history/orders` (status/fill notional/price).
- Added explicit Trading212 monitor pacing config in `src/config.py` for endpoint limits: `TRADING212_POSITIONS_POLL_SEC` (default `1.0`), `TRADING212_HISTORY_POLL_SEC` (default `10.0`), `TRADING212_HISTORY_PAGE_LIMIT` (default `50`), and `TRADING212_HISTORY_MAX_PAGES` (default `1`).
- Extended `tests/test_daily_run_trading212_execution.py` to validate that position reconciliation still resolves fill status/quantity while `exec_price` is sourced from `history.orders.fill.price`.
- Fixed broker execution-cost calculation in `scripts/production/daily_run.py` to compute notionals from order `filledValue`/`exec_price` with explicit USD↔GBP conversion into broker-account currency before cost reconciliation.
- Added cost-safety guard in `scripts/production/daily_run.py`: when one or more filled broker orders lack usable notional/currency conversion, broker execution cost totals are skipped and a warning is persisted instead of storing inflated values.
- Added regression coverage in `tests/test_daily_run_trading212_execution.py` for broker notional conversion and incomplete-notional handling.

## 2026-02-02
- Sanitized VIX merge columns in `src/production.py` to coalesce legacy `*_x`/`*_y` fields and prevent duplicate-column merge errors during `update_market_data`.
- Dropped preexisting VIX columns from update rows before merging fresh VIX stats in `src/production.py` to avoid suffix collisions.

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

- Rewrote git history to purge the tracked `venv/` directory (removes large files blocking GitHub push).

- Rewrote README to match current project structure, scripts, and production workflow.

## 2026-01-30
- Added Postgres-backed price snapshots (`production_prices`) and helpers in `src/production_db.py`.
- Daily production runs now persist per-ticker close prices to Postgres for the run date.
- Updated `/portfolio` to price holdings from Postgres when available, with file fallback if no DB prices exist.
- Stopped ignoring `alphas/` in `.gitignore`, while still ignoring `alphas/_ensembles/` outputs for deploys.

## 2026-01-31
- Added buy-scaling guards in `src/rule_backtester.py` and `src/production.py` to prevent trades from driving cash negative, with recalculated post-trade cash weight.
- Increased `CASH_RESERVE` to 2% in `src/config.py`.
- Added `/summaries` API endpoint in `scripts/production/api_server.py` to fetch net worth history with optional date filtering and limits.
- Added `fields` query param to `/summaries` to return a reduced payload (e.g., only `date` and `net_worth_usd`).

## 2026-02-04
- Treated TradingView updates with no new bar as resolved (instead of missing) so `require_all_tickers` only fails when analysis data is absent, not just unchanged.
- Added `/stale-tickers` API endpoint to report tickers missing data for a target date.
- Resolved no-new-bar tickers before validating OHLCV so missing indicator fields don’t trigger false partial-update errors.
- Marked tickers with missing TradingView analysis as resolved to avoid blocking updates.
- Added excluded tickers support with `/exclude-tickers` and `/excluded-tickers`, plus a file-based exclusion list applied to production and backtests.
- Added a pre-commit backtest requirement to `AGENTS.md`.
- Fixed `RuleBasedBacktester` method indentation after adding excluded-ticker loading so backtests run again.
- Updated config defaults: TOP_K 10 -> 4; ROLLING_WINDOW_FOR_VOL 21 -> 14; USE_REGIME_SYSTEM True -> False; CASH_RESERVE 0.02 -> 0.0; SLIPPAGE_COEFF 0.005 -> 0.008; REGIME_DISPERSION_COL ROC_10 -> ROC_50; REGIME_TREND_BAND 0.02 -> 0.008.
- Updated CASH_RESERVE 0.0 -> 0.02.

## 2026-02-06
- Added Trading212 client helpers and config toggles (base URL, timeouts, FX rate, instrument cache, mapping file), plus `requests` dependency.
- Integrated Trading212 into daily production runs as the source-of-truth for cash/positions, with FX conversion from GBP to USD.
- Added broker discrepancies tracking, broker order logging, and side-by-side summary fields (broker cash/net worth/currency).
- Added Postgres tables and API endpoints for broker account/positions/orders snapshots.
- Added a Trading212 mapping builder script to generate the ticker map and missing list from the local universe.

## 2026-02-07
- Expanded Trading212 instrument indexing to include `shortName` aliases so renamed tickers map correctly.
- Added a Trading212 universe pre-check in the daily production run, halting if any tradable tickers lack a mapping, and stopped using broker cash/positions as the state-of-truth for signal generation.
- Removed the Trading212 mapping builder script from the repository per request.
- Added `src/market_data_validation.py` with strict parquet preflight checks (MultiIndex shape, required columns, duplicates, datetime compatibility) and wired it into backtesting/production entrypoints.
- Added `src/production_market_data.py` to modularize production market-data responsibilities (`update_market_data`, `add_universe_tickers`, indicator prep helpers), and re-exported market-data APIs from `src/production.py`.
- Added structured per-ticker market-data update diagnostics (status/reason metadata) and persisted them from `scripts/production/daily_run.py` via `--update-diagnostics-file`.
- Hardened fallback exception handling with explicit context/logging in regime HMM flows (`src/regime.py`) and narrowed overly broad exception catches in `scripts/production/api_server.py`.
- Fixed config consistency issues by removing a duplicate `_env_int` definition and aligning the default `TRADING_REGION` to `us` in `src/config.py`; removed hardcoded env defaults from `scripts/production/daily_run.py`.
- Added test baseline under `tests/`:
  - `tests/test_market_data_validation.py`
  - `tests/test_regime.py`
  - `tests/test_rule_backtester.py`
  - `tests/test_backtester_cli_smoke.py`
  - `tests/conftest.py` path bootstrap.
- Added CI workflow `.github/workflows/ci.yml` to run syntax lint (`compileall`) plus pytest (including CLI smoke backtest).
- Updated `README.md` with the new update diagnostics option and test command.
- Removed unused/deprecated LLM artifacts: deleted `src/llm_interface.py` and `GEMINI.md`.
- Cleaned local/generated clutter from the working tree: removed `__pycache__/`, `*.pyc`, `.pytest_cache`, and project-level `.DS_Store` files (except permission-restricted `.git` internals).
- Removed run artifacts under `runs/` per cleanup request.
- Updated stale guidance in `AGENTS.md` (removed references to `_candidates/` and `scripts/orchestration/orchestrator.py`; refreshed testing guidance to use the current pytest suite).
- Added `params_inventory.json` at repo root with a generated inventory of environment variables (shell, `.env`, defaults, resolved values), config constants, and CLI parameters/defaults across project entrypoints.

## 2026-02-09
- Hardened Trading212 symbol resolution in `src/trading212.py`: `resolve_t212_ticker(...)` now treats overrides as authoritative only when the override ticker exists in the current instrument index, and otherwise falls back to symbol-based matching.
- Updated Trading212 mapping calls in `scripts/production/daily_run.py` (both universe preflight validation and order routing) to pass the instrument index (`by_ticker`) into ticker resolution, preventing stale override entries from blocking mapping.
- Added `tests/test_trading212.py` with coverage for valid-override behavior and stale-override fallback behavior (including the `DAY`/`CDAY` style mismatch scenario).
- Hardened Trading212 order polling for production runs: `src/trading212.py` now raises structured `Trading212ApiError`, URL-encodes order IDs in `get_order`, and treats immediate `404 Order not found` after placement as transient during `wait_for_fill` retries.
- Added defensive order-id extraction in `scripts/production/daily_run.py` (`id`, `orderId`, `order_id`, nested `order.id`) before polling.
- Extended `tests/test_trading212.py` with order-polling regression tests covering transient `404` retry behavior and non-404 propagation.
- Increased default Trading212 fill wait window in `src/config.py` (`TRADING212_ORDER_TIMEOUT` from 60s to 300s) to reduce false failures on slow broker status transitions.
- Increased default Trading212 order poll interval in `src/config.py` (`TRADING212_ORDER_POLL_SEC` from 2s to 5s) to reduce API polling pressure.
- Improved unfilled-order failure message in `scripts/production/daily_run.py` to include actionable guidance (`TRADING212_ORDER_TIMEOUT`, market-hours scheduling, `TRADING212_EXTENDED_HOURS`).
- Updated Trading212 order execution flow in `scripts/production/daily_run.py` so non-filled/errored broker orders are captured as issues (instead of crashing), order placement halts after the first unresolved order in a run, and persisted state is synced from post-trade broker snapshot when broker issues occur to avoid model/broker drift on subsequent runs.
- Added focused execution-flow tests in `tests/test_daily_run_trading212_execution.py` with a lightweight `psycopg2` stub so these unit tests run even in environments without Postgres driver dependencies.
- Added high-visibility stage logging throughout `scripts/production/daily_run.py` with UTC timestamps and structured context (startup, DB init, data loading/updating, state load, broker context, universe validation, trade generation, order execution milestones, snapshot refresh, DB persistence, completion, and fatal exception logging with traceback) to make Railway runtime failures and stalls pinpointable.
- Reworked Trading212 execution in `scripts/production/daily_run.py` to run in two phases (`SELL` first, then `BUY`), placing each phase's orders first and then monitoring those placed IDs in bulk.
- Added bulk orders API support in `src/trading212.py` (`get_orders()` -> `GET /equity/orders`) plus `wait_for_orders(...)` for timeout/poll-driven bulk status tracking of multiple order IDs.
- Updated execution tests to validate bulk monitoring behavior and `SELL`-before-`BUY` placement ordering.
- Added GET request retry/backoff handling in `src/trading212.py` for temporary broker/API throttling (`429`) and transient server errors (`5xx`), with configurable retry limits/backoff via new config keys (`TRADING212_HTTP_MAX_RETRIES`, `TRADING212_HTTP_RETRY_BASE_SEC`, `TRADING212_HTTP_RETRY_MAX_SEC`).
- Hardened post-trade broker snapshot refresh in `scripts/production/daily_run.py` so rate-limited snapshot calls no longer crash the run; the workflow now logs snapshot failures and falls back to pre-trade snapshot data when necessary.
- Extended `tests/test_trading212.py` with retry behavior coverage (GET retry-on-429 success path and no-retry-on-POST 429 safety check).
- Improved bulk order monitoring reliability in `src/trading212.py::wait_for_orders(...)`: when `GET /equity/orders` does not include pending/filled IDs, the monitor now periodically falls back to `GET /equity/orders/{id}` for unresolved order IDs and emits periodic monitor progress logs.
- Added additional Trading212 placement logging in `scripts/production/daily_run.py` to include the immediate broker response status value (`response_status`) for each placed order.
- Added regression coverage in `tests/test_trading212.py` for the bulk-monitor fallback path when bulk-list responses omit tracked order IDs.
- Added centralized per-call Trading212 HTTP logging in `src/trading212.py::_request(...)` with explicit request start/success/error lines including method, URL, status, latency, retryability, and compact payload/error previews so Railway logs show the result of every broker API call.
- Adjusted Trading212 retry behavior in `src/trading212.py::_request(...)` to ignore zero-second `Retry-After` values and use exponential backoff instead, reducing immediate retry storms under broker rate limiting.
- Tuned `src/trading212.py::wait_for_orders(...)` to reduce fallback pressure (`GET /equity/orders/{id}` less frequently and in small batches) and treat fallback `429` as transient rather than failing the whole monitor cycle.
- Extended `tests/test_trading212.py` with coverage for fallback `429` handling in bulk monitoring and positive backoff behavior when `Retry-After: 0` is returned.
- Improved terminal resolution in `src/trading212.py::wait_for_orders(...)` for broker-side “seen then disappears” behavior: if an order was previously observed and repeated fallback lookups return `404`, it is now marked as terminal `UNKNOWN` (with explicit resolution metadata) instead of staying unresolved until timeout.
- Added regression coverage in `tests/test_trading212.py` for the “seen in bulk list, then missing + fallback 404” resolution path.
- Further hardened `src/trading212.py::wait_for_orders(...)` with fair round-robin fallback checks (avoids repeatedly probing only the same unresolved IDs), faster fallback cadence, and terminal `UNKNOWN` resolution for repeated fallback `404` responses even when an order ID was never seen in bulk listings.
- Added Trading212 unresolved-order reconciliation in `scripts/production/daily_run.py`: when bulk order monitoring does not confirm fill, the run now queries `GET /equity/positions?ticker=...` and marks orders as `FILLED` if position quantity matches expected post-trade quantity, with explicit reconciliation logs/metadata.
- Extended `tests/test_daily_run_trading212_execution.py` with a regression test that verifies unresolved `NEW` order status can be correctly reconciled to `FILLED` via the positions endpoint.

## 2026-02-08
- Added a standalone Technology candidate monitor workflow (`scripts/backtesting/tech_universe_monitor.py`) that scans a broad TradingView catalog, applies TradingView prechecks + sector filtering + existing universe quality gates, tracks pass streaks across runs, and outputs manual review/potential-addition lists without mutating production universe files.
- Added reusable monitor helpers in `src/universe_monitor.py` (catalog parsing, technology-sector matching, streak-state updates) with unit tests in `tests/test_universe_monitor.py`.
- Updated the monitor pipeline to apply low-cost TradingView prechecks first, then fetch yfinance metadata only for survivors and enforce market-cap gating there, reducing metadata calls while preserving manual-review outputs.
- Fixed a runtime bug in `scripts/backtesting/tech_universe_monitor.py` where quality-history fetches referenced `yf` without importing `yfinance`, which caused `history_error:NameError` for all quality checks.
- Added DB persistence for universe-monitor snapshots in `src/production_db.py` (state + candidates tables with overwrite semantics) and exposed monitor APIs in `scripts/production/api_server.py` (`/universe-monitor/summary`, `/universe-monitor/candidates`, `/universe-monitor/potential`) for app consumption.
- Updated `scripts/backtesting/tech_universe_monitor.py` to write monitor results to Postgres by default when DB is configured, plus `--skip-file-output`/`--skip-db` controls.
- Added DB-backed universe mapping (`production_universe_map`) and wired the cron runner (`scripts/production/daily_run.py`) to refresh it from the exchange-map file each run.
- Updated API stale/universe endpoints to use DB-first data (`/stale-tickers` now reads latest DB price snapshots + DB universe map, with file/parquet fallback), which fixes split-service deployments where the API service has no shared parquet volume.
- Added DB-backed excluded tickers (`production_excluded_tickers`) and updated API endpoints to read/write exclusions from Postgres by default, mirroring to file for compatibility.
- Updated `scripts/production/daily_run.py` to sync DB exclusions to the local excluded-tickers file at run start so universe filtering stays consistent in split-service deployments.
- Added broker execution-cost reconciliation in `scripts/production/daily_run.py` using broker cash delta + buy/sell notional + external cash flow adjustments, persisted as daily and cumulative metrics.
- Extended `production_runs` schema and DB summaries (`src/production_db.py`) with broker cost fields (`broker_execution_cost`, `broker_total_execution_cost`, USD counterparts, and reconciliation components) so app/API can compare broker-realized costs vs model costs over time.
- Removed file-system fallback paths from production workflows so DB is the single source of truth: `scripts/production/daily_run.py` now requires DB config and reads/writes state/pending data only via Postgres.
- Converted `scripts/production/api_server.py` to DB-only behavior across run/summary/trade/portfolio/universe/monitor/adjustment endpoints and made DB configuration mandatory at startup.
- Updated `scripts/production/queue_adjustments.py` to require DB configuration and queue adjustments only in Postgres.
- Updated `README.md` production notes to reflect DB-required behavior for production run/API paths.
- Removed remaining local artifact/support-file writes from `scripts/production/daily_run.py` (no `runs/production` outputs, no cash-log/universe-registry/exchange-map/excluded sync files); run outputs now persist to DB only.
- Added DB-native universe filtering/exclusion path in `scripts/production/daily_run.py` and `src/production.py` (`generate_trades_for_date(..., excluded_tickers=...)`) to avoid file-based excluded ticker reads in production runs.
- Extended `src/production_market_data.py::update_market_data` with optional in-memory `exchange_map` input so daily runs can use DB universe mapping without reading a mapping file.
- Fixed a payload corruption bug in `scripts/production/queue_adjustments.py` where `--add-tickers-exchange` reused the `entries` accumulator, which could enqueue non-dict pending adjustment rows.
- Changed DB connection handling in `src/production_db.py` to explicit transaction commit/rollback (instead of autocommit) so multi-step replacement writes are atomic.
- Narrowed `reset_production_data()` in `src/production_db.py` to clear run/state/trade/price/pending/broker runtime tables only, preserving universe map, excluded tickers, and universe monitor tables.

## 2026-02-14
- Optimized trade generation in `src/production.py` to precompute scores only for active strategies selected for the current regime, avoiding unnecessary scoring work for inactive strategies.
- Optimized price snapshot building in `scripts/production/daily_run.py` by replacing row-wise iteration (`iterrows`) with a vectorized ticker/close extraction path.
- Optimized `src/production_market_data.py::update_market_data` by precomputing per-ticker last local dates once and only slicing ticker history when a new bar actually exists.
- Optimized VIX beta enrichment in `src/production_market_data.py` with per-ticker log-return caching and a reusable `_compute_vix_beta_from_series(...)` helper to cut repeated `xs(...)` work.
- Removed duplicated cross-sectional z-score passes in `src/production_market_data.py::_add_swing_features`, eliminating redundant groupby transforms during feature preparation.
- Fixed a compatibility regression in `scripts/production/daily_run.py::_build_price_snapshot` by replacing `Index.ne("")` with `(index != "")`, preserving behavior on the project pandas version.
- Restored prior `generate_trades_for_date(...)` behavior in `src/production.py` for custom selector edge-cases (`strategy_selector` returning an empty list) by removing the new hard failure so results remain baseline-compatible.
- Added broker-aware signal-state support in `scripts/production/daily_run.py`: when Trading212 integration is enabled and `TRADING212_USE_BROKER_STATE_FOR_SIGNALS=true` (default), trade generation now uses broker-derived USD cash and broker positions as the input `ProductionState`.
- Added `TRADING212_USE_BROKER_STATE_FOR_SIGNALS` config flag in `src/config.py` to allow disabling broker-state-driven signal generation without disabling broker execution.
- Hardened Trading212 account-currency handling in `scripts/production/daily_run.py::_build_trading212_context(...)` by supporting USD accounts directly, preserving USD conversion for discrepancy checks, and validating unsupported currencies explicitly.
- Added `_state_from_broker_context(...)` adapter in `scripts/production/daily_run.py` to keep broker-awareness in the production/execution layer while leaving strategy scoring logic unchanged.
- Added unit coverage in `tests/test_daily_run_trading212_execution.py` for broker-state adapter behavior (broker cash/positions adoption and fallback behavior).
- Updated broker-aware signal input in `scripts/production/daily_run.py::_state_from_broker_context(...)` to clear `prev_weights`, preventing stale local-weight smoothing bias when broker state is the source of truth.
- Added broker position-integrity monitoring in `scripts/production/daily_run.py::_state_from_broker_context(...)` that logs `broker_fractional_positions_detected` when non-integer holdings are observed.
- Added `_state_from_broker_snapshot(...)` and updated `scripts/production/daily_run.py` to persist broker-synced cash/positions after every non-dry-run Trading212 session (not only when order issues occur), reducing model/broker drift.
- Extended `tests/test_daily_run_trading212_execution.py` with coverage for fractional-position alert logging and broker-snapshot state sync behavior.
- Updated broker-aware smoothing behavior in `scripts/production/daily_run.py`: `_state_from_broker_context(...)` now derives `prev_weights` from broker positions + latest price snapshot so turnover smoothing remains active while anchored to actual holdings.
- Added broker-prev-weight diagnostics (`broker_prev_weights_computed`, `broker_prev_weights_missing_prices`, `broker_prev_weights_skipped`) and wired price lookup from the target-date snapshot into broker-state preparation.
- Extended `tests/test_daily_run_trading212_execution.py` with coverage for broker-derived `prev_weights` computation.
