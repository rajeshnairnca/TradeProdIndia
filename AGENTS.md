# Repository Guidelines

## Project Structure & Modules
- Core Python code lives in `src/` (strategies, backtester, regime, utils, config). Scripts under `scripts/` orchestrate workflows: `backtesting/`, `production/`, and `data_extraction/`.  
- Promoted alphas sit in `alphas/` with their `strategy.py`, descriptions, and results.  
- Data inputs are under `data/`; logs and run artifacts accumulate in `logs/` and `runs/`. Keep large artifacts out of git.

## Build, Run, and Key Commands
- Activate the project venv before running anything.  
- Long-running commands should be prefixed with `caffeinate -i` to avoid sleep. Common entry points:  
  - `caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following rule_mean_reversion` — backtest single or multiple strategies.  
  - `caffeinate -i python3 scripts/backtesting/strategy_sweep.py` — try strategy combinations and report the best result.  
  - `caffeinate -i python3 src/walkforward.py --alpha-name <strategy>` — walk-forward validation.  
- Keep `.env` for secrets/keys; do not commit it.

## Coding Style & Naming
- Python 3.11, 4-space indents, prefer type hints and explicit imports.  
- Follow existing patterns in `src/`: small, testable helpers; configuration from `src/config.py`; avoid hard-coded paths.  
- Names: snake_case for files/functions/variables; PascalCase for classes; `rule_*` or descriptive strategy names in `alphas/`.

## Testing Guidelines
- Use the existing `pytest` suite under `tests/` for fast validation; add deterministic unit/smoke tests for logic-heavy changes.  
- For data-dependent code, stub or downsample inputs to keep runs fast.  
- Validate backtesting changes with short date ranges before long runs.

## Commit & Pull Request Practices
- Use concise, imperative commits (e.g., `Add walkforward gating`); group related changes per commit.  
- Before any commit, run: `TRADING_REGION=us DATA_FILE=data/daily_data_us.parquet REGIME_MODE=heuristic caffeinate -i python3 scripts/backtesting/sector_experiments.py --sectors Technology --start-date 2010-01-01 --end-date 2025-01-01 --regime-scope global --regime-mapping '{"bear_high_vol":"rule_mean_reversion","bear_low_vol":"rule_low_vol_defensive","bull_high_vol":"rule_quality_min_vol","bull_low_vol":"rule_momentum_acceleration","sideways_high_vol":"rule_range_reversion","sideways_low_vol":"rule_trend_strength"}'`.  
- PRs should describe the goal, key changes, and how you validated (commands run, datasets used, any shortcuts). Include links to logs/artifacts in `runs/` or `logs/` when relevant.  
- Note any deviations from default workflows (custom hyperparams, truncated datasets) so others can reproduce.

## Change Log
- Log every code change in `AGENT_CHANGES.md` (agent_changes.md) with a brief note on what changed and why.

## Latest Backtest Snapshot (2026-03-08)
- Baseline run (India mapping set, `WEIGHT_SMOOTHING=0.85`, `rebalance-every=28`, `2013-01-01` to `2026-01-01`):
  - `CAGR 53.9592%`, `Sharpe 1.5702`, `Max Drawdown -66.5310%`, `Final Net Worth 242,696,010.50`.
- Improved run (same strategies/mapping/date range, `WEIGHT_SMOOTHING=0.00`, `rebalance-every=14`):
  - `CAGR 59.5616%`, `Sharpe 1.5978`, `Max Drawdown -60.8747%`, `Final Net Worth 382,533,520.47`.
