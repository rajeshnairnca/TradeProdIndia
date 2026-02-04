# Repository Guidelines

## Project Structure & Modules
- Core Python code lives in `src/` (strategies, backtester, regime, utils, config). Scripts under `scripts/` orchestrate workflows: `backtesting/` (ensemble evaluation) and `orchestration/` (LLM-driven generation).  
- Candidate strategies stage in `_candidates/`; promoted alphas sit in `alphas/` with their `strategy.py`, descriptions, and results.  
- Data inputs are under `data/`; logs and run artifacts accumulate in `logs/` and `runs/`. Keep large artifacts out of git.

## Build, Run, and Key Commands
- Activate the project venv before running anything.  
- Long-running commands should be prefixed with `caffeinate -i` to avoid sleep. Common entry points:  
  - `caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following rule_mean_reversion` — backtest single or multiple strategies.  
  - `caffeinate -i python3 scripts/backtesting/strategy_sweep.py` — try strategy combinations and report the best result.  
  - `caffeinate -i python3 scripts/orchestration/orchestrator.py` — full generate-backtest-promote loop.  
  - `caffeinate -i python3 src/walkforward.py --alpha-name <strategy>` — walk-forward validation.  
- Keep `.env` for secrets/keys; do not commit it.

## Coding Style & Naming
- Python 3.11, 4-space indents, prefer type hints and explicit imports.  
- Follow existing patterns in `src/`: small, testable helpers; configuration from `src/config.py`; avoid hard-coded paths.  
- Names: snake_case for files/functions/variables; PascalCase for classes; `rule_*` or descriptive strategy names in `alphas/`.

## Testing Guidelines
- No dedicated test harness is present; when adding features, include lightweight checks (e.g., sanity asserts, small deterministic runs) or a minimal `pytest` under `tests/` if introducing logic-heavy changes.  
- For data-dependent code, stub or downsample inputs to keep runs fast.  
- Validate backtesting changes with short date ranges before long runs.

## Commit & Pull Request Practices
- Use concise, imperative commits (e.g., `Add walkforward gating`); group related changes per commit.  
- Before any commit, run: `TRADING_REGION=us DATA_FILE=data/daily_data_us.parquet REGIME_MODE=heuristic caffeinate -i python3 scripts/backtesting/sector_experiments.py --sectors Technology --start-date 2010-01-01 --end-date 2025-01-01 --regime-scope global --regime-mapping '{"bear_high_vol":"rule_mean_reversion","bear_low_vol":"rule_low_vol_defensive","bull_high_vol":"rule_quality_min_vol","bull_low_vol":"rule_momentum_acceleration","sideways_high_vol":"rule_range_reversion","sideways_low_vol":"rule_trend_strength"}'`.  
- PRs should describe the goal, key changes, and how you validated (commands run, datasets used, any shortcuts). Include links to logs/artifacts in `runs/` or `logs/` when relevant.  
- Note any deviations from default workflows (custom hyperparams, truncated datasets) so others can reproduce.

## Change Log
- Log every code change in `AGENT_CHANGES.md` (agent_changes.md) with a brief note on what changed and why.
