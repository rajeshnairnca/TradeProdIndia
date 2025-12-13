# Repository Guidelines

## Project Structure & Modules
- Core Python code lives in `src/` (environments, models, utils, config). Scripts under `scripts/` orchestrate workflows: `training/` (train new alphas), `backtesting/` (ensemble evaluation), `orchestration/` (LLM-driven generation), and `tuning/` (hyperparameter search).  
- Trained candidates stage in `_candidates/`; promoted alphas sit in `alphas/` with their models, features, and results.  
- Data inputs are under `data/`; logs and run artifacts accumulate in `logs/` and `runs/`. Keep large artifacts out of git.

## Build, Run, and Key Commands
- Activate the project venv before running anything.  
- Long-running commands should be prefixed with `caffeinate -i` to avoid sleep. Common entry points:  
  - `caffeinate -i python3 scripts/training/train_alpha.py --alpha-name alpha_base --timesteps 350000` — train a single alpha.  
  - `caffeinate -i python3 scripts/backtesting/backtester.py --alphas alpha_base another_alpha` — backtest single or multiple alphas.  
  - `caffeinate -i python3 scripts/orchestration/orchestrator.py` — full generate-train-validate-promote loop.  
  - `caffeinate -i python3 src/walkforward.py --alpha-name <candidate>` — walk-forward validation.  
- Keep `.env` for secrets/keys; do not commit it.

## Coding Style & Naming
- Python 3.11, 4-space indents, prefer type hints and explicit imports.  
- Follow existing patterns in `src/`: small, testable helpers; configuration from `src/config.py`; avoid hard-coded paths.  
- Names: snake_case for files/functions/variables; PascalCase for classes; `alpha_*` for strategies in `alphas/`.

## Testing Guidelines
- No dedicated test harness is present; when adding features, include lightweight checks (e.g., sanity asserts, small deterministic runs) or a minimal `pytest` under `tests/` if introducing logic-heavy changes.  
- For data-dependent code, stub or downsample inputs to keep runs fast.  
- Validate training/backtesting changes with short timesteps before long jobs.

## Commit & Pull Request Practices
- Use concise, imperative commits (e.g., `Add walkforward gating`); group related changes per commit.  
- PRs should describe the goal, key changes, and how you validated (commands run, datasets used, any shortcuts). Include links to logs/artifacts in `runs/` or `logs/` when relevant.  
- Note any deviations from default workflows (custom hyperparams, truncated datasets) so others can reproduce.
