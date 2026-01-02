# Autonomous Trading Ensemble System (Deterministic, Rule-Based)

## Project Overview

This project is a deterministic, rule-based trading system that runs an ensemble of strategies and adapts which ones are active based on market regime (trend/volatility/breadth/dispersion). A large language model (LLM) can propose new rule-based strategies, which are then backtested and promoted if they improve performance.

Key technologies:

* **Python**
* **Pandas & NumPy** for data manipulation
* **Matplotlib** for plotting
* **Google Generative AI** for LLM strategy generation

## Project Structure

```
/
├─── _candidates/      # Staging area for LLM-proposed strategies.
├─── alphas/           # Home for promoted strategies (each has strategy.py + description.txt).
│    └─── _ensembles/  # Stores backtest results for strategy combinations.
├─── data/             # Raw data files (daily prices, indicators).
├─── scripts/          # Executable scripts (backtesting, orchestration).
├─── src/              # Core strategy + backtesting logic.
├─── README.md         # This file.
└─── requirements.txt
```

## Strategy Format

Each strategy lives under `alphas/<strategy_name>/strategy.py` and must define:

* `DESCRIPTION`: one-line description
* `REGIME_TAGS`: list of regimes (bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol)
* `generate_scores(df: pd.DataFrame) -> pd.Series`: returns a score per row (same index as `df`)

## Ensemble Logic

On each day:

1. Determine market regime from `src/regime.py`.
2. Select strategies whose `REGIME_TAGS` match.
3. Z-score each strategy’s cross-sectional signal and average them.
4. Build a top‑K portfolio with optional volatility parity and regime-based sizing.

## Usage

### 1. Install Dependencies

```bash
/path/to/your/venv/bin/pip install -r requirements.txt
```

### 2. Backtest Strategies

```bash
# Backtest a single strategy
caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following

# Backtest multiple strategies as an ensemble
caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following rule_mean_reversion rule_volatility_breakout
```

By default, the backtester filters the universe to the Nasdaq 100 list. To disable the filter:

```bash
UNIVERSE_FILTER=all caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following
```

### 2b. Sweep Strategy Combinations

```bash
# Find the best combo among all strategies
caffeinate -i python3 scripts/backtesting/strategy_sweep.py

# Limit combo sizes and samples
caffeinate -i python3 scripts/backtesting/strategy_sweep.py --min-size 2 --max-size 4 --max-combos 100
```

### 3. Walk-Forward Validation

```bash
caffeinate -i python3 src/walkforward.py --alpha-name rule_trend_following
```

### 4. Orchestrate LLM Strategy Generation

```bash
caffeinate -i python3 scripts/orchestration/orchestrator.py
```

The orchestrator will:

1. Summarize current strategies.
2. Ask the LLM for a new rule-based strategy.
3. Backtest the candidate with the ensemble.
4. Promote the candidate if it improves results.

## Command Reference (Scripts + Flags)

Use `caffeinate -i` for long runs (examples below use it where helpful).

### Backtester

Command: `python3 scripts/backtesting/backtester.py`

Flags:
- `--strategies`: list of strategy names to include (default: all in `alphas/`).
- `--strategy-roots`: repeatable; root dirs to load strategies from (default: `alphas`).
- `--output-root`: root directory for outputs (default: `alphas`).
- `--use-full-history`: ignore holdout split and backtest full history.
- `--start-date`: YYYY-MM-DD inclusive start date.
- `--end-date`: YYYY-MM-DD exclusive end date.

Examples:
```bash
caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following
```

### Strategy Sweep

Command: `python3 scripts/backtesting/strategy_sweep.py`

Flags:
- `--strategies`: list of strategy names to consider (default: all in `alphas/`).
- `--strategy-roots`: repeatable; root dirs to load strategies from.
- `--min-size`: minimum strategies per combo (default: 1).
- `--max-size`: maximum strategies per combo (default: all).
- `--max-combos`: cap combinations evaluated.
- `--seed`: RNG seed for sampling combos (default: 42).
- `--metric`: metric to optimize: `cagr`, `sharpe`, `max_drawdown`.
- `--start-date`: YYYY-MM-DD inclusive start.
- `--end-date`: YYYY-MM-DD exclusive end.
- `--output-root`: output root (default: `alphas`).
- `--jobs`: parallel workers (1 = serial).

Example:
```bash
caffeinate -i python3 scripts/backtesting/strategy_sweep.py --min-size 2 --max-size 4 --max-combos 200 --jobs 7
```

### Walk-Forward Validation

Command: `python3 src/walkforward.py`

Flags:
- `--alpha-name`: single strategy name.
- `--strategies`: list of strategy names to validate together.
- `--strategy-roots`: repeatable; root dirs to load strategies from.

Example:
```bash
caffeinate -i python3 src/walkforward.py --alpha-name rule_trend_following
```

### LLM Orchestration

Command: `python3 scripts/orchestration/orchestrator.py`

Flags:
- `--llm-guidance`: extra prompt guidance passed to the LLM.
- `--sweep-baseline`: run a combo sweep to pick the baseline ensemble.
- `--sweep-min-size`: minimum strategies in sweep combos.
- `--sweep-max-size`: maximum strategies in sweep combos.
- `--sweep-max-combos`: max combos to evaluate per sweep.
- `--sweep-metric`: metric for sweep (`cagr`, `sharpe`, `max_drawdown`).

### ML Regime Strategy Selector

Command: `python3 scripts/ml_regime/ml_strategy_selector.py`

Flags:
- `--strategies`: list of strategy names to include (default: all in `alphas/`).
- `--strategy-roots`: repeatable; root dirs to load strategies from.
- `--start-date`: YYYY-MM-DD inclusive start.
- `--end-date`: YYYY-MM-DD exclusive end.
- `--train-ratio`: train/test split (default: `config.TRAIN_RATIO`).
- `--confirm-days`: consecutive days before a regime switch is confirmed (default: 10).
- `--output-root`: output root (default: `runs`).
- `--jobs`: CPU threads for XGBoost training.
- `--mapping-search`: exhaustive search: one strategy per regime, pick best CAGR.
- `--mapping-allow-reuse`: allow strategies to repeat across regimes during mapping.
- `--mapping-max`: cap number of mappings evaluated.
- `--mapping-jobs`: parallel workers for mapping search.
- `--eval-full-period`: run the hybrid backtest (and mapping eval) over the full filtered period instead of test split.
- `--skip-ml`: bypass XGBoost training and use regime labels directly (optionally confirm-smoothed); skips train/test split.

Example (HMM rolling, 7 regimes, mapping search):
```bash
REGIME_MODE=hmm_rolling HMM_N_COMPONENTS=7 HMM_STATE_LABELS=1 TRADING_REGION=us \
caffeinate -i python3 scripts/ml_regime/ml_strategy_selector.py --mapping-search --mapping-jobs 7
```

Outputs:
- Per-run results under `runs/ml_regime/<timestamp>/`
- Rolling summary appended to `runs/ml_regime/summary.csv`

### Regime Visualization

Command: `python3 scripts/visualize_regimes.py`

Flags:
- `--region`: `us` or `india` (defaults to `TRADING_REGION`).
- `--data-file`: override parquet path.
- `--output`: output file for 4‑state comparison (default: `regime_comparison_4states.png`).
- `--output-states`: output file for raw HMM state plot.
- `--skip-hmm`: skip full-history HMM panel.
- `--skip-hmm-rolling`: skip rolling HMM panel.
- `--plot-hmm-states`: render HMM states (uses `hmm_state`).

Example:
```bash
TRADING_REGION=us HMM_N_COMPONENTS=7 caffeinate -i \
python3 scripts/visualize_regimes.py --plot-hmm-states --output my_regime_plot.png
```

### Data Extraction (yfinance)

Command: `python3 "scripts/data_extraction/data_extract_yfinance - days - v6.py"`

Flags:
- `--stocks-file`: newline-delimited ticker list (default: built‑in list).
- `--output-file`: output parquet (default: `data/daily_data.parquet`).
- `--period`: yfinance period (default: `20y`).
- `--interval`: yfinance interval (default: `1d`).
- `--min-trading-days`: minimum rows per ticker (default: 50).
- `--rolling-window`: rolling window for ADV/vol/VIX z (default: 21).
- `--vix-ticker`: VIX ticker (default: `^VIX`).

### XGBoost Alpha (Train)

Command: `python3 scripts/xgboost/train_xgb_alpha.py`

Flags:
- `--alpha-name`: name for saved XGB alpha (required).
- `--features-file`: optional feature engineering Python file.
- `--description`: description stored in results.
- `--walkforward-folds`: number of walk-forward folds (0 = skip).

### XGBoost Alpha (Backtest)

Command: `python3 scripts/xgboost/backtest_xgb_alpha.py`

Flags:
- `--alpha-name`: base alpha name (required; without `_xgb`).
- `--features-file`: optional feature engineering Python file.
- `--top-k`: number of tickers to allocate each day.
- `--gross-cap`: gross exposure cap (1.0 = fully invested).
- `--max-weight`: max per‑name weight.
- `--cost-bps`: round‑trip cost in bps.
- `--sector-cap`: max sector weight as fraction of gross.
- `--target-vol`: annualized target vol (0 to disable).
- `--vol-window`: lookback window (days) for realized vol.
- `--min-gross`: minimum gross when vol targeting is on.
- `--max-gross`: maximum gross when vol targeting is on.

### Hyperparameter Tuning

Command: `python3 scripts/tuning/tune_hyperparameters.py`

Flags:
- none (runs with internal Optuna settings).

### Analysis Helper (Regime/Strategy Fit)

Command: `python3 scripts/analysis/regime_strategy_fit.py`

Flags:
- none.

## Environment Flags (Global)

These are read from the environment via `src/config.py`:
- `TRADING_REGION`: `us` or `india` (drives default data file and output routing).
- `DATA_FILE`: override data parquet path.
- `UNIVERSE_FILTER`: `all`, `none`, `nasdaq100`, or comma‑separated tickers.
- `REGIME_MODE`: `heuristic`, `hmm`, or `hmm_rolling`.
- `HMM_N_COMPONENTS`: number of HMM states (regimes).
- `HMM_WARMUP_PERIOD`: initial HMM training window (days).
- `HMM_STEP_SIZE`: HMM rolling prediction block size (days).
- `HMM_STATE_LABELS`: `1/true` to use `hmm_state_*` labels instead of 4‑state bull/bear.
- `HMM_COVARIANCE_TYPE`: `full` or `diag` (HMM fit stability).
- `HMM_MIN_COVAR`: minimum covariance floor for HMM.
- `USE_REGIME_SYSTEM`: enable/disable regime overlays.
- `BACKTEST_USE_FULL_HISTORY`: default to full history in backtests.
