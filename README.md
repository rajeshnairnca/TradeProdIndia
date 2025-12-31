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
