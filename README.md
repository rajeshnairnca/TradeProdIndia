# Autonomous Trading Ensemble System

## Project Overview

This project is an autonomous trading system that uses reinforcement learning (RL) to discover and manage a **portfolio of diverse trading strategies (alphas)**. The system is designed to be self-improving, using a large language model (LLM) to iteratively propose new alphas, train them, and add them to the ensemble if they improve the portfolio's overall risk-adjusted performance.

The main technologies used are:

*   **Python:** The primary programming language.
*   **Stable Baselines3 & Gymnasium:** For building and training the individual reinforcement learning alphas.
*   **PyTorch:** As the underlying deep learning framework for the RL agents.
*   **Pandas & NumPy:** For data manipulation and numerical operations.
*   **Google Generative AI:** To use a Gemini model for inventing new alpha strategies.

## Project Structure

```
/
├─── _candidates/      # Staging area for newly generated alphas during training.
├─── alphas/           # Home for all successful, promoted alphas.
│    ├─── alpha_base/  # An example of a single alpha's directory.
│    └─── _ensembles/  # Stores the backtest results of alpha combinations.
├─── data/             # Raw data files (e.g., daily prices, sector mappings).
├─── scripts/          # All executable scripts for running the system.
│    ├─── backtesting/
│    ├─── orchestration/
│    └─── training/
├─── src/              # Core source code for the trading system.
├─── README.md         # This file.
└─── requirements.txt
```

## Core Concepts

### Ensemble Strategy: Weighted Average

The system's final trading signal is not from a single model, but from a combination of all alphas in the `alphas/` directory. The strategy is a **performance-weighted average**:

1.  Each individual alpha has a historical performance metric (its Compound Annual Growth Rate, or CAGR) which is calculated during its initial backtest.
2.  When the ensemble is backtested, each alpha is assigned a "weight" proportional to its CAGR. Alphas with a higher CAGR have more influence on the final trading decision.
3.  This allows the system to favor its most successful strategies while still benefiting from the diversification of having many alphas.

### Alpha Lifecycle & Validation

The system uses a two-stage process to prevent bad alphas from entering the portfolio and to ensure only robust strategies are promoted:

1.  **Generation:** The orchestrator asks the LLM to generate a new, creative alpha strategy based on the performance of the current ensemble.
2.  **Training & Screening:** The new alpha is created in the `_candidates/` directory and undergoes initial training. After training, it is subjected to a quick backtest on unseen data. This acts as a fast, efficient filter to discard obviously flawed strategies early.
3.  **Walk-Forward Validation:** If the candidate passes the initial screening, it undergoes a much more rigorous walk-forward validation. This method involves training and testing on multiple, rolling windows of time, which is a more realistic simulation of real-world performance.
4.  **Promotion:** If the candidate demonstrates a strong positive performance in the walk-forward validation, it is permanently promoted from the `_candidates/` directory to the `alphas/` directory, officially joining the ensemble.

## Usage

### 1. Setting up the Environment

First, install the required Python packages from your virtual environment.

```bash
# Ensure your virtual environment is active
/path/to/your/venv/bin/pip install -r requirements.txt
```

### 2. Training a Single Alpha (Optional)

You can train or retrain a single alpha strategy using the `train_alpha.py` script. This is useful for testing a new idea or for retraining an existing alpha with more data or for more timesteps.

```bash
# Example of retraining the base alpha
caffeinate -i python3 scripts/training/train_alpha.py --alpha-name alpha_base --timesteps 350000 --description "Base alpha strategy."


### 3. Backtesting an Ensemble

You can evaluate the performance of any combination of trained alphas using the `backtester.py` script. This script now uses the CAGR-weighted average method to combine signals.

```bash
# Backtest a single alpha
caffeinate -i python3 scripts/backtesting/backtester.py --alphas my_first_alpha

# Backtest an ensemble of multiple alphas
caffeinate -i python3 scripts/backtesting/backtester.py --alphas alpha_base candidate_1756738092 candidate_1756780384 candidate_1756818712
```


### 3. Walkforward analysis for an alpha
caffeinate -i python3 src/walkforward.py --alpha-name candidate_1756780384


### 4. Evolving the Ensemble with the Orchestrator

The main entry point of the project is the `orchestrator.py` script. This script manages the end-to-end process of automatically discovering, validating, and promoting new alphas to the ensemble.

To start the orchestration process, run:

```bash
caffeinate -i python3 scripts/orchestration/orchestrator.py
```
