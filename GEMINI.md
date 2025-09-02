# Autonomous Trading Ensemble System

## Project Overview

This project is an autonomous trading system that uses reinforcement learning (RL) to discover and manage a **portfolio of diverse trading strategies (alphas)**. The system is designed to be self-improving, using a large language model (LLM) to iteratively propose new alphas, train them, and add them to the ensemble if they improve the portfolio's overall risk-adjusted performance.

The core of the project is an **ensemble of RL agents**. Instead of relying on a single model, the system combines the signals from many different specialist agents to make more robust trading decisions. The final trading signal is a **performance-weighted average** of the signals from all alphas, giving more influence to historically successful strategies.

The main technologies used are:

*   **Python:** The primary programming language.
*   **Stable Baselines3 & Gymnasium:** For building and training the individual reinforcement learning alphas.
*   **PyTorch:** As the underlying deep learning framework for the RL agents.
*   **Pandas & NumPy:** For data manipulation and numerical operations.
*   **Google Generative AI:** To use a Gemini model for inventing new alpha strategies.

## Building and Running

The system has been refactored into a modular, multi-alpha architecture. Here is the primary workflow:

### 1. Setting up the Environment

First, install the required Python packages from the virtual environment.

```bash
# Ensure your virtual environment is active
/path/to/your/venv/bin/pip install -r requirements.txt
```

### 2. Training a Single Alpha (Optional)

You can train a single, individual alpha strategy using the `train_alpha.py` script. This is useful for testing a new idea before letting the orchestrator manage it.

```bash
python train_alpha.py --alpha-name "my_first_alpha" --description "A simple momentum strategy."
```

This will create a new directory under `alphas/my_first_alpha/` containing the trained model, logs, and a backtest performance chart.

### 3. Backtesting an Ensemble

You can evaluate the performance of any combination of trained alphas using the `backtester.py` script. This script combines the alphas' signals using a weighted average based on their individual historical CAGR.

```bash
# Backtest an ensemble of multiple alphas
python backtester.py --alphas my_first_alpha another_alpha
```

This will output the CAGR and save a performance chart for that specific ensemble.

### 4. Evolving the Ensemble with the Orchestrator

The main entry point of the project is the `orchestrator.py` script. This script manages the end-to-end process of automatically evolving the alpha ensemble.

To start the orchestration process, run:

```bash
python orchestrator.py
```

The orchestrator follows a two-stage validation process:
1.  **Generation & Screening:** It asks the LLM to invent a new alpha strategy and trains it. A quick initial backtest is performed. If the alpha is clearly not viable, it is discarded immediately.
2.  **Walk-Forward Validation:** If the alpha passes the initial screen, it is subjected to a more rigorous walk-forward validation to test its robustness over time.
3.  **Promotion:** If the walk-forward performance is strong, the candidate is permanently added to the portfolio in the `alphas` directory. Otherwise, it is discarded.

## Development Conventions

*   **Code Style:** The project follows the PEP 8 style guide for Python code.
*   **Modularity:** The code is organized into several modules, each with a specific responsibility:
    *   `config.py`: Centralized configuration for all hyperparameters and settings.
    *   `trading_environment.py`: Contains the core `gymnasium` environment for market simulation.
    *   `models.py`: Defines the neural network architectures for the RL agents.
    *   `portfolio.py`: Contains logic for portfolio construction (currently unused as agents learn weights directly, but available for future strategies).
    *   `train_alpha.py`: A script to train a single alpha strategy and perform an initial screening backtest.
    *   `backtester.py`: A script to evaluate the performance of an ensemble of alphas using a performance-weighted average.
    *   `orchestrator.py`: The main orchestration logic that manages the two-stage validation and evolution of the ensemble.
    *   `llm_interface.py`: The interface to the Gemini model.
*   **Data:** The financial data used for training and backtesting is stored in the `data/` directory.
*   **Alphas:** All trained and successful alphas are stored as subdirectories in the `alphas/` directory. The `_candidates` directory is a temporary staging area for alphas undergoing training and validation.