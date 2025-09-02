# Description: Centralized configuration file for the autonomous trading project.

import torch

# ---- Data Configuration ----
DATA_FILE = "data/daily_data.parquet"
SECTOR_MAP_FILE = 'data/symbol_to_sector_map.json'
NUM_SECTORS = 11

# ---- Environment & Portfolio Configuration ----
INITIAL_CAPITAL = 1_000_000.0
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TOP_K = 7
ADV_LOOKBACK = 21
ROLLING_WINDOW_FOR_VOL = 21
USE_VOL_PARITY = True

# ---- Penalties & Costs ----
# From empirical analysis of Indian market brokerage charges
TURNOVER_PENALTY = 0.000718
LEVERAGE_PENALTY = 1e-4
RISK_PENALTY_COEFF = 0.5021
SLIPPAGE_COEFF = 0.005

# ---- Model Architecture ----
FEATURES_DIM = 256
EMBED_DIM = 32
SECTOR_EMBED_DIM = 8
LSTM_N_STACK = 8 # Number of historical steps for the recurrent model's memory

# ---- Training Hyperparameters ----
TRAINING_TIMESTEPS = 350000
N_ENVS = 7  # Number of parallel environments
SEED = 42

# Walk-Forward Validation Parameters (in number of unique dates/periods)
WF_TRAIN_WINDOW_SIZE = 252 * 3 # e.g., 3 years of trading days
WF_VALIDATION_WINDOW_SIZE = 252 # e.g., 1 year of trading days
WF_STEP_SIZE = 252 # e.g., advance by 1 year

# PPO parameters tuned via Optuna
PPO_PARAMS = dict(
    learning_rate=3.217e-05, 
    n_steps=1024, 
    batch_size=64, 
    n_epochs=8, 
    gamma=0.9898, 
    ent_coef=0.0157,
    verbose=1
)

# ---- Walk-Forward Validation Configuration ----
# If False, the system will use the simple TRAIN_RATIO split.
# If True, the orchestrator will use walk-forward validation to evaluate candidates.
ENABLE_WALK_FORWARD_VALIDATION = True
N_SPLITS = 5  # Number of chronological splits
TRAIN_YEARS = 3  # Number of years for each training set
VALIDATION_YEARS = 1  # Number of years for each validation set


# ---- System Configuration ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
