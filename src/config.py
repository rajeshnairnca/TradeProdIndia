# Description: Centralized configuration file for the autonomous trading project.

import os
import torch

# ---- Mode Selection ----
# Default to full mode for compatibility with existing trained models; set TRAINING_MODE=fast to use fast settings.
MODE = os.getenv("TRAINING_MODE", "full").strip().lower()
FAST_MODE = MODE == "fast"
ACTIVE_MODE = "fast" if FAST_MODE else "full"

# Overrides applied when FAST_MODE is True
FAST_OVERRIDES = {
    "FEATURES_DIM": 128,
    "EMBED_DIM": 16,
    "LSTM_N_STACK": 4,
    "TRAINING_TIMESTEPS": 150_000,
    "N_ENVS": 2,
    "ENABLE_WALK_FORWARD_VALIDATION": False,
    "PPO_PARAMS": {
        "n_steps": 512,
        "batch_size": 32,
        "n_epochs": 4,
    },
}

def _maybe_override(key, default):
    if FAST_MODE and key in FAST_OVERRIDES:
        override = FAST_OVERRIDES[key]
        if isinstance(default, dict) and isinstance(override, dict):
            merged = default.copy()
            merged.update(override)
            return merged
        return override
    return default

# ---- Data Configuration ----
DATA_FILE = "data/daily_data.parquet"
SECTOR_MAP_FILE = 'data/symbol_to_sector_map.json'
NUM_SECTORS = 11

# ---- Environment & Portfolio Configuration ----
INITIAL_CAPITAL = 1_000_000.0
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TOP_K = 10
ADV_LOOKBACK = 21
ROLLING_WINDOW_FOR_VOL = 21
USE_VOL_PARITY = True
def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

USE_REGIME_SYSTEM = _env_bool("USE_REGIME_SYSTEM", True)

# ---- Penalties & Costs ----
# From empirical analysis of Indian market brokerage charges
TURNOVER_PENALTY = 0.000718
LEVERAGE_PENALTY = 1e-4
RISK_PENALTY_COEFF = 0.5021
SLIPPAGE_COEFF = 0.005
CASH_DRAG_COEFF = 8e-5  # per-step penalty scaled by cash_weight to discourage idle cash
WEIGHT_CHANGE_PENALTY = 0.0005  # penalize large day-over-day weight changes (swing-friendly)
WEIGHT_SMOOTHING = 0.2  # blend factor for previous weights vs new weights (higher = smoother)

# ---- Model Architecture ----
FEATURES_DIM = _maybe_override("FEATURES_DIM", 256)
EMBED_DIM = _maybe_override("EMBED_DIM", 32)
SECTOR_EMBED_DIM = 8
LSTM_N_STACK = _maybe_override("LSTM_N_STACK", 8) # Number of historical steps for the recurrent model's memory

# ---- Training Hyperparameters ----
# Base full-mode steps trimmed for quicker convergence; fast mode overrides below.
TRAINING_TIMESTEPS = _maybe_override("TRAINING_TIMESTEPS", 250000)
N_ENVS = _maybe_override("N_ENVS", 7)  # Number of parallel environments
SEED = 42

# Walk-Forward Validation Parameters (in number of unique dates/periods)
WF_TRAIN_WINDOW_SIZE = 252 * 3 # e.g., 3 years of trading days
WF_VALIDATION_WINDOW_SIZE = 252 # e.g., 1 year of trading days
WF_STEP_SIZE = 504 # e.g., advance by 1 year

# PPO parameters tuned via Optuna
PPO_PARAMS = _maybe_override("PPO_PARAMS", dict(
    learning_rate=3.217e-05,
    n_steps=1024,
    batch_size=64,
    n_epochs=8,
    gamma=0.9898,
    ent_coef=0.014,  # slightly lower entropy to reduce churn when sizing is more aggressive
    verbose=1
))

# ---- Walk-Forward Validation Configuration ----
# If False, the system will use the simple TRAIN_RATIO split.
# If True, the orchestrator will use walk-forward validation to evaluate candidates.
ENABLE_WALK_FORWARD_VALIDATION = bool(_maybe_override("ENABLE_WALK_FORWARD_VALIDATION", True))
N_SPLITS = 1  # Number of chronological splits
TRAIN_YEARS = 3  # Number of years for each training set
VALIDATION_YEARS = 1  # Number of years for each validation set


# ---- System Configuration ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
