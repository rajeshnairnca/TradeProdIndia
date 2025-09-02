import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.trading_environment import DailyCrossSectionalEnv
from src.models import TransformerFeatureExtractor
from src import config
from src.utils import calculate_performance_metrics

def run_walk_forward(alpha_name: str):
    """
    Performs walk-forward validation for a given alpha strategy.

    Args:
        alpha_name: The name of the alpha to be evaluated.
    """
    print(f"--- Starting Walk-Forward Validation for: {alpha_name} ---")

    # 1. Load Data
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, config.DATA_FILE))
    
    # --- Robust Sector Mapping ---
    with open(os.path.join(PROJECT_ROOT, config.SECTOR_MAP_FILE), 'r') as f:
        symbol_to_sector_name = json.load(f)
    
    all_sectors = sorted(list(set(symbol_to_sector_name.values())))
    if "Unknown" not in all_sectors:
        all_sectors.append("Unknown")

    sector_name_to_id = {name: i for i, name in enumerate(all_sectors)}
    num_sectors = len(all_sectors)

    universe = df.index.get_level_values('ticker').unique().tolist()
    symbol_to_sector_id = {
        sym: sector_name_to_id[symbol_to_sector_name.get(sym, "Unknown")]
        for sym in universe
    }

    all_dates = df.index.get_level_values('date').unique().sort_values()
    
    # 2. Calculate Splits
    trading_days_per_year = 252
    train_window = config.TRAIN_YEARS * trading_days_per_year
    validation_window = config.VALIDATION_YEARS * trading_days_per_year
    
    overall_net_worths = []
    all_fold_results = {}

    for i in range(config.N_SPLITS):
        print(f"--- Running Fold {i+1}/{config.N_SPLITS} ---")
        
        train_start_idx = i * validation_window 
        train_end_idx = train_start_idx + train_window
        validation_start_idx = train_end_idx
        validation_end_idx = validation_start_idx + validation_window

        if validation_end_idx > len(all_dates):
            print("Not enough data for this fold. Stopping walk-forward validation.")
            break

        train_start_date, train_end_date = all_dates[train_start_idx], all_dates[train_end_idx]
        validation_start_date, validation_end_date = all_dates[validation_start_idx], all_dates[validation_end_idx]

        print(f"Train Period: {train_start_date.date()} to {train_end_date.date()}")
        print(f"Validation Period: {validation_start_date.date()} to {validation_end_date.date()}")

        train_df = df[(df.index.get_level_values('date') >= train_start_date) & (df.index.get_level_values('date') < train_end_date)]
        validation_df = df[(df.index.get_level_values('date') >= validation_start_date) & (df.index.get_level_values('date') < validation_end_date)]

        # 4. Train Model for this fold
        train_env_raw = DailyCrossSectionalEnv(train_df, symbol_to_sector_id, num_sectors, is_backtest=False)
        train_env = DummyVecEnv([lambda: train_env_raw])
        train_env = VecFrameStack(train_env, n_stack=config.LSTM_N_STACK)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

        policy_kwargs = {
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": dict(features_dim=config.FEATURES_DIM, num_sectors=num_sectors, sector_embed_dim=config.SECTOR_EMBED_DIM)
        }

        model = RecurrentPPO(
            RecurrentActorCriticPolicy,
            train_env,
            policy_kwargs=policy_kwargs,
            **config.PPO_PARAMS,
            tensorboard_log=None,
            device=config.DEVICE,
            seed=config.SEED
        )
        model.learn(total_timesteps=config.TRAINING_TIMESTEPS)

        train_env.save("vec_normalize_temp.pkl")

        # 5. Evaluate Model on the validation fold
        validation_env_raw = DailyCrossSectionalEnv(validation_df, symbol_to_sector_id, num_sectors, is_backtest=True)
        validation_env = DummyVecEnv([lambda: validation_env_raw])
        validation_env = VecFrameStack(validation_env, n_stack=config.LSTM_N_STACK)
        validation_env = VecNormalize.load("vec_normalize_temp.pkl", validation_env)
        validation_env.training = False
        validation_env.norm_reward = False

        model.save("temp_model.zip")
        model = RecurrentPPO.load("temp_model.zip", env=validation_env)

        obs = validation_env.reset()
        done, lstm_states = False, None
        fold_net_worths = [config.INITIAL_CAPITAL]

        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, _, done, info = validation_env.step(action)
            net_worth = info[0]['net_worth']
            fold_net_worths.append(net_worth)

        # 6. Store results for the fold
        fold_cagr = calculate_performance_metrics(pd.Series(fold_net_worths), len(validation_df.index.get_level_values('date').unique()))["CAGR"]
        print(f"Fold {i+1} CAGR: {fold_cagr:.2f}%")
        all_fold_results[f"Fold_{i+1}"] = {"CAGR": fold_cagr}

        if not overall_net_worths:
            overall_net_worths.extend(fold_net_worths)
        else:
            last_overall_worth = overall_net_worths[-1]
            scaled_fold_worths = [w * (last_overall_worth / fold_net_worths[0]) for w in fold_net_worths[1:]]
            overall_net_worths.extend(scaled_fold_worths)

    # 7. Aggregate and Report Final Results
    print("\n--- Walk-Forward Validation Summary ---")
    
    if os.path.exists("temp_model.zip"):
        os.remove("temp_model.zip")
    if os.path.exists("vec_normalize_temp.pkl"):
        os.remove("vec_normalize_temp.pkl")

    if not overall_net_worths:
        print("Validation could not be completed. No results to show.")
        return None

    total_days = config.N_SPLITS * validation_window
    final_metrics = calculate_performance_metrics(pd.Series(overall_net_worths), total_days)

    print(f"Overall CAGR: {final_metrics['CAGR']:.2f}%")
    print(f"Overall Sharpe Ratio: {final_metrics['Sharpe Ratio']:.2f}")
    print(f"Overall Max Drawdown: {final_metrics['Max Drawdown']:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(overall_net_worths)
    plt.title(f'Walk-Forward Validation Equity Curve for "{alpha_name}"')
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig(f"{alpha_name}_walkforward_performance.png")
    print(f"Saved walk-forward performance chart to {alpha_name}_walkforward_performance.png")
    
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run walk-forward validation for a single alpha.")
    parser.add_argument("--alpha-name", type=str, required=True, help="Name of the alpha to validate.")
    args = parser.parse_args()
    
    run_walk_forward(args.alpha_name)
