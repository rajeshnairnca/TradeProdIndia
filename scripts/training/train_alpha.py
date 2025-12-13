import os
import sys
import json
import glob
import re
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime
import importlib.util

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from src import config
from src.trading_environment import DailyCrossSectionalEnv
from src.models import TransformerFeatureExtractor
from src.utils import calculate_cagr
from src.regime import compute_market_regime_table


def validate_feature_file_safe(path: str):
    """Best-effort guardrail: reject obvious future-peeking constructs in feature scripts."""
    with open(path, "r") as f:
        content = f.read()
    patterns = [
        r"center\s*=\s*True",      # centered rolling windows
        r"shift\s*\(\s*-\s*\d",    # negative shifts (lookahead)
        r"lead\s*\(",              # lead/lag forward-looking
    ]
    for pat in patterns:
        if re.search(pat, content):
            raise RuntimeError(f"Feature file '{path}' appears to use future-looking operation: pattern '{pat}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-name", type=str, required=True, help="A unique name for the alpha being trained.")
    parser.add_argument("--description", type=str, default="", help="A brief description of the alpha's strategy.")
    parser.add_argument("--timesteps", type=int, default=config.TRAINING_TIMESTEPS, help="Number of timesteps to train.")
    parser.add_argument("--output-dir", type=str, default="alphas", help="The root directory to save alpha artifacts.")
    parser.add_argument("--features-file", type=str, help="Path to the feature engineering python file.")
    args = parser.parse_args()

    # --- DIRECTORY SETUP ---
    alpha_dir = os.path.join(args.output_dir, args.alpha_name)
    log_dir = os.path.join(alpha_dir, "logs")
    model_dir = os.path.join(alpha_dir, "model")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save description
    with open(os.path.join(alpha_dir, "description.txt"), "w") as f:
        f.write(args.description)

    # --- TRAINING SETUP ---
    multiprocessing.set_start_method('spawn', force=True)
    set_random_seed(config.SEED)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    data_file_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    if not os.path.exists(data_file_path): raise FileNotFoundError(f"Data file not found: {data_file_path}")
    full_data = pd.read_parquet(data_file_path)
    all_dates = full_data.index.get_level_values('date').unique().sort_values()
    split_index = int(len(all_dates) * config.TRAIN_RATIO)
    train_dates, test_dates = all_dates[:split_index], all_dates[split_index:]
    if len(test_dates) == 0:
        raise RuntimeError("Dataset too small to create a holdout set; adjust config.TRAIN_RATIO or provide more data.")

    # --- Load and apply feature engineering ---
    if args.features_file and os.path.exists(args.features_file):
        print(f"Loading feature engineering from {args.features_file}")
        validate_feature_file_safe(args.features_file)

        # Save a copy of the features file to the alpha directory for reproducibility
        try:
            dest_path = os.path.join(alpha_dir, 'feature_engineering.py')
            if os.path.abspath(args.features_file) != os.path.abspath(dest_path):
                with open(args.features_file, 'r') as src:
                    with open(dest_path, 'w') as dst:
                        dst.write(src.read())
                print(f"Copied feature engineering file to {dest_path}")
            else:
                print("Source and destination feature files are the same; skipping copy.")
        except IOError as e:
            print(f"Error: Could not copy feature file: {e}", file=sys.stderr)

        spec = importlib.util.spec_from_file_location("feature_engineering", args.features_file)
        feature_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_module)
        feature_engineering = getattr(feature_module, "feature_engineering", None)
        if feature_engineering:
            full_data = feature_engineering(full_data)
            print("Feature engineering applied.")
        else:
            print("Warning: feature_engineering not found; proceeding without applying.")

    # Recreate splits after potential feature engineering
    train_df = full_data[full_data.index.get_level_values('date').isin(train_dates)]
    test_df = full_data[full_data.index.get_level_values('date').isin(test_dates)]

    # Use full history for regime table so test dates have coverage, but computations are still backward-looking.
    regime_table = compute_market_regime_table(pd.concat([train_df, test_df]))

    # Save the feature keys used for this alpha
    feature_keys = sorted([c for c in full_data.columns if c.endswith('_z')])
    with open(os.path.join(alpha_dir, "feature_keys.json"), "w") as f:
        json.dump(feature_keys, f)

    # Sanity check: warn if feature keys reference missing columns in base data before FE
    missing_base = [k for k in feature_keys if k not in full_data.columns]
    if missing_base:
        print(f"Warning: missing engineered features in data after applying feature file: {missing_base}")

    # --- True Sector Loading ---
    with open(os.path.join(PROJECT_ROOT, config.SECTOR_MAP_FILE), 'r') as f:
        symbol_to_sector_name = json.load(f)
    all_sectors = sorted(list(set(symbol_to_sector_name.values())))
    if "Unknown" not in all_sectors:
        all_sectors.append("Unknown")
    sector_name_to_id = {name: i for i, name in enumerate(all_sectors)}
    NUM_SECTORS = len(all_sectors)
    universe = full_data.index.get_level_values('ticker').unique().tolist()
    symbol_to_sector_id = {sym: sector_name_to_id.get(symbol_to_sector_name.get(sym, "Unknown"), 0) for sym in universe}

    def make_env(rank, seed=0, df=None, is_backtest=False):
        def _init():
            env = DailyCrossSectionalEnv(
                df=df,
                symbol_to_sector_id=symbol_to_sector_id,
                num_sectors=NUM_SECTORS,
                universe=universe,
                feature_keys=feature_keys,
                regime_table=regime_table,
                apply_regime_top_k=True,
                is_backtest=is_backtest
            )
            env = Monitor(env, filename=os.path.join(log_dir, f"{rank}.monitor.csv"))
            return env
        set_random_seed(seed)
        return _init

    # --- SANITY CHECK FOR FINITE OBS/REGIME ---
    sanity_env = DailyCrossSectionalEnv(
        df=train_df,
        symbol_to_sector_id=symbol_to_sector_id,
        num_sectors=NUM_SECTORS,
        universe=universe,
        feature_keys=feature_keys,
        regime_table=regime_table,
        apply_regime_top_k=True,
        is_backtest=True,
    )
    obs, _ = sanity_env.reset()
    for k in ("stock_features", "global_features", "mask", "sector_ids"):
        arr = np.asarray(obs[k])
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Non-finite values detected in observation key '{k}' during sanity check.")
    dummy_action = np.zeros(len(universe) + 1, dtype=np.float32)
    if len(universe) > 0:
        dummy_action[:-1] = 1.0 / len(universe)
    _, _, _, _, info = sanity_env.step(dummy_action)
    if not np.isfinite(info.get("net_worth", 0.0)):
        raise RuntimeError("Non-finite net_worth detected during sanity step.")

    train_env = SubprocVecEnv([make_env(i, config.SEED + i, df=train_df) for i in range(config.N_ENVS)])
    train_env = VecFrameStack(train_env, n_stack=config.LSTM_N_STACK)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # --- MODEL TRAINING ---
    policy_kwargs = {
        "features_extractor_class": TransformerFeatureExtractor, 
        "features_extractor_kwargs": dict(features_dim=config.FEATURES_DIM, num_sectors=NUM_SECTORS, sector_embed_dim=config.SECTOR_EMBED_DIM)
    }
    
    model = RecurrentPPO(
        RecurrentActorCriticPolicy, 
        train_env, 
        policy_kwargs=policy_kwargs, 
        device=config.DEVICE, 
        tensorboard_log=log_dir, 
        **config.PPO_PARAMS
    )

    

    eval_env = SubprocVecEnv([make_env(config.N_ENVS, config.SEED, df=train_df)])
    eval_env = VecFrameStack(eval_env, n_stack=config.LSTM_N_STACK)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.)
    eval_env.obs_rms = train_env.obs_rms

    # Stop training if there is no improvement after a given number of evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=55, min_evals=20, verbose=1)
    
    checkpoint_callback = CheckpointCallback(save_freq=max(config.PPO_PARAMS['n_steps'] * 5 // config.N_ENVS, 1), save_path=log_dir, name_prefix="rl_model", save_vecnormalize=True)
    # The EvalCallback will save the best model and trigger the stop callback
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=model_dir, 
                                 log_path=log_dir, 
                                 eval_freq=max(config.PPO_PARAMS['n_steps'] * 5 // config.N_ENVS, 1), 
                                 deterministic=True, 
                                 render=False, 
                                 callback_after_eval=stop_train_callback)

    print(f"--- Training alpha '{args.alpha_name}' for {args.timesteps} timesteps ---")
    model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback], progress_bar=True)

    # --- SAVE FINAL MODEL & ENV STATS ---
    model.save(os.path.join(model_dir, "final_model.zip"))
    train_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"Final model for '{args.alpha_name}' saved to {model_dir}")

    # --- BACKTESTING ---
    print("\n--- Backtesting on Unseen Test Data ---")
    model = RecurrentPPO.load(os.path.join(model_dir, "best_model.zip"), device=config.DEVICE)

    test_env_raw = DummyVecEnv([make_env(0, config.SEED, df=test_df, is_backtest=True)])
    test_env = VecFrameStack(test_env_raw, n_stack=config.LSTM_N_STACK)
    test_env = VecNormalize.load(os.path.join(model_dir, "vec_normalize.pkl"), test_env)
    test_env.training = False
    test_env.norm_reward = False

    obs = test_env.reset()
    done, lstm_states = False, None
    net_worth_history, date_history = [], []

    initial_net_worth = test_env.get_attr("net_worth")[0]
    initial_date_idx = test_env.get_attr("current_idx")[0]
    initial_date = test_env.get_attr("dates")[0][initial_date_idx - 1]
    net_worth_history.append(initial_net_worth)
    date_history.append(initial_date)

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, _, dones, infos = test_env.step(action)
        done = dones[0]
        net_worth_history.append(infos[0]['net_worth'])
        date_history.append(datetime.strptime(infos[0]['date'], '%Y-%m-%d'))

    print("\n--- Backtest Results ---")
    final_net_worth = net_worth_history[-1]
    start_date, end_date = date_history[0], date_history[-1]
    print(f"Backtest Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"Backtest End Date: {end_date.strftime('%Y-%m-%d')}")
    num_years = (end_date - start_date).days / 365.25
    cagr = calculate_cagr(initial_net_worth, final_net_worth, num_years)
    print(f"Final Net Worth: ${final_net_worth:,.2f} | CAGR: {cagr:.2%}")

    # --- SAVE RESULTS ---
    plt.figure(figsize=(12, 6))
    plt.plot(date_history, net_worth_history)
    plt.title(f'Backtest Performance for Alpha: {args.alpha_name}')
    plt.xlabel('Date'); plt.ylabel('Net Worth ($)')
    plt.grid(True)
    plot_path = os.path.join(alpha_dir, 'backtest_performance.png')
    plt.savefig(plot_path)
    print(f"Performance plot saved to {plot_path}")

    results = {"cagr": cagr, "final_net_worth": final_net_worth}
    with open(os.path.join(alpha_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"--- Alpha '{args.alpha_name}' finished successfully. ---")

if __name__ == "__main__":
    main()
