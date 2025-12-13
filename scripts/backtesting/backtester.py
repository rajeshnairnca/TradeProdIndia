import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
import importlib.util

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from src import config
from src.regime import compute_market_regime_table
from src.trading_environment import DailyCrossSectionalEnv
from src.utils import calculate_cagr

def make_ensemble_dirname(alpha_names: list[str]) -> str:
    """Generate a short, stable ensemble folder name with a hash suffix."""
    sorted_names = sorted(alpha_names)
    if len(sorted_names) == 1:
        return sorted_names[0]

    digest = hashlib.sha1("::".join(sorted_names).encode("utf-8")).hexdigest()[:8]
    preview_parts = [name[:12] for name in sorted_names[:3]]
    preview = "_vs_".join(preview_parts)
    if len(sorted_names) > 3:
        preview += f"_plus{len(sorted_names) - 3}"
    return f"{preview}__{digest}"

class EnsembleBacktester:
    def __init__(self, alpha_names: list[str], alphas_root_dir="alphas"):
        self.alphas_root_dir = os.path.join(PROJECT_ROOT, alphas_root_dir)
        self.alpha_names = alpha_names
        self.alphas = self._load_alphas()
        if not self.alphas:
            raise ValueError("No valid alphas found for the given names. Please check the directory and names.")
        self.regime_table = self._compute_regime_table()
        self.test_envs = self._setup_test_envs()
        self.alpha_weights = self._calculate_weights()

    def _load_alphas(self):
        """Loads a specific list of trained alpha models, their feature keys, and their performance."""
        alphas = {}
        for alpha_name in self.alpha_names:
            model_dir = os.path.join(self.alphas_root_dir, alpha_name, "model")
            model_path = os.path.join(model_dir, "best_model.zip")
            stats_path = os.path.join(model_dir, "vec_normalize.pkl")
            features_path = os.path.join(self.alphas_root_dir, alpha_name, "feature_keys.json")
            results_path = os.path.join(self.alphas_root_dir, alpha_name, "results.json")

            if os.path.exists(model_path) and os.path.exists(stats_path) and os.path.exists(features_path):
                print(f"Loading alpha: {alpha_name}")
                # Monkey-patching sys.modules to load model saved with old structure
                from src import models
                sys.modules['models'] = models
                model = RecurrentPPO.load(model_path, device=config.DEVICE)
                
                with open(features_path, 'r') as f:
                    feature_keys = json.load(f)
                
                cagr = 0.0
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        try:
                            results_data = json.load(f)
                            cagr = results_data.get("cagr", 0.0)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from {results_path}. Defaulting CAGR to 0.")
                else:
                    print(f"Warning: results.json not found for alpha '{alpha_name}'. Defaulting CAGR to 0.")

                alphas[alpha_name] = {"model": model, "stats_path": stats_path, "feature_keys": feature_keys, "cagr": cagr}
            else:
                print(f"Warning: Could not find model, stats, or feature files for alpha '{alpha_name}'. It will be skipped.")
        return alphas

    def _compute_regime_table(self):
        """Pre-compute daily market regime metrics for regime-aware sizing."""
        full_data_orig = pd.read_parquet(os.path.join(PROJECT_ROOT, config.DATA_FILE))
        return compute_market_regime_table(full_data_orig)

    def _calculate_weights(self):
        """Calculates weights for each alpha based on its CAGR."""
        cagrs = np.array([alpha_info["cagr"] for alpha_info in self.alphas.values()])
        print(f"Individual Alpha CAGRs for weighting: {list(cagrs)}")

        # Set any negative CAGRs to 0 to avoid negative weights
        cagrs = np.maximum(cagrs, 0)
        
        total_cagr = np.sum(cagrs)
        
        if total_cagr > 0:
            weights = cagrs / total_cagr
        else:
            # If all CAGRs are <= 0, fall back to equal weighting
            print("Warning: All alpha CAGRs are zero or negative. Falling back to equal weighting.")
            weights = np.ones(len(self.alphas)) / len(self.alphas)
        
        print(f"Calculated Ensemble Weights: {list(weights)}")
        return weights

    def _get_current_regime(self, current_date):
        """Lookup regime info for a given date; returns dict with flags."""
        if current_date not in self.regime_table.index:
            return {"vol_high": False, "dispersion_high": False, "dispersion_low": False, "breadth_low": False, "breadth_high": False}
        row = self.regime_table.loc[current_date]
        return {
            "vol_high": bool(row["vol_high"]),
            "dispersion_high": bool(row["dispersion_high"]),
            "dispersion_low": bool(row["dispersion_low"]),
            "breadth_low": bool(row["breadth_low"]),
            "breadth_high": bool(row["breadth_high"]),
        }

    def _dynamic_top_k(self, regime_flags):
        """Regime-aware top_k selection."""
        if regime_flags["vol_high"] and regime_flags["dispersion_low"] and regime_flags["breadth_low"]:
            return 5  # concentrated in choppy/high-vol, low-dispersion conditions
        if (not regime_flags["vol_high"]) and regime_flags["dispersion_high"] and regime_flags["breadth_high"]:
            return 12  # broaden when trends are clean and dispersion is strong
        return config.TOP_K

    def _apply_top_k(self, weights, k):
        """Keep top-k weights, renormalize; if k >= len, return normalized weights."""
        weights = np.array(weights, dtype=np.float32)
        if k >= len(weights):
            return weights / (np.sum(weights) + 1e-9) if np.sum(weights) > 0 else weights
        idx = np.argpartition(weights, -k)[-k:]
        new_w = np.zeros_like(weights)
        new_w[idx] = weights[idx]
        total = np.sum(new_w)
        return new_w / (total + 1e-9) if total > 0 else new_w

    def _setup_test_envs(self):
        """Prepares a separate test environment for each alpha, ensuring feature consistency."""
        full_data_orig = pd.read_parquet(os.path.join(PROJECT_ROOT, config.DATA_FILE))
        
        with open(os.path.join(PROJECT_ROOT, config.SECTOR_MAP_FILE), 'r') as f: 
            symbol_to_sector_name = json.load(f)
        
        universe = full_data_orig.index.get_level_values('ticker').unique().tolist()
        all_sectors = sorted(list(set(symbol_to_sector_name.values())))
        if "Unknown" not in all_sectors:
            all_sectors.append("Unknown")
        sector_name_to_id = {name: i for i, name in enumerate(all_sectors)}
        NUM_SECTORS = len(all_sectors)
        symbol_to_sector_id = {sym: sector_name_to_id.get(symbol_to_sector_name.get(sym, "Unknown"), 0) for sym in universe}

        envs = {}
        for alpha_name in self.alpha_names:
            full_data = full_data_orig.copy()
            features_file = os.path.join(self.alphas_root_dir, alpha_name, "feature_engineering.py")
            
            if os.path.exists(features_file):
                print(f"Applying feature engineering from {alpha_name}")
                spec = importlib.util.spec_from_file_location(f"{alpha_name}.feature_engineering", features_file)
                feature_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(feature_module)
                feature_engineering = feature_module.feature_engineering
                full_data = feature_engineering(full_data)

            all_dates = full_data.index.get_level_values('date').unique().sort_values()
            split_index = int(len(all_dates) * config.TRAIN_RATIO)
            _ , test_dates = all_dates[:split_index], all_dates[split_index:]
            test_data = full_data[full_data.index.get_level_values('date').isin(test_dates)].copy()

            # Enforce feature consistency
            expected_features = self.alphas[alpha_name]["feature_keys"]
            current_features = sorted([c for c in test_data.columns if c.endswith('_z')])
            
            missing_features = set(expected_features) - set(current_features)
            for f in missing_features:
                test_data[f] = 0.0
                
            extra_features = set(current_features) - set(expected_features)
            if extra_features:
                test_data = test_data.drop(columns=list(extra_features))

            env = DummyVecEnv([
                lambda: DailyCrossSectionalEnv(
                    df=test_data,
                    symbol_to_sector_id=symbol_to_sector_id,
                    num_sectors=NUM_SECTORS,
                    universe=universe,
                    feature_keys=expected_features,
                    is_backtest=True,
                    regime_table=self.regime_table,
                    apply_regime_top_k=config.USE_REGIME_SYSTEM
                )
            ])
            env = VecFrameStack(env, n_stack=config.LSTM_N_STACK)
            
            stats_path = self.alphas[alpha_name]["stats_path"]
            env = VecNormalize.load(stats_path, env)
            env.training = False
            env.norm_reward = False
            envs[alpha_name] = env
            
        return envs

    def run(self):
        """Executes the backtest for the loaded ensemble of alphas."""
        print(f"\n--- Running Ensemble Backtest with {len(self.alphas)} Alphas: {list(self.alphas.keys())} ---")
        
        obs_dict = {name: env.reset() for name, env in self.test_envs.items()}
        lstm_states = {name: None for name in self.alphas.keys()}
        done = False
        net_worth_history, date_history, transaction_log = [], [], []

        main_env = list(self.test_envs.values())[0]
        universe = main_env.get_attr("universe")[0]
        initial_net_worth = main_env.get_attr("net_worth")[0]
        initial_date_idx = main_env.get_attr("current_idx")[0]
        initial_date = main_env.get_attr("dates")[0][initial_date_idx - 1]
        net_worth_history.append(initial_net_worth)
        date_history.append(initial_date)

        while not done:
            all_weights = []
            for name, alpha in self.alphas.items():
                obs = obs_dict[name]
                action, lstm_states[name] = alpha["model"].predict(obs, state=lstm_states[name], deterministic=True)
                all_weights.append(action[0])
            
            combined_weights = np.average(all_weights, axis=0, weights=self.alpha_weights)

            if np.sum(combined_weights) > 1e-9:
                base_weights = combined_weights / np.sum(combined_weights)
            else:
                base_weights = np.zeros_like(combined_weights)

            # Regime-aware top-k handled inside the environment via regime_table
            final_weights = base_weights

            new_obs_dict = {}
            infos = None
            for i, (name, env) in enumerate(self.test_envs.items()):
                obs, _, dones, current_infos = env.step([final_weights])
                new_obs_dict[name] = obs
                done = dones[0]
                if i == 0: # Use first env for history
                    infos = current_infos
            
            obs_dict = new_obs_dict
            
            if infos:
                info_dict = infos[0]
                net_worth_history.append(info_dict['net_worth'])
                date_history.append(datetime.strptime(info_dict['date'], '%Y-%m-%d'))
                
                # Log transactions
                trade_dollars = info_dict.get('trade_dollars', [])
                prices = info_dict.get('prices', [])
                net_worth = info_dict.get('net_worth')
                cash_balance = info_dict.get('cash')
                portfolio_value = info_dict.get('portfolio_value')
                gross_exposure = info_dict.get('gross_exposure')
                leverage = info_dict.get('leverage')
                turnover = info_dict.get('turnover')
                for i in range(len(trade_dollars)):
                    if abs(trade_dollars[i]) > 1.0: # Log only trades over $1
                        transaction_log.append({
                            "date": info_dict['date'],
                            "ticker": universe[i],
                            "action": "BUY" if trade_dollars[i] > 0 else "SELL",
                            "shares": trade_dollars[i] / prices[i] if prices[i] > 0 else 0,
                            "price_usd": prices[i],
                            "value_usd": trade_dollars[i],
                            "net_worth_usd": net_worth,
                            "cash_usd": cash_balance,
                            "portfolio_value_usd": portfolio_value,
                            "gross_exposure": gross_exposure,
                            "leverage": leverage,
                            "turnover": turnover
                        })

        print("\n--- Ensemble Backtest Results ---")
        final_net_worth = net_worth_history[-1]
        start_date, end_date = date_history[0], date_history[-1]
        num_years = (end_date - start_date).days / 365.25
        cagr = calculate_cagr(initial_net_worth, final_net_worth, num_years)
        print(f"Final Net Worth: ${final_net_worth:,.2f} | CAGR: {cagr:.2%}")

        ensemble_dirname = make_ensemble_dirname(list(self.alphas.keys()))
        output_dir = os.path.join(self.alphas_root_dir, "_ensembles", ensemble_dirname)
        os.makedirs(output_dir, exist_ok=True)

        # Save transaction log
        if transaction_log:
            log_df = pd.DataFrame(transaction_log)
            csv_path = os.path.join(output_dir, 'transactions.csv')
            log_df.to_csv(csv_path, index=False)
            print(f"Transaction log saved to {csv_path}")

        plt.figure(figsize=(12, 6))
        plt.plot(date_history, net_worth_history)
        plt.title(f'Ensemble Backtest Performance ({len(self.alphas)} Alphas)\n{", ".join(self.alphas.keys())}')
        plt.xlabel('Date'); plt.ylabel('Net Worth ($)')
        plt.grid(True)
        plot_path = os.path.join(output_dir, 'backtest_performance.png')
        plt.savefig(plot_path)
        print(f"Performance plot saved to {plot_path}")

        results = {
            "cagr": cagr,
            "final_net_worth": final_net_worth,
            "num_alphas": len(self.alphas),
            "alphas": list(self.alphas.keys()),
            "ensemble_dir": ensemble_dirname
        }
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs='+', required=True, help="List of alpha names to include in the ensemble.")
    parser.add_argument("--alphas-dir", type=str, default="alphas", help="Root directory containing the trained alpha models.")
    args = parser.parse_args()

    try:
        backtester = EnsembleBacktester(alpha_names=args.alphas, alphas_root_dir=args.alphas_dir)
        backtester.run()
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
