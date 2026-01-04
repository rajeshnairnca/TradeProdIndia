import os
import pandas as pd
import numpy as np
import importlib.util
from tqdm import tqdm
from src import regime, config, strategy as strat_module

def load_all_strategies():
    # Use the existing helper to list and load
    strategy_names = strat_module.list_strategy_names(["alphas"])
    strategies = strat_module.load_strategies(strategy_names, ["alphas"])
    return strategies

def run_analysis(region="us"):
    # 1. Load Data
    data_path = config.REGION_DATA_FILES.get(region, "data/daily_data.parquet")
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Compute Regime
    # Force HMM mode for this analysis as we want to test the new logic
    print("Computing Regimes (HMM)...")
    regime_table = regime.compute_market_regime_table(df, mode="hmm")
    
    common_idx = df.index.get_level_values("date").intersection(regime_table.index)
    # Filter df to match regime table dates
    # Note: df is multi-index (date, ticker), regime_table is index (date)
    df = df[df.index.get_level_values("date").isin(common_idx)]
    regime_table = regime_table.loc[common_idx]
    
    # 3. Compute Scores & Returns
    strategies = load_all_strategies()
    print(f"Analyzing {len(strategies)} strategies: {[s.name for s in strategies]}")
    
    # Calculate Forward 1-Day Return
    # Using 'Close' pct_change shifted -1 grouping by ticker
    df["ret_1d"] = df.groupby("ticker")["Close"].pct_change().shift(-1)
    
    results = {s.name: {} for s in strategies}
    
    # Precompute scores
    score_dict = {}
    for s in strategies:
        print(f"Scoring {s.name}...")
        score_dict[s.name] = s.score_func(df)

    print("Aggregating performance by regime...")
    # Group dates by regime
    regime_groups = regime_table.groupby("regime_label")
    
    for label, group in regime_groups:
        print(f"Processing regime: {label} ({len(group)} days)")
        dates = group.index
        
        for s in strategies:
            try:
                # subset scores for these dates
                # slice using .loc with pd.IndexSlice or just checking date level
                # Much faster to join?
                
                # Get the scores for strategy 's'
                s_series = score_dict[s.name]
                
                # Filter s_series for the current regime dates
                # s_series index is (date, ticker)
                s_subset = s_series[s_series.index.get_level_values("date").isin(dates)]
                
                # Align returns
                ret_subset = df.loc[s_subset.index, "ret_1d"]
                
                if s_subset.empty:
                    continue

                # Combine into a temporary DF
                combined = pd.DataFrame({"score": s_subset, "ret": ret_subset})
                
                # Top 10 per day
                def get_top_k_mean(g):
                    return g.nlargest(10, "score")["ret"].mean()
                
                daily_means = combined.groupby(level="date").apply(get_top_k_mean)
                
                avg_daily_ret = daily_means.mean()
                ann_ret = avg_daily_ret * 252
                results[s.name][label] = ann_ret
                
            except Exception as e:
                print(f"Error processing {s.name} in {label}: {e}")
                results[s.name][label] = np.nan

    # 4. Print Report
    print("\n--- Strategy Performance by Regime (Annualized Return of Top 10) ---")
    
    # DataFrame for display
    res_df = pd.DataFrame(results).T
    res_df = res_df * 100 # percentage
    # Reorder columns if standard labels
    standard_cols = [
        "bull_low_vol",
        "bull_high_vol",
        "bear_low_vol",
        "bear_high_vol",
        "sideways_low_vol",
        "sideways_high_vol",
    ]
    cols = [c for c in standard_cols if c in res_df.columns] + [c for c in res_df.columns if c not in standard_cols]
    res_df = res_df[cols]
    
    print(res_df.round(2))
    
    # Recommend Best
    print("\n--- Best Strategy per Regime ---")
    for col in res_df.columns:
        if res_df[col].notna().any():
            best_strat = res_df[col].idxmax()
            best_ret = res_df.loc[best_strat, col]
            print(f"{col}: {best_strat} ({best_ret:.2f}%)")

if __name__ == "__main__":
    run_analysis()
