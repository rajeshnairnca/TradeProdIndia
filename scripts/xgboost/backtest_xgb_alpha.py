import argparse
import importlib.util
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys

sys.path.append(PROJECT_ROOT)
from src import config
from src.utils import calculate_cagr


def load_feature_engineer(path: str):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("candidate.feature_engineering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.feature_engineering


def prepare_data(features_file: str | None):
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, config.DATA_FILE))
    if features_file:
        fe_fn = load_feature_engineer(features_file)
        df = fe_fn(df)
    feature_cols = [c for c in df.columns if c.endswith("_z")]
    df["fwd_return"] = df.groupby(level="ticker")["log_return"].shift(-1)
    df = df.dropna(subset=["fwd_return"])
    return df, feature_cols


def split_by_date(df: pd.DataFrame):
    dates = df.index.get_level_values("date").unique().sort_values()
    split_idx = int(len(dates) * config.TRAIN_RATIO)
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]
    test_df = df[df.index.get_level_values("date").isin(test_dates)]
    return test_df


def load_model(alpha_name: str):
    alpha_dir = os.path.join(PROJECT_ROOT, "alphas", f"{alpha_name}_xgb")
    model_path = os.path.join(alpha_dir, "model.json")
    feature_path = os.path.join(alpha_dir, "feature_keys.json")
    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        raise FileNotFoundError(f"Could not find model or feature keys in {alpha_dir}")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(feature_path, "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def compute_gross_target(past_returns: List[float], base_gross: float, target_vol: float, vol_window: int, min_gross: float, max_gross: float) -> float:
    if target_vol <= 0 or len(past_returns) < max(vol_window, 1):
        return base_gross
    realized = np.std(past_returns[-vol_window:]) * np.sqrt(252)
    if realized <= 1e-8:
        return base_gross
    adj = target_vol / realized
    gross = base_gross * adj
    return float(np.clip(gross, min_gross, max_gross))


def allocate_day(day_df: pd.DataFrame, preds: np.ndarray, top_k: int, gross_cap: float, max_weight: float, sector_cap: float, symbol_to_sector: Dict[str, str]) -> Dict:
    day_df = day_df.copy()
    day_df["pred"] = preds
    # Attach sector info
    if "ticker" in day_df.columns:
        tickers = day_df["ticker"]
    else:
        tickers = [idx[1] if isinstance(idx, tuple) and len(idx) > 1 else "" for idx in day_df.index]
    day_df["sector"] = [symbol_to_sector.get(t, "Unknown") for t in tickers]

    day_df = day_df.sort_values("pred", ascending=False)
    selected = day_df.head(top_k)
    if selected.empty:
        return {"daily_return": 0.0, "selected": []}

    n = len(selected)
    weights = np.full(n, gross_cap / n if n > 0 else 0.0, dtype=np.float32)
    weights = np.minimum(weights, max_weight)

    # Sector caps
    sectors = selected["sector"].tolist()
    sector_weights: Dict[str, float] = {}
    for i, sec in enumerate(sectors):
        sector_weights[sec] = sector_weights.get(sec, 0.0) + weights[i]
    for i, sec in enumerate(sectors):
        cap = sector_cap * gross_cap
        if sector_weights[sec] > cap and sector_weights[sec] > 0:
            scale = cap / sector_weights[sec]
            weights[i] *= scale
    total_w = np.sum(weights)
    if total_w > gross_cap and total_w > 0:
        weights = weights * (gross_cap / total_w)

    realized_returns = selected["fwd_return"].to_numpy(dtype=np.float32)
    daily_return = float(np.nansum(realized_returns * weights))
    selected = selected.copy()
    selected["weight"] = weights
    return {"daily_return": daily_return, "selected": selected, "weights": weights}


def backtest(model, feature_cols: List[str], test_df: pd.DataFrame, top_k: int, gross_cap: float, max_weight: float, cost_bps: float, sector_cap: float, target_vol: float, vol_window: int, min_gross: float, max_gross: float):
    if test_df.empty:
        raise RuntimeError("No test data available for backtest.")

    with open(os.path.join(PROJECT_ROOT, config.SECTOR_MAP_FILE), "r") as f:
        symbol_to_sector_name = json.load(f)
    symbol_to_sector = symbol_to_sector_name

    dates = test_df.index.get_level_values("date").unique().sort_values()
    net_worth = config.INITIAL_CAPITAL
    history = [net_worth]
    date_history = []
    transactions = []
    prev_alloc: Dict[str, float] = {}
    past_returns: List[float] = []

    for date in dates:
        day_df = test_df[test_df.index.get_level_values("date") == date].copy()
        if day_df.empty:
            continue

        # Ensure feature columns exist
        for col in feature_cols:
            if col not in day_df.columns:
                day_df[col] = 0.0

        gross_target = compute_gross_target(past_returns, gross_cap, target_vol, vol_window, min_gross, max_gross)

        preds = model.predict(day_df[feature_cols].to_numpy(dtype=np.float32))
        alloc = allocate_day(day_df, preds, top_k, gross_target, max_weight, sector_cap, symbol_to_sector)
        selected = alloc["selected"]
        daily_return = alloc["daily_return"]

        # Transaction cost: charge per turnover vs prev day
        turnover = 0.0
        current_alloc = {}
        prev_net = net_worth
        if not isinstance(selected, list) and not selected.empty:
            for _, row in selected.iterrows():
                ticker = row.name[1] if isinstance(row.name, tuple) and len(row.name) > 1 else row.get("ticker", "")
                current_alloc[ticker] = float(row.get("weight", 0.0))
        tickers = set(prev_alloc.keys()) | set(current_alloc.keys())
        for t in tickers:
            prev_w = prev_alloc.get(t, 0.0)
            curr_w = current_alloc.get(t, 0.0)
            turnover += abs(curr_w - prev_w)
        cost = turnover * (cost_bps / 10000.0)

        net_worth *= (1.0 + daily_return - cost)
        history.append(net_worth)
        date_history.append(date)
        prev_alloc = current_alloc
        past_returns.append(daily_return - cost)

        if not isinstance(selected, list) and not selected.empty:
            for _, row in selected.iterrows():
                price = float(row["Close"]) if "Close" in row else 0.0
                alloc_usd = row.get("weight", 0.0) * prev_net
                transactions.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": row.name[1] if isinstance(row.name, tuple) and len(row.name) > 1 else row.get("ticker", ""),
                        "pred": float(row["pred"]),
                        "fwd_return": float(row["fwd_return"]),
                        "alloc_usd": alloc_usd,
                        "weight": float(row.get("weight", 0.0)),
                        "price": price,
                        "turnover": turnover,
                        "cost_bps": cost_bps,
                        "gross_target": gross_target,
                    }
                )

    start_date, end_date = date_history[0], date_history[-1]
    num_years = (end_date - start_date).days / 365.25 if len(date_history) > 1 else 0.0
    cagr = calculate_cagr(config.INITIAL_CAPITAL, net_worth, num_years) if num_years > 0 else 0.0

    return {
        "history": history,
        "dates": date_history,
        "transactions": transactions,
        "final_net_worth": net_worth,
        "cagr": cagr,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest an XGBoost cross-sectional alpha.")
    parser.add_argument("--alpha-name", required=True, help="Base alpha name used for training (without _xgb).")
    parser.add_argument("--features-file", help="Optional feature file to apply before prediction.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Number of tickers to allocate each day.")
    parser.add_argument("--gross-cap", type=float, default=1.0, help="Max gross exposure (1.0 = fully invested).")
    parser.add_argument("--max-weight", type=float, default=0.2, help="Max per-name weight cap.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Round-trip cost in basis points used in backtest.")
    parser.add_argument("--sector-cap", type=float, default=0.35, help="Max sector weight as a fraction of gross.")
    parser.add_argument("--target-vol", type=float, default=0.20, help="Annualized target vol for gross scaling (0 to disable).")
    parser.add_argument("--vol-window", type=int, default=63, help="Lookback window (days) for realized vol.")
    parser.add_argument("--min-gross", type=float, default=0.5, help="Minimum gross when vol targeting is enabled.")
    parser.add_argument("--max-gross", type=float, default=1.2, help="Maximum gross when vol targeting is enabled.")
    args = parser.parse_args()

    model, model_feature_cols = load_model(args.alpha_name)
    df, data_feature_cols = prepare_data(args.features_file)

    # Use intersection of model features and data; fill missing with zeros
    feature_cols = model_feature_cols
    test_df = split_by_date(df)

    results = backtest(
        model,
        feature_cols,
        test_df,
        top_k=args.top_k,
        gross_cap=args.gross_cap,
        max_weight=args.max_weight,
        cost_bps=args.cost_bps,
        sector_cap=args.sector_cap,
        target_vol=args.target_vol,
        vol_window=args.vol_window,
        min_gross=args.min_gross,
        max_gross=args.max_gross,
    )

    ensemble_dir = os.path.join(PROJECT_ROOT, "alphas", "_ensembles", f"{args.alpha_name}_xgb_simple")
    os.makedirs(ensemble_dir, exist_ok=True)

    pd.DataFrame(results["transactions"]).to_csv(os.path.join(ensemble_dir, "transactions.csv"), index=False)
    out = {
        "cagr": results["cagr"],
        "final_net_worth": results["final_net_worth"],
        "num_days": len(results["dates"]),
        "alpha": f"{args.alpha_name}_xgb",
        "top_k": args.top_k,
        "gross_cap": args.gross_cap,
        "max_weight": args.max_weight,
        "cost_bps": args.cost_bps,
        "sector_cap": args.sector_cap,
        "target_vol": args.target_vol,
        "vol_window": args.vol_window,
    }
    with open(os.path.join(ensemble_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"Backtest complete. CAGR: {results['cagr']:.2%}, Final Net Worth: ${results['final_net_worth']:,.2f}")
    print(f"Results saved to {ensemble_dir}")


if __name__ == "__main__":
    main()
