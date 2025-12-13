import argparse
import importlib.util
import json
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys

sys.path.append(PROJECT_ROOT)
from src import config


def load_feature_engineer(path: str):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("candidate.feature_engineering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.feature_engineering


def prepare_data(features_file: str | None) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, config.DATA_FILE))
    if features_file:
        fe_fn = load_feature_engineer(features_file)
        df = fe_fn(df)
    feature_cols = [c for c in df.columns if c.endswith("_z")]
    df["fwd_return"] = df.groupby(level="ticker")["log_return"].shift(-1)
    df = df.dropna(subset=["fwd_return"])
    return df, feature_cols


def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = df.index.get_level_values("date").unique().sort_values()
    split_idx = int(len(dates) * config.TRAIN_RATIO)
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]
    train_df = df[df.index.get_level_values("date").isin(train_dates)]
    test_df = df[df.index.get_level_values("date").isin(test_dates)]
    return train_df, test_df


def train_xgb(train_df: pd.DataFrame, feature_cols: list[str]):
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = train_df["fwd_return"].to_numpy(dtype=np.float32)
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=config.SEED,
        n_jobs=4,
    )
    model.fit(X, y)
    return model


def evaluate(model, test_df: pd.DataFrame, feature_cols: list[str]):
    if test_df.empty:
        return {}
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_true = test_df["fwd_return"].to_numpy(dtype=np.float32)
    preds = model.predict(X_test)
    mse = float(np.mean((preds - y_true) ** 2))
    return {"mse": mse}


def walkforward_eval(df: pd.DataFrame, feature_cols: List[str], n_splits: int = 3):
    """Simple rolling walk-forward: train on expanding window, test on next block."""
    dates = df.index.get_level_values("date").unique().sort_values()
    block = len(dates) // (n_splits + 1)
    results = []
    for i in range(n_splits):
        train_end_idx = block * (i + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(len(dates), val_start_idx + block)
        if val_end_idx - val_start_idx < 50:
            break
        train_dates = dates[:train_end_idx]
        val_dates = dates[val_start_idx:val_end_idx]
        train_df = df[df.index.get_level_values("date").isin(train_dates)]
        val_df = df[df.index.get_level_values("date").isin(val_dates)]
        model = train_xgb(train_df, feature_cols)
        metrics = evaluate(model, val_df, feature_cols)
        results.append({
            "fold": i,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "mse": metrics.get("mse", None)
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Train an XGBoost cross-sectional return model.")
    parser.add_argument("--alpha-name", required=True, help="Name for the saved XGBoost alpha.")
    parser.add_argument("--features-file", help="Optional path to feature_engineering.py for this alpha.")
    parser.add_argument("--description", default="", help="Description of the alpha.")
    parser.add_argument("--walkforward-folds", type=int, default=0, help="Number of rolling folds for walk-forward validation (0 to skip).")
    args = parser.parse_args()

    df, feature_cols = prepare_data(args.features_file)
    train_df, test_df = split_by_date(df)

    wf_results = walkforward_eval(df, feature_cols, n_splits=args.walkforward_folds) if args.walkforward_folds > 0 else []
    model = train_xgb(train_df, feature_cols)
    metrics = evaluate(model, test_df, feature_cols)

    output_dir = os.path.join(PROJECT_ROOT, "alphas", f"{args.alpha_name}_xgb")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.json")
    model.save_model(model_path)

    with open(os.path.join(output_dir, "feature_keys.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    results = {
        "description": args.description,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "walkforward": wf_results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved XGBoost alpha to {output_dir}")
    if metrics:
        print(f"Test MSE: {metrics['mse']:.6f}")


if __name__ == "__main__":
    main()
