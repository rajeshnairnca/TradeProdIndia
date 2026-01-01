import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import list_strategy_names, load_strategies


def _build_features(df: pd.DataFrame, regime_table: pd.DataFrame) -> pd.DataFrame:
    market_close = df.groupby(level="date")["Close"].mean()
    market_return = market_close.pct_change()
    market_vol = market_return.rolling(
        window=config.ROLLING_WINDOW_FOR_VOL,
        min_periods=config.ROLLING_WINDOW_FOR_VOL,
    ).std()
    features = pd.DataFrame(
        {
            "market_return": market_return,
            "market_vol": market_vol,
            "breadth": regime_table["breadth"],
            "dispersion": regime_table["dispersion"],
            "trend_up": regime_table["trend_up"].astype(float),
            "vol_high": regime_table["vol_high"].astype(float),
        }
    )
    return features.sort_index()


def _apply_confirmed_switch(pred_labels: pd.Series, confirm_days: int) -> pd.Series:
    confirm_days = max(1, confirm_days)
    if pred_labels.empty or confirm_days <= 1:
        return pred_labels
    held = pd.Series(index=pred_labels.index, dtype=object)
    current = None
    candidate = None
    streak = 0
    for i, value in enumerate(pred_labels.to_numpy()):
        if current is None:
            current = value
            held.iloc[i] = current
            continue
        if value == current:
            candidate = None
            streak = 0
            held.iloc[i] = current
            continue
        if candidate == value:
            streak += 1
        else:
            candidate = value
            streak = 1
        if streak >= confirm_days:
            current = candidate
            candidate = None
            streak = 0
        held.iloc[i] = current
    return held


def _split_by_ratio(index: pd.Index, train_ratio: float) -> tuple[pd.Index, pd.Index]:
    split_idx = int(len(index) * train_ratio)
    train_idx = index[:split_idx]
    test_idx = index[split_idx:]
    return train_idx, test_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised ML classifier to predict market regimes."
    )
    parser.add_argument("--strategies", nargs="+", help="List of strategy names to include.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root directories for strategies.")
    parser.add_argument("--start-date", type=str, help="Optional YYYY-MM-DD start date.")
    parser.add_argument("--end-date", type=str, help="Optional YYYY-MM-DD end date (exclusive).")
    parser.add_argument("--train-ratio", type=float, default=config.TRAIN_RATIO, help="Train split ratio.")
    parser.add_argument("--confirm-days", type=int, default=10, help="Required consecutive days before switching.")
    parser.add_argument("--output-root", default="runs", help="Root directory for outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy_roots = args.strategy_roots or ["alphas"]
    strategy_names = args.strategies or list_strategy_names(strategy_roots)
    if not strategy_names:
        raise ValueError("No strategies found to train.")

    strategies = load_strategies(strategy_names, strategy_roots)
    if not strategies:
        raise ValueError("No valid strategies loaded.")

    data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    df = pd.read_parquet(data_path)

    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    regime_table = compute_market_regime_table(df)
    features = _build_features(df, regime_table)
    labels = regime_table["regime_label"].copy()

    if start_date is not None:
        features = features[features.index >= start_date]
        labels = labels[labels.index >= start_date]
    if end_date is not None:
        features = features[features.index < end_date]
        labels = labels[labels.index < end_date]

    common_index = features.index.intersection(labels.index)
    features = features.loc[common_index].dropna()
    labels = labels.loc[features.index]

    train_idx, test_idx = _split_by_ratio(features.index, args.train_ratio)
    if len(test_idx) == 0:
        raise RuntimeError("Train/test split produced an empty test set.")

    label_names = sorted(labels.unique())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}

    y = labels.map(label_to_id).astype(int)
    X_train, y_train = features.loc[train_idx], y.loc[train_idx]
    X_test = features.loc[test_idx]

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(label_names),
        eval_metric="mlogloss",
        random_state=config.SEED,
    )
    model.fit(X_train, y_train)

    pred_ids = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)
    pred_labels = pd.Series(pred_ids, index=X_test.index).map(id_to_label)
    confirm_days = max(1, args.confirm_days)
    held_pred_labels = _apply_confirmed_switch(pred_labels, confirm_days)

    actual_labels = labels.loc[test_idx]
    decision_mask = held_pred_labels.ne(held_pred_labels.shift(1))
    decision_actual = actual_labels[decision_mask]
    decision_pred = held_pred_labels[decision_mask]
    accuracy = float((decision_pred == decision_actual).mean())
    majority_acc = float(decision_actual.value_counts(normalize=True).iloc[0]) if not decision_actual.empty else 0.0
    random_acc = 1.0 / max(len(label_names), 1)

    topk_results = {}
    for k in (2, 3):
        kk = min(k, len(label_names))
        if decision_mask.any():
            topk_ids = np.argsort(pred_probs[decision_mask.values], axis=1)[:, -kk:]
            true_ids = decision_actual.map(label_to_id).to_numpy()
            hit = np.any(topk_ids == true_ids[:, None], axis=1)
            topk_results[f"top_{kk}_accuracy"] = float(hit.mean())
        else:
            topk_results[f"top_{kk}_accuracy"] = 0.0

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, args.output_root, "ml_regime", stamp)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "regime_labels": label_names,
        "train_ratio": args.train_ratio,
        "confirm_days": confirm_days,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "accuracy": accuracy,
        "daily_accuracy": float((pred_labels == actual_labels).mean()),
        "majority_accuracy": majority_acc,
        "random_accuracy": random_acc,
        **topk_results,
        "note": "Supervised classifier predicts regime labels; accuracy uses confirmed switches.",
    }
    with open(os.path.join(output_dir, "ml_regime_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    out_df = pd.DataFrame(
        {
            "predicted_regime": held_pred_labels,
            "actual_regime": labels.loc[test_idx],
        }
    )
    out_df.to_csv(os.path.join(output_dir, "ml_regime_daily.csv"))

    pred_by_date = held_pred_labels.copy()
    strategies_by_name = {strategy.name: strategy for strategy in strategies}

    def _select_for_regime(regime_label, available):
        active = [s for s in available if not s.regime_tags or regime_label in s.regime_tags]
        return active if active else available

    def selector(current_date, state, _available):
        predicted = pred_by_date.get(current_date)
        if not predicted:
            return None
        state["regime_label"] = predicted
        if predicted == "bull_low_vol":
            state["trend_up"] = True
            state["vol_high"] = False
        elif predicted == "bull_high_vol":
            state["trend_up"] = True
            state["vol_high"] = True
        elif predicted == "bear_high_vol":
            state["trend_up"] = False
            state["vol_high"] = True
        elif predicted == "bear_low_vol":
            state["trend_up"] = False
            state["vol_high"] = False
        return _select_for_regime(predicted, list(strategies_by_name.values()))

    hybrid_start = test_idx.min()
    hybrid_end = test_idx.max() + pd.Timedelta(days=1)
    hybrid_backtester = RuleBasedBacktester(
        df,
        strategies,
        regime_table=regime_table,
        strategy_selector=selector,
    )
    hybrid_result = hybrid_backtester.run(start_date=hybrid_start, end_date=hybrid_end)
    if hybrid_result.transactions:
        pd.DataFrame(hybrid_result.transactions).to_csv(
            os.path.join(output_dir, "transactions.csv"), index=False
        )
    hybrid_payload = {
        "cagr": hybrid_result.metrics.get("CAGR", 0.0) / 100.0,
        "final_net_worth": hybrid_result.metrics.get("final_net_worth", config.INITIAL_CAPITAL),
        "metrics": hybrid_result.metrics,
        "num_strategies": len(strategies),
        "strategies": [s.name for s in strategies],
        "start_date": hybrid_start.strftime("%Y-%m-%d") if isinstance(hybrid_start, pd.Timestamp) else None,
        "end_date": hybrid_end.strftime("%Y-%m-%d") if isinstance(hybrid_end, pd.Timestamp) else None,
        "note": "Hybrid backtest uses ML-predicted regimes to select strategies with full trade simulation.",
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(hybrid_payload, f, indent=2)

    print(f"Results saved to {output_dir}")
    print(f"Decision Accuracy (confirm): {results['accuracy']:.2%}")
    print(f"Daily Accuracy: {results['daily_accuracy']:.2%}")
    print(f"Hybrid CAGR: {hybrid_result.metrics.get('CAGR', 0.0):.2f}%")


if __name__ == "__main__":
    main()
