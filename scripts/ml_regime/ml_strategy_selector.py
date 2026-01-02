import argparse
import csv
import itertools
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import list_strategy_names, load_strategies

_WORK_DF = None
_WORK_REGIME_TABLE = None
_WORK_STRATEGIES_BY_NAME = None
_WORK_STRATEGIES = None
_WORK_PRED_BY_DATE = None
_WORK_HYBRID_START = None
_WORK_HYBRID_END = None


def _get_cmap(name: str, count: int):
    try:
        cmap = plt.colormaps.get_cmap(name)
        if hasattr(cmap, "resampled"):
            return cmap.resampled(count)
        return cmap
    except Exception:
        return plt.get_cmap(name, count)


def _init_mapping_worker(data_path, strategy_names, strategy_roots, pred_by_date, hybrid_start, hybrid_end):
    global _WORK_DF
    global _WORK_REGIME_TABLE
    global _WORK_STRATEGIES_BY_NAME
    global _WORK_STRATEGIES
    global _WORK_PRED_BY_DATE
    global _WORK_HYBRID_START
    global _WORK_HYBRID_END

    df = pd.read_parquet(data_path)
    regime_table = compute_market_regime_table(df)
    strategies = load_strategies(strategy_names, strategy_roots)
    strategies_by_name = {strategy.name: strategy for strategy in strategies}

    _WORK_DF = df
    _WORK_REGIME_TABLE = regime_table
    _WORK_STRATEGIES = strategies
    _WORK_STRATEGIES_BY_NAME = strategies_by_name
    _WORK_PRED_BY_DATE = pred_by_date
    _WORK_HYBRID_START = hybrid_start
    _WORK_HYBRID_END = hybrid_end


def _append_summary_row(summary_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _apply_predicted_state_worker(state, predicted: str) -> None:
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


def _evaluate_mapping_worker(args):
    regime_labels, combo = args
    mapping = dict(zip(regime_labels, combo))

    def selector(current_date, state, _available):
        predicted = _WORK_PRED_BY_DATE.get(current_date)
        if not predicted:
            return None
        _apply_predicted_state_worker(state, predicted)
        chosen = mapping.get(predicted)
        if not chosen:
            return None
        strategy = _WORK_STRATEGIES_BY_NAME.get(chosen)
        return [strategy] if strategy else None

    backtester = RuleBasedBacktester(
        _WORK_DF,
        _WORK_STRATEGIES,
        regime_table=_WORK_REGIME_TABLE,
        strategy_selector=selector,
    )
    result = backtester.run(start_date=_WORK_HYBRID_START, end_date=_WORK_HYBRID_END)
    return result.metrics.get("CAGR", 0.0), mapping


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
    parser.add_argument("--jobs", type=int, default=1, help="CPU threads for model training (1 = serial).")
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip XGBoost training and use regime labels directly (optionally confirm-smoothed).",
    )
    parser.add_argument(
        "--eval-full-period",
        action="store_true",
        help="Evaluate hybrid backtest (and mapping search) over the full filtered period.",
    )
    parser.add_argument(
        "--mapping-search",
        action="store_true",
        help="Exhaustively map one strategy per regime and pick the best CAGR.",
    )
    parser.add_argument(
        "--mapping-allow-reuse",
        action="store_true",
        help="Allow strategies to repeat across regimes during mapping search.",
    )
    parser.add_argument(
        "--mapping-max",
        type=int,
        default=None,
        help="Optional cap on the number of mappings to evaluate.",
    )
    parser.add_argument(
        "--mapping-jobs",
        type=int,
        default=1,
        help="Parallel workers for mapping search (1 = serial).",
    )
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
    features_full = features
    labels_full = labels

    if args.skip_ml:
        train_idx = features.index
        test_idx = features.index
    else:
        train_idx, test_idx = _split_by_ratio(features.index, args.train_ratio)
    if len(test_idx) == 0:
        raise RuntimeError("Train/test split produced an empty test set.")

    label_names = sorted(labels.unique())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}

    y = labels.map(label_to_id).astype(int)
    X_train, y_train = features.loc[train_idx], y.loc[train_idx]
    X_test = features.loc[test_idx]
    X_full = features_full

    jobs = max(1, int(args.jobs))
    train_label_ids = sorted(y_train.unique().tolist())
    model_trained = False
    pred_probs_test = None
    if args.skip_ml:
        pred_labels = labels.loc[test_idx]
        pred_labels_full = labels_full
    else:
        if len(train_label_ids) < 2:
            only_id = int(train_label_ids[0]) if train_label_ids else 0
            pred_ids_test = np.full(len(X_test), only_id, dtype=int)
            pred_ids_full = np.full(len(X_full), only_id, dtype=int)
            pred_probs_test = np.zeros((len(X_test), len(label_names)), dtype=float)
            if len(label_names) > 0:
                pred_probs_test[:, only_id] = 1.0
        else:
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
                n_jobs=jobs,
            )
            model.fit(X_train, y_train)
            model_trained = True
            pred_ids_test = model.predict(X_test)
            pred_probs_test = model.predict_proba(X_test)
            pred_ids_full = model.predict(X_full)
        pred_labels = pd.Series(pred_ids_test, index=X_test.index).map(id_to_label)
        pred_labels_full = pd.Series(pred_ids_full, index=X_full.index).map(id_to_label)

    confirm_days = max(1, args.confirm_days)
    held_pred_labels = _apply_confirmed_switch(pred_labels, confirm_days)
    held_pred_labels_full = _apply_confirmed_switch(pred_labels_full, confirm_days)

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
        if decision_mask.any() and pred_probs_test is not None:
            topk_ids = np.argsort(pred_probs_test[decision_mask.values], axis=1)[:, -kk:]
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
        "skip_ml": bool(args.skip_ml),
        "data_start": features_full.index.min().strftime("%Y-%m-%d"),
        "data_end": features_full.index.max().strftime("%Y-%m-%d"),
        "test_start": test_idx.min().strftime("%Y-%m-%d"),
        "test_end": test_idx.max().strftime("%Y-%m-%d"),
        "eval_full_period": bool(args.eval_full_period),
        "model_trained": model_trained,
        "train_label_count": len(train_label_ids),
        "train_labels": [id_to_label[idx] for idx in train_label_ids],
        "accuracy": accuracy,
        "daily_accuracy": float((pred_labels == actual_labels).mean()),
        "majority_accuracy": majority_acc,
        "random_accuracy": random_acc,
        **topk_results,
        "note": "Supervised classifier predicts regime labels; accuracy uses confirmed switches.",
    }
    if args.skip_ml:
        results["note"] = "ML skipped; regime labels used directly (confirm-smoothed)."
    results["mapping_search"] = bool(args.mapping_search)
    results["mapping_jobs"] = max(1, int(args.mapping_jobs))

    pred_by_date = held_pred_labels_full.copy() if args.eval_full_period else held_pred_labels.copy()
    strategies_by_name = {strategy.name: strategy for strategy in strategies}

    def _select_for_regime(regime_label, available):
        active = [s for s in available if not s.regime_tags or regime_label in s.regime_tags]
        return active if active else available

    def _apply_predicted_state(state: dict, predicted: str) -> None:
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

    def _run_hybrid(mapping: dict[str, str] | None):
        def selector(current_date, state, _available):
            predicted = pred_by_date.get(current_date)
            if not predicted:
                return None
            _apply_predicted_state(state, predicted)
            if mapping is None:
                return _select_for_regime(predicted, list(strategies_by_name.values()))
            chosen = mapping.get(predicted)
            if not chosen:
                return None
            strategy = strategies_by_name.get(chosen)
            return [strategy] if strategy else None

        hybrid_backtester = RuleBasedBacktester(
            df,
            strategies,
            regime_table=regime_table,
            strategy_selector=selector,
        )
        return hybrid_backtester.run(start_date=hybrid_start, end_date=hybrid_end)

    if args.eval_full_period:
        hybrid_start = features_full.index.min()
        hybrid_end = features_full.index.max() + pd.Timedelta(days=1)
    else:
        hybrid_start = test_idx.min()
        hybrid_end = test_idx.max() + pd.Timedelta(days=1)
    best_mapping = None
    best_mapping_cagr = None
    mapping_total = None
    mappings_evaluated = None

    if args.mapping_search:
        regime_labels = list(label_names)
        strategy_names = [s.name for s in strategies]
        if not args.mapping_allow_reuse and len(strategy_names) < len(regime_labels):
            raise ValueError(
                "Not enough strategies for unique mapping per regime. "
                f"Strategies={len(strategy_names)}, Regimes={len(regime_labels)}."
            )
        if args.mapping_allow_reuse:
            mapping_iter = itertools.product(strategy_names, repeat=len(regime_labels))
            mapping_total = len(strategy_names) ** len(regime_labels)
        else:
            mapping_iter = itertools.permutations(strategy_names, len(regime_labels))
            mapping_total = math.perm(len(strategy_names), len(regime_labels))

        if args.mapping_max and mapping_total and args.mapping_max < mapping_total:
            mapping_iter = itertools.islice(mapping_iter, args.mapping_max)
            mapping_total = args.mapping_max

        best_metrics = None
        mappings_evaluated = 0
        progress_interval = max(1, min(500, mapping_total // 20)) if mapping_total else 1

        mapping_jobs = max(1, int(args.mapping_jobs))
        if mapping_jobs == 1:
            for combo in mapping_iter:
                mapping = dict(zip(regime_labels, combo))
                result = _run_hybrid(mapping)
                mappings_evaluated += 1
                cagr = result.metrics.get("CAGR", 0.0)
                if best_mapping_cagr is None or cagr > best_mapping_cagr:
                    best_mapping_cagr = cagr
                    best_mapping = mapping
                    best_metrics = result
                if mappings_evaluated % progress_interval == 0 or mappings_evaluated == mapping_total:
                    print(f"Mapping search: {mappings_evaluated}/{mapping_total} evaluated")
        else:
            with ProcessPoolExecutor(
                max_workers=mapping_jobs,
                initializer=_init_mapping_worker,
                initargs=(
                    data_path,
                    [s.name for s in strategies],
                    strategy_roots,
                    pred_by_date,
                    hybrid_start,
                    hybrid_end,
                ),
            ) as executor:
                args_iter = ((regime_labels, combo) for combo in mapping_iter)
                for cagr, mapping in executor.map(_evaluate_mapping_worker, args_iter, chunksize=1):
                    mappings_evaluated += 1
                    if best_mapping_cagr is None or cagr > best_mapping_cagr:
                        best_mapping_cagr = cagr
                        best_mapping = mapping
                    if mappings_evaluated % progress_interval == 0 or mappings_evaluated == mapping_total:
                        print(f"Mapping search: {mappings_evaluated}/{mapping_total} evaluated")

        if best_mapping is None:
            raise RuntimeError("Mapping search did not produce any results.")
        if best_metrics is None:
            hybrid_result = _run_hybrid(best_mapping)
        else:
            hybrid_result = best_metrics
        results["mapping_search"] = True
        results["mapping_allow_reuse"] = args.mapping_allow_reuse
        results["mapping_total"] = mapping_total
        results["mappings_evaluated"] = mappings_evaluated
        results["best_mapping"] = best_mapping
    else:
        hybrid_result = _run_hybrid(None)

    with open(os.path.join(output_dir, "ml_regime_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.eval_full_period:
        out_df = pd.DataFrame(
            {"predicted_regime": held_pred_labels_full, "actual_regime": labels_full}
        )
    else:
        out_df = pd.DataFrame(
            {"predicted_regime": held_pred_labels, "actual_regime": labels.loc[test_idx]}
        )
    if best_mapping:
        out_df["mapped_strategy"] = out_df["predicted_regime"].map(best_mapping)
    out_df.to_csv(os.path.join(output_dir, "ml_regime_daily.csv"))

    if hybrid_result.transactions:
        pd.DataFrame(hybrid_result.transactions).to_csv(
            os.path.join(output_dir, "transactions.csv"), index=False
        )
    if hybrid_result.equity_curve and hybrid_result.dates:
        dates = pd.to_datetime(hybrid_result.dates)
        equity = np.array(hybrid_result.equity_curve, dtype=float)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, equity, color="black", linewidth=1.2)

        if pred_by_date is not None:
            regimes = pred_by_date.reindex(dates).astype("object")
            regimes = regimes.fillna("unknown")
            unique_labels = [str(x) for x in pd.Series(regimes.unique()).tolist()]
            known_order = ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "unknown"]
            ordered = [label for label in known_order if label in unique_labels]
            ordered += [label for label in sorted(unique_labels) if label not in ordered]

            color_map = {
                "bull_low_vol": "green",
                "bull_high_vol": "gold",
                "bear_high_vol": "red",
                "bear_low_vol": "orange",
                "unknown": "gray",
            }
            cmap = _get_cmap("tab20", max(3, len(ordered)))
            denom = max(1, len(ordered) - 1)
            y_min, y_max = np.nanmin(equity), np.nanmax(equity)
            for idx, label in enumerate(ordered):
                mask = regimes.eq(label).fillna(False)
                if not mask.any():
                    continue
                color = color_map.get(label, cmap(idx / denom))
                ax.fill_between(
                    dates,
                    y_min,
                    y_max,
                    where=mask.to_numpy(),
                    color=color,
                    alpha=0.18,
                    label=label,
                )
            ax.legend(loc="upper left", ncol=2)

        ax.set_title("Hybrid Backtest Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Worth ($)")
        ax.grid(True)
        plot_path = os.path.join(output_dir, "performance.png")
        plt.savefig(plot_path)
        print(f"Performance plot saved to {plot_path}")
    hybrid_payload = {
        "cagr": hybrid_result.metrics.get("CAGR", 0.0) / 100.0,
        "final_net_worth": hybrid_result.metrics.get("final_net_worth", config.INITIAL_CAPITAL),
        "metrics": hybrid_result.metrics,
        "num_strategies": len(strategies),
        "strategies": [s.name for s in strategies],
        "start_date": hybrid_start.strftime("%Y-%m-%d") if isinstance(hybrid_start, pd.Timestamp) else None,
        "end_date": hybrid_end.strftime("%Y-%m-%d") if isinstance(hybrid_end, pd.Timestamp) else None,
        "data_start": features_full.index.min().strftime("%Y-%m-%d"),
        "data_end": features_full.index.max().strftime("%Y-%m-%d"),
        "test_start": test_idx.min().strftime("%Y-%m-%d"),
        "test_end": test_idx.max().strftime("%Y-%m-%d"),
        "eval_full_period": bool(args.eval_full_period),
        "note": "Hybrid backtest uses ML-predicted regimes to select strategies with full trade simulation.",
    }
    if best_mapping:
        hybrid_payload["best_mapping"] = best_mapping
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(hybrid_payload, f, indent=2)

    eval_days = len(hybrid_result.dates) if hybrid_result.dates else 0
    eval_years = eval_days / 252.0 if eval_days else 0.0
    switch_count = None
    switch_rate = None
    unique_regimes = None
    if pred_by_date is not None and hybrid_result.dates:
        date_index = pd.to_datetime(hybrid_result.dates)
        regime_series = pred_by_date.reindex(date_index)
        if not regime_series.empty:
            switch_mask = regime_series.ne(regime_series.shift(1))
            switch_count = max(0, int(switch_mask.sum()) - 1)
            switch_rate = switch_count / max(len(regime_series), 1)
            unique_regimes = int(regime_series.nunique(dropna=True))

    summary_path = os.path.join(PROJECT_ROOT, args.output_root, "ml_regime", "summary.csv")
    summary_row = {
        "run_id": os.path.basename(output_dir),
        "output_dir": output_dir,
        "timestamp_utc": stamp,
        "command": " ".join(sys.argv),
        "trading_region": config.TRADING_REGION,
        "data_file": config.DATA_FILE,
        "universe_filter": config.UNIVERSE_FILTER,
        "regime_mode": config.REGIME_MODE,
        "hmm_n_components": config.HMM_N_COMPONENTS,
        "hmm_state_labels": config.HMM_STATE_LABELS,
        "hmm_warmup_period": config.HMM_WARMUP_PERIOD,
        "hmm_step_size": config.HMM_STEP_SIZE,
        "hmm_covariance_type": config.HMM_COVARIANCE_TYPE,
        "hmm_min_covar": config.HMM_MIN_COVAR,
        "start_date_arg": args.start_date,
        "end_date_arg": args.end_date,
        "data_start": results.get("data_start"),
        "data_end": results.get("data_end"),
        "test_start": results.get("test_start"),
        "test_end": results.get("test_end"),
        "eval_full_period": results.get("eval_full_period"),
        "train_ratio": args.train_ratio,
        "confirm_days": confirm_days,
        "skip_ml": results.get("skip_ml"),
        "jobs": args.jobs,
        "mapping_search": results.get("mapping_search"),
        "mapping_allow_reuse": results.get("mapping_allow_reuse"),
        "mapping_max": args.mapping_max,
        "mapping_jobs": args.mapping_jobs,
        "mapping_total": results.get("mapping_total"),
        "mappings_evaluated": results.get("mappings_evaluated"),
        "model_trained": results.get("model_trained"),
        "train_label_count": results.get("train_label_count"),
        "train_labels": json.dumps(results.get("train_labels")),
        "regime_labels": json.dumps(results.get("regime_labels")),
        "accuracy": results.get("accuracy"),
        "daily_accuracy": results.get("daily_accuracy"),
        "majority_accuracy": results.get("majority_accuracy"),
        "random_accuracy": results.get("random_accuracy"),
        "top_2_accuracy": results.get("top_2_accuracy"),
        "top_3_accuracy": results.get("top_3_accuracy"),
        "hybrid_start": hybrid_payload.get("start_date"),
        "hybrid_end": hybrid_payload.get("end_date"),
        "eval_days": eval_days,
        "eval_years": eval_years,
        "switch_count": switch_count,
        "switch_rate": switch_rate,
        "unique_regimes": unique_regimes,
        "cagr_pct": hybrid_payload.get("metrics", {}).get("CAGR"),
        "sharpe": hybrid_payload.get("metrics", {}).get("Sharpe Ratio"),
        "max_drawdown": hybrid_payload.get("metrics", {}).get("Max Drawdown"),
        "final_net_worth": hybrid_payload.get("final_net_worth"),
        "num_strategies": hybrid_payload.get("num_strategies"),
        "strategies": json.dumps(hybrid_payload.get("strategies")),
        "best_mapping": json.dumps(hybrid_payload.get("best_mapping")),
        "note": results.get("note"),
    }
    _append_summary_row(summary_path, summary_row)

    print(f"Results saved to {output_dir}")
    print(f"Decision Accuracy (confirm): {results['accuracy']:.2%}")
    print(f"Daily Accuracy: {results['daily_accuracy']:.2%}")
    print(f"Hybrid CAGR: {hybrid_result.metrics.get('CAGR', 0.0):.2f}%")


if __name__ == "__main__":
    main()
