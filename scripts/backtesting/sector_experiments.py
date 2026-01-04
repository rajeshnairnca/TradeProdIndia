import argparse
import csv
import itertools
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.regime import compute_market_regime_table
from src.rule_backtester import RuleBasedBacktester
from src.strategy import list_strategy_names, load_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sector-aware backtest experiments.")
    parser.add_argument("--strategies", nargs="+", help="List of strategy names to include.")
    parser.add_argument("--strategy-roots", action="append", default=[], help="Root directories for strategies.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD start date.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional YYYY-MM-DD end date (exclusive).")
    parser.add_argument(
        "--sectors",
        type=str,
        default=None,
        help="Comma-separated sectors to include (default: all except 'unknown').",
    )
    parser.add_argument(
        "--min-tickers",
        type=int,
        default=10,
        help="Minimum tickers required to include a sector.",
    )
    parser.add_argument(
        "--regime-scope",
        choices=("global", "sector", "both"),
        default="both",
        help="Use global regime, sector regime, or both.",
    )
    parser.add_argument(
        "--best-single-strategy",
        action="store_true",
        help="Evaluate each strategy per sector/scope and keep the best by CAGR.",
    )
    parser.add_argument(
        "--regime-mapping-search",
        action="store_true",
        help="Search for the best strategy-per-regime mapping per sector/scope.",
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
        help="Optional cap on mappings to evaluate per sector/scope.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers (1 = serial).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (enables resume).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing summary CSV in --output-dir.",
    )
    return parser.parse_args()


def _sector_ticker_counts(df: pd.DataFrame) -> pd.Series:
    sector_by_ticker = df["sector"].groupby(level="ticker").first()
    return sector_by_ticker.value_counts()


def _resolve_sector_list(df: pd.DataFrame, sectors_arg: str | None, min_tickers: int) -> list[str]:
    counts = _sector_ticker_counts(df)
    sectors = [s for s in counts.index if s != "unknown" and counts[s] >= min_tickers]
    if sectors_arg:
        wanted = {s.strip() for s in sectors_arg.split(",") if s.strip()}
        sectors = [s for s in sectors if s in wanted]
    return sectors


def _run_backtest(
    df: pd.DataFrame,
    strategies,
    regime_table: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> dict:
    backtester = RuleBasedBacktester(df, strategies, regime_table=regime_table)
    result = backtester.run(start_date=start_date, end_date=end_date)
    metrics = result.metrics
    return {
        "cagr_pct": metrics.get("CAGR"),
        "sharpe": metrics.get("Sharpe Ratio"),
        "max_drawdown": metrics.get("Max Drawdown"),
        "final_net_worth": metrics.get("final_net_worth"),
        "metrics": metrics,
    }


def _mapping_iter(strategy_names: list[str], regime_labels: list[str], allow_reuse: bool):
    if allow_reuse:
        return itertools.product(strategy_names, repeat=len(regime_labels))
    return itertools.permutations(strategy_names, len(regime_labels))


def _mapping_total(strategy_count: int, regime_count: int, allow_reuse: bool) -> int:
    if allow_reuse:
        return strategy_count ** regime_count
    total = 1
    for i in range(regime_count):
        total *= max(strategy_count - i, 0)
    return total


def _evaluate_sector_scope_task(payload: dict) -> list[dict]:
    data_path = payload["data_path"]
    sector = payload["sector"]
    scope = payload["scope"]
    strategy_names = payload["strategy_names"]
    strategy_roots = payload["strategy_roots"]
    mode = payload["mode"]
    mapping_allow_reuse = payload.get("mapping_allow_reuse", False)
    mapping_max = payload.get("mapping_max")
    skip_strategies = set(payload.get("skip_strategies", []))
    start_date = pd.to_datetime(payload["start_date"]) if payload["start_date"] else None
    end_date = pd.to_datetime(payload["end_date"]) if payload["end_date"] else None

    df = pd.read_parquet(data_path)
    if "sector" not in df.columns:
        raise ValueError("Sector column missing; re-run data extraction with sector enabled.")
    sector_df = df[df["sector"] == sector]
    if sector_df.empty:
        return []
    sector_tickers = sector_df.index.get_level_values("ticker").unique().tolist()

    if scope == "global":
        regime_table = compute_market_regime_table(df)
    else:
        regime_table = compute_market_regime_table(sector_df)

    strategies = load_strategies(strategy_names, strategy_roots)
    strategies_by_name = {strategy.name: strategy for strategy in strategies}

    rows = []
    if mode == "mapping":
        regime_labels = (
            pd.Series(regime_table["regime_label"])
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not regime_labels:
            return []
        if not mapping_allow_reuse and len(strategy_names) < len(regime_labels):
            raise ValueError(
                "Not enough strategies for unique mapping per regime. "
                f"Strategies={len(strategy_names)}, Regimes={len(regime_labels)}."
            )

        mapping_total = _mapping_total(len(strategy_names), len(regime_labels), mapping_allow_reuse)
        mapping_iter = _mapping_iter(strategy_names, regime_labels, mapping_allow_reuse)
        if mapping_max and mapping_total > mapping_max:
            mapping_iter = itertools.islice(mapping_iter, mapping_max)
            mapping_total = mapping_max

        best_row = None
        evaluated = 0
        for combo in mapping_iter:
            mapping = dict(zip(regime_labels, combo))

            def selector(current_date, state, _available):
                label = state.get("regime_label")
                chosen = mapping.get(str(label)) if label is not None else None
                if not chosen:
                    return None
                strategy = strategies_by_name.get(chosen)
                return [strategy] if strategy else None

            backtester = RuleBasedBacktester(
                sector_df,
                strategies,
                regime_table=regime_table,
                strategy_selector=selector,
            )
            result = backtester.run(start_date=start_date, end_date=end_date)
            evaluated += 1
            row = {
                "sector": sector,
                "regime_scope": scope,
                "tickers": len(sector_tickers),
                "start_date": payload["start_date"],
                "end_date": payload["end_date"],
                "strategy": "__mapping__",
                "mapping": json.dumps(mapping, sort_keys=True),
                "mappings_evaluated": evaluated,
                "mapping_total": mapping_total,
                "cagr_pct": result.metrics.get("CAGR"),
                "sharpe": result.metrics.get("Sharpe Ratio"),
                "max_drawdown": result.metrics.get("Max Drawdown"),
                "final_net_worth": result.metrics.get("final_net_worth"),
            }
            if best_row is None or (row["cagr_pct"] or -1e9) > (best_row["cagr_pct"] or -1e9):
                best_row = row
        if best_row:
            rows.append(best_row)
    elif mode == "best_single":
        for strategy in strategies:
            if strategy.name in skip_strategies:
                continue
            result = _run_backtest(sector_df, [strategy], regime_table, start_date, end_date)
            rows.append(
                {
                    "sector": sector,
                    "regime_scope": scope,
                    "tickers": len(sector_tickers),
                    "start_date": payload["start_date"],
                    "end_date": payload["end_date"],
                    "strategy": strategy.name,
                    "cagr_pct": result["cagr_pct"],
                    "sharpe": result["sharpe"],
                    "max_drawdown": result["max_drawdown"],
                    "final_net_worth": result["final_net_worth"],
                }
            )
    else:
        result = _run_backtest(sector_df, strategies, regime_table, start_date, end_date)
        rows.append(
            {
                "sector": sector,
                "regime_scope": scope,
                "tickers": len(sector_tickers),
                "start_date": payload["start_date"],
                "end_date": payload["end_date"],
                "strategy": "__ensemble__",
                "cagr_pct": result["cagr_pct"],
                "sharpe": result["sharpe"],
                "max_drawdown": result["max_drawdown"],
                "final_net_worth": result["final_net_worth"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.best_single_strategy and args.regime_mapping_search:
        raise ValueError("Use only one of --best-single-strategy or --regime-mapping-search.")
    if not args.strategy_roots:
        args.strategy_roots = ["alphas"]
    strategy_names = args.strategies or list_strategy_names(args.strategy_roots)
    strategies = load_strategies(strategy_names, args.strategy_roots)
    if not strategies:
        raise ValueError("No strategies loaded.")
    strategies_by_name = {strategy.name: strategy for strategy in strategies}

    data_path = os.path.join(PROJECT_ROOT, config.DATA_FILE)
    df = pd.read_parquet(data_path)
    if "sector" not in df.columns:
        raise ValueError("Sector column missing; re-run data extraction with sector enabled.")

    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    sectors = _resolve_sector_list(df, args.sectors, args.min_tickers)
    if not sectors:
        raise ValueError("No sectors matched the requested filters.")

    print(f"Sectors to evaluate: {', '.join(sectors)}")
    global_regime = None
    if args.jobs <= 1 and args.regime_scope in ("global", "both"):
        print("Computing global regime table...")
        global_regime = compute_market_regime_table(df)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, args.output_root, "sector_experiments", stamp)
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "sector_experiment_summary.csv")
    mapping_summary_path = os.path.join(output_dir, "sector_regime_mapping_summary.csv")
    summary_fieldnames = [
        "sector",
        "regime_scope",
        "tickers",
        "start_date",
        "end_date",
        "strategy",
        "cagr_pct",
        "sharpe",
        "max_drawdown",
        "final_net_worth",
        "strategies",
    ]
    mapping_fieldnames = [
        "sector",
        "regime_scope",
        "tickers",
        "start_date",
        "end_date",
        "strategy",
        "mapping",
        "mappings_evaluated",
        "mapping_total",
        "cagr_pct",
        "sharpe",
        "max_drawdown",
        "final_net_worth",
    ]
    completed = set()
    if args.resume:
        resume_path = mapping_summary_path if args.regime_mapping_search else summary_path
        if os.path.exists(resume_path):
            existing = pd.read_csv(resume_path)
            if "strategy" in existing.columns:
                completed = {
                    (row["sector"], row["regime_scope"], row["strategy"])
                    for _, row in existing.iterrows()
                }
            else:
                completed = {
                    (row["sector"], row["regime_scope"], None)
                    for _, row in existing.iterrows()
                }
    scopes = []
    if args.regime_scope in ("global", "both"):
        scopes.append("global")
    if args.regime_scope in ("sector", "both"):
        scopes.append("sector")

    if args.jobs > 1:
        mode = "mapping" if args.regime_mapping_search else "best_single" if args.best_single_strategy else "ensemble"
        completed_by_scope = {}
        if args.best_single_strategy:
            for sector, scope, strategy in completed:
                if strategy:
                    completed_by_scope.setdefault((sector, scope), set()).add(strategy)

        tasks = []
        for sector in sectors:
            for scope in scopes:
                if args.regime_mapping_search:
                    key = (sector, scope, "__mapping__")
                    if key in completed:
                        continue
                    tasks.append(
                        {
                            "data_path": data_path,
                            "sector": sector,
                            "scope": scope,
                            "strategy_names": strategy_names,
                            "strategy_roots": args.strategy_roots,
                            "mode": mode,
                            "mapping_allow_reuse": args.mapping_allow_reuse,
                            "mapping_max": args.mapping_max,
                            "start_date": args.start_date,
                            "end_date": args.end_date,
                        }
                    )
                elif args.best_single_strategy:
                    done = completed_by_scope.get((sector, scope), set())
                    if len(done) == len(strategy_names):
                        continue
                    tasks.append(
                        {
                            "data_path": data_path,
                            "sector": sector,
                            "scope": scope,
                            "strategy_names": strategy_names,
                            "strategy_roots": args.strategy_roots,
                            "mode": mode,
                            "skip_strategies": sorted(done),
                            "start_date": args.start_date,
                            "end_date": args.end_date,
                        }
                    )
                else:
                    key = (sector, scope, None)
                    if key in completed:
                        continue
                    tasks.append(
                        {
                            "data_path": data_path,
                            "sector": sector,
                            "scope": scope,
                            "strategy_names": strategy_names,
                            "strategy_roots": args.strategy_roots,
                            "mode": mode,
                            "start_date": args.start_date,
                            "end_date": args.end_date,
                        }
                    )

        if tasks:
            with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
                for rows in tqdm(
                    executor.map(_evaluate_sector_scope_task, tasks),
                    total=len(tasks),
                    desc="Sector tasks",
                ):
                    for row in rows:
                        if args.regime_mapping_search:
                            needs_header = not os.path.exists(mapping_summary_path)
                            with open(mapping_summary_path, "a", newline="") as f:
                                writer = csv.DictWriter(f, fieldnames=mapping_fieldnames)
                                if needs_header:
                                    writer.writeheader()
                                writer.writerow(row)
                        else:
                            needs_header = not os.path.exists(summary_path)
                            with open(summary_path, "a", newline="") as f:
                                writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                                if needs_header:
                                    writer.writeheader()
                                writer.writerow(row)
        else:
            print("No tasks to run (already complete).")
    else:
        for sector in sectors:
            sector_df = df[df["sector"] == sector]
            if sector_df.empty:
                continue
            sector_tickers = sector_df.index.get_level_values("ticker").unique().tolist()
            regime_tables = {}
            if args.regime_scope in ("global", "both"):
                regime_tables["global"] = global_regime
            if args.regime_scope in ("sector", "both"):
                regime_tables["sector"] = compute_market_regime_table(sector_df)

            for scope, regime_table in regime_tables.items():
                if args.regime_mapping_search:
                    key = (sector, scope, "__mapping__")
                    if key in completed:
                        continue
                    regime_labels = (
                        pd.Series(regime_table["regime_label"])
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if not regime_labels:
                        continue
                    if not args.mapping_allow_reuse and len(strategy_names) < len(regime_labels):
                        raise ValueError(
                            "Not enough strategies for unique mapping per regime. "
                            f"Strategies={len(strategy_names)}, Regimes={len(regime_labels)}."
                        )

                    mapping_total = _mapping_total(len(strategy_names), len(regime_labels), args.mapping_allow_reuse)
                    mapping_iter = _mapping_iter(strategy_names, regime_labels, args.mapping_allow_reuse)
                    if args.mapping_max and mapping_total > args.mapping_max:
                        mapping_iter = itertools.islice(mapping_iter, args.mapping_max)
                        mapping_total = args.mapping_max

                    print(
                        f"Running {sector} ({scope}) mapping search "
                        f"[regimes={len(regime_labels)}, mappings={mapping_total}]..."
                    )

                    best_row = None
                    evaluated = 0
                    for combo in tqdm(
                        mapping_iter,
                        total=mapping_total,
                        desc=f"{sector} {scope} mappings",
                    ):
                        mapping = dict(zip(regime_labels, combo))

                        def selector(current_date, state, _available):
                            label = state.get("regime_label")
                            chosen = mapping.get(str(label)) if label is not None else None
                            if not chosen:
                                return None
                            strategy = strategies_by_name.get(chosen)
                            return [strategy] if strategy else None

                        backtester = RuleBasedBacktester(
                            sector_df,
                            strategies,
                            regime_table=regime_table,
                            strategy_selector=selector,
                        )
                        result = backtester.run(start_date=start_date, end_date=end_date)
                        evaluated += 1
                        row = {
                            "sector": sector,
                            "regime_scope": scope,
                            "tickers": len(sector_tickers),
                            "start_date": args.start_date,
                            "end_date": args.end_date,
                            "strategy": "__mapping__",
                            "mapping": json.dumps(mapping, sort_keys=True),
                            "mappings_evaluated": evaluated,
                            "mapping_total": mapping_total,
                            "cagr_pct": result.metrics.get("CAGR"),
                            "sharpe": result.metrics.get("Sharpe Ratio"),
                            "max_drawdown": result.metrics.get("Max Drawdown"),
                            "final_net_worth": result.metrics.get("final_net_worth"),
                        }
                        if best_row is None or (row["cagr_pct"] or -1e9) > (best_row["cagr_pct"] or -1e9):
                            best_row = row

                    if best_row:
                        needs_header = not os.path.exists(mapping_summary_path)
                        with open(mapping_summary_path, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=mapping_fieldnames)
                            if needs_header:
                                writer.writeheader()
                            writer.writerow(best_row)
                        completed.add(key)
                elif args.best_single_strategy:
                    for strategy in strategies:
                        key = (sector, scope, strategy.name)
                        if key in completed:
                            continue
                        print(
                            f"Running {sector} ({scope}) | {strategy.name} "
                            f"with {len(sector_tickers)} tickers..."
                        )
                        result = _run_backtest(sector_df, [strategy], regime_table, start_date, end_date)
                        row = {
                            "sector": sector,
                            "regime_scope": scope,
                            "tickers": len(sector_tickers),
                            "start_date": args.start_date,
                            "end_date": args.end_date,
                            "strategy": strategy.name,
                            "cagr_pct": result["cagr_pct"],
                            "sharpe": result["sharpe"],
                            "max_drawdown": result["max_drawdown"],
                            "final_net_worth": result["final_net_worth"],
                        }
                        needs_header = not os.path.exists(summary_path)
                        with open(summary_path, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                            if needs_header:
                                writer.writeheader()
                            writer.writerow(row)
                        completed.add(key)
                else:
                    key = (sector, scope, None)
                    if key in completed:
                        continue
                    print(f"Running {sector} ({scope}) with {len(sector_tickers)} tickers...")
                    result = _run_backtest(sector_df, strategies, regime_table, start_date, end_date)
                    row = {
                        "sector": sector,
                        "regime_scope": scope,
                        "tickers": len(sector_tickers),
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "cagr_pct": result["cagr_pct"],
                        "sharpe": result["sharpe"],
                        "max_drawdown": result["max_drawdown"],
                        "final_net_worth": result["final_net_worth"],
                        "strategies": [s.name for s in strategies],
                    }
                    needs_header = not os.path.exists(summary_path)
                    with open(summary_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                        if needs_header:
                            writer.writeheader()
                        writer.writerow(row)
                    completed.add(key)

    if args.regime_mapping_search and os.path.exists(mapping_summary_path):
        summary = pd.read_csv(mapping_summary_path)
        with open(os.path.join(output_dir, "sector_regime_mapping_summary.json"), "w") as f:
            json.dump(summary.to_dict(orient="records"), f, indent=2)
        top = summary.sort_values(by="cagr_pct", ascending=False).head(10)
        print("\nTop regime-mapping results by CAGR:")
        print(top[["sector", "regime_scope", "cagr_pct", "sharpe", "max_drawdown"]].to_string(index=False))
        print(f"\nSummary saved to {mapping_summary_path}")
    elif os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        with open(os.path.join(output_dir, "sector_experiment_summary.json"), "w") as f:
            json.dump(summary.to_dict(orient="records"), f, indent=2)

        if args.best_single_strategy:
            best_idx = summary.groupby(["sector", "regime_scope"])["cagr_pct"].idxmax()
            best_df = summary.loc[best_idx].copy()
            best_df = best_df.rename(columns={"strategy": "best_strategy"})
            best_path = os.path.join(output_dir, "sector_best_single_strategy.csv")
            best_df.to_csv(best_path, index=False)
            print(f"\nBest single-strategy summary saved to {best_path}")
            top = best_df.sort_values(by="cagr_pct", ascending=False).head(10)
            print("\nTop best-single results by CAGR:")
            print(
                top[["sector", "regime_scope", "best_strategy", "cagr_pct", "sharpe", "max_drawdown"]]
                .to_string(index=False)
            )
        elif not summary.empty:
            top = summary.sort_values(by="cagr_pct", ascending=False).head(10)
            print("\nTop results by CAGR:")
            print(top[["sector", "regime_scope", "cagr_pct", "sharpe", "max_drawdown"]].to_string(index=False))

        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
