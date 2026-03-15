import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.entry_indicator import (
    DEFAULT_BASE_STRATEGIES,
    DEFAULT_REGIME_MAPPING,
    compute_entry_indicator_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a data-driven market-entry indicator using historical forward returns "
            "of the configured strategy stack."
        )
    )
    parser.add_argument(
        "--strategy-roots",
        action="append",
        default=[],
        help="Root directory containing strategies.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_BASE_STRATEGIES),
        help="Base strategies to include (regime mapping strategies are auto-added).",
    )
    parser.add_argument(
        "--regime-mapping",
        type=str,
        default=json.dumps(DEFAULT_REGIME_MAPPING),
        help="JSON mapping of regime_label -> strategy name.",
    )
    parser.add_argument("--start-date", type=str, default="2013-01-01", help="Backtest start date.")
    parser.add_argument("--end-date", type=str, help="Optional backtest end date (exclusive).")
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Date to score (default: latest available backtest date).",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=126,
        help="Forward horizon in trading days used for entry-quality calibration.",
    )
    parser.add_argument(
        "--confirm-days",
        type=int,
        default=config.CONFIRM_DAYS,
        help="Regime confirmation days for non-sideways regimes.",
    )
    parser.add_argument(
        "--confirm-days-sideways",
        type=int,
        default=config.CONFIRM_DAYS_SIDEWAYS,
        help="Regime confirmation days for sideways regimes.",
    )
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=config.REBALANCE_EVERY,
        help="Rebalance cadence in trading days.",
    )
    parser.add_argument(
        "--ignore-stock-filters",
        action="store_true",
        help="Ignore excluded ticker file and quality-filter gating while building indicator history.",
    )
    parser.add_argument(
        "--persist-db",
        action="store_true",
        help="Persist the computed indicator payload to Postgres (if DATABASE_URL/POSTGRES_URL is configured).",
    )
    parser.add_argument("--output-json", type=str, help="Optional output file for JSON payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        mapping = json.loads(args.regime_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --regime-mapping JSON: {exc}") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("--regime-mapping must be a non-empty JSON object.")
    mapping = {str(k): str(v) for k, v in mapping.items()}

    payload = compute_entry_indicator_payload(
        strategy_roots=args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS),
        strategies=list(args.strategies or []),
        regime_mapping=mapping,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of_date=args.as_of_date,
        lookahead_days=int(args.lookahead_days),
        confirm_days=int(args.confirm_days),
        confirm_days_sideways=int(args.confirm_days_sideways),
        rebalance_every=int(args.rebalance_every),
        ignore_stock_filters=bool(args.ignore_stock_filters),
    )

    out = json.dumps(payload, indent=2)
    print(out)
    if args.output_json:
        out_path = args.output_json
        if not os.path.isabs(out_path):
            out_path = os.path.join(PROJECT_ROOT, out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
            f.write("\n")
        print(f"\nSaved indicator payload to {out_path}")
    if args.persist_db:
        try:
            from src.production_db import db_enabled, upsert_entry_indicator_snapshot
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not import DB helpers for --persist-db: {exc}")
            return
        if not db_enabled():
            print("Warning: --persist-db requested but DATABASE_URL/POSTGRES_URL is not configured.")
            return
        try:
            upsert_entry_indicator_snapshot(
                payload=dict(payload),
                start_date=str(args.start_date),
                end_date=str(args.end_date).strip() if args.end_date else None,
                as_of_date=str(args.as_of_date).strip() if args.as_of_date else None,
                lookahead_days=int(args.lookahead_days),
                confirm_days=int(args.confirm_days),
                confirm_days_sideways=int(args.confirm_days_sideways),
                rebalance_every=int(args.rebalance_every),
                strategy_roots=args.strategy_roots or list(config.DEFAULT_STRATEGY_ROOTS),
                strategies=list(args.strategies or []),
                regime_mapping=mapping,
            )
            print("Persisted entry indicator snapshot to DB.")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to persist entry indicator snapshot to DB: {exc}")


if __name__ == "__main__":
    main()
