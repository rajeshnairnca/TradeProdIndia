import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.production import load_state
from src.production_db import (
    db_enabled,
    init_db,
    replace_trades,
    upsert_run_summary,
    upsert_state,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill production runs/trades/state into Postgres.")
    parser.add_argument("--output-dir", default="runs/production", help="Root output directory for daily runs.")
    parser.add_argument("--state-file", default="runs/production/state.json", help="Path to state file.")
    return parser.parse_args()


def _resolve_path(path: str) -> str:
    resolved = config.resolve_path(path)
    if os.path.isabs(resolved):
        return resolved
    return os.path.join(PROJECT_ROOT, resolved)


def main():
    if not db_enabled():
        print("DATABASE_URL/POSTGRES_URL not configured. Aborting.")
        return

    init_db()
    args = parse_args()
    output_dir = Path(_resolve_path(args.output_dir))
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    run_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    runs_loaded = 0
    trades_loaded = 0
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        trades_path = run_dir / "trades.csv"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                summary = None
            if summary:
                if "date" not in summary:
                    summary["date"] = run_dir.name
                upsert_run_summary(summary)
                runs_loaded += 1
        if trades_path.exists():
            df = pd.read_csv(trades_path)
            if not df.empty:
                run_date = run_dir.name
                replace_trades(run_date, df.to_dict(orient="records"))
                trades_loaded += len(df)

    state_path = Path(_resolve_path(args.state_file))
    if state_path.exists():
        state = load_state(state_path, config.INITIAL_CAPITAL)
        upsert_state(state)

    print(f"Backfill complete. Runs: {runs_loaded}, Trades: {trades_loaded}")


if __name__ == "__main__":
    main()
