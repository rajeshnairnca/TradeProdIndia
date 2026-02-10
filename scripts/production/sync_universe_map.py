import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync TradingView exchange map JSON into production_universe_map."
    )
    parser.add_argument(
        "--map-file",
        default=config.TRADINGVIEW_EXCHANGE_MAP_FILE,
        help="Path to ticker->exchange JSON map (default: TRADINGVIEW_EXCHANGE_MAP_FILE).",
    )
    parser.add_argument(
        "--mode",
        choices=["merge", "replace"],
        default="merge",
        help="merge: update DB map with file values; replace: overwrite DB map from file only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only, do not write to database.",
    )
    return parser.parse_args()


def _resolve_path(path: str) -> Path:
    resolved = config.resolve_path(path)
    if os.path.isabs(resolved):
        return Path(resolved)
    return Path(PROJECT_ROOT) / resolved


def _load_exchange_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Exchange map file not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in exchange map file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Exchange map must be a JSON object of ticker->exchange: {path}")

    cleaned: dict[str, str] = {}
    for ticker, exchange in payload.items():
        t = str(ticker).strip().upper()
        ex = str(exchange).strip().upper()
        if not t:
            continue
        cleaned[t] = ex or "UNKNOWN"
    return cleaned


def _count_unknown(mapping: dict[str, str]) -> int:
    return sum(1 for exchange in mapping.values() if str(exchange).strip().upper() == "UNKNOWN")


def _build_merged_map(
    db_map: dict[str, str],
    file_map: dict[str, str],
) -> tuple[dict[str, str], int]:
    merged = dict(db_map)
    skipped_unknown_overwrites = 0
    for ticker, exchange in file_map.items():
        current = str(merged.get(ticker, "")).strip().upper()
        if exchange == "UNKNOWN" and current and current != "UNKNOWN":
            skipped_unknown_overwrites += 1
            continue
        merged[ticker] = exchange
    return merged, skipped_unknown_overwrites


def main():
    args = parse_args()
    try:
        from src.production_db import (
            db_enabled,
            init_db,
            load_universe_map,
            replace_universe_map,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing DB dependencies for production DB access. Install requirements and retry."
        ) from exc

    if not db_enabled():
        raise RuntimeError("Database is required. Set DATABASE_URL or POSTGRES_URL.")
    map_path = _resolve_path(args.map_file)
    file_map = _load_exchange_map(map_path)

    init_db()
    db_map = load_universe_map()
    if args.mode == "replace":
        target_map = dict(file_map)
        skipped_unknown_overwrites = 0
    else:
        target_map, skipped_unknown_overwrites = _build_merged_map(db_map, file_map)

    inserted = sum(1 for ticker in target_map if ticker not in db_map)
    updated = sum(
        1
        for ticker, exchange in target_map.items()
        if ticker in db_map and str(db_map.get(ticker, "")).strip().upper() != exchange
    )
    removed = sum(1 for ticker in db_map if ticker not in target_map)
    unknown_before = _count_unknown(db_map)
    unknown_after = _count_unknown(target_map)

    print(f"Map file: {map_path}")
    print(f"Mode: {args.mode}")
    print(f"DB rows before: {len(db_map)} | UNKNOWN before: {unknown_before}")
    print(f"File rows: {len(file_map)}")
    print(
        f"Target rows: {len(target_map)} | inserted: {inserted} | updated: {updated} | removed: {removed}"
    )
    print(
        f"UNKNOWN after: {unknown_after} | UNKNOWN delta: {unknown_after - unknown_before:+d}"
    )
    if skipped_unknown_overwrites:
        print(f"Skipped UNKNOWN overwrites of known exchanges: {skipped_unknown_overwrites}")

    if args.dry_run:
        print("Dry run complete. No database changes were written.")
        return

    replace_universe_map(target_map)
    print("Universe mapping sync complete.")


if __name__ == "__main__":
    main()
