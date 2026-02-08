import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd

from src.production_db import (
    append_pending_adjustments as db_append_pending_adjustments,
    db_enabled,
    init_db as db_init,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Queue cash or universe adjustments for the next run.")
    parser.add_argument("--add-cash", type=float, default=0.0, help="Cash to add on next production run.")
    parser.add_argument("--cash-note", default="app", help="Note for the cash injection.")
    parser.add_argument(
        "--add-tickers",
        default=None,
        help="Comma-separated tickers to add on next production run.",
    )
    parser.add_argument(
        "--add-tickers-file",
        default=None,
        help="Path to newline-delimited tickers to add on next production run.",
    )
    parser.add_argument(
        "--add-tickers-exchange",
        default=None,
        help="Comma-separated EXCHANGE:TICKER entries to add with exchange info.",
    )
    parser.add_argument(
        "--add-tickers-exchange-file",
        default=None,
        help="Path to newline-delimited EXCHANGE:TICKER entries.",
    )
    parser.add_argument("--source", default="app", help="Source label for the adjustment.")
    return parser.parse_args()


def main():
    if not db_enabled():
        raise RuntimeError(
            "Database is required for queued adjustments. Set DATABASE_URL or POSTGRES_URL."
        )
    db_init()
    args = parse_args()
    entries: list[dict] = []
    timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.add_cash:
        entries.append(
            {
                "type": "cash",
                "amount": float(args.add_cash),
                "note": args.cash_note,
                "source": args.source,
                "created_at": timestamp,
            }
        )

    tickers: list[str] = []
    exchange_map: dict[str, str] = {}
    if args.add_tickers:
        tickers.extend([t.strip() for t in args.add_tickers.split(",") if t.strip()])
    if args.add_tickers_file:
        with open(args.add_tickers_file, "r") as f:
            tickers.extend([line.strip() for line in f if line.strip()])
    if args.add_tickers_exchange:
        exchange_entries = [t.strip() for t in args.add_tickers_exchange.split(",") if t.strip()]
        for exchange_entry in exchange_entries:
            if ":" not in exchange_entry:
                continue
            exchange, ticker = exchange_entry.split(":", 1)
            exchange_map[ticker.strip().upper()] = exchange.strip().upper()
    if args.add_tickers_exchange_file:
        with open(args.add_tickers_exchange_file, "r") as f:
            for line in f:
                entry = line.strip()
                if not entry or ":" not in entry:
                    continue
                exchange, ticker = entry.split(":", 1)
                exchange_map[ticker.strip().upper()] = exchange.strip().upper()
    if exchange_map:
        tickers.extend(exchange_map.keys())
    tickers = sorted(set(tickers))
    if tickers:
        payload = {
            "type": "tickers",
            "tickers": tickers,
            "source": args.source,
            "created_at": timestamp,
        }
        if exchange_map:
            payload["exchanges"] = exchange_map
        entries.append(payload)

    if not entries:
        print("No adjustments queued.")
        return

    db_append_pending_adjustments(entries)
    print(f"Queued {len(entries)} adjustment(s) to database.")


if __name__ == "__main__":
    main()
