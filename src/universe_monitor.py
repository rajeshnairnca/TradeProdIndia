from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

_SYMBOL_RE = re.compile(r"^([A-Z]+):([A-Z0-9.\-]+)$")


def parse_tradingview_catalog(
    catalog_path: str | Path,
    allowed_exchanges: Iterable[str],
    exchange_priority: Iterable[str] | None = None,
    max_candidates: int | None = None,
) -> list[dict[str, str]]:
    """Parse a TradingView symbol catalog and return one exchange/symbol per ticker."""
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")

    payload = json.loads(path.read_text())
    raw_items = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(raw_items, list):
        raise ValueError("Invalid catalog format: expected top-level 'data' list.")

    allowed = {str(exchange).strip().upper() for exchange in allowed_exchanges if str(exchange).strip()}
    if not allowed:
        raise ValueError("At least one allowed exchange is required.")

    priority_list = [str(exchange).strip().upper() for exchange in (exchange_priority or []) if str(exchange).strip()]
    if not priority_list:
        priority_list = list(allowed)
    priority_rank = {exchange: idx for idx, exchange in enumerate(priority_list)}

    best_by_ticker: dict[str, tuple[int, int, dict[str, str]]] = {}
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("s", "")).strip().upper()
        match = _SYMBOL_RE.match(symbol)
        if not match:
            continue
        exchange, ticker = match.groups()
        if exchange not in allowed:
            continue

        rank = priority_rank.get(exchange, len(priority_rank) + 1)
        candidate = {
            "ticker": ticker,
            "exchange": exchange,
            "symbol": f"{exchange}:{ticker}",
        }
        current = best_by_ticker.get(ticker)
        if current is None or (rank, idx) < (current[0], current[1]):
            best_by_ticker[ticker] = (rank, idx, candidate)

    selected = [record[2] for record in sorted(best_by_ticker.values(), key=lambda x: (x[0], x[1]))]
    if max_candidates is not None and max_candidates > 0:
        selected = selected[:max_candidates]
    return selected


def is_technology_sector(sector: str | None, keywords: Iterable[str]) -> bool:
    if not sector:
        return False
    sector_lc = str(sector).lower()
    for keyword in keywords:
        token = str(keyword).strip().lower()
        if token and token in sector_lc:
            return True
    return False


def update_monitor_records(
    records: dict[str, dict],
    evaluations: Iterable[dict],
    run_date: str,
) -> dict[str, dict]:
    updated = dict(records)
    for item in evaluations:
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        passed = bool(item.get("monitor_pass", False))

        prev = dict(updated.get(ticker, {}))
        prev_streak = int(prev.get("pass_streak", 0) or 0)
        prev_total = int(prev.get("total_pass_days", 0) or 0)
        prev_status = str(prev.get("last_status", "")).lower() == "pass"
        prev_seen = str(prev.get("last_seen", ""))

        streak = prev_streak
        total = prev_total
        if prev_seen == run_date:
            if passed:
                if not prev_status:
                    streak = 1
                    total += 1
            else:
                streak = 0
        else:
            if passed:
                streak = (prev_streak + 1) if prev_status else 1
                total += 1
            else:
                streak = 0

        updated[ticker] = {
            "pass_streak": streak,
            "total_pass_days": total,
            "last_status": "pass" if passed else "fail",
            "last_seen": run_date,
        }
    return updated
