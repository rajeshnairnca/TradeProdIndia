from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

RUN_CALENDAR_ACTION_SKIP = "skip"
RUN_CALENDAR_ACTION_FORCE_RUN = "force_run"
RUN_CALENDAR_ACTIONS = {
    RUN_CALENDAR_ACTION_SKIP,
    RUN_CALENDAR_ACTION_FORCE_RUN,
}


def normalize_run_calendar_action(action: str | None) -> str | None:
    if action is None:
        return None
    normalized = str(action).strip().lower()
    if not normalized:
        return None
    if normalized not in RUN_CALENDAR_ACTIONS:
        raise ValueError(
            f"Invalid run-calendar action '{action}'. Supported: {sorted(RUN_CALENDAR_ACTIONS)}"
        )
    return normalized


def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(value).tz_localize(None)
    return pd.Timestamp(ts.date())


@lru_cache(maxsize=32)
def _market_holidays_for_year(year: int) -> frozenset[pd.Timestamp]:
    all_holidays = _load_market_holidays()
    return frozenset(ts for ts in all_holidays if ts.year == int(year))


def _parse_holiday_date(value: object) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return _normalize_date(text)
    except (TypeError, ValueError):
        return None


def _extract_holiday_dates(payload: object) -> set[pd.Timestamp]:
    parsed: set[pd.Timestamp] = set()
    if isinstance(payload, dict):
        for _, values in payload.items():
            if isinstance(values, list):
                parsed.update(_extract_holiday_dates(values))
            else:
                item = _parse_holiday_date(values)
                if item is not None:
                    parsed.add(item)
        return parsed
    if isinstance(payload, list):
        for item in payload:
            parsed.update(_extract_holiday_dates(item))
        return parsed
    item = _parse_holiday_date(payload)
    if item is not None:
        parsed.add(item)
    return parsed


def _candidate_holiday_file_paths() -> list[Path]:
    explicit = os.getenv("RUN_CALENDAR_MARKET_HOLIDAYS_FILE", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        project_root = Path(__file__).resolve().parent.parent
        candidates.append(project_root / "data" / "india_market_holidays.json")
    return candidates


@lru_cache(maxsize=1)
def _load_market_holidays() -> frozenset[pd.Timestamp]:
    holidays: set[pd.Timestamp] = set()

    csv_env = os.getenv("RUN_CALENDAR_MARKET_HOLIDAYS", "").strip()
    if csv_env:
        for item in csv_env.split(","):
            parsed = _parse_holiday_date(item)
            if parsed is not None:
                holidays.add(parsed)

    json_env = os.getenv("RUN_CALENDAR_MARKET_HOLIDAYS_JSON", "").strip()
    if json_env:
        try:
            payload = json.loads(json_env)
        except json.JSONDecodeError:
            payload = None
        holidays.update(_extract_holiday_dates(payload))

    for candidate in _candidate_holiday_file_paths():
        path = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                holidays.update(_extract_holiday_dates(payload))
            else:
                for line in path.read_text(encoding="utf-8").splitlines():
                    parsed = _parse_holiday_date(line)
                    if parsed is not None:
                        holidays.add(parsed)
        except OSError:
            continue
    return frozenset(holidays)


def is_us_federal_holiday(value: str | pd.Timestamp) -> bool:
    run_date = _normalize_date(value)
    return run_date in _market_holidays_for_year(run_date.year)


def list_us_federal_holidays(year: int) -> list[str]:
    return sorted(ts.strftime("%Y-%m-%d") for ts in _market_holidays_for_year(int(year)))


def resolve_schedule_date(
    explicit_date: str | None,
    timezone_name: str = "Asia/Kolkata",
) -> pd.Timestamp:
    if explicit_date:
        return _normalize_date(explicit_date)
    try:
        now_local = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        now_local = datetime.utcnow()
    return pd.Timestamp(now_local.date())


def evaluate_run_day(
    run_date: str | pd.Timestamp,
    *,
    override_action: str | None = None,
    override_reason: str | None = None,
    skip_weekends: bool = False,
    skip_us_federal_holidays: bool = True,
) -> dict[str, object]:
    normalized_date = _normalize_date(run_date)
    date_str = normalized_date.strftime("%Y-%m-%d")

    action = normalize_run_calendar_action(override_action)
    if action == RUN_CALENDAR_ACTION_FORCE_RUN:
        return {
            "date": date_str,
            "should_run": True,
            "reason_code": "override_force_run",
            "reason": override_reason or "Date explicitly allowed by run-calendar override.",
            "source": "override",
            "override_action": action,
        }
    if action == RUN_CALENDAR_ACTION_SKIP:
        return {
            "date": date_str,
            "should_run": False,
            "reason_code": "override_skip",
            "reason": override_reason or "Date explicitly blocked by run-calendar override.",
            "source": "override",
            "override_action": action,
        }

    if skip_weekends and normalized_date.dayofweek >= 5:
        return {
            "date": date_str,
            "should_run": False,
            "reason_code": "weekend",
            "reason": "Weekend is blocked by run-calendar settings.",
            "source": "default",
            "override_action": None,
        }

    if skip_us_federal_holidays and is_us_federal_holiday(normalized_date):
        return {
            "date": date_str,
            "should_run": False,
            # Keep legacy reason_code for backward compatibility with existing clients.
            "reason_code": "us_federal_holiday",
            "reason": "Market holiday is blocked by run-calendar settings.",
            "source": "default",
            "override_action": None,
        }

    return {
        "date": date_str,
        "should_run": True,
        "reason_code": "allowed",
        "reason": "Date allowed by run-calendar settings.",
        "source": "default",
        "override_action": None,
    }
