from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

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
def _us_federal_holidays_for_year(year: int) -> frozenset[pd.Timestamp]:
    holidays = USFederalHolidayCalendar().holidays(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
    )
    return frozenset(pd.Timestamp(ts).tz_localize(None).normalize() for ts in holidays)


def is_us_federal_holiday(value: str | pd.Timestamp) -> bool:
    run_date = _normalize_date(value)
    return run_date in _us_federal_holidays_for_year(run_date.year)


def list_us_federal_holidays(year: int) -> list[str]:
    return sorted(ts.strftime("%Y-%m-%d") for ts in _us_federal_holidays_for_year(int(year)))


def resolve_schedule_date(
    explicit_date: str | None,
    timezone_name: str = "America/New_York",
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
            "reason_code": "us_federal_holiday",
            "reason": "US federal holiday is blocked by run-calendar settings.",
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
