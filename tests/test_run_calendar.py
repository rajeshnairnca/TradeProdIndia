from __future__ import annotations

from src.run_calendar import (
    RUN_CALENDAR_ACTION_FORCE_RUN,
    RUN_CALENDAR_ACTION_SKIP,
    evaluate_run_day,
    is_us_federal_holiday,
    list_us_federal_holidays,
)


def test_us_federal_holiday_is_blocked_by_default() -> None:
    decision = evaluate_run_day("2026-01-01")
    assert decision["should_run"] is False
    assert decision["reason_code"] == "us_federal_holiday"


def test_non_holiday_weekday_runs_by_default() -> None:
    decision = evaluate_run_day("2026-01-02")
    assert decision["should_run"] is True
    assert decision["reason_code"] == "allowed"


def test_weekend_blocking_is_opt_in() -> None:
    default_decision = evaluate_run_day("2026-01-03")
    assert default_decision["should_run"] is True

    weekend_blocked_decision = evaluate_run_day("2026-01-03", skip_weekends=True)
    assert weekend_blocked_decision["should_run"] is False
    assert weekend_blocked_decision["reason_code"] == "weekend"


def test_force_run_override_wins_over_default_holiday_block() -> None:
    decision = evaluate_run_day(
        "2026-01-01",
        override_action=RUN_CALENDAR_ACTION_FORCE_RUN,
        override_reason="manual release",
    )
    assert decision["should_run"] is True
    assert decision["reason_code"] == "override_force_run"


def test_skip_override_blocks_even_normal_weekday() -> None:
    decision = evaluate_run_day(
        "2026-01-02",
        override_action=RUN_CALENDAR_ACTION_SKIP,
        override_reason="maintenance",
    )
    assert decision["should_run"] is False
    assert decision["reason_code"] == "override_skip"


def test_holiday_helpers_return_expected_dates() -> None:
    assert is_us_federal_holiday("2026-01-01") is True
    assert "2026-01-01" in list_us_federal_holidays(2026)
