from __future__ import annotations

import json

from src.universe_monitor import is_technology_sector, parse_tradingview_catalog, update_monitor_records


def test_parse_tradingview_catalog_prefers_exchange_priority(tmp_path):
    payload = {
        "data": [
            {"s": "NYSE:ABC"},
            {"s": "NASDAQ:ABC"},
            {"s": "OTC:ZZZZ"},
            {"s": "AMEX:QQQ"},
            {"s": "INVALID"},
        ]
    }
    catalog_file = tmp_path / "tv_catalog.json"
    catalog_file.write_text(json.dumps(payload))

    rows = parse_tradingview_catalog(
        catalog_path=catalog_file,
        allowed_exchanges=["NASDAQ", "NYSE", "AMEX"],
        exchange_priority=["NASDAQ", "NYSE", "AMEX"],
    )

    assert any(row["ticker"] == "ABC" and row["exchange"] == "NASDAQ" for row in rows)
    assert any(row["ticker"] == "QQQ" and row["exchange"] == "AMEX" for row in rows)
    assert all(row["ticker"] != "ZZZZ" for row in rows)


def test_is_technology_sector_matches_keywords():
    keywords = ["technology", "software", "semiconductor"]
    assert is_technology_sector("Information Technology", keywords)
    assert is_technology_sector("Application Software", keywords)
    assert not is_technology_sector("Utilities", keywords)


def test_update_monitor_records_tracks_streak_and_resets():
    records = {}
    records = update_monitor_records(records, [{"ticker": "AAA", "monitor_pass": True}], "2026-02-08")
    assert records["AAA"]["pass_streak"] == 1
    assert records["AAA"]["total_pass_days"] == 1

    records = update_monitor_records(records, [{"ticker": "AAA", "monitor_pass": True}], "2026-02-09")
    assert records["AAA"]["pass_streak"] == 2
    assert records["AAA"]["total_pass_days"] == 2

    records = update_monitor_records(records, [{"ticker": "AAA", "monitor_pass": False}], "2026-02-10")
    assert records["AAA"]["pass_streak"] == 0
    assert records["AAA"]["total_pass_days"] == 2


def test_update_monitor_records_same_day_rerun_does_not_double_increment():
    records = {}
    records = update_monitor_records(records, [{"ticker": "AAA", "monitor_pass": True}], "2026-02-08")
    records = update_monitor_records(records, [{"ticker": "AAA", "monitor_pass": True}], "2026-02-08")
    assert records["AAA"]["pass_streak"] == 1
    assert records["AAA"]["total_pass_days"] == 1
