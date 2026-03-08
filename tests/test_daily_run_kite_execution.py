from __future__ import annotations

import sys
import types

if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_extras = types.ModuleType("psycopg2.extras")
    fake_sql = types.ModuleType("psycopg2.sql")

    def _json_passthrough(value):  # type: ignore[no-untyped-def]
        return value

    def _execute_values_noop(*args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    class _RealDictCursor:
        pass

    fake_extras.Json = _json_passthrough  # type: ignore[attr-defined]
    fake_extras.execute_values = _execute_values_noop  # type: ignore[attr-defined]
    fake_extras.RealDictCursor = _RealDictCursor  # type: ignore[attr-defined]

    def _connect_forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("psycopg2.connect should not be used in this unit test")

    fake_psycopg2.connect = _connect_forbidden  # type: ignore[attr-defined]
    fake_psycopg2.extras = fake_extras  # type: ignore[attr-defined]
    fake_sql.SQL = lambda value: value  # type: ignore[attr-defined]
    fake_sql.Identifier = lambda value: value  # type: ignore[attr-defined]
    fake_psycopg2.sql = fake_sql  # type: ignore[attr-defined]
    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_extras
    sys.modules["psycopg2.sql"] = fake_sql

from scripts.production.daily_run import _execute_kite_orders


class _FakeKiteClient:
    def __init__(self, fill_status: str = "FILLED") -> None:
        self.placed: list[dict] = []
        self.fill_status = fill_status

    def place_market_order(  # type: ignore[no-untyped-def]
        self,
        *,
        exchange: str,
        tradingsymbol: str,
        transaction_type: str,
        quantity: int,
    ) -> dict:
        order_id = f"order-{len(self.placed) + 1}"
        self.placed.append(
            {
                "order_id": order_id,
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
            }
        )
        return {"order_id": order_id}

    def wait_for_orders(self, order_ids: list[int | str]) -> dict[str, dict]:
        snapshots: dict[str, dict] = {}
        for order_id in order_ids:
            snapshots[str(order_id)] = {
                "id": str(order_id),
                "status": self.fill_status,
                "filledQuantity": 1.0,
                "fillPrice": 100.0,
                "filledValue": 100.0,
                "currency": "INR",
            }
        return snapshots


def test_execute_kite_orders_places_sells_before_buys() -> None:
    client = _FakeKiteClient(fill_status="FILLED")
    context = {
        "client": client,
        "by_ticker": {},
        "by_symbol": {},
        "overrides": {},
    }
    trades = [
        {"ticker": "RELIANCE.NS", "action": "BUY", "shares": 1},
        {"ticker": "INFY.NS", "action": "SELL", "shares": -1},
    ]
    orders, missing, issues = _execute_kite_orders(trades, context, dry_run=False)
    assert missing == []
    assert issues == []
    assert len(orders) == 2
    assert client.placed[0]["transaction_type"] == "SELL"
    assert client.placed[1]["transaction_type"] == "BUY"
    assert client.placed[0]["tradingsymbol"] == "INFY"
    assert client.placed[1]["tradingsymbol"] == "RELIANCE"


def test_execute_kite_orders_reports_missing_mapping() -> None:
    client = _FakeKiteClient(fill_status="FILLED")
    context = {
        "client": client,
        "by_ticker": {"NSE:KNOWN": {"exchange": "NSE", "tradingsymbol": "KNOWN"}},
        "by_symbol": {},
        "overrides": {},
    }
    trades = [{"ticker": "UNKNOWN.NS", "action": "BUY", "shares": 1}]
    orders, missing, issues = _execute_kite_orders(trades, context, dry_run=False)
    assert orders == []
    assert issues == []
    assert missing == ["UNKNOWN.NS"]


def test_execute_kite_orders_marks_unfilled_issue() -> None:
    client = _FakeKiteClient(fill_status="REJECTED")
    context = {
        "client": client,
        "by_ticker": {},
        "by_symbol": {},
        "overrides": {},
    }
    trades = [{"ticker": "RELIANCE.NS", "action": "BUY", "shares": 1}]
    orders, missing, issues = _execute_kite_orders(trades, context, dry_run=False)
    assert missing == []
    assert len(orders) == 1
    assert orders[0]["status"] == "REJECTED"
    assert len(issues) == 1

