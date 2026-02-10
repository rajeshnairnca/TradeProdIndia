from __future__ import annotations

import sys
import types

if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_extras = types.ModuleType("psycopg2.extras")

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
    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_extras

from scripts.production.daily_run import _execute_trading212_orders


class _FakeClient:
    def __init__(
        self,
        bulk_statuses: list[dict[str, dict]] | None = None,
        raise_on_wait: Exception | None = None,
        positions_by_ticker: dict[str, list[dict]] | None = None,
    ) -> None:
        self.place_calls = 0
        self.wait_calls = 0
        self.position_calls = 0
        self.placed: list[dict] = []
        self.bulk_statuses = bulk_statuses or []
        self.raise_on_wait = raise_on_wait
        self.positions_by_ticker = positions_by_ticker or {}

    def place_market_order(self, ticker: str, quantity: float) -> dict:
        self.place_calls += 1
        order_id = f"order-{self.place_calls}"
        self.placed.append({"id": order_id, "ticker": ticker, "quantity": quantity})
        return {"id": order_id, "ticker": ticker, "quantity": quantity}

    def wait_for_orders(self, order_ids: list[int | str]) -> dict[str, dict]:
        self.wait_calls += 1
        if self.raise_on_wait is not None:
            raise self.raise_on_wait
        if self.bulk_statuses:
            payload = self.bulk_statuses.pop(0)
            return {str(key): value for key, value in payload.items()}
        return {
            str(order_id): {"id": str(order_id), "status": "FILLED", "filledQuantity": 1.0}
            for order_id in order_ids
        }

    def get_positions(self, ticker: str | None = None) -> list[dict]:
        self.position_calls += 1
        if not ticker:
            return []
        rows = self.positions_by_ticker.get(str(ticker), [])
        return [dict(row) for row in rows]


def _context_with_two_tickers(client: _FakeClient, positions_raw: list[dict] | None = None) -> dict:
    by_ticker = {
        "AAPL_US_EQ": {"ticker": "AAPL_US_EQ", "currencyCode": "USD", "type": "STOCK"},
        "MSFT_US_EQ": {"ticker": "MSFT_US_EQ", "currencyCode": "USD", "type": "STOCK"},
    }
    by_symbol = {
        "AAPL": [by_ticker["AAPL_US_EQ"]],
        "MSFT": [by_ticker["MSFT_US_EQ"]],
    }
    return {
        "client": client,
        "by_ticker": by_ticker,
        "by_symbol": by_symbol,
        "overrides": {},
        "account_currency": "USD",
        "positions_raw": positions_raw or [],
    }


def test_execute_orders_halts_on_non_filled_order_without_raising() -> None:
    client = _FakeClient(
        bulk_statuses=[
            {
                "order-1": {"id": "order-1", "status": "PENDING", "filledQuantity": 0.0},
                "order-2": {"id": "order-2", "status": "FILLED", "filledQuantity": 10.0, "filledValue": 2000.0},
            }
        ]
    )
    context = _context_with_two_tickers(client)
    trades = [
        {"ticker": "AAPL", "action": "BUY", "shares": 25},
        {"ticker": "MSFT", "action": "BUY", "shares": 10},
    ]
    orders, missing, issues = _execute_trading212_orders(trades, context, dry_run=False)
    assert missing == []
    assert len(orders) == 2
    assert orders[0]["ticker"] == "AAPL"
    assert orders[0]["status"] == "PENDING"
    assert len(issues) == 1
    assert issues[0]["ticker"] == "AAPL"
    assert client.place_calls == 2
    assert client.wait_calls == 1


def test_execute_orders_collects_error_issue_on_execution_exception() -> None:
    client = _FakeClient(raise_on_wait=RuntimeError("network glitch"))
    context = _context_with_two_tickers(client)
    trades = [{"ticker": "AAPL", "action": "BUY", "shares": 25}]
    orders, missing, issues = _execute_trading212_orders(trades, context, dry_run=False)
    assert missing == []
    assert len(orders) == 1
    assert orders[0]["status"] == "UNKNOWN"
    assert len(issues) == 1
    assert issues[0]["error"] == "bulk_status_error"


def test_execute_orders_places_sells_before_buys() -> None:
    client = _FakeClient(
        bulk_statuses=[
            {"order-1": {"id": "order-1", "status": "FILLED", "filledQuantity": 4.0, "filledValue": 800.0}},
            {"order-2": {"id": "order-2", "status": "FILLED", "filledQuantity": 3.0, "filledValue": 600.0}},
        ]
    )
    context = _context_with_two_tickers(client)
    trades = [
        {"ticker": "AAPL", "action": "BUY", "shares": 3},
        {"ticker": "MSFT", "action": "SELL", "shares": -4},
    ]
    orders, missing, issues = _execute_trading212_orders(trades, context, dry_run=False)
    assert missing == []
    assert issues == []
    assert len(orders) == 2
    assert client.placed[0]["ticker"] == "MSFT_US_EQ"
    assert client.placed[1]["ticker"] == "AAPL_US_EQ"


def test_execute_orders_reconciles_unresolved_status_with_positions_endpoint() -> None:
    client = _FakeClient(
        bulk_statuses=[
            {"order-1": {"id": "order-1", "status": "NEW", "filledQuantity": 0.0}},
        ],
        positions_by_ticker={
            "AAPL_US_EQ": [
                {
                    "ticker": "AAPL_US_EQ",
                    "quantity": 3.0,
                    "instrument": {"ticker": "AAPL_US_EQ"},
                }
            ]
        },
    )
    context = _context_with_two_tickers(client, positions_raw=[])
    trades = [{"ticker": "AAPL", "action": "BUY", "shares": 3}]
    orders, missing, issues = _execute_trading212_orders(trades, context, dry_run=False)
    assert missing == []
    assert issues == []
    assert len(orders) == 1
    assert orders[0]["status"] == "FILLED"
    assert orders[0]["filled_quantity"] == 3.0
    assert orders[0]["payload"]["resolution"] == "reconciled_via_positions_endpoint"
    assert client.position_calls >= 1
