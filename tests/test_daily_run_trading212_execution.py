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

from scripts.production.daily_run import (
    _build_retry_pending_ticker_entries,
    _broker_notionals,
    _execute_trading212_orders,
    _state_from_broker_context,
    _state_from_broker_snapshot,
)
from scripts.production import daily_run as daily_run_module
from src.production import ProductionState


class _FakeClient:
    def __init__(
        self,
        raise_on_positions: Exception | None = None,
        raise_on_history: Exception | None = None,
        positions_by_ticker: dict[str, list[dict]] | None = None,
        history_pages: list[dict] | None = None,
    ) -> None:
        self.place_calls = 0
        self.wait_calls = 0
        self.position_calls = 0
        self.history_calls = 0
        self.placed: list[dict] = []
        self.raise_on_positions = raise_on_positions
        self.raise_on_history = raise_on_history
        self.positions_by_ticker = positions_by_ticker or {}
        self.history_pages = history_pages or []

    def place_market_order(self, ticker: str, quantity: float) -> dict:
        self.place_calls += 1
        order_id = f"order-{self.place_calls}"
        self.placed.append({"id": order_id, "ticker": ticker, "quantity": quantity})
        return {"id": order_id, "ticker": ticker, "quantity": quantity}

    def wait_for_orders(self, order_ids: list[int | str]) -> dict[str, dict]:
        self.wait_calls += 1
        raise AssertionError("wait_for_orders should not be called by execution flow")

    def get_positions(self, ticker: str | None = None) -> list[dict]:
        self.position_calls += 1
        if self.raise_on_positions is not None:
            raise self.raise_on_positions
        if not ticker:
            rows: list[dict] = []
            for per_ticker_rows in self.positions_by_ticker.values():
                rows.extend(dict(row) for row in per_ticker_rows)
            return rows
        rows = self.positions_by_ticker.get(str(ticker), [])
        return [dict(row) for row in rows]

    def get_historical_orders(
        self,
        limit: int = 50,
        cursor: str | None = None,
        ticker: str | None = None,
    ) -> dict:
        self.history_calls += 1
        if self.raise_on_history is not None:
            raise self.raise_on_history
        if self.history_pages:
            return dict(self.history_pages.pop(0))
        return {"items": []}


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
        history_pages=[
            {
                "items": [
                    {"order": {"id": "order-1", "status": "REJECTED"}},
                    {"order": {"id": "order-2", "status": "FILLED"}, "fill": {"quantity": 10.0, "price": 200.0}},
                ]
            }
        ],
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
    assert orders[0]["status"] == "REJECTED"
    assert len(issues) == 1
    assert issues[0]["ticker"] == "AAPL"
    assert client.place_calls == 2
    assert client.wait_calls == 0


def test_execute_orders_collects_error_issue_on_execution_exception() -> None:
    client = _FakeClient(raise_on_positions=RuntimeError("network glitch"))
    context = _context_with_two_tickers(client)
    trades = [{"ticker": "AAPL", "action": "BUY", "shares": 25}]
    orders, missing, issues = _execute_trading212_orders(trades, context, dry_run=False)
    assert missing == []
    assert len(orders) == 1
    assert orders[0]["status"] == "UNKNOWN"
    assert len(issues) == 1
    assert issues[0]["error"] == "order_monitor_error"


def test_execute_orders_places_sells_before_buys() -> None:
    client = _FakeClient(
        history_pages=[
            {
                "items": [
                    {"order": {"id": "order-1", "status": "FILLED"}, "fill": {"quantity": 4.0, "price": 200.0}},
                ]
            },
            {
                "items": [
                    {"order": {"id": "order-2", "status": "FILLED"}, "fill": {"quantity": 3.0, "price": 200.0}},
                ]
            },
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
    assert client.wait_calls == 0


def test_execute_orders_reconciles_unresolved_status_with_positions_endpoint() -> None:
    client = _FakeClient(
        history_pages=[
            {
                "items": [
                    {
                        "order": {"id": "order-1"},
                        "fill": {"price": 251.25, "quantity": 3.0},
                    }
                ]
            }
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
    assert orders[0]["exec_price"] == 251.25
    assert orders[0]["payload"]["resolution"] == "reconciled_via_positions_endpoint"
    assert (
        orders[0]["payload"]["resolutionPriceSource"]
        == "history.orders.fill.price"
    )
    assert client.position_calls >= 1
    assert client.history_calls >= 1


def test_execute_orders_normalizes_negative_sell_fill_snapshot() -> None:
    client = _FakeClient(
        history_pages=[
            {
                "items": [
                    {
                        "order": {"id": "order-1", "status": "FILLED"},
                        "fill": {
                            "quantity": -4.0,
                            "price": 200.0,
                            "value": -800.0,
                            "currencyCode": "USD",
                        },
                    }
                ]
            }
        ],
    )
    context = _context_with_two_tickers(client)
    trades = [{"ticker": "MSFT", "action": "SELL", "shares": -4}]
    orders, missing, issues = _execute_trading212_orders(trades, context, dry_run=False)
    assert missing == []
    assert issues == []
    assert len(orders) == 1
    assert orders[0]["status"] == "FILLED"
    assert orders[0]["filled_quantity"] == 4.0
    assert orders[0]["exec_price"] == 200.0
    assert orders[0]["currency"] == "USD"
    assert orders[0]["payload"]["filledValue"] == 800.0


def test_execute_orders_prefers_fill_price_for_exec_price() -> None:
    client = _FakeClient(
        history_pages=[
            {
                "items": [
                    {
                        "order": {"id": "order-1", "status": "FILLED"},
                        "fill": {"quantity": 2.0, "price": 120.0, "value": 180.0, "currencyCode": "USD"},
                    }
                ]
            }
        ],
    )
    context = _context_with_two_tickers(client)
    trades = [{"ticker": "AAPL", "action": "BUY", "shares": 2}]
    orders, missing, issues = _execute_trading212_orders(trades, context, dry_run=False)
    assert missing == []
    assert issues == []
    assert len(orders) == 1
    assert orders[0]["status"] == "FILLED"
    assert orders[0]["filled_quantity"] == 2.0
    # Prefer explicit fill price over implied value/quantity division.
    assert orders[0]["exec_price"] == 120.0


def test_build_retry_pending_ticker_entries_filters_to_failed_tickers() -> None:
    pending_entries = [
        {
            "type": "cash",
            "amount": 100.0,
            "source": "app",
        },
        {
            "type": "tickers",
            "tickers": ["SNDK", "AAPL"],
            "exchanges": {"SNDK": "NASDAQ", "AAPL": "NASDAQ"},
            "source": "app",
        },
        {
            "type": "tickers",
            "tickers": ["XYZ", "MSFT"],
            "ticker_exchanges": [
                {"ticker": "XYZ", "exchange": "NYSE"},
                {"ticker": "MSFT", "exchange": "NASDAQ"},
            ],
            "source": "app",
        },
    ]
    retry = _build_retry_pending_ticker_entries(pending_entries, {"SNDK", "XYZ"})
    assert len(retry) == 2
    assert retry[0]["tickers"] == ["SNDK"]
    assert retry[0]["exchanges"] == {"SNDK": "NASDAQ"}
    assert retry[1]["tickers"] == ["XYZ"]
    assert retry[1]["exchanges"] == {"XYZ": "NYSE"}


def test_broker_notionals_converts_order_currency_to_broker_currency() -> None:
    orders = [
        {
            "action": "BUY",
            "quantity": 10.0,
            "filled_quantity": 10.0,
            "exec_price": 100.0,
            "currency": "USD",
            "payload": {},
        },
        {
            "action": "SELL",
            "quantity": -2.0,
            "filled_quantity": 2.0,
            "exec_price": None,
            "currency": "GBP",
            "payload": {"filledValue": 350.0, "currency": "GBP"},
        },
    ]
    buy_notional, sell_notional, filled_orders, covered_orders = _broker_notionals(
        orders=orders,
        broker_currency="GBP",
        fx_rate_gbp_per_usd=0.8,
    )
    assert buy_notional == 800.0
    assert sell_notional == 350.0
    assert filled_orders == 2
    assert covered_orders == 2


def test_broker_notionals_marks_uncovered_when_notional_or_currency_missing() -> None:
    orders = [
        {
            "action": "BUY",
            "quantity": 5.0,
            "filled_quantity": 5.0,
            "exec_price": None,
            "currency": "USD",
            "payload": {},
        },
        {
            "action": "SELL",
            "quantity": -3.0,
            "filled_quantity": 3.0,
            "exec_price": None,
            "currency": "EUR",
            "payload": {"filledValue": 300.0, "currency": "EUR"},
        },
    ]
    buy_notional, sell_notional, filled_orders, covered_orders = _broker_notionals(
        orders=orders,
        broker_currency="GBP",
        fx_rate_gbp_per_usd=0.8,
    )
    assert buy_notional == 0.0
    assert sell_notional == 0.0
    assert filled_orders == 2
    assert covered_orders == 0


def test_state_from_broker_context_uses_broker_cash_and_positions() -> None:
    state = ProductionState(
        last_date="2025-01-01",
        cash=100.0,
        positions={"AAPL": 1},
        prev_weights={"AAPL": 0.6},
        total_costs_usd=42.0,
    )
    context = {
        "broker_cash_usd": 250.5,
        "broker_positions": {"AAPL": 2.0, "MSFT": 0.0, "": 1.0, "NVDA": -1.5},
    }
    out = _state_from_broker_context(state, context)
    assert out.cash == 250.5
    assert out.positions == {"AAPL": 2.0, "NVDA": -1.5}
    assert out.prev_weights == {}
    assert out.total_costs_usd == 42.0


def test_state_from_broker_context_falls_back_to_state_cash_when_missing() -> None:
    state = ProductionState(
        last_date="2025-01-01",
        cash=123.0,
        positions={"AAPL": 1},
        prev_weights={},
        total_costs_usd=0.0,
    )
    out = _state_from_broker_context(state, context={})
    assert out.cash == 123.0
    assert out.positions == {}


def test_state_from_broker_context_logs_fractional_positions(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []

    def _capture(message: str, **fields) -> None:
        events.append((message, fields))

    monkeypatch.setattr(daily_run_module, "_log", _capture)
    state = ProductionState(
        last_date="2025-01-01",
        cash=100.0,
        positions={},
        prev_weights={},
        total_costs_usd=0.0,
    )
    _state_from_broker_context(
        state,
        {"broker_cash_usd": 100.0, "broker_positions": {"AAPL": 1.25, "MSFT": 2.0}},
    )
    assert any(msg == "broker_fractional_positions_detected" for msg, _ in events)


def test_state_from_broker_snapshot_syncs_cash_and_positions() -> None:
    new_state = ProductionState(
        last_date="2025-01-02",
        cash=50.0,
        positions={"AAPL": 1},
        prev_weights={"AAPL": 0.5},
        total_costs_usd=7.0,
    )
    summary = {
        "broker_currency": "GBP",
        "broker_fx_rate_gbp_per_usd": 0.8,
        "broker_cash_after": 80.0,
    }
    context = {"overrides": {}}
    post_positions = [
        {"ticker": "AAPL_US_EQ", "quantity": 3.0, "instrument": {"ticker": "AAPL_US_EQ"}},
        {"ticker": "MSFT_US_EQ", "quantity": 1.0, "instrument": {"ticker": "MSFT_US_EQ"}},
    ]
    out = _state_from_broker_snapshot(new_state, summary, context, post_positions)
    assert out.last_date == "2025-01-02"
    assert out.cash == 100.0
    assert out.positions == {"AAPL": 3.0, "MSFT": 1.0}
    assert out.prev_weights == {"AAPL": 0.5}
    assert out.total_costs_usd == 7.0


def test_state_from_broker_context_computes_prev_weights_from_broker_holdings() -> None:
    state = ProductionState(
        last_date="2025-01-01",
        cash=10.0,
        positions={},
        prev_weights={},
        total_costs_usd=3.0,
    )
    context = {
        "broker_cash_usd": 100.0,
        "broker_positions": {"AAPL": 2.0, "MSFT": 1.0},
    }
    price_lookup = {"AAPL": 50.0, "MSFT": 100.0}
    out = _state_from_broker_context(state, context, price_lookup=price_lookup)
    assert out.cash == 100.0
    assert out.positions == {"AAPL": 2.0, "MSFT": 1.0}
    assert out.prev_weights == {
        "AAPL": 100.0 / 300.0,
        "MSFT": 100.0 / 300.0,
    }
    assert out.total_costs_usd == 3.0
