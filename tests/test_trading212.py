import json

import pytest

from src.trading212 import (
    Trading212ApiError,
    Trading212Client,
    Trading212Credentials,
    resolve_t212_ticker,
)


def test_resolve_t212_ticker_uses_valid_override_when_present() -> None:
    by_ticker = {"DAY_US_EQ": {"ticker": "DAY_US_EQ", "currencyCode": "USD", "type": "STOCK"}}
    by_symbol = {"DAY": [by_ticker["DAY_US_EQ"]]}
    overrides = {"DAY": "DAY_US_EQ"}
    resolved = resolve_t212_ticker(
        "DAY",
        by_symbol=by_symbol,
        overrides=overrides,
        preferred_currency="USD",
        by_ticker=by_ticker,
    )
    assert resolved == "DAY_US_EQ"


def test_resolve_t212_ticker_falls_back_when_override_not_in_instruments() -> None:
    by_ticker = {"DAY_US_EQ": {"ticker": "DAY_US_EQ", "currencyCode": "USD", "type": "STOCK"}}
    by_symbol = {"DAY": [by_ticker["DAY_US_EQ"]]}
    overrides = {"DAY": "CDAY_CA_EQ"}
    resolved = resolve_t212_ticker(
        "DAY",
        by_symbol=by_symbol,
        overrides=overrides,
        preferred_currency="USD",
        by_ticker=by_ticker,
    )
    assert resolved == "DAY_US_EQ"


def test_wait_for_fill_retries_on_transient_404(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Trading212Client(
        credentials=Trading212Credentials(api_key="k", api_secret="s"),
        base_url="https://demo.trading212.com/api/v0",
        timeout=0.1,
    )
    calls = {"count": 0}

    def fake_get_order(order_id: int | str) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            raise Trading212ApiError(
                status_code=404,
                response_text='{"detail":"Order not found"}',
                method="GET",
                url=f"https://demo.trading212.com/api/v0/equity/orders/{order_id}",
            )
        return {"id": str(order_id), "status": "FILLED", "filledQuantity": 2.0}

    monkeypatch.setattr(client, "get_order", fake_get_order)
    filled = client.wait_for_fill(order_id="abc-123", expected_qty=2.0, timeout_sec=1.0, poll_sec=0.0)
    assert filled.get("status") == "FILLED"
    assert calls["count"] == 2


def test_wait_for_fill_raises_non_404_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Trading212Client(
        credentials=Trading212Credentials(api_key="k", api_secret="s"),
        base_url="https://demo.trading212.com/api/v0",
        timeout=0.1,
    )

    def fake_get_order(order_id: int | str) -> dict:
        raise Trading212ApiError(
            status_code=500,
            response_text='{"detail":"oops"}',
            method="GET",
            url=f"https://demo.trading212.com/api/v0/equity/orders/{order_id}",
        )

    monkeypatch.setattr(client, "get_order", fake_get_order)
    with pytest.raises(Trading212ApiError):
        client.wait_for_fill(order_id=123, expected_qty=1.0, timeout_sec=0.1, poll_sec=0.0)


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        text: str,
        reason: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.reason = reason
        self.headers = headers or {}

    def json(self) -> dict:
        return json.loads(self.text)


def test_request_retries_get_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Trading212Client(
        credentials=Trading212Credentials(api_key="k", api_secret="s"),
        base_url="https://demo.trading212.com/api/v0",
        timeout=0.1,
    )
    responses = [
        _FakeResponse(
            status_code=429,
            text='{"errorMessage":"too many requests"}',
            reason="Too Many Requests",
            headers={"Retry-After": "0"},
        ),
        _FakeResponse(status_code=200, text='{"currencyCode":"GBP"}', reason="OK"),
    ]
    calls = {"count": 0}

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return responses.pop(0)

    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("src.trading212.time.sleep", lambda _: None)
    data = client.get_account_summary()
    assert data.get("currencyCode") == "GBP"
    assert calls["count"] == 2


def test_request_does_not_retry_post_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Trading212Client(
        credentials=Trading212Credentials(api_key="k", api_secret="s"),
        base_url="https://demo.trading212.com/api/v0",
        timeout=0.1,
    )
    calls = {"count": 0}

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return _FakeResponse(
            status_code=429,
            text='{"errorMessage":"too many requests"}',
            reason="Too Many Requests",
            headers={"Retry-After": "0"},
        )

    monkeypatch.setattr(client._session, "request", fake_request)
    with pytest.raises(Trading212ApiError):
        client.place_market_order("AAPL_US_EQ", 1.0)
    assert calls["count"] == 1


def test_wait_for_orders_uses_get_order_fallback_for_missing_bulk_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Trading212Client(
        credentials=Trading212Credentials(api_key="k", api_secret="s"),
        base_url="https://demo.trading212.com/api/v0",
        timeout=0.1,
    )
    bulk_calls = {"count": 0}

    def fake_get_orders() -> list[dict]:
        bulk_calls["count"] += 1
        return []

    def fake_get_order(order_id: int | str) -> dict:
        return {"id": str(order_id), "status": "FILLED", "filledQuantity": 1.0}

    monkeypatch.setattr(client, "get_orders", fake_get_orders)
    monkeypatch.setattr(client, "get_order", fake_get_order)
    monkeypatch.setattr("src.trading212.time.sleep", lambda _: None)

    snapshots = client.wait_for_orders(["abc-1"], timeout_sec=0.2, poll_sec=0.0)
    assert snapshots["abc-1"]["status"] == "FILLED"
    assert bulk_calls["count"] >= 3


def test_wait_for_orders_handles_fallback_429_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Trading212Client(
        credentials=Trading212Credentials(api_key="k", api_secret="s"),
        base_url="https://demo.trading212.com/api/v0",
        timeout=0.1,
    )
    calls = {"get_order": 0}

    def fake_get_orders() -> list[dict]:
        return []

    def fake_get_order(order_id: int | str) -> dict:
        calls["get_order"] += 1
        if calls["get_order"] < 2:
            raise Trading212ApiError(
                status_code=429,
                response_text='{"errorMessage":"too many requests"}',
                method="GET",
                url=f"https://demo.trading212.com/api/v0/equity/orders/{order_id}",
                reason="Too Many Requests",
            )
        return {"id": str(order_id), "status": "FILLED", "filledQuantity": 1.0}

    monkeypatch.setattr(client, "get_orders", fake_get_orders)
    monkeypatch.setattr(client, "get_order", fake_get_order)
    monkeypatch.setattr("src.trading212.time.sleep", lambda _: None)

    snapshots = client.wait_for_orders(["abc-2"], timeout_sec=0.5, poll_sec=0.0)
    assert snapshots["abc-2"]["status"] == "FILLED"
    assert calls["get_order"] >= 2


def test_request_uses_exponential_backoff_when_retry_after_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Trading212Client(
        credentials=Trading212Credentials(api_key="k", api_secret="s"),
        base_url="https://demo.trading212.com/api/v0",
        timeout=0.1,
    )
    responses = [
        _FakeResponse(
            status_code=429,
            text='{"errorMessage":"too many requests"}',
            reason="Too Many Requests",
            headers={"Retry-After": "0"},
        ),
        _FakeResponse(status_code=200, text='{"currencyCode":"GBP"}', reason="OK"),
    ]
    sleeps: list[float] = []

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        return responses.pop(0)

    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("src.trading212.time.sleep", lambda sec: sleeps.append(float(sec)))

    data = client.get_account_summary()
    assert data.get("currencyCode") == "GBP"
    assert sleeps
    assert sleeps[0] > 0.0
