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
