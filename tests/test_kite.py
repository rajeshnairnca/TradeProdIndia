import json

import pytest

from src.kite import (
    KiteClient,
    KiteCredentials,
    account_cash_available,
    account_net_worth,
    normalize_kite_order_snapshot,
    positions_to_internal_positions,
    resolve_kite_instrument,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str, reason: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.reason = reason

    def json(self):  # type: ignore[no-untyped-def]
        return json.loads(self.text)


def test_resolve_kite_instrument_honors_override_and_suffix() -> None:
    overrides = {
        "SBIN.NS": ("NSE", "SBIN"),
    }
    resolved_override = resolve_kite_instrument("SBIN.NS", overrides, by_key=None, by_symbol=None)
    assert resolved_override == ("NSE", "SBIN")

    resolved_suffix = resolve_kite_instrument("RELIANCE.NS", {}, by_key=None, by_symbol=None)
    assert resolved_suffix == ("NSE", "RELIANCE")


def test_positions_to_internal_positions_maps_symbols_to_internal_tickers() -> None:
    overrides = {"SBIN.NS": ("NSE", "SBIN")}
    positions = [
        {"exchange": "NSE", "tradingsymbol": "SBIN", "quantity": 12, "t1_quantity": 1},
        {"exchange": "BSE", "tradingsymbol": "INFY", "quantity": 5, "t1_quantity": 0},
    ]
    mapped = positions_to_internal_positions(positions, overrides)
    assert mapped["SBIN.NS"] == 13.0
    assert mapped["INFY.BO"] == 5.0


def test_normalize_kite_order_snapshot_marks_complete_as_filled() -> None:
    snapshot = normalize_kite_order_snapshot(
        {
            "order_id": "123",
            "status": "COMPLETE",
            "filled_quantity": 10,
            "average_price": 100.5,
        }
    )
    assert snapshot["status"] == "FILLED"
    assert snapshot["filledQuantity"] == 10.0
    assert snapshot["fillPrice"] == 100.5
    assert snapshot["currency"] == "INR"


def test_account_helpers_compute_cash_and_net_worth() -> None:
    margins = {
        "equity": {
            "available": {"live_balance": 1000.0},
        }
    }
    holdings = [
        {"quantity": 2, "t1_quantity": 0, "last_price": 100.0},
        {"quantity": 1, "t1_quantity": 1, "last_price": 50.0},
    ]
    cash = account_cash_available(margins)
    net_worth = account_net_worth(margins, holdings)
    assert cash == 1000.0
    assert net_worth == 1300.0


def test_kite_client_place_market_order_unwraps_data(monkeypatch: pytest.MonkeyPatch) -> None:
    client = KiteClient(
        credentials=KiteCredentials(
            api_key="api",
            api_secret="secret",
            access_token="token",
            request_token="",
        ),
        base_url="https://api.kite.trade",
        timeout=0.1,
    )

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeResponse(
            status_code=200,
            text='{"status":"success","data":{"order_id":"abc-1"}}',
            reason="OK",
        )

    monkeypatch.setattr(client._session, "request", fake_request)
    payload = client.place_market_order(
        exchange="NSE",
        tradingsymbol="SBIN",
        transaction_type="BUY",
        quantity=3,
    )
    assert payload["order_id"] == "abc-1"


def test_kite_client_get_quote_ohlc_unwraps_data(monkeypatch: pytest.MonkeyPatch) -> None:
    client = KiteClient(
        credentials=KiteCredentials(
            api_key="api",
            api_secret="secret",
            access_token="token",
            request_token="",
        ),
        base_url="https://api.kite.trade",
        timeout=0.1,
    )

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeResponse(
            status_code=200,
            text=(
                '{"status":"success","data":{"NSE:SBIN":{"instrument_token":779521,'
                '"ohlc":{"open":741.0,"high":748.0,"low":739.0,"close":745.0},"volume":1200}}}'
            ),
            reason="OK",
        )

    monkeypatch.setattr(client._session, "request", fake_request)
    payload = client.get_quote_ohlc(["nse:sbin"])
    assert "NSE:SBIN" in payload
    assert payload["NSE:SBIN"]["ohlc"]["close"] == 745.0
