from src.trading212 import resolve_t212_ticker


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
