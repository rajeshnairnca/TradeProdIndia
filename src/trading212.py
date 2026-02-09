from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from . import config


@dataclass
class Trading212Credentials:
    api_key: str
    api_secret: str


def trading212_enabled() -> bool:
    return bool(
        config.USE_TRADING212
        and os.getenv("TRADING212_API_KEY")
        and os.getenv("TRADING212_API_SECRET")
    )


def _load_credentials() -> Trading212Credentials:
    api_key = os.getenv("TRADING212_API_KEY", "").strip()
    api_secret = os.getenv("TRADING212_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("Trading212 API credentials not configured.")
    return Trading212Credentials(api_key=api_key, api_secret=api_secret)


def _auth_header(credentials: Trading212Credentials) -> str:
    token = f"{credentials.api_key}:{credentials.api_secret}".encode("utf-8")
    encoded = base64.b64encode(token).decode("ascii")
    return f"Basic {encoded}"


class Trading212Client:
    def __init__(
        self,
        credentials: Trading212Credentials | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.credentials = credentials or _load_credentials()
        self.base_url = (base_url or config.TRADING212_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.TRADING212_TIMEOUT
        self._session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            if not path.startswith("/"):
                path = f"/{path}"
            url = f"{self.base_url}{path}"
        headers = {
            "Authorization": _auth_header(self.credentials),
            "Content-Type": "application/json",
        }
        response = self._session.request(
            method=method,
            url=url,
            params=params,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Trading212 API error {response.status_code}: {response.text}")
        if not response.text:
            return None
        return response.json()

    def get_account_summary(self) -> dict[str, Any]:
        data = self._request("GET", "/equity/account/summary")
        return data if isinstance(data, dict) else {}

    def get_positions(self, ticker: str | None = None) -> list[dict[str, Any]]:
        params = {"ticker": ticker} if ticker else None
        data = self._request("GET", "/equity/positions", params=params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        return []

    def get_instruments(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/equity/metadata/instruments")
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        return []

    def place_market_order(
        self,
        ticker: str,
        quantity: float,
        extended_hours: bool | None = None,
    ) -> dict[str, Any]:
        payload = {
            "ticker": ticker,
            "quantity": float(quantity),
            "extendedHours": bool(
                extended_hours if extended_hours is not None else config.TRADING212_EXTENDED_HOURS
            ),
        }
        data = self._request("POST", "/equity/orders/market", payload=payload)
        return data if isinstance(data, dict) else {}

    def get_order(self, order_id: int | str) -> dict[str, Any]:
        data = self._request("GET", f"/equity/orders/{order_id}")
        return data if isinstance(data, dict) else {}

    def wait_for_fill(
        self,
        order_id: int | str,
        expected_qty: float | None = None,
        timeout_sec: float | None = None,
        poll_sec: float | None = None,
    ) -> dict[str, Any]:
        timeout = timeout_sec if timeout_sec is not None else config.TRADING212_ORDER_TIMEOUT
        poll = poll_sec if poll_sec is not None else config.TRADING212_ORDER_POLL_SEC
        deadline = time.time() + float(timeout)
        last = {}
        while time.time() <= deadline:
            last = self.get_order(order_id)
            status = str(last.get("status", "")).upper()
            filled_qty = _safe_float(last.get("filledQuantity"))
            if status == "FILLED":
                if expected_qty is None or _float_close(filled_qty, expected_qty):
                    return last
            if status in {"REJECTED", "CANCELLED"}:
                return last
            time.sleep(float(poll))
        return last


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def load_instruments_cache(client: Trading212Client) -> list[dict[str, Any]]:
    cache_path = Path(config.resolve_path(config.TRADING212_INSTRUMENTS_CACHE))
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if isinstance(payload, list):
                return payload
        except json.JSONDecodeError:
            pass
    instruments = client.get_instruments()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(instruments))
    return instruments


def load_ticker_overrides() -> dict[str, str]:
    path = Path(config.resolve_path(config.TRADING212_TICKER_MAP_FILE))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in payload.items():
        tv = str(key).strip().upper()
        t212 = str(value).strip().upper()
        if tv and t212:
            cleaned[tv] = t212
    return cleaned


def build_instrument_index(
    instruments: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_ticker: dict[str, dict[str, Any]] = {}
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for inst in instruments:
        ticker = str(inst.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        by_ticker[ticker] = inst
        symbols: set[str] = set()
        symbol = ticker.split("_", 1)[0].strip().upper()
        if symbol:
            symbols.add(symbol)
        short_name = str(inst.get("shortName", "")).strip().upper()
        if short_name:
            symbols.add(short_name)
        for sym in symbols:
            by_symbol.setdefault(sym, []).append(inst)
    return by_ticker, by_symbol


def resolve_t212_ticker(
    tv_ticker: str,
    by_symbol: dict[str, list[dict[str, Any]]],
    overrides: dict[str, str],
    preferred_currency: str | None = None,
    by_ticker: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    tv = str(tv_ticker).strip().upper()
    if not tv:
        return None
    if tv in overrides:
        override = str(overrides[tv]).strip().upper()
        if override and (by_ticker is None or override in by_ticker):
            return override
    candidates = by_symbol.get(tv, [])
    if not candidates:
        return None
    preferred = (preferred_currency or config.TRADING212_PREFERRED_CURRENCY).upper()

    def score(inst: dict[str, Any]) -> tuple[int, str]:
        currency = str(inst.get("currencyCode", "")).upper()
        inst_type = str(inst.get("type", "")).upper()
        score_val = 0
        if currency == preferred:
            score_val += 2
        if inst_type in {"STOCK", "ETF"}:
            score_val += 1
        return (score_val, currency)

    best = sorted(candidates, key=score, reverse=True)[0]
    return str(best.get("ticker", "")).strip().upper() or None


def map_position_to_internal(
    t212_ticker: str,
    overrides: dict[str, str],
) -> str:
    ticker = str(t212_ticker).strip().upper()
    for tv, t212 in overrides.items():
        if t212 == ticker:
            return tv
    return ticker.split("_", 1)[0] if "_" in ticker else ticker


def positions_to_internal_positions(
    positions: list[dict[str, Any]],
    overrides: dict[str, str],
) -> dict[str, float]:
    mapped: dict[str, float] = {}
    for pos in positions:
        instrument = pos.get("instrument") or {}
        t212_ticker = pos.get("ticker") or instrument.get("ticker")
        if not t212_ticker:
            continue
        internal = map_position_to_internal(t212_ticker, overrides)
        quantity = _safe_float(pos.get("quantity"))
        if not internal:
            continue
        mapped[internal] = mapped.get(internal, 0.0) + quantity
    return mapped


def compare_positions(
    internal: dict[str, float],
    broker: dict[str, float],
    cash_internal: float,
    cash_broker: float,
    qty_tol: float = 1e-4,
    cash_tol: float = 1e-2,
) -> dict[str, Any]:
    internal_keys = set(internal.keys())
    broker_keys = set(broker.keys())
    missing = sorted(internal_keys - broker_keys)
    extra = sorted(broker_keys - internal_keys)
    mismatched = {}
    for key in sorted(internal_keys & broker_keys):
        if abs(internal.get(key, 0.0) - broker.get(key, 0.0)) > qty_tol:
            mismatched[key] = {
                "internal": internal.get(key, 0.0),
                "broker": broker.get(key, 0.0),
            }
    cash_diff = cash_broker - cash_internal
    return {
        "cash_diff": cash_diff,
        "cash_diff_abs": abs(cash_diff),
        "cash_matches": abs(cash_diff) <= cash_tol,
        "positions_missing": missing,
        "positions_extra": extra,
        "quantity_mismatches": mismatched,
    }


def extract_fx_rates(
    positions: list[dict[str, Any]],
    account_currency: str | None = None,
) -> dict[str, float]:
    rates: dict[str, list[float]] = {}
    for pos in positions:
        instrument = pos.get("instrument") or {}
        instrument_currency = str(instrument.get("currencyCode", "")).upper()
        wallet = pos.get("walletImpact") or {}
        wallet_currency = str(wallet.get("currencyCode", "")).upper()
        if account_currency:
            wallet_currency = str(account_currency).upper()
        if not instrument_currency or not wallet_currency:
            continue
        if instrument_currency == wallet_currency:
            continue
        quantity = _safe_float(pos.get("quantity"))
        current_price = _safe_float(pos.get("currentPrice"))
        wallet_value = _safe_float(wallet.get("currentValue"))
        if quantity <= 0 or current_price <= 0 or wallet_value <= 0:
            continue
        rate = wallet_value / (current_price * quantity)
        rates.setdefault(instrument_currency, []).append(rate)
    averaged: dict[str, float] = {}
    for currency, vals in rates.items():
        if vals:
            averaged[currency] = sum(vals) / len(vals)
    return averaged


def account_cash_available(summary: dict[str, Any]) -> float:
    cash = summary.get("cash") or {}
    return _safe_float(cash.get("availableToTrade"))


def account_net_worth(summary: dict[str, Any]) -> float:
    total = summary.get("totalValue")
    if total is not None:
        return _safe_float(total)
    cash = summary.get("cash") or {}
    investments = summary.get("investments") or {}
    return _safe_float(cash.get("availableToTrade")) + _safe_float(cash.get("inPies")) + _safe_float(
        cash.get("reservedForOrders")
    ) + _safe_float(investments.get("currentValue"))
