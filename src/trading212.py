from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from . import config


class Trading212ApiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        response_text: str,
        method: str,
        url: str,
        reason: str | None = None,
    ) -> None:
        self.status_code = int(status_code)
        self.response_text = response_text
        self.method = method.upper()
        self.url = url
        self.reason = (reason or "").strip()
        reason_text = f" {self.reason}" if self.reason else ""
        super().__init__(
            f"Trading212 API error {self.status_code}{reason_text}: "
            f"{self.method} {self.url} -> {self.response_text}"
        )


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
        method_upper = str(method).upper()
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
        retries = max(0, int(config.TRADING212_HTTP_MAX_RETRIES))
        base_sleep = max(0.0, float(config.TRADING212_HTTP_RETRY_BASE_SEC))
        max_sleep = max(base_sleep, float(config.TRADING212_HTTP_RETRY_MAX_SEC))
        attempt = 0
        while True:
            start = time.monotonic()
            print(
                (
                    f"[Trading212Client] request_start method={method_upper} url={url} "
                    f"attempt={attempt + 1}/{retries + 1}"
                ),
                flush=True,
            )
            response = self._session.request(
                method=method_upper,
                url=url,
                params=params,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            elapsed_ms = round((time.monotonic() - start) * 1000.0, 1)
            if response.status_code < 400:
                if not response.text:
                    print(
                        (
                            f"[Trading212Client] request_success method={method_upper} url={url} "
                            f"status={response.status_code} elapsed_ms={elapsed_ms} body=empty"
                        ),
                        flush=True,
                    )
                    return None
                try:
                    parsed = response.json()
                except ValueError:
                    preview = _preview_text(response.text)
                    print(
                        (
                            f"[Trading212Client] request_success method={method_upper} url={url} "
                            f"status={response.status_code} elapsed_ms={elapsed_ms} body_type=text "
                            f"body_preview={preview}"
                        ),
                        flush=True,
                    )
                    return response.text
                summary = _payload_summary(parsed)
                print(
                    (
                        f"[Trading212Client] request_success method={method_upper} url={url} "
                        f"status={response.status_code} elapsed_ms={elapsed_ms} {summary}"
                    ),
                    flush=True,
                )
                return parsed
            can_retry = (
                method_upper == "GET"
                and response.status_code in {429, 500, 502, 503, 504}
                and attempt < retries
            )
            error_preview = _preview_text(response.text)
            print(
                (
                    f"[Trading212Client] request_error method={method_upper} url={url} "
                    f"status={response.status_code} elapsed_ms={elapsed_ms} retryable={str(can_retry).lower()} "
                    f"body_preview={error_preview}"
                ),
                flush=True,
            )
            if not can_retry:
                raise Trading212ApiError(
                    status_code=response.status_code,
                    response_text=response.text,
                    method=method_upper,
                    url=url,
                    reason=response.reason,
                )
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            if retry_after is None or retry_after <= 0:
                retry_after = min(max_sleep, base_sleep * (2 ** attempt))
            retry_after = max(0.0, retry_after)
            print(
                (
                    f"[Trading212Client] retrying {method_upper} {url} after {retry_after:.2f}s "
                    f"(status={response.status_code}, attempt={attempt + 1}/{retries})"
                ),
                flush=True,
            )
            time.sleep(retry_after)
            attempt += 1

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

    def get_orders(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/equity/orders")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return [item for item in data["items"] if isinstance(item, dict)]
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
        quoted_order_id = quote(str(order_id), safe="")
        data = self._request("GET", f"/equity/orders/{quoted_order_id}")
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
            try:
                last = self.get_order(order_id)
            except Trading212ApiError as exc:
                if exc.status_code == 404:
                    # Newly-created orders can briefly return 404 before they are queryable.
                    last = {
                        "id": str(order_id),
                        "status": "PENDING",
                        "poll_error": "Order not found yet",
                    }
                    time.sleep(float(poll))
                    continue
                raise
            status = str(last.get("status", "")).upper()
            filled_qty = _safe_float(last.get("filledQuantity"))
            if status == "FILLED":
                if expected_qty is None or _float_close(filled_qty, expected_qty):
                    return last
            if status in {"REJECTED", "CANCELLED"}:
                return last
            time.sleep(float(poll))
        return last

    def wait_for_orders(
        self,
        order_ids: list[int | str],
        timeout_sec: float | None = None,
        poll_sec: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        normalized_ids = [str(order_id) for order_id in order_ids if str(order_id).strip()]
        snapshots: dict[str, dict[str, Any]] = {
            order_id: {"id": order_id, "status": "PENDING"} for order_id in normalized_ids
        }
        if not normalized_ids:
            return snapshots

        timeout = timeout_sec if timeout_sec is not None else config.TRADING212_ORDER_TIMEOUT
        poll = poll_sec if poll_sec is not None else config.TRADING212_ORDER_POLL_SEC
        deadline = time.time() + float(timeout)
        remaining = set(normalized_ids)
        seen_order: dict[str, bool] = {order_id: False for order_id in normalized_ids}
        fallback_404_count: dict[str, int] = {order_id: 0 for order_id in normalized_ids}
        round_idx = 0
        fallback_every = 3
        fallback_batch_size = 2
        fallback_404_terminal_threshold_seen = 2
        fallback_404_terminal_threshold_unseen = 3
        fallback_cursor = 0

        while time.time() <= deadline and remaining:
            round_idx += 1
            try:
                orders = self.get_orders()
            except Trading212ApiError as exc:
                if exc.status_code in {404, 429}:
                    time.sleep(float(poll))
                    continue
                raise
            by_id: dict[str, dict[str, Any]] = {}
            for order in orders:
                oid = order.get("id")
                if oid is None:
                    continue
                by_id[str(oid)] = order
            unresolved: list[str] = []
            for order_id in normalized_ids:
                if order_id not in remaining:
                    continue
                payload = by_id.get(order_id)
                if payload is None:
                    unresolved.append(order_id)
                    continue
                seen_order[order_id] = True
                fallback_404_count[order_id] = 0
                snapshots[order_id] = payload
                status = str(payload.get("status", "")).upper()
                if status in {"FILLED", "REJECTED", "CANCELLED"}:
                    remaining.remove(order_id)

            if unresolved and round_idx % fallback_every == 0:
                fallback_unresolved = list(unresolved)
                fallback_count = min(fallback_batch_size, len(fallback_unresolved))
                for _ in range(fallback_count):
                    if not fallback_unresolved:
                        break
                    idx = fallback_cursor % len(fallback_unresolved)
                    order_id = fallback_unresolved.pop(idx)
                    fallback_cursor += 1
                    try:
                        payload = self.get_order(order_id)
                    except Trading212ApiError as exc:
                        if exc.status_code == 404:
                            fallback_404_count[order_id] = fallback_404_count.get(order_id, 0) + 1
                            threshold = (
                                fallback_404_terminal_threshold_seen
                                if seen_order.get(order_id, False)
                                else fallback_404_terminal_threshold_unseen
                            )
                            if (
                                fallback_404_count[order_id] >= threshold
                                and order_id in remaining
                            ):
                                resolution = (
                                    "not_returned_by_orders_endpoints_after_seen"
                                    if seen_order.get(order_id, False)
                                    else "not_returned_by_orders_endpoints"
                                )
                                snapshots[order_id] = {
                                    "id": order_id,
                                    "status": "UNKNOWN",
                                    "resolution": resolution,
                                }
                                remaining.remove(order_id)
                                print(
                                    (
                                        f"[Trading212Client] bulk order monitor inferred terminal "
                                        f"status=UNKNOWN for order_id={order_id} after repeated 404 fallback lookups"
                                    ),
                                    flush=True,
                                )
                            continue
                        if exc.status_code == 429:
                            continue
                        raise
                    snapshots[order_id] = payload
                    seen_order[order_id] = True
                    fallback_404_count[order_id] = 0
                    status = str(payload.get("status", "")).upper()
                    if status in {"FILLED", "REJECTED", "CANCELLED"} and order_id in remaining:
                        remaining.remove(order_id)

            if round_idx == 1 or round_idx % 6 == 0:
                print(
                    (
                        f"[Trading212Client] bulk order monitor round={round_idx} "
                        f"remaining={len(remaining)} timeout_sec={float(timeout):.0f}"
                    ),
                    flush=True,
                )
            if remaining:
                time.sleep(float(poll))
        return snapshots


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _preview_text(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...[truncated]"


def _payload_summary(payload: Any) -> str:
    if isinstance(payload, dict):
        keys = sorted(payload.keys())
        preview_keys = keys[:8]
        suffix = "" if len(keys) <= 8 else ",..."
        return f"body_type=dict keys={','.join(preview_keys)}{suffix}"
    if isinstance(payload, list):
        return f"body_type=list items={len(payload)}"
    return f"body_type={type(payload).__name__}"


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
