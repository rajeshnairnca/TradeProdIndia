from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from . import config


class KiteApiError(RuntimeError):
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
        self.method = str(method).upper()
        self.url = str(url)
        self.reason = (reason or "").strip()
        reason_text = f" {self.reason}" if self.reason else ""
        super().__init__(
            f"Kite API error {self.status_code}{reason_text}: "
            f"{self.method} {self.url} -> {self.response_text}"
        )


@dataclass
class KiteCredentials:
    api_key: str
    api_secret: str
    access_token: str
    request_token: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


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


def kite_enabled() -> bool:
    if not config.USE_KITE:
        return False
    api_key = os.getenv("KITE_API_KEY", "").strip()
    if not api_key:
        return False
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if access_token:
        return True
    access_token_file = str(config.KITE_ACCESS_TOKEN_FILE or "").strip()
    if access_token_file:
        token_path = Path(config.resolve_path(access_token_file))
        if token_path.exists() and token_path.read_text().strip():
            return True
    request_token = os.getenv("KITE_REQUEST_TOKEN", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    return bool(request_token and api_secret)


def _load_credentials() -> KiteCredentials:
    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    request_token = os.getenv("KITE_REQUEST_TOKEN", "").strip()
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if not access_token:
        access_token_file = str(config.KITE_ACCESS_TOKEN_FILE or "").strip()
        if access_token_file:
            token_path = Path(config.resolve_path(access_token_file))
            if token_path.exists():
                access_token = token_path.read_text().strip()
    if not api_key:
        raise RuntimeError("Kite API key is not configured. Set KITE_API_KEY.")
    return KiteCredentials(
        api_key=api_key,
        api_secret=api_secret,
        access_token=access_token,
        request_token=request_token,
    )


def _auth_header(credentials: KiteCredentials) -> str:
    token = f"{credentials.api_key}:{credentials.access_token}".strip()
    return f"token {token}"


class KiteClient:
    def __init__(
        self,
        credentials: KiteCredentials | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        auto_auth: bool = True,
    ) -> None:
        self.credentials = credentials or _load_credentials()
        self.base_url = (base_url or config.KITE_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.KITE_TIMEOUT
        self._session = requests.Session()
        if auto_auth:
            self._ensure_access_token()

    def _persist_access_token(self, access_token: str) -> None:
        token = str(access_token or "").strip()
        if not token:
            return
        access_token_file = str(config.KITE_ACCESS_TOKEN_FILE or "").strip()
        if not access_token_file:
            return
        path = Path(config.resolve_path(access_token_file))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n")

    def _ensure_access_token(self) -> None:
        if self.credentials.access_token:
            return
        if not config.KITE_SESSION_GENERATE_ON_START:
            raise RuntimeError(
                "Kite access token missing. Set KITE_ACCESS_TOKEN (or token file) "
                "or enable KITE_SESSION_GENERATE_ON_START with request token + secret."
            )
        if not self.credentials.request_token or not self.credentials.api_secret:
            raise RuntimeError(
                "Kite access token missing. Provide KITE_ACCESS_TOKEN or set "
                "KITE_REQUEST_TOKEN + KITE_API_SECRET to generate a session token."
            )
        session = self.generate_session(self.credentials.request_token)
        token = str(session.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("Kite session token generation succeeded but access_token was missing.")
        self.credentials.access_token = token
        self._persist_access_token(token)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        *,
        form_encoded: bool = False,
        auth_required: bool = True,
    ) -> Any:
        method_upper = str(method).upper()
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            if not path.startswith("/"):
                path = f"/{path}"
            url = f"{self.base_url}{path}"
        headers = {"X-Kite-Version": "3"}
        if auth_required:
            headers["Authorization"] = _auth_header(self.credentials)
        if not form_encoded:
            headers["Content-Type"] = "application/json"
        retries = max(0, int(config.KITE_HTTP_MAX_RETRIES))
        base_sleep = max(0.0, float(config.KITE_HTTP_RETRY_BASE_SEC))
        max_sleep = max(base_sleep, float(config.KITE_HTTP_RETRY_MAX_SEC))
        attempt = 0
        while True:
            start = time.monotonic()
            print(
                (
                    f"[KiteClient] request_start method={method_upper} url={url} "
                    f"attempt={attempt + 1}/{retries + 1}"
                ),
                flush=True,
            )
            response = self._session.request(
                method=method_upper,
                url=url,
                params=params,
                data=payload if form_encoded else None,
                json=None if form_encoded else payload,
                headers=headers,
                timeout=self.timeout,
            )
            elapsed_ms = round((time.monotonic() - start) * 1000.0, 1)
            if response.status_code < 400:
                if not response.text:
                    print(
                        (
                            f"[KiteClient] request_success method={method_upper} url={url} "
                            f"status={response.status_code} elapsed_ms={elapsed_ms} body=empty"
                        ),
                        flush=True,
                    )
                    return {}
                try:
                    parsed = response.json()
                except ValueError:
                    preview = _preview_text(response.text)
                    print(
                        (
                            f"[KiteClient] request_success method={method_upper} url={url} "
                            f"status={response.status_code} elapsed_ms={elapsed_ms} body_type=text "
                            f"body_preview={preview}"
                        ),
                        flush=True,
                    )
                    return response.text
                if isinstance(parsed, dict) and str(parsed.get("status", "")).lower() == "error":
                    message = str(parsed.get("message") or parsed.get("error_type") or "").strip()
                    raise KiteApiError(
                        status_code=response.status_code,
                        response_text=message or response.text,
                        method=method_upper,
                        url=url,
                        reason=message or response.reason,
                    )
                data = parsed.get("data") if isinstance(parsed, dict) and "data" in parsed else parsed
                summary = _payload_summary(data)
                print(
                    (
                        f"[KiteClient] request_success method={method_upper} url={url} "
                        f"status={response.status_code} elapsed_ms={elapsed_ms} {summary}"
                    ),
                    flush=True,
                )
                return data
            can_retry = (
                method_upper == "GET"
                and response.status_code in {429, 500, 502, 503, 504}
                and attempt < retries
            )
            error_preview = _preview_text(response.text)
            print(
                (
                    f"[KiteClient] request_error method={method_upper} url={url} "
                    f"status={response.status_code} elapsed_ms={elapsed_ms} retryable={str(can_retry).lower()} "
                    f"body_preview={error_preview}"
                ),
                flush=True,
            )
            if not can_retry:
                raise KiteApiError(
                    status_code=response.status_code,
                    response_text=response.text,
                    method=method_upper,
                    url=url,
                    reason=response.reason,
                )
            retry_after = min(max_sleep, base_sleep * (2 ** attempt))
            retry_after = max(0.0, retry_after)
            print(
                (
                    f"[KiteClient] retrying {method_upper} {url} after {retry_after:.2f}s "
                    f"(status={response.status_code}, attempt={attempt + 1}/{retries})"
                ),
                flush=True,
            )
            time.sleep(retry_after)
            attempt += 1

    def _request_csv(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": _auth_header(self.credentials),
            "X-Kite-Version": "3",
        }
        start = time.monotonic()
        response = self._session.request(
            method="GET",
            url=url,
            headers=headers,
            timeout=self.timeout,
        )
        elapsed_ms = round((time.monotonic() - start) * 1000.0, 1)
        if response.status_code >= 400:
            raise KiteApiError(
                status_code=response.status_code,
                response_text=response.text,
                method="GET",
                url=url,
                reason=response.reason,
            )
        print(
            (
                f"[KiteClient] request_success method=GET url={url} "
                f"status={response.status_code} elapsed_ms={elapsed_ms} body_type=csv"
            ),
            flush=True,
        )
        return response.text

    def generate_session(self, request_token: str) -> dict[str, Any]:
        token = str(request_token or "").strip()
        if not token:
            raise RuntimeError("Kite request token is required to generate a session.")
        secret = str(self.credentials.api_secret or "").strip()
        if not secret:
            raise RuntimeError("Kite API secret is required to generate a session.")
        checksum = hashlib.sha256(
            f"{self.credentials.api_key}{token}{secret}".encode("utf-8")
        ).hexdigest()
        data = self._request(
            "POST",
            "/session/token",
            payload={
                "api_key": self.credentials.api_key,
                "request_token": token,
                "checksum": checksum,
            },
            form_encoded=True,
            auth_required=False,
        )
        payload = data if isinstance(data, dict) else {}
        access_token = str(payload.get("access_token") or "").strip()
        if access_token:
            self.credentials.access_token = access_token
            self._persist_access_token(access_token)
        return payload

    def get_margins(self) -> dict[str, Any]:
        data = self._request("GET", "/user/margins")
        return data if isinstance(data, dict) else {}

    def get_holdings(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/portfolio/holdings")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def get_positions(self) -> dict[str, Any]:
        data = self._request("GET", "/portfolio/positions")
        return data if isinstance(data, dict) else {}

    def get_orders(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/orders")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def get_order_history(self, order_id: int | str) -> list[dict[str, Any]]:
        quoted = quote(str(order_id), safe="")
        data = self._request("GET", f"/orders/{quoted}")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def get_instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        ex = str(exchange or "").strip().upper()
        path = f"/instruments/{ex}" if ex else "/instruments"
        raw = self._request_csv(path)
        rows = list(csv.DictReader(io.StringIO(raw)))
        cleaned: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            payload = {str(k or "").strip(): v for k, v in row.items() if str(k or "").strip()}
            if payload:
                cleaned.append(payload)
        return cleaned

    def get_quote_ohlc(self, instruments: list[str]) -> dict[str, dict[str, Any]]:
        cleaned = [str(item).strip().upper() for item in instruments if str(item).strip()]
        if not cleaned:
            return {}
        data = self._request("GET", "/quote/ohlc", params={"i": cleaned})
        if not isinstance(data, dict):
            return {}
        output: dict[str, dict[str, Any]] = {}
        for key, payload in data.items():
            instrument = str(key or "").strip().upper()
            if not instrument or not isinstance(payload, dict):
                continue
            output[instrument] = payload
        return output

    def place_market_order(
        self,
        *,
        exchange: str,
        tradingsymbol: str,
        transaction_type: str,
        quantity: int,
        product: str | None = None,
        variety: str | None = None,
    ) -> dict[str, Any]:
        data = self._request(
            "POST",
            f"/orders/{str(variety or config.KITE_ORDER_VARIETY).strip().lower()}",
            payload={
                "exchange": str(exchange).strip().upper(),
                "tradingsymbol": str(tradingsymbol).strip().upper(),
                "transaction_type": str(transaction_type).strip().upper(),
                "quantity": int(quantity),
                "product": str(product or config.KITE_PRODUCT).strip().upper(),
                "order_type": "MARKET",
                "validity": "DAY",
            },
            form_encoded=True,
        )
        return data if isinstance(data, dict) else {}

    def wait_for_fill(
        self,
        order_id: int | str,
        expected_qty: float | None = None,
        timeout_sec: float | None = None,
        poll_sec: float | None = None,
    ) -> dict[str, Any]:
        timeout = timeout_sec if timeout_sec is not None else config.KITE_ORDER_TIMEOUT
        poll = poll_sec if poll_sec is not None else config.KITE_ORDER_POLL_SEC
        deadline = time.time() + float(timeout)
        fallback = {"order_id": str(order_id), "status": "PENDING"}
        while time.time() <= deadline:
            history = self.get_order_history(order_id)
            payload = history[-1] if history else fallback
            normalized = normalize_kite_order_snapshot(payload)
            status = str(normalized.get("status", "")).upper()
            filled_qty = _safe_float(normalized.get("filledQuantity"))
            if status == "FILLED":
                if expected_qty is None or _float_close(filled_qty, expected_qty):
                    return normalized
            if status in {"REJECTED", "CANCELLED"}:
                return normalized
            time.sleep(float(poll))
        return normalize_kite_order_snapshot(fallback)

    def wait_for_orders(
        self,
        order_ids: list[int | str],
        timeout_sec: float | None = None,
        poll_sec: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        normalized_ids = [str(order_id).strip() for order_id in order_ids if str(order_id).strip()]
        snapshots: dict[str, dict[str, Any]] = {
            order_id: {"order_id": order_id, "status": "PENDING"} for order_id in normalized_ids
        }
        if not normalized_ids:
            return snapshots
        timeout = timeout_sec if timeout_sec is not None else config.KITE_ORDER_TIMEOUT
        poll = poll_sec if poll_sec is not None else config.KITE_ORDER_POLL_SEC
        deadline = time.time() + float(timeout)
        remaining = set(normalized_ids)
        round_idx = 0
        while time.time() <= deadline and remaining:
            round_idx += 1
            orders = self.get_orders()
            by_id: dict[str, dict[str, Any]] = {}
            for order in orders:
                order_id = order.get("order_id") or order.get("id")
                if order_id is None:
                    continue
                by_id[str(order_id)] = order
            unresolved = []
            for order_id in list(remaining):
                payload = by_id.get(order_id)
                if payload is None:
                    unresolved.append(order_id)
                    continue
                normalized = normalize_kite_order_snapshot(payload)
                snapshots[order_id] = normalized
                if str(normalized.get("status", "")).upper() in {"FILLED", "REJECTED", "CANCELLED"}:
                    remaining.discard(order_id)
            if unresolved and round_idx % 3 == 0:
                for order_id in unresolved[:2]:
                    history = self.get_order_history(order_id)
                    payload = history[-1] if history else {"order_id": order_id, "status": "UNKNOWN"}
                    normalized = normalize_kite_order_snapshot(payload)
                    snapshots[order_id] = normalized
                    if str(normalized.get("status", "")).upper() in {"FILLED", "REJECTED", "CANCELLED"}:
                        remaining.discard(order_id)
            if remaining:
                time.sleep(float(poll))
        return snapshots


def _normalize_kite_status(status: str) -> str:
    raw = str(status or "").strip().upper()
    if raw in {"COMPLETE", "TRADED"}:
        return "FILLED"
    if raw in {"CANCELLED", "CANCELED"}:
        return "CANCELLED"
    if raw in {"REJECTED"}:
        return "REJECTED"
    if raw in {
        "OPEN",
        "TRIGGER PENDING",
        "PUT ORDER REQ RECEIVED",
        "VALIDATION PENDING",
        "MODIFY VALIDATION PENDING",
        "MODIFY PENDING",
        "AMO REQ RECEIVED",
        "AMO REQ REJECTED",
        "PENDING",
    }:
        return "PENDING"
    if raw:
        return raw
    return "UNKNOWN"


def normalize_kite_order_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    order_id = payload.get("order_id") or payload.get("id")
    status = _normalize_kite_status(str(payload.get("status") or ""))
    filled_qty = abs(_safe_float(payload.get("filled_quantity")))
    if filled_qty <= 0 and status == "FILLED":
        filled_qty = abs(_safe_float(payload.get("quantity")))
    fill_price = abs(_safe_float(payload.get("average_price")))
    if fill_price <= 0:
        fill_price = abs(_safe_float(payload.get("price")))
    filled_value = abs(_safe_float(payload.get("filled_value")))
    if filled_value <= 0 and filled_qty > 0 and fill_price > 0:
        filled_value = filled_qty * fill_price
    snapshot: dict[str, Any] = {
        "id": str(order_id) if order_id is not None else None,
        "status": status,
    }
    if filled_qty > 0:
        snapshot["filledQuantity"] = filled_qty
    if fill_price > 0:
        snapshot["fillPrice"] = fill_price
        snapshot["resolutionPriceSource"] = "orders.average_price"
    if filled_value > 0:
        snapshot["filledValue"] = filled_value
    snapshot["currency"] = "INR"
    snapshot["filledValueCurrency"] = "INR"
    snapshot["fillPriceCurrency"] = "INR"
    snapshot["raw"] = payload
    return snapshot


def load_instruments_cache(client: KiteClient) -> list[dict[str, Any]]:
    cache_path = Path(config.resolve_path(config.KITE_INSTRUMENTS_CACHE))
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if isinstance(payload, list):
                return payload
        except json.JSONDecodeError:
            pass
    exchange = str(config.KITE_INSTRUMENTS_EXCHANGE or "").strip().upper() or None
    instruments = client.get_instruments(exchange=exchange)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(instruments))
    return instruments


def _parse_override_target(value: Any) -> tuple[str, str] | None:
    if isinstance(value, dict):
        exchange = str(value.get("exchange") or config.KITE_DEFAULT_EXCHANGE).strip().upper()
        tradingsymbol = str(value.get("tradingsymbol") or value.get("symbol") or "").strip().upper()
        if exchange and tradingsymbol:
            return exchange, tradingsymbol
        return None
    raw = str(value or "").strip().upper()
    if not raw:
        return None
    if ":" in raw:
        exchange, symbol = raw.split(":", 1)
        exchange = exchange.strip().upper()
        symbol = symbol.strip().upper()
        if exchange and symbol:
            return exchange, symbol
        return None
    exchange = str(config.KITE_DEFAULT_EXCHANGE).strip().upper()
    return (exchange, raw) if exchange and raw else None


def load_ticker_overrides() -> dict[str, tuple[str, str]]:
    path = Path(config.resolve_path(config.KITE_TICKER_MAP_FILE))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, tuple[str, str]] = {}
    for key, value in payload.items():
        ticker = str(key).strip().upper()
        if not ticker:
            continue
        target = _parse_override_target(value)
        if target:
            cleaned[ticker] = target
    return cleaned


def build_instrument_index(
    instruments: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_key: dict[str, dict[str, Any]] = {}
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for inst in instruments:
        exchange = str(inst.get("exchange") or "").strip().upper()
        tradingsymbol = str(inst.get("tradingsymbol") or "").strip().upper()
        if not exchange or not tradingsymbol:
            continue
        key = f"{exchange}:{tradingsymbol}"
        by_key[key] = inst
        names = {tradingsymbol}
        name = str(inst.get("name") or "").strip().upper()
        if name:
            names.add(name)
        for symbol in names:
            by_symbol.setdefault(symbol, []).append(inst)
    return by_key, by_symbol


def _infer_from_tv_ticker(tv_ticker: str) -> tuple[str, str] | None:
    ticker = str(tv_ticker or "").strip().upper()
    if not ticker:
        return None
    if ":" in ticker:
        exchange, symbol = ticker.split(":", 1)
        exchange = exchange.strip().upper()
        symbol = symbol.strip().upper()
        if exchange and symbol:
            return exchange, symbol
        return None
    suffix_map = {
        ".NS": "NSE",
        ".NSE": "NSE",
        ".BO": "BSE",
        ".BSE": "BSE",
    }
    for suffix, exchange in suffix_map.items():
        if ticker.endswith(suffix) and len(ticker) > len(suffix):
            return exchange, ticker[: -len(suffix)]
    default_exchange = str(config.KITE_DEFAULT_EXCHANGE).strip().upper()
    if not default_exchange:
        return None
    return default_exchange, ticker


def resolve_kite_instrument(
    tv_ticker: str,
    overrides: dict[str, tuple[str, str]],
    by_key: dict[str, dict[str, Any]] | None = None,
    by_symbol: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, str] | None:
    ticker = str(tv_ticker or "").strip().upper()
    if not ticker:
        return None
    target = overrides.get(ticker)
    if target:
        exchange, symbol = target
        key = f"{exchange}:{symbol}"
        if by_key is None or key in by_key:
            return exchange, symbol
    inferred = _infer_from_tv_ticker(ticker)
    if inferred is None:
        return None
    exchange, symbol = inferred
    key = f"{exchange}:{symbol}"
    if by_key is None or key in by_key:
        return exchange, symbol
    if by_symbol is None:
        return None
    candidates = by_symbol.get(symbol) or []
    if not candidates:
        return None
    preferred_exchange = str(config.KITE_DEFAULT_EXCHANGE).strip().upper()
    ranked = sorted(
        candidates,
        key=lambda inst: (
            str(inst.get("exchange") or "").strip().upper() == preferred_exchange,
            str(inst.get("exchange") or "").strip().upper() == exchange,
        ),
        reverse=True,
    )
    best = ranked[0]
    resolved_exchange = str(best.get("exchange") or "").strip().upper()
    resolved_symbol = str(best.get("tradingsymbol") or "").strip().upper()
    if not resolved_exchange or not resolved_symbol:
        return None
    return resolved_exchange, resolved_symbol


def map_position_to_internal(
    exchange: str,
    tradingsymbol: str,
    overrides: dict[str, tuple[str, str]],
) -> str:
    ex = str(exchange or "").strip().upper()
    symbol = str(tradingsymbol or "").strip().upper()
    if not symbol:
        return ""
    for internal_ticker, target in overrides.items():
        if target == (ex, symbol):
            return internal_ticker
    if ex == "NSE":
        return f"{symbol}.NS"
    if ex == "BSE":
        return f"{symbol}.BO"
    return symbol


def positions_to_internal_positions(
    positions: list[dict[str, Any]],
    overrides: dict[str, tuple[str, str]],
) -> dict[str, float]:
    mapped: dict[str, float] = {}
    for pos in positions:
        exchange = str(pos.get("exchange") or config.KITE_DEFAULT_EXCHANGE).strip().upper()
        tradingsymbol = str(pos.get("tradingsymbol") or "").strip().upper()
        if not tradingsymbol:
            continue
        quantity = _safe_float(pos.get("quantity"))
        quantity += _safe_float(pos.get("t1_quantity"))
        if abs(quantity) <= 1e-9:
            continue
        internal = map_position_to_internal(exchange, tradingsymbol, overrides)
        if not internal:
            continue
        mapped[internal] = mapped.get(internal, 0.0) + quantity
    return mapped


def account_cash_available(margins: dict[str, Any]) -> float:
    equity = margins.get("equity")
    block = equity if isinstance(equity, dict) else margins
    available = block.get("available")
    available = available if isinstance(available, dict) else {}
    for key in ("live_balance", "cash", "opening_balance"):
        value = _safe_float(available.get(key), default=float("nan"))
        if value == value:
            return value
    net = _safe_float(block.get("net"), default=float("nan"))
    if net == net:
        return net
    return 0.0


def account_net_worth(
    margins: dict[str, Any],
    holdings: list[dict[str, Any]],
) -> float:
    cash = account_cash_available(margins)
    investments = 0.0
    for row in holdings:
        quantity = _safe_float(row.get("quantity")) + _safe_float(row.get("t1_quantity"))
        price = _safe_float(row.get("last_price"))
        if quantity <= 0 or price <= 0:
            continue
        investments += quantity * price
    return cash + investments
