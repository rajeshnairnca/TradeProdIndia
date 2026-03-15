import os
import sys
import time
import json
import secrets
from datetime import datetime, timezone
from typing import List, Literal, Optional
from urllib.parse import urlencode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from src import config
from src.cagr_metrics import compute_cagr_summary
from src.entry_indicator import (
    DEFAULT_BASE_STRATEGIES as ENTRY_DEFAULT_BASE_STRATEGIES,
    DEFAULT_REGIME_MAPPING as ENTRY_DEFAULT_REGIME_MAPPING,
    compute_entry_indicator_payload,
)
from src.production_db import (
    RUN_SUMMARY_FIELDS as DB_RUN_SUMMARY_FIELDS,
    append_pending_adjustments as db_append_pending_adjustments,
    count_broker_orders as db_count_broker_orders,
    count_latest_broker_orders as db_count_latest_broker_orders,
    clear_pending_adjustments as db_clear_pending_adjustments,
    db_enabled,
    delete_run_calendar_override as db_delete_run_calendar_override,
    init_db as db_init,
    list_excluded_tickers as db_list_excluded_tickers,
    list_latest_broker_orders as db_list_latest_broker_orders,
    list_latest_broker_positions as db_list_latest_broker_positions,
    list_latest_trades as db_list_latest_trades,
    latest_entry_indicator_snapshot as db_latest_entry_indicator_snapshot,
    list_pending_adjustments as db_list_pending_adjustments,
    list_run_calendar_overrides_paginated as db_list_run_calendar_overrides_paginated,
    list_run_summaries_paginated as db_list_run_summaries_paginated,
    list_universe_map as db_list_universe_map,
    latest_run_date as db_latest_run_date,
    latest_prices as db_latest_prices,
    latest_price_run_date as db_latest_price_run_date,
    latest_summary as db_latest_summary,
    latest_broker_account as db_latest_broker_account,
    latest_universe_monitor_summary as db_latest_universe_monitor_summary,
    latest_universe_selection_diagnostics_state as db_latest_universe_selection_diagnostics_state,
    list_run_summaries as db_list_run_summaries,
    list_broker_orders as db_list_broker_orders,
    list_trades as db_list_trades,
    list_universe_monitor_candidates as db_list_universe_monitor_candidates,
    list_universe_selection_diagnostics_records as db_list_universe_selection_diagnostics_records,
    load_broker_auth_state as db_load_broker_auth_state,
    load_excluded_tickers as db_load_excluded_tickers,
    load_run_calendar_override as db_load_run_calendar_override,
    load_universe_map as db_load_universe_map,
    price_tickers_for_date as db_price_tickers_for_date,
    replace_excluded_tickers as db_replace_excluded_tickers,
    load_state as db_load_state,
    reset_production_data as db_reset_production_data,
    upsert_broker_auth_state as db_upsert_broker_auth_state,
    upsert_entry_indicator_snapshot as db_upsert_entry_indicator_snapshot,
    upsert_run_calendar_override as db_upsert_run_calendar_override,
)
from src.kite import KiteClient, KiteCredentials
from src.run_calendar import (
    RUN_CALENDAR_ACTION_FORCE_RUN,
    RUN_CALENDAR_ACTION_SKIP,
    evaluate_run_day,
    list_us_federal_holidays,
)

API_KEY = os.getenv("API_KEY", "").strip()
FX_PROVIDER_URL = os.getenv("FX_PROVIDER_URL", "https://api.frankfurter.app/latest")
FX_TIMEOUT_SECONDS = float(os.getenv("FX_TIMEOUT_SECONDS", "4.0"))
FX_CACHE_TTL_SECONDS = int(os.getenv("FX_CACHE_TTL_SECONDS", "3600"))
SUPPORTED_FX_CURRENCIES = {"USD", "GBP"}
_fx_cache: dict[tuple[str, str], dict[str, object]] = {}
DEFAULT_PAGE_LIMIT = int(os.getenv("API_DEFAULT_PAGE_LIMIT", "200"))
MAX_PAGE_LIMIT = int(os.getenv("API_MAX_PAGE_LIMIT", "1000"))
BASE_CURRENCY = os.getenv("BASE_CURRENCY", "INR").strip().upper() or "INR"
_INDIA_REGION = str(getattr(config, "TRADING_REGION", "")).strip().lower() == "india"
ENTRY_INDICATOR_CACHE_TTL_SECONDS = int(
    os.getenv("ENTRY_INDICATOR_CACHE_TTL_SECONDS", "1800")
)
_entry_indicator_cache: dict[tuple, dict[str, object]] = {}
KITE_CONNECT_LOGIN_URL = os.getenv(
    "KITE_CONNECT_LOGIN_URL",
    "https://kite.zerodha.com/connect/login",
).strip()


def _resolve_default_broker_query() -> str:
    explicit = os.getenv("DEFAULT_BROKER", "").strip().lower()
    if explicit:
        return explicit
    # India-only deployment default.
    return "kite"


DEFAULT_BROKER_QUERY = _resolve_default_broker_query()

if not db_enabled():
    raise RuntimeError(
        "Database is required for the production API. Set DATABASE_URL or POSTGRES_URL."
    )
db_init()


def _normalize_currency(code: str | None, default: str = "USD") -> str:
    value = str(code or default).strip().upper()
    if value not in SUPPORTED_FX_CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported currency '{value}'. Supported: {sorted(SUPPORTED_FX_CURRENCIES)}",
        )
    return value


def _to_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(parsed):
        return None
    return parsed


def _derive_usd_to_gbp_from_db() -> float | None:
    rows = db_list_run_summaries()
    if not rows:
        return None
    candidate_pairs = (
        ("broker_net_worth", "broker_net_worth_usd"),
        ("broker_execution_cost", "broker_execution_cost_usd"),
        ("broker_total_execution_cost", "broker_total_execution_cost_usd"),
    )
    for row in reversed(rows):
        currency = str(row.get("broker_currency") or "").strip().upper()
        if currency == "USD":
            return 1.0
        if currency != "GBP":
            continue
        for native_key, usd_key in candidate_pairs:
            native = _to_float(row.get(native_key))
            usd = _to_float(row.get(usd_key))
            if native is None or usd is None or abs(usd) <= 1e-12:
                continue
            ratio = native / usd
            if 0.1 <= ratio <= 10.0:
                return ratio
    return None


def _fetch_usd_to_gbp_from_provider() -> float | None:
    try:
        response = requests.get(
            FX_PROVIDER_URL,
            params={"from": "USD", "to": "GBP"},
            timeout=FX_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None
    if response.status_code < 200 or response.status_code >= 300:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    rates = payload.get("rates")
    if not isinstance(rates, dict):
        return None
    rate = _to_float(rates.get("GBP"))
    if rate is None or rate <= 0:
        return None
    return rate


def _get_usd_to_gbp_rate() -> tuple[float | None, str]:
    cache_key = ("USD", "GBP")
    now = time.time()
    cached = _fx_cache.get(cache_key)
    if cached:
        fetched_at = _to_float(cached.get("fetched_at_epoch"))
        rate = _to_float(cached.get("rate"))
        if (
            fetched_at is not None
            and rate is not None
            and (now - fetched_at) <= FX_CACHE_TTL_SECONDS
        ):
            return rate, "cache"

    provider_rate = _fetch_usd_to_gbp_from_provider()
    if provider_rate is not None:
        _fx_cache[cache_key] = {
            "rate": provider_rate,
            "fetched_at_epoch": now,
        }
        return provider_rate, "provider"

    db_rate = _derive_usd_to_gbp_from_db()
    if db_rate is not None:
        return db_rate, "db_fallback"

    return None, "unavailable"


def _load_effective_excluded_tickers() -> tuple[set[str], str]:
    return db_load_excluded_tickers(), "db"


def _normalize_exchange_token(value: str | None) -> str | None:
    token = str(value or "").strip().upper()
    return token or None


def _is_exchange_like(value: str | None) -> bool:
    token = _normalize_exchange_token(value)
    if not token:
        return False
    if token in {"NSE", "BSE", "NFO", "BFO", "MCX"}:
        return True
    return token.isalpha() and len(token) <= 5


def _parse_ticker_exchange_pair(
    raw_ticker: str | None,
    raw_exchange: str | None = None,
) -> tuple[str | None, str | None]:
    ticker = _normalize_exchange_token(raw_ticker)
    exchange = _normalize_exchange_token(raw_exchange)
    if ticker and ":" in ticker:
        left, right = ticker.split(":", 1)
        left = _normalize_exchange_token(left)
        right = _normalize_exchange_token(right)
        if not exchange:
            if _is_exchange_like(left) and right:
                exchange, ticker = left, right
            elif _is_exchange_like(right) and left:
                exchange, ticker = right, left
            else:
                ticker, exchange = left, right
    if exchange and ":" in exchange:
        left, right = exchange.split(":", 1)
        left = _normalize_exchange_token(left)
        right = _normalize_exchange_token(right)
        if _is_exchange_like(left):
            exchange = left
            if not ticker and right:
                ticker = right
        elif _is_exchange_like(right):
            exchange = right
            if not ticker and left:
                ticker = left
    return ticker, exchange


def _force_base_currency(value: str | None) -> str:
    if _INDIA_REGION:
        return BASE_CURRENCY
    token = str(value or "").strip().upper()
    return token or BASE_CURRENCY


def _apply_summary_currency_fields(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if "broker_currency" in out:
        out["broker_currency"] = _force_base_currency(out.get("broker_currency"))
    return out


def _apply_broker_summary_currency(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if "currency" in out:
        out["currency"] = _force_base_currency(out.get("currency"))
    return out


def _apply_broker_positions_currency(rows: list[dict]) -> list[dict]:
    if not _INDIA_REGION:
        return rows
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        if "account_currency" in item:
            item["account_currency"] = BASE_CURRENCY
        out.append(item)
    return out


def _apply_broker_orders_currency(rows: list[dict]) -> list[dict]:
    if not _INDIA_REGION:
        return rows
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        if "currency" in item:
            item["currency"] = BASE_CURRENCY
        out.append(item)
    return out


class AdjustmentRequest(BaseModel):
    cash_amount: float = 0.0
    cash_note: str = "app"
    tickers: Optional[List[str]] = None
    ticker_exchanges: Optional[dict[str, str] | List[str]] = None
    source: str = "app"


class ExcludeTickersRequest(BaseModel):
    tickers: List[str]


class ResetRequest(BaseModel):
    confirm: str
    preserve_universe_monitor: bool = True


class RunCalendarOverrideRequest(BaseModel):
    date: str
    action: Literal["skip", "force_run"] = RUN_CALENDAR_ACTION_SKIP
    reason: str = ""
    source: str = "app"


def _normalize_iso_date(value: str, field_name: str = "date") -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    try:
        return str(pd.to_datetime(text).date())
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} format. Use YYYY-MM-DD.",
        )


def _normalize_pagination(
    limit: int,
    offset: int,
    *,
    default_limit: int = DEFAULT_PAGE_LIMIT,
    max_limit: int = MAX_PAGE_LIMIT,
) -> tuple[int, int]:
    normalized_limit = default_limit if limit is None else int(limit)
    normalized_offset = 0 if offset is None else int(offset)
    if normalized_limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")
    if normalized_offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if normalized_limit > max_limit:
        raise HTTPException(status_code=400, detail=f"limit must be <= {max_limit}")
    return normalized_limit, normalized_offset


def _parse_csv_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [value.strip() for value in str(raw).split(",") if value.strip()]


BROKER_SUMMARY_FIELDS: tuple[str, ...] = (
    "run_date",
    "broker",
    "currency",
    "cash",
    "investments",
    "net_worth",
    "payload",
)
BROKER_POSITION_FIELDS: tuple[str, ...] = (
    "run_date",
    "broker",
    "ticker",
    "quantity",
    "average_price",
    "current_price",
    "instrument_currency",
    "account_currency",
    "wallet_current_value",
    "payload",
)
BROKER_ORDER_FIELDS: tuple[str, ...] = (
    "run_date",
    "broker",
    "ticker",
    "action",
    "quantity",
    "filled_quantity",
    "exec_price",
    "currency",
    "status",
    "order_id",
    "payload",
    "created_at",
)
SUMMARY_FIELDS: tuple[str, ...] = DB_RUN_SUMMARY_FIELDS


def _normalize_broker_fields(
    raw_fields: str | None,
    *,
    allowed: tuple[str, ...],
    include_payload: bool,
) -> list[str] | None:
    requested = _parse_csv_values(raw_fields)
    if not requested:
        return None
    allowed_set = set(allowed)
    unknown = sorted({field for field in requested if field not in allowed_set})
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fields: {', '.join(unknown)}",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for field in requested:
        if not include_payload and field == "payload":
            continue
        if field in seen:
            continue
        seen.add(field)
        normalized.append(field)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="No valid fields requested after applying include_payload=false.",
        )
    return normalized


def _normalize_summary_fields(raw_fields: str | None) -> list[str] | None:
    requested = _parse_csv_values(raw_fields)
    if not requested:
        return None
    allowed_set = set(SUMMARY_FIELDS)
    unknown = sorted({field for field in requested if field not in allowed_set})
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fields: {', '.join(unknown)}",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for field in requested:
        if field in seen:
            continue
        seen.add(field)
        normalized.append(field)
    return normalized


def _normalize_regime_mapping(raw: str | None) -> dict[str, str]:
    if not raw:
        return {str(k): str(v) for k, v in ENTRY_DEFAULT_REGIME_MAPPING.items()}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid regime_mapping JSON: {exc}")
    if not isinstance(parsed, dict) or not parsed:
        raise HTTPException(status_code=400, detail="regime_mapping must be a non-empty JSON object.")
    normalized = {
        str(k).strip(): str(v).strip()
        for k, v in parsed.items()
        if str(k).strip() and str(v).strip()
    }
    if not normalized:
        raise HTTPException(status_code=400, detail="regime_mapping must contain non-empty keys/values.")
    return normalized


def _entry_indicator_cache_key(
    *,
    strategy_roots: list[str],
    strategies: list[str],
    regime_mapping: dict[str, str],
    start_date: str,
    end_date: str | None,
    as_of_date: str | None,
    lookahead_days: int,
    confirm_days: int,
    confirm_days_sideways: int,
    rebalance_every: int,
    ignore_stock_filters: bool,
) -> tuple:
    return (
        tuple(strategy_roots),
        tuple(strategies),
        tuple(sorted(regime_mapping.items())),
        str(start_date),
        str(end_date) if end_date else None,
        str(as_of_date) if as_of_date else None,
        int(lookahead_days),
        int(confirm_days),
        int(confirm_days_sideways),
        int(rebalance_every),
        bool(ignore_stock_filters),
    )


def _load_kite_auth_state() -> dict[str, object]:
    state = db_load_broker_auth_state("kite")
    if isinstance(state, dict):
        return state
    return {}


def _resolve_kite_redirect_url(request: Request) -> str:
    configured = str(getattr(config, "KITE_REDIRECT_URL", "") or "").strip()
    if configured:
        return configured
    base = str(request.base_url).rstrip("/")
    return f"{base}/kite/auth/callback"


def _read_kite_access_token_file() -> str:
    token_file = str(getattr(config, "KITE_ACCESS_TOKEN_FILE", "") or "").strip()
    if not token_file:
        return ""
    path = config.resolve_path(token_file)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return str(handle.read()).strip()
    except OSError:
        return ""


def _utc_from_epoch(epoch_value: int | None) -> str | None:
    if epoch_value is None:
        return None
    try:
        epoch_int = int(epoch_value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(epoch_int, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    if not API_KEY:
        return
    if x_api_key and x_api_key == API_KEY:
        return
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        if token == API_KEY:
            return
    raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title="Trading Production API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/kite/auth/login", dependencies=[Depends(_require_api_key)])
def kite_auth_login(request: Request):
    api_key = os.getenv("KITE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="KITE_API_KEY is not configured.")

    redirect_url = _resolve_kite_redirect_url(request)
    state = secrets.token_urlsafe(24)
    now_epoch = int(time.time())
    expires_epoch = now_epoch + int(getattr(config, "KITE_AUTH_STATE_TTL_SECONDS", 900))

    db_upsert_broker_auth_state(
        "kite",
        api_key=api_key,
        pending_state=state,
        pending_state_expires_at_epoch=expires_epoch,
    )

    login_params = {
        "v": "3",
        "api_key": api_key,
        "redirect_url": redirect_url,
        "state": state,
    }
    login_url = f"{KITE_CONNECT_LOGIN_URL}?{urlencode(login_params)}"

    return {
        "broker": "kite",
        "login_url": login_url,
        "redirect_url": redirect_url,
        "state": state,
        "state_expires_at_epoch": expires_epoch,
        "state_expires_at_utc": _utc_from_epoch(expires_epoch),
    }


@app.get("/kite/auth/callback")
def kite_auth_callback(
    request_token: str | None = None,
    status: str | None = None,
    action: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_type: str | None = None,
):
    error_value = str(error or "").strip()
    if error_value:
        detail = str(error_type or "kite_auth_error").strip()
        raise HTTPException(status_code=400, detail=f"Kite auth failed ({detail}): {error_value}")

    request_token_value = str(request_token or "").strip()
    if not request_token_value:
        raise HTTPException(status_code=400, detail="Missing request_token in Kite callback.")

    auth_state = _load_kite_auth_state()
    expected_state = str(auth_state.get("pending_state") or "").strip()
    expected_expiry = auth_state.get("pending_state_expires_at_epoch")
    if expected_state:
        callback_state = str(state or "").strip()
        if not callback_state or callback_state != expected_state:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid callback state. Start a fresh login via "
                    "/kite/auth/login and retry."
                ),
            )
        try:
            expiry_epoch = int(expected_expiry) if expected_expiry is not None else None
        except (TypeError, ValueError):
            expiry_epoch = None
        if expiry_epoch is not None and int(time.time()) > expiry_epoch:
            raise HTTPException(
                status_code=400,
                detail="Kite login state has expired. Start a fresh login via /kite/auth/login.",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="No pending Kite login state found. Start via /kite/auth/login first.",
        )

    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise HTTPException(
            status_code=503,
            detail="KITE_API_KEY and KITE_API_SECRET must be configured before callback exchange.",
        )

    credentials = KiteCredentials(
        api_key=api_key,
        api_secret=api_secret,
        access_token="",
        request_token=request_token_value,
    )
    client = KiteClient(credentials=credentials, auto_auth=False)
    try:
        session = client.generate_session(request_token_value)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Kite session exchange failed: {type(exc).__name__}: {exc}",
        )

    access_token = str(session.get("access_token") or client.credentials.access_token or "").strip()
    if not access_token:
        raise HTTPException(
            status_code=502,
            detail="Kite session exchange did not return an access_token.",
        )

    db_upsert_broker_auth_state(
        "kite",
        access_token=access_token,
        request_token=request_token_value,
        api_key=api_key,
        user_id=str(session.get("user_id") or "").strip() or None,
        session_payload=session if isinstance(session, dict) else {"raw_session": str(session)},
        pending_state=None,
        pending_state_expires_at_epoch=None,
    )

    return {
        "ok": True,
        "broker": "kite",
        "status": "authenticated",
        "kite_status": str(status or "").strip() or None,
        "kite_action": str(action or "").strip() or None,
        "user_id": str(session.get("user_id") or "").strip() or None,
        "has_access_token": True,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/kite/auth/status", dependencies=[Depends(_require_api_key)])
def kite_auth_status():
    auth_state = _load_kite_auth_state()
    db_access_token = str(auth_state.get("access_token") or "").strip()
    env_access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    file_access_token = _read_kite_access_token_file()

    token_source = "none"
    if db_access_token:
        token_source = "db"
    elif env_access_token:
        token_source = "env"
    elif file_access_token:
        token_source = "file"

    pending_state = str(auth_state.get("pending_state") or "").strip()
    pending_expiry_raw = auth_state.get("pending_state_expires_at_epoch")
    try:
        pending_expiry = int(pending_expiry_raw) if pending_expiry_raw is not None else None
    except (TypeError, ValueError):
        pending_expiry = None
    now_epoch = int(time.time())

    return {
        "broker": "kite",
        "has_access_token": bool(db_access_token or env_access_token or file_access_token),
        "access_token_source": token_source,
        "db_token_present": bool(db_access_token),
        "env_token_present": bool(env_access_token),
        "file_token_present": bool(file_access_token),
        "user_id": str(auth_state.get("user_id") or "").strip() or None,
        "pending_login": bool(pending_state),
        "pending_state_expires_at_epoch": pending_expiry,
        "pending_state_expires_at_utc": _utc_from_epoch(pending_expiry),
        "pending_state_is_expired": bool(
            pending_state and pending_expiry is not None and pending_expiry < now_epoch
        ),
    }


@app.get("/fx-rate", dependencies=[Depends(_require_api_key)])
def fx_rate(base: str = "USD", quote: str = "GBP"):
    base_code = _normalize_currency(base, default="USD")
    quote_code = _normalize_currency(quote, default="GBP")
    if base_code == quote_code:
        return {
            "base": base_code,
            "quote": quote_code,
            "rate": 1.0,
            "source": "identity",
        }

    if {base_code, quote_code} != {"USD", "GBP"}:
        raise HTTPException(
            status_code=400,
            detail="Only USD/GBP exchange rates are currently supported.",
        )

    usd_to_gbp, source = _get_usd_to_gbp_rate()
    if usd_to_gbp is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to resolve USD/GBP FX rate from provider or database fallback. "
                "Retry shortly."
            ),
        )

    rate = usd_to_gbp if (base_code, quote_code) == ("USD", "GBP") else 1.0 / usd_to_gbp
    return {
        "base": base_code,
        "quote": quote_code,
        "rate": rate,
        "source": source,
    }


@app.get("/latest-run", dependencies=[Depends(_require_api_key)])
def latest_run():
    latest = db_latest_run_date()
    if not latest:
        raise HTTPException(status_code=404, detail="No production runs found.")
    return {"latest_run": latest}


@app.get("/latest-summary", dependencies=[Depends(_require_api_key)])
def latest_summary():
    payload = db_latest_summary()
    if not payload:
        raise HTTPException(status_code=404, detail="Latest summary not found.")
    return _apply_summary_currency_fields(payload)


@app.get("/summaries", dependencies=[Depends(_require_api_key)])
def summaries(
    start: str | None = None,
    end: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    fields: str | None = None,
):
    limit, offset = _normalize_pagination(limit, offset)
    start_date = _normalize_iso_date(start, field_name="start") if start else None
    end_date = _normalize_iso_date(end, field_name="end") if end else None
    requested_fields = _normalize_summary_fields(fields)
    total, filtered = db_list_run_summaries_paginated(
        start=start_date,
        end=end_date,
        limit=limit,
        offset=offset,
        fields=requested_fields,
    )

    normalized = [_apply_summary_currency_fields(item) for item in filtered]
    return {
        "total": total,
        "count": len(normalized),
        "limit": limit,
        "offset": offset,
        "summaries": normalized,
    }


@app.get("/latest-trades", dependencies=[Depends(_require_api_key)])
def latest_trades(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, rows = db_list_latest_trades(limit=limit, offset=offset)
    if total <= 0:
        raise HTTPException(status_code=404, detail="Latest trades not found.")
    if not rows:
        return {"total": total, "count": 0, "limit": limit, "offset": offset, "trades": []}
    return {
        "total": total,
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "trades": rows,
    }


@app.get("/trades", dependencies=[Depends(_require_api_key)])
def trades(limit: int = 200, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)

    total, records = db_list_trades(limit, offset)
    return {
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "trades": records,
    }


@app.get("/cagr", dependencies=[Depends(_require_api_key)])
def cagr_summary():
    latest = db_latest_summary()
    if latest:
        payload = latest.get("cagr_payload")
        if isinstance(payload, dict):
            return payload

    summaries = db_list_run_summaries()
    return compute_cagr_summary(summaries)


@app.get("/entry-indicator", dependencies=[Depends(_require_api_key)])
def entry_indicator(
    strategy_roots: str | None = None,
    strategies: str | None = None,
    regime_mapping: str | None = None,
    start_date: str = "2013-01-01",
    end_date: str | None = None,
    as_of_date: str | None = None,
    lookahead_days: int = 126,
    confirm_days: int = config.CONFIRM_DAYS,
    confirm_days_sideways: int = config.CONFIRM_DAYS_SIDEWAYS,
    rebalance_every: int = config.REBALANCE_EVERY,
    ignore_stock_filters: bool = False,
    refresh: bool = False,
):
    if lookahead_days < 20:
        raise HTTPException(status_code=400, detail="lookahead_days must be >= 20.")

    start_date_value = _normalize_iso_date(start_date, field_name="start_date")
    end_date_value = _normalize_iso_date(end_date, field_name="end_date") if end_date else None
    as_of_date_value = _normalize_iso_date(as_of_date, field_name="as_of_date") if as_of_date else None

    strategy_roots_list = _parse_csv_values(strategy_roots) or list(config.DEFAULT_STRATEGY_ROOTS)
    strategies_list = _parse_csv_values(strategies) or list(ENTRY_DEFAULT_BASE_STRATEGIES)
    mapping = _normalize_regime_mapping(regime_mapping)

    cache_key = _entry_indicator_cache_key(
        strategy_roots=strategy_roots_list,
        strategies=strategies_list,
        regime_mapping=mapping,
        start_date=start_date_value,
        end_date=end_date_value,
        as_of_date=as_of_date_value,
        lookahead_days=int(lookahead_days),
        confirm_days=int(confirm_days),
        confirm_days_sideways=int(confirm_days_sideways),
        rebalance_every=int(rebalance_every),
        ignore_stock_filters=bool(ignore_stock_filters),
    )
    now_epoch = time.time()
    ttl = max(0, int(ENTRY_INDICATOR_CACHE_TTL_SECONDS))
    if not refresh and ttl > 0:
        cached = _entry_indicator_cache.get(cache_key)
        if cached:
            fetched_at = _to_float(cached.get("fetched_at_epoch"))
            cached_payload = cached.get("payload")
            if (
                fetched_at is not None
                and isinstance(cached_payload, dict)
                and (now_epoch - fetched_at) <= ttl
            ):
                payload = dict(cached_payload)
                payload["cache"] = {"hit": True, "ttl_seconds": ttl}
                return payload

    is_default_request = (
        strategy_roots is None
        and strategies is None
        and regime_mapping is None
        and end_date is None
        and as_of_date is None
        and int(lookahead_days) == 126
        and int(confirm_days) == int(config.CONFIRM_DAYS)
        and int(confirm_days_sideways) == int(config.CONFIRM_DAYS_SIDEWAYS)
        and int(rebalance_every) == int(config.REBALANCE_EVERY)
        and (not bool(ignore_stock_filters))
    )
    if not refresh and is_default_request:
        snapshot = db_latest_entry_indicator_snapshot()
        if snapshot and isinstance(snapshot.get("payload"), dict):
            payload = dict(snapshot.get("payload") or {})
            payload["generated_at_utc"] = payload.get("generated_at_utc") or (
                str(snapshot.get("generated_at_utc") or "").strip() or None
            )
            payload["cache"] = {"hit": True, "source": "db"}
            return payload

    try:
        payload = compute_entry_indicator_payload(
            strategy_roots=strategy_roots_list,
            strategies=strategies_list,
            regime_mapping=mapping,
            start_date=start_date_value,
            end_date=end_date_value,
            as_of_date=as_of_date_value,
            lookahead_days=int(lookahead_days),
            confirm_days=int(confirm_days),
            confirm_days_sideways=int(confirm_days_sideways),
            rebalance_every=int(rebalance_every),
            ignore_stock_filters=bool(ignore_stock_filters),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute entry indicator: {type(exc).__name__}: {exc}",
        )

    if ttl > 0:
        _entry_indicator_cache[cache_key] = {
            "fetched_at_epoch": now_epoch,
            "payload": dict(payload),
        }
    try:
        db_upsert_entry_indicator_snapshot(
            payload=dict(payload),
            start_date=start_date_value,
            end_date=end_date_value,
            as_of_date=as_of_date_value,
            lookahead_days=int(lookahead_days),
            confirm_days=int(confirm_days),
            confirm_days_sideways=int(confirm_days_sideways),
            rebalance_every=int(rebalance_every),
            strategy_roots=strategy_roots_list,
            strategies=strategies_list,
            regime_mapping=mapping,
        )
    except Exception:
        # Avoid failing API calls on snapshot persistence issues.
        pass
    payload["cache"] = {"hit": False, "ttl_seconds": ttl}
    payload["generated_at_utc"] = payload.get("generated_at_utc") or datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return payload


@app.get("/portfolio", dependencies=[Depends(_require_api_key)])
def portfolio_snapshot():
    state = None
    db_state = db_load_state()
    if db_state:
        state = {
            "last_date": db_state.last_date,
            "cash": db_state.cash,
            "positions": db_state.positions,
            "prev_weights": db_state.prev_weights,
            "total_costs_usd": db_state.total_costs_usd,
        }
    if not state:
        raise HTTPException(status_code=404, detail="Portfolio state not found.")

    cash = float(state.get("cash", 0.0))
    positions = state.get("positions", {}) or {}
    if not isinstance(positions, dict):
        positions = {}

    as_of, price_map = db_latest_prices(list(positions.keys()))
    if as_of is None:
        raise HTTPException(status_code=404, detail="No price snapshot found in database.")

    portfolio_rows = []
    missing_prices = []
    portfolio_value = 0.0

    for ticker, shares in positions.items():
        try:
            shares_val = int(shares)
        except (TypeError, ValueError):
            shares_val = 0
        ticker_key = str(ticker).strip().upper()
        price = price_map.get(ticker_key)
        value = None
        if price is not None:
            value = price * shares_val
            portfolio_value += value
        else:
            missing_prices.append(str(ticker))
        portfolio_rows.append(
            {
                "ticker": str(ticker),
                "shares": shares_val,
                "quantity": shares_val,
                "price_usd": price,
                "value_usd": value,
            }
        )

    return {
        "as_of": as_of,
        "cash_usd": cash,
        "portfolio_value_usd": portfolio_value,
        "net_worth_usd": cash + portfolio_value,
        "positions": portfolio_rows,
        "missing_prices": missing_prices,
    }


@app.get("/universe", dependencies=[Depends(_require_api_key)])
def universe_listing(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, universe = db_list_universe_map(limit=limit, offset=offset)
    if total <= 0:
        raise HTTPException(status_code=404, detail="Universe mapping not found.")
    return {
        "total": total,
        "count": len(universe),
        "limit": limit,
        "offset": offset,
        "universe": universe,
    }


@app.get("/universe/selection-diagnostics", dependencies=[Depends(_require_api_key)])
def universe_selection_diagnostics(
    sector: str = "Technology",
    date: str | None = None,
    regime_scope: Literal["global", "sector"] = "global",
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
):
    limit, offset = _normalize_pagination(limit, offset)
    sector_name = str(sector or "").strip()
    if not sector_name:
        raise HTTPException(status_code=400, detail="sector is required.")
    target_date = _normalize_iso_date(date, field_name="date") if date else None
    state = db_latest_universe_selection_diagnostics_state(
        sector=sector_name,
        regime_scope=regime_scope,
        run_date=target_date,
    )
    if not state:
        raise HTTPException(
            status_code=404,
            detail=(
                "Selection diagnostics snapshot not found in database for the requested scope/date. "
                "Run production daily_run to generate and persist this snapshot."
            ),
        )
    resolved_run_date = str(state.get("run_date"))
    payload = state.get("payload")
    payload = dict(payload) if isinstance(payload, dict) else {}
    total, records = db_list_universe_selection_diagnostics_records(
        run_date=resolved_run_date,
        sector=sector_name,
        regime_scope=regime_scope,
        limit=limit,
        offset=offset,
    )
    payload["date"] = resolved_run_date
    payload["sector"] = sector_name
    payload["regime_scope"] = regime_scope
    payload["total"] = total
    payload["count"] = len(records)
    payload["limit"] = limit
    payload["offset"] = offset
    payload["records"] = records
    payload["source"] = "db"
    return payload


@app.get("/universe-monitor/summary", dependencies=[Depends(_require_api_key)])
def universe_monitor_summary():
    payload = db_latest_universe_monitor_summary()
    if not payload:
        raise HTTPException(status_code=404, detail="Universe monitor summary not found.")
    full_payload = payload.get("payload")
    if isinstance(full_payload, dict):
        return full_payload
    return payload


@app.get("/universe-monitor/candidates", dependencies=[Depends(_require_api_key)])
def universe_monitor_candidates(
    limit: int = 200,
    offset: int = 0,
    watchlist: bool = False,
    potential: bool = False,
):
    limit, offset = _normalize_pagination(limit, offset)

    total, records = db_list_universe_monitor_candidates(
        limit=limit,
        offset=offset,
        watchlist=watchlist,
        potential=potential,
    )
    return {
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "watchlist": watchlist,
        "potential": potential,
        "candidates": records,
    }


@app.get("/universe-monitor/potential", dependencies=[Depends(_require_api_key)])
def universe_monitor_potential(limit: int = 200, offset: int = 0):
    return universe_monitor_candidates(
        limit=limit,
        offset=offset,
        watchlist=False,
        potential=True,
    )


@app.get("/broker-summary", dependencies=[Depends(_require_api_key)])
def broker_summary(
    broker: str = DEFAULT_BROKER_QUERY,
    include_payload: bool = False,
    fields: str | None = None,
):
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_SUMMARY_FIELDS,
        include_payload=include_payload,
    )
    summary = db_latest_broker_account(
        broker,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if summary:
        return _apply_broker_summary_currency(summary)
    raise HTTPException(status_code=404, detail="Broker summary not found.")


@app.get("/broker-positions", dependencies=[Depends(_require_api_key)])
def broker_positions(
    broker: str = DEFAULT_BROKER_QUERY,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    include_payload: bool = False,
    fields: str | None = None,
):
    limit, offset = _normalize_pagination(limit, offset)
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_POSITION_FIELDS,
        include_payload=include_payload,
    )
    total, positions = db_list_latest_broker_positions(
        broker=broker,
        limit=limit,
        offset=offset,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if total > 0:
        normalized_positions = _apply_broker_positions_currency(positions)
        return {
            "total": total,
            "count": len(normalized_positions),
            "limit": limit,
            "offset": offset,
            "positions": normalized_positions,
        }
    raise HTTPException(status_code=404, detail="Broker positions not found.")


@app.get("/broker-orders", dependencies=[Depends(_require_api_key)])
def broker_orders(
    broker: str = DEFAULT_BROKER_QUERY,
    limit: int = 200,
    offset: int = 0,
    include_payload: bool = False,
    fields: str | None = None,
    meta_only: bool = False,
):
    limit, offset = _normalize_pagination(limit, offset)
    if meta_only:
        total = db_count_broker_orders(broker)
        return {
            "total": total,
            "count": 0,
            "limit": limit,
            "offset": offset,
            "orders": [],
            "meta_only": True,
        }
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_ORDER_FIELDS,
        include_payload=include_payload,
    )
    total, records = db_list_broker_orders(
        broker,
        limit,
        offset,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if total > 0:
        normalized_records = _apply_broker_orders_currency(records)
        return {
            "total": total,
            "count": len(normalized_records),
            "limit": limit,
            "offset": offset,
            "orders": normalized_records,
        }
    return {"total": 0, "count": 0, "limit": limit, "offset": offset, "orders": []}


@app.get("/latest-broker-orders", dependencies=[Depends(_require_api_key)])
def latest_broker_orders(
    broker: str = DEFAULT_BROKER_QUERY,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    include_payload: bool = False,
    fields: str | None = None,
    meta_only: bool = False,
):
    limit, offset = _normalize_pagination(limit, offset)
    if meta_only:
        run_date, total = db_count_latest_broker_orders(broker)
        if total > 0:
            return {
                "broker": broker,
                "run_date": str(run_date),
                "total": total,
                "count": 0,
                "limit": limit,
                "offset": offset,
                "orders": [],
                "meta_only": True,
            }
        raise HTTPException(status_code=404, detail="Broker orders not found.")
    requested_fields = _normalize_broker_fields(
        fields,
        allowed=BROKER_ORDER_FIELDS,
        include_payload=include_payload,
    )
    total, records = db_list_latest_broker_orders(
        broker=broker,
        limit=limit,
        offset=offset,
        include_payload=include_payload,
        fields=requested_fields,
    )
    if total > 0:
        normalized_records = _apply_broker_orders_currency(records)
        return {
            "total": total,
            "count": len(normalized_records),
            "limit": limit,
            "offset": offset,
            "orders": normalized_records,
        }
    raise HTTPException(status_code=404, detail="Broker orders not found.")


@app.get("/latest-broker-orders/count", dependencies=[Depends(_require_api_key)])
def latest_broker_orders_count(broker: str = DEFAULT_BROKER_QUERY):
    run_date, total = db_count_latest_broker_orders(broker)
    if total <= 0:
        raise HTTPException(status_code=404, detail="Broker orders not found.")
    return {
        "broker": broker,
        "run_date": str(run_date),
        "total": total,
    }


@app.get("/excluded-tickers", dependencies=[Depends(_require_api_key)])
def list_excluded_tickers(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, excluded = db_list_excluded_tickers(limit=limit, offset=offset)
    return {
        "total": total,
        "count": len(excluded),
        "limit": limit,
        "offset": offset,
        "excluded_tickers": excluded,
        "source": "db",
    }


@app.post("/exclude-tickers", dependencies=[Depends(_require_api_key)])
def exclude_tickers(payload: ExcludeTickersRequest):
    tickers = [str(t).strip().upper() for t in (payload.tickers or []) if str(t).strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided.")
    excluded, source = _load_effective_excluded_tickers()
    excluded.update(tickers)
    db_replace_excluded_tickers(excluded)
    return {"count": len(excluded), "excluded_tickers": sorted(excluded), "source": source}


@app.get("/run-calendar/decision", dependencies=[Depends(_require_api_key)])
def run_calendar_decision(date: str):
    run_date = _normalize_iso_date(date, field_name="date")
    override = db_load_run_calendar_override(run_date) or {}
    decision = evaluate_run_day(
        run_date,
        override_action=str(override.get("action") or "").strip().lower() or None,
        override_reason=str(override.get("reason") or "").strip() or None,
        skip_weekends=config.RUN_CALENDAR_SKIP_WEEKENDS,
        skip_us_federal_holidays=config.RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS,
    )
    return {
        "date": run_date,
        "should_run": bool(decision.get("should_run")),
        "reason_code": decision.get("reason_code"),
        "reason": decision.get("reason"),
        "source": decision.get("source"),
        "override": override or None,
        "settings": {
            "skip_weekends": bool(config.RUN_CALENDAR_SKIP_WEEKENDS),
            "skip_us_federal_holidays": bool(config.RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS),
        },
    }


@app.get("/run-calendar/overrides", dependencies=[Depends(_require_api_key)])
def run_calendar_overrides(
    start: str | None = None,
    end: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
):
    limit, offset = _normalize_pagination(limit, offset)
    start_date = _normalize_iso_date(start, field_name="start") if start else None
    end_date = _normalize_iso_date(end, field_name="end") if end else None
    total, rows = db_list_run_calendar_overrides_paginated(
        start=start_date,
        end=end_date,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "overrides": rows,
    }


@app.post("/run-calendar/overrides", dependencies=[Depends(_require_api_key)])
def upsert_run_calendar_override(payload: RunCalendarOverrideRequest):
    run_date = _normalize_iso_date(payload.date, field_name="date")
    action = str(payload.action).strip().lower()
    if action not in (RUN_CALENDAR_ACTION_SKIP, RUN_CALENDAR_ACTION_FORCE_RUN):
        raise HTTPException(status_code=400, detail="Invalid action. Use 'skip' or 'force_run'.")
    reason = str(payload.reason or "").strip()
    source = str(payload.source or "app").strip() or "app"
    db_upsert_run_calendar_override(
        run_date=run_date,
        action=action,
        reason=reason or None,
        source=source,
    )
    decision = evaluate_run_day(
        run_date,
        override_action=action,
        override_reason=reason or None,
        skip_weekends=config.RUN_CALENDAR_SKIP_WEEKENDS,
        skip_us_federal_holidays=config.RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS,
    )
    return {
        "upserted": True,
        "date": run_date,
        "action": action,
        "reason": reason,
        "source": source,
        "decision": decision,
    }


@app.delete("/run-calendar/overrides/{run_date}", dependencies=[Depends(_require_api_key)])
def delete_run_calendar_override(run_date: str):
    date_value = _normalize_iso_date(run_date, field_name="run_date")
    deleted = db_delete_run_calendar_override(date_value)
    return {"deleted": bool(deleted), "date": date_value}


@app.get("/run-calendar/us-federal-holidays", dependencies=[Depends(_require_api_key)])
def run_calendar_us_federal_holidays(year: int):
    if year < 1970 or year > 2100:
        raise HTTPException(status_code=400, detail="year must be between 1970 and 2100.")
    holidays = list_us_federal_holidays(year)
    return {"year": year, "count": len(holidays), "holidays": holidays}


@app.get("/stale-tickers", dependencies=[Depends(_require_api_key)])
def stale_tickers(
    date: Optional[str] = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
):
    limit, offset = _normalize_pagination(limit, offset)
    mapping = db_load_universe_map()
    if not mapping:
        raise HTTPException(status_code=404, detail="Universe mapping not found.")

    if date:
        try:
            target_date = pd.to_datetime(date).tz_localize(None)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid date format.")
    else:
        latest = db_latest_price_run_date()
        if not latest:
            raise HTTPException(status_code=404, detail="No production price snapshots found.")
        target_date = pd.to_datetime(latest).tz_localize(None)

    target_run_date = str(pd.to_datetime(target_date).date())
    tickers = set(mapping.keys())
    excluded, _ = _load_effective_excluded_tickers()
    if excluded:
        tickers -= excluded
    observed = db_price_tickers_for_date(target_run_date)
    if observed:
        tickers -= observed
    stale_sorted = sorted(tickers)
    paged = stale_sorted[offset : offset + limit]

    return {
        "as_of": target_run_date,
        "total": len(stale_sorted),
        "count": len(paged),
        "limit": limit,
        "offset": offset,
        "stale_tickers": paged,
        "source": "db",
    }


@app.post("/queue-adjustments", dependencies=[Depends(_require_api_key)])
def queue_adjustments(payload: AdjustmentRequest):
    entries: list[dict] = []
    timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if payload.cash_amount:
        entries.append(
            {
                "type": "cash",
                "amount": float(payload.cash_amount),
                "note": payload.cash_note,
                "source": payload.source,
                "created_at": timestamp,
            }
        )
    tickers: list[str] = []
    exchange_map: dict[str, str] = {}
    for token in (payload.tickers or []):
        ticker, exchange = _parse_ticker_exchange_pair(str(token), None)
        if not ticker:
            continue
        tickers.append(ticker)
        if exchange:
            exchange_map[ticker] = exchange
    if payload.ticker_exchanges:
        if isinstance(payload.ticker_exchanges, dict):
            for ticker_key, exchange_value in payload.ticker_exchanges.items():
                ticker, exchange = _parse_ticker_exchange_pair(
                    str(ticker_key),
                    exchange_value if exchange_value is not None else None,
                )
                if not ticker:
                    continue
                tickers.append(ticker)
                if exchange:
                    exchange_map[ticker] = exchange
        elif isinstance(payload.ticker_exchanges, list):
            for item in payload.ticker_exchanges:
                ticker, exchange = _parse_ticker_exchange_pair(str(item), None)
                if not ticker:
                    continue
                tickers.append(ticker)
                if exchange:
                    exchange_map[ticker] = exchange
    if exchange_map:
        tickers.extend(exchange_map.keys())
    tickers = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    exchange_map = {
        str(ticker).strip().upper(): str(exchange).strip().upper() or "UNKNOWN"
        for ticker, exchange in exchange_map.items()
        if str(ticker).strip()
    }
    if tickers:
        entry = {
            "type": "tickers",
            "tickers": sorted(set(tickers)),
            "source": payload.source,
            "created_at": timestamp,
        }
        if exchange_map:
            entry["exchanges"] = exchange_map
        entries.append(entry)
    if not entries:
        raise HTTPException(status_code=400, detail="No adjustments provided.")

    db_append_pending_adjustments(entries)

    return {"queued": len(entries)}


@app.get("/pending-adjustments", dependencies=[Depends(_require_api_key)])
def list_pending_adjustments(limit: int = DEFAULT_PAGE_LIMIT, offset: int = 0):
    limit, offset = _normalize_pagination(limit, offset)
    total, entries = db_list_pending_adjustments(limit=limit, offset=offset)
    return {
        "total": total,
        "count": len(entries),
        "limit": limit,
        "offset": offset,
        "pending": entries,
    }


@app.post("/clear-pending", dependencies=[Depends(_require_api_key)])
def clear_pending_adjustments():
    db_clear_pending_adjustments()
    return {"cleared": True}


@app.delete("/pending-adjustments", dependencies=[Depends(_require_api_key)])
def delete_pending_adjustments():
    db_clear_pending_adjustments()
    return {"cleared": True}


@app.post("/reset-production", dependencies=[Depends(_require_api_key)])
def reset_production(payload: ResetRequest):
    preserve_monitor = bool(payload.preserve_universe_monitor)
    if preserve_monitor:
        if payload.confirm != "RESET_PRODUCTION_DATA":
            raise HTTPException(status_code=400, detail="Confirmation token missing or invalid.")
    else:
        if payload.confirm != "RESET_PRODUCTION_AND_MONITOR_DATA":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid confirmation token for full reset. "
                    "Use RESET_PRODUCTION_AND_MONITOR_DATA."
                ),
            )
    db_reset_production_data(preserve_universe_monitor=preserve_monitor)
    return {
        "reset": True,
        "preserve_universe_monitor": preserve_monitor,
    }
