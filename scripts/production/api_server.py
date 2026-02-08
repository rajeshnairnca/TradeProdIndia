import json
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src import config
from src.production_db import (
    append_pending_adjustments as db_append_pending_adjustments,
    clear_pending_adjustments as db_clear_pending_adjustments,
    db_enabled,
    init_db as db_init,
    latest_run_date as db_latest_run_date,
    latest_prices as db_latest_prices,
    latest_price_run_date as db_latest_price_run_date,
    latest_summary as db_latest_summary,
    latest_trades as db_latest_trades,
    latest_broker_account as db_latest_broker_account,
    latest_broker_orders as db_latest_broker_orders,
    latest_broker_positions as db_latest_broker_positions,
    latest_universe_monitor_summary as db_latest_universe_monitor_summary,
    list_run_summaries as db_list_run_summaries,
    list_broker_orders as db_list_broker_orders,
    list_trades as db_list_trades,
    list_universe_monitor_candidates as db_list_universe_monitor_candidates,
    load_excluded_tickers as db_load_excluded_tickers,
    load_universe_map as db_load_universe_map,
    load_pending_adjustments as db_load_pending_adjustments,
    price_tickers_for_date as db_price_tickers_for_date,
    replace_excluded_tickers as db_replace_excluded_tickers,
    load_state as db_load_state,
    reset_production_data as db_reset_production_data,
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "runs/production")
PENDING_FILE = os.getenv("PENDING_FILE", "runs/production/pending_adjustments.jsonl")
STATE_FILE = os.getenv("STATE_FILE", "runs/production/state.json")
EXCHANGE_MAP_FILE = os.getenv("EXCHANGE_MAP_FILE", config.TRADINGVIEW_EXCHANGE_MAP_FILE)
EXCLUDED_TICKERS_FILE = os.getenv("EXCLUDED_TICKERS_FILE", config.EXCLUDED_TICKERS_FILE)
UNIVERSE_MONITOR_OUTPUT_DIR = os.getenv("UNIVERSE_MONITOR_OUTPUT_DIR", "runs/universe_monitor")
API_KEY = os.getenv("API_KEY", "").strip()


def _db_ready() -> bool:
    if not db_enabled():
        return False
    db_init()
    return True


def _resolve_path(path: str) -> str:
    resolved = config.resolve_path(path)
    if os.path.isabs(resolved):
        return resolved
    return os.path.join(PROJECT_ROOT, resolved)


def _latest_run_dir() -> Path:
    base = Path(_resolve_path(OUTPUT_DIR))
    if not base.exists():
        raise HTTPException(status_code=404, detail="No production output directory found.")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise HTTPException(status_code=404, detail="No production runs found.")
    return sorted(dirs, key=lambda d: d.name)[-1]


def _list_run_dirs() -> list[Path]:
    base = Path(_resolve_path(OUTPUT_DIR))
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: d.name)


def _latest_universe_monitor_dir() -> Path:
    base = Path(_resolve_path(UNIVERSE_MONITOR_OUTPUT_DIR))
    if not base.exists():
        raise HTTPException(status_code=404, detail="No universe monitor output directory found.")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise HTTPException(status_code=404, detail="No universe monitor runs found.")
    return sorted(dirs, key=lambda d: d.name)[-1]


def _load_exchange_map(path: str) -> dict[str, str]:
    resolved = _resolve_path(path)
    exchange_path = Path(resolved)
    if not exchange_path.exists():
        return {}
    try:
        payload = json.loads(exchange_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        ticker = str(key).strip().upper()
        exchange = str(value).strip().upper()
        if not ticker:
            continue
        mapping[ticker] = exchange or "UNKNOWN"
    return mapping


def _load_excluded_tickers(path: str) -> set[str]:
    resolved = _resolve_path(path)
    excluded_path = Path(resolved)
    if not excluded_path.exists():
        return set()
    try:
        lines = excluded_path.read_text().splitlines()
    except OSError:
        return set()
    return {line.strip().upper() for line in lines if line.strip()}


def _write_excluded_tickers(path: str, tickers: set[str]) -> None:
    resolved = _resolve_path(path)
    excluded_path = Path(resolved)
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    if not tickers:
        excluded_path.write_text("")
        return
    excluded_path.write_text("\n".join(sorted(tickers)) + "\n")


def _load_effective_excluded_tickers() -> tuple[set[str], str]:
    if _db_ready():
        return db_load_excluded_tickers(), "db"
    return _load_excluded_tickers(EXCLUDED_TICKERS_FILE), "file"


def _load_state(path: str) -> dict | None:
    state_path = Path(_resolve_path(path))
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return None


def _xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    if len(cashflows) < 2:
        return None
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]

    def _year_frac(ts: pd.Timestamp) -> float:
        return (ts - t0).days / 365.25

    def npv(rate: float) -> float:
        total = 0.0
        for dt, amount in cashflows:
            total += amount / ((1.0 + rate) ** _year_frac(dt))
        return total

    def d_npv(rate: float) -> float:
        total = 0.0
        for dt, amount in cashflows:
            yf = _year_frac(dt)
            total += -yf * amount / ((1.0 + rate) ** (yf + 1.0))
        return total

    rate = 0.1
    for _ in range(50):
        f_val = npv(rate)
        if abs(f_val) < 1e-7:
            return rate
        deriv = d_npv(rate)
        if abs(deriv) < 1e-12:
            break
        rate -= f_val / deriv
        if rate <= -0.9999:
            rate = -0.9999

    low, high = -0.9999, 10.0
    f_low, f_high = npv(low), npv(high)
    if f_low * f_high > 0:
        return None
    mid = 0.0
    for _ in range(100):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return mid


class AdjustmentRequest(BaseModel):
    cash_amount: float = 0.0
    cash_note: str = "app"
    tickers: Optional[List[str]] = None
    ticker_exchanges: Optional[dict[str, str]] = None
    source: str = "app"


class ExcludeTickersRequest(BaseModel):
    tickers: List[str]


class ResetRequest(BaseModel):
    confirm: str


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


@app.get("/latest-run", dependencies=[Depends(_require_api_key)])
def latest_run():
    if _db_ready():
        latest = db_latest_run_date()
        if latest:
            return {"latest_run": latest}
    run_dir = _latest_run_dir()
    return {"latest_run": run_dir.name}


@app.get("/latest-summary", dependencies=[Depends(_require_api_key)])
def latest_summary():
    if _db_ready():
        payload = db_latest_summary()
        if payload:
            return payload
    run_dir = _latest_run_dir()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="summary.json not found.")
    with summary_path.open("r") as f:
        return json.load(f)


@app.get("/summaries", dependencies=[Depends(_require_api_key)])
def summaries(
    start: str | None = None,
    end: str | None = None,
    limit: int = 0,
    fields: str | None = None,
):
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")

    start_dt = pd.to_datetime(start) if start else None
    end_dt = pd.to_datetime(end) if end else None

    rows: list[dict] = []
    if _db_ready():
        rows = db_list_run_summaries()
    if not rows:
        for run_dir in _list_run_dirs():
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                payload = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                continue
            if "date" not in payload or "net_worth_usd" not in payload:
                continue
            rows.append(payload)

    if not rows:
        return {"count": 0, "summaries": []}

    def _parse_date(value: str | None):
        try:
            return pd.to_datetime(value) if value else None
        except (TypeError, ValueError):
            return None

    filtered: list[dict] = []
    for row in rows:
        row_date = _parse_date(row.get("date"))
        if row_date is None:
            continue
        if start_dt and row_date < start_dt:
            continue
        if end_dt and row_date > end_dt:
            continue
        filtered.append(row)

    filtered = sorted(filtered, key=lambda r: str(r.get("date")))
    if limit:
        filtered = filtered[-limit:]

    field_list: list[str] | None = None
    if fields:
        field_list = [f.strip() for f in fields.split(",") if f.strip()]

    if field_list:
        trimmed = []
        for row in filtered:
            trimmed.append({key: row.get(key) for key in field_list})
        filtered = trimmed

    return {"count": len(filtered), "summaries": filtered}


@app.get("/latest-trades", dependencies=[Depends(_require_api_key)])
def latest_trades(limit: int = 0):
    if _db_ready():
        trades = db_latest_trades(limit=limit)
        if trades:
            return {"trades": trades}
    run_dir = _latest_run_dir()
    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        raise HTTPException(status_code=404, detail="trades.csv not found.")
    df = pd.read_csv(trades_path)
    if limit and limit > 0:
        df = df.head(limit)
    return {"trades": df.to_dict(orient="records")}


@app.get("/trades", dependencies=[Depends(_require_api_key)])
def trades(limit: int = 200, offset: int = 0):
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    if _db_ready():
        total, records = db_list_trades(limit, offset)
        if total > 0:
            return {"total": total, "limit": limit, "offset": offset, "trades": records}

    frames: list[pd.DataFrame] = []
    for run_dir in _list_run_dirs():
        trades_path = run_dir / "trades.csv"
        if not trades_path.exists():
            continue
        df = pd.read_csv(trades_path)
        if df.empty:
            continue
        if "run_date" not in df.columns:
            df["run_date"] = run_dir.name
        frames.append(df)

    if not frames:
        return {"total": 0, "limit": limit, "offset": offset, "trades": []}

    all_trades = pd.concat(frames, ignore_index=True)
    if "date" in all_trades.columns:
        all_trades = all_trades.sort_values(["date", "ticker"], ascending=[False, True])
    else:
        all_trades = all_trades.sort_values(["run_date", "ticker"], ascending=[False, True])

    total = int(len(all_trades))
    if limit == 0:
        page = all_trades.iloc[offset:]
    else:
        page = all_trades.iloc[offset : offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "trades": page.to_dict(orient="records"),
    }


@app.get("/cagr", dependencies=[Depends(_require_api_key)])
def cagr_summary():
    summaries = []
    if _db_ready():
        summaries = db_list_run_summaries()
    if not summaries:
        for run_dir in _list_run_dirs():
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                payload = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                continue
            if "date" not in payload or "net_worth_usd" not in payload:
                continue
            summaries.append(payload)

    if len(summaries) < 2:
        return {
            "cagr": None,
            "cagr_adjusted": None,
            "detail": "Not enough history to compute CAGR.",
        }

    summaries = sorted(summaries, key=lambda s: s["date"])
    start = summaries[0]
    end = summaries[-1]
    start_date = pd.to_datetime(start["date"])
    end_date = pd.to_datetime(end["date"])
    years = (end_date - start_date).days / 365.25 if end_date > start_date else 0.0
    start_value = float(start.get("net_worth_usd", 0.0))
    end_value = float(end.get("net_worth_usd", 0.0))
    if years <= 0 or start_value <= 0:
        return {
            "cagr": None,
            "cagr_adjusted": None,
            "detail": "Invalid dates or starting value for CAGR calculation.",
        }

    cagr = (end_value / start_value) ** (1.0 / years) - 1.0
    cash_adjustments = sum(float(s.get("cash_adjustment", 0.0)) for s in summaries)

    twr_growth = 1.0
    for i in range(1, len(summaries)):
        start_val = float(summaries[i - 1].get("net_worth_usd", 0.0))
        end_val = float(summaries[i].get("net_worth_usd", 0.0))
        flow = float(summaries[i].get("cash_adjustment", 0.0))
        denom = start_val + flow
        if denom <= 0:
            twr_growth = None
            break
        period_return = (end_val / denom) - 1.0
        twr_growth *= 1.0 + period_return

    cagr_adjusted = None
    if twr_growth is not None:
        cagr_adjusted = twr_growth ** (1.0 / years) - 1.0

    cashflows = [(start_date, -start_value)]
    for item in summaries[1:]:
        flow = float(item.get("cash_adjustment", 0.0))
        if flow != 0.0:
            cashflows.append((pd.to_datetime(item["date"]), -flow))
    cashflows.append((end_date, end_value))
    irr = _xirr(cashflows)

    broker_payload = None
    broker_summaries = [
        s for s in summaries if s.get("broker_net_worth") is not None
    ]
    if len(broker_summaries) >= 2:
        broker_summaries = sorted(broker_summaries, key=lambda s: s["date"])
        b_start = broker_summaries[0]
        b_end = broker_summaries[-1]
        b_start_date = pd.to_datetime(b_start["date"])
        b_end_date = pd.to_datetime(b_end["date"])
        b_years = (b_end_date - b_start_date).days / 365.25 if b_end_date > b_start_date else 0.0
        b_start_val = float(b_start.get("broker_net_worth", 0.0))
        b_end_val = float(b_end.get("broker_net_worth", 0.0))
        if b_years > 0 and b_start_val > 0:
            b_cagr = (b_end_val / b_start_val) ** (1.0 / b_years) - 1.0
        else:
            b_cagr = None

        b_twr_growth = 1.0
        for i in range(1, len(broker_summaries)):
            start_val = float(broker_summaries[i - 1].get("broker_net_worth", 0.0))
            end_val = float(broker_summaries[i].get("broker_net_worth", 0.0))
            fx_rate = float(broker_summaries[i].get("broker_fx_rate_gbp_per_usd") or 0.0)
            flow_usd = float(broker_summaries[i].get("cash_adjustment", 0.0))
            flow = flow_usd * fx_rate if fx_rate else flow_usd
            denom = start_val + flow
            if denom <= 0:
                b_twr_growth = None
                break
            period_return = (end_val / denom) - 1.0
            b_twr_growth *= 1.0 + period_return

        b_cagr_adjusted = None
        if b_twr_growth is not None and b_years > 0:
            b_cagr_adjusted = b_twr_growth ** (1.0 / b_years) - 1.0

        b_cashflows = [(b_start_date, -b_start_val)]
        for item in broker_summaries[1:]:
            fx_rate = float(item.get("broker_fx_rate_gbp_per_usd") or 0.0)
            flow_usd = float(item.get("cash_adjustment", 0.0))
            flow = flow_usd * fx_rate if fx_rate else flow_usd
            if flow != 0.0:
                b_cashflows.append((pd.to_datetime(item["date"]), -flow))
        b_cashflows.append((b_end_date, b_end_val))
        b_irr = _xirr(b_cashflows)
        broker_payload = {
            "currency": b_start.get("broker_currency"),
            "cagr": b_cagr,
            "cagr_adjusted": b_cagr_adjusted,
            "irr": b_irr,
        }

    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "years": years,
        "start_value_usd": start_value,
        "end_value_usd": end_value,
        "cagr": cagr,
        "cagr_adjusted": cagr_adjusted,
        "irr": irr,
        "cash_adjustments_total_usd": cash_adjustments,
        "broker": broker_payload,
        "note": "cagr_adjusted uses time-weighted returns with cash flows applied at period start; irr is money-weighted using net worth and cash adjustments.",
    }


@app.get("/portfolio", dependencies=[Depends(_require_api_key)])
def portfolio_snapshot():
    state = None
    if _db_ready():
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
        state = _load_state(STATE_FILE)
    if not state:
        raise HTTPException(status_code=404, detail="State file not found.")

    cash = float(state.get("cash", 0.0))
    positions = state.get("positions", {}) or {}
    if not isinstance(positions, dict):
        positions = {}

    as_of = None
    price_map: dict[str, float] = {}
    if _db_ready():
        as_of, price_map = db_latest_prices(list(positions.keys()))

    if as_of is not None:
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

    data_path = _resolve_path(config.DATA_FILE)
    if not Path(data_path).exists():
        raise HTTPException(status_code=404, detail="Data file not found.")

    df = pd.read_parquet(data_path, columns=["Close"])
    if not isinstance(df.index, pd.MultiIndex) or "ticker" not in df.index.names:
        raise HTTPException(status_code=500, detail="Unexpected data index format.")

    latest_date = df.index.get_level_values("date").max()
    portfolio_rows = []
    missing_prices = []
    portfolio_value = 0.0

    for ticker, shares in positions.items():
        try:
            shares_val = int(shares)
        except (TypeError, ValueError):
            shares_val = 0
        price = None
        value = None
        try:
            series = df.xs(str(ticker), level="ticker")["Close"]
            if not series.empty:
                price = float(series.iloc[-1])
                value = price * shares_val
                portfolio_value += value
        except (KeyError, TypeError, ValueError):
            price = None
        if price is None:
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
        "as_of": str(pd.to_datetime(latest_date).date()) if latest_date is not None else None,
        "cash_usd": cash,
        "portfolio_value_usd": portfolio_value,
        "net_worth_usd": cash + portfolio_value,
        "positions": portfolio_rows,
        "missing_prices": missing_prices,
    }


@app.get("/universe", dependencies=[Depends(_require_api_key)])
def universe_listing():
    if _db_ready():
        mapping = db_load_universe_map()
        if mapping:
            universe = [{"ticker": k, "exchange": v} for k, v in sorted(mapping.items())]
            return {"count": len(universe), "universe": universe}
    mapping = _load_exchange_map(EXCHANGE_MAP_FILE)
    if not mapping:
        raise HTTPException(status_code=404, detail="Exchange map not found.")
    universe = [{"ticker": k, "exchange": v} for k, v in sorted(mapping.items())]
    return {"count": len(universe), "universe": universe}


def _as_bool_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "on"})


@app.get("/universe-monitor/summary", dependencies=[Depends(_require_api_key)])
def universe_monitor_summary():
    if _db_ready():
        payload = db_latest_universe_monitor_summary()
        if payload:
            full_payload = payload.get("payload")
            if isinstance(full_payload, dict):
                return full_payload
            return payload

    run_dir = _latest_universe_monitor_dir()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Universe monitor summary not found.")
    with summary_path.open("r") as f:
        return json.load(f)


@app.get("/universe-monitor/candidates", dependencies=[Depends(_require_api_key)])
def universe_monitor_candidates(
    limit: int = 200,
    offset: int = 0,
    watchlist: bool = False,
    potential: bool = False,
):
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    if _db_ready():
        total, records = db_list_universe_monitor_candidates(
            limit=limit,
            offset=offset,
            watchlist=watchlist,
            potential=potential,
        )
        if total > 0:
            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "watchlist": watchlist,
                "potential": potential,
                "candidates": records,
            }

    run_dir = _latest_universe_monitor_dir()
    csv_path = run_dir / "tech_candidates_all.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Universe monitor candidates CSV not found.")

    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            "total": 0,
            "limit": limit,
            "offset": offset,
            "watchlist": watchlist,
            "potential": potential,
            "candidates": [],
        }

    if watchlist:
        df = df[
            _as_bool_mask(df, "tv_pass")
            & _as_bool_mask(df, "tech_sector")
            & _as_bool_mask(df, "market_cap_pass")
        ]
    if potential:
        min_pass_days = 0
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text())
                min_pass_days = int(payload.get("min_pass_days") or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                min_pass_days = 0
        if "pass_streak" not in df.columns:
            df["pass_streak"] = 0
        df["pass_streak"] = pd.to_numeric(df["pass_streak"], errors="coerce").fillna(0)
        df = df[_as_bool_mask(df, "monitor_pass") & (df["pass_streak"] >= min_pass_days)]

    sort_cols = []
    ascending = []
    if "pass_streak" in df.columns:
        sort_cols.append("pass_streak")
        ascending.append(False)
    if "monitor_pass" in df.columns:
        sort_cols.append("monitor_pass")
        ascending.append(False)
    if "recommend_all" in df.columns:
        sort_cols.append("recommend_all")
        ascending.append(False)
    if "dollar_volume" in df.columns:
        sort_cols.append("dollar_volume")
        ascending.append(False)
    if "ticker" in df.columns:
        sort_cols.append("ticker")
        ascending.append(True)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, kind="stable")

    total = int(len(df))
    if limit == 0:
        page = df.iloc[offset:]
    else:
        page = df.iloc[offset : offset + limit]
    page = page.where(pd.notna(page), None)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "watchlist": watchlist,
        "potential": potential,
        "candidates": page.to_dict(orient="records"),
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
def broker_summary(broker: str = "trading212"):
    if _db_ready():
        summary = db_latest_broker_account(broker)
        if summary:
            return summary
    raise HTTPException(status_code=404, detail="Broker summary not found.")


@app.get("/broker-positions", dependencies=[Depends(_require_api_key)])
def broker_positions(broker: str = "trading212"):
    if _db_ready():
        positions = db_latest_broker_positions(broker)
        if positions:
            return {"count": len(positions), "positions": positions}
    raise HTTPException(status_code=404, detail="Broker positions not found.")


@app.get("/broker-orders", dependencies=[Depends(_require_api_key)])
def broker_orders(broker: str = "trading212", limit: int = 200, offset: int = 0):
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if _db_ready():
        total, records = db_list_broker_orders(broker, limit, offset)
        if total > 0:
            return {"total": total, "limit": limit, "offset": offset, "orders": records}
    return {"total": 0, "limit": limit, "offset": offset, "orders": []}


@app.get("/latest-broker-orders", dependencies=[Depends(_require_api_key)])
def latest_broker_orders(broker: str = "trading212", limit: int = 0):
    if _db_ready():
        records = db_latest_broker_orders(broker, limit=limit)
        if records:
            return {"orders": records}
    raise HTTPException(status_code=404, detail="Broker orders not found.")


@app.get("/excluded-tickers", dependencies=[Depends(_require_api_key)])
def list_excluded_tickers():
    excluded, source = _load_effective_excluded_tickers()
    return {"count": len(excluded), "excluded_tickers": sorted(excluded), "source": source}


@app.post("/exclude-tickers", dependencies=[Depends(_require_api_key)])
def exclude_tickers(payload: ExcludeTickersRequest):
    tickers = [str(t).strip().upper() for t in (payload.tickers or []) if str(t).strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided.")
    excluded, source = _load_effective_excluded_tickers()
    excluded.update(tickers)
    if _db_ready():
        db_replace_excluded_tickers(excluded)
    _write_excluded_tickers(EXCLUDED_TICKERS_FILE, excluded)
    return {"count": len(excluded), "excluded_tickers": sorted(excluded), "source": source}


@app.get("/stale-tickers", dependencies=[Depends(_require_api_key)])
def stale_tickers(date: Optional[str] = None):
    if _db_ready():
        mapping = db_load_universe_map()
        if mapping:
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
            return {
                "as_of": target_run_date,
                "count": len(tickers),
                "stale_tickers": sorted(tickers),
                "source": "db",
            }

    data_path = _resolve_path(config.DATA_FILE)
    if not Path(data_path).exists():
        raise HTTPException(status_code=404, detail="Data file not found.")

    mapping = _load_exchange_map(EXCHANGE_MAP_FILE)
    if not mapping:
        raise HTTPException(status_code=404, detail="Exchange map not found.")

    df = pd.read_parquet(data_path, columns=["Close"])
    if not isinstance(df.index, pd.MultiIndex) or "ticker" not in df.index.names:
        raise HTTPException(status_code=500, detail="Unexpected data index format.")

    if date:
        try:
            target_date = pd.to_datetime(date).tz_localize(None)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid date format.")
    else:
        target_date = df.index.get_level_values("date").max()
    if target_date is None:
        raise HTTPException(status_code=404, detail="No dates found in data.")
    target_date = pd.to_datetime(target_date).tz_localize(None)

    tickers = set(mapping.keys())
    excluded, _ = _load_effective_excluded_tickers()
    if excluded:
        tickers -= excluded
    date_values = df.index.get_level_values("date")
    day_data = df.loc[date_values == target_date]
    if not day_data.empty:
        day_data = day_data.reset_index()
        if "ticker" in day_data.columns:
            tickers -= set(day_data["ticker"].astype(str).str.upper().tolist())

    return {
        "as_of": str(pd.to_datetime(target_date).date()),
        "count": len(tickers),
        "stale_tickers": sorted(tickers),
        "source": "file",
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
    tickers = [t.strip().upper() for t in (payload.tickers or []) if t and t.strip()]
    exchange_map: dict[str, str] = {}
    if payload.ticker_exchanges:
        for ticker, exchange in payload.ticker_exchanges.items():
            t = str(ticker).strip().upper()
            ex = str(exchange).strip().upper()
            if t:
                exchange_map[t] = ex or "UNKNOWN"
    if exchange_map:
        tickers.extend(exchange_map.keys())
        tickers = [t for t in tickers if t and t.strip()]
        tickers = sorted(set(tickers))
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

    if _db_ready():
        db_append_pending_adjustments(entries)
    else:
        pending_path = Path(_resolve_path(PENDING_FILE))
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        with pending_path.open("a") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    return {"queued": len(entries)}


@app.get("/pending-adjustments", dependencies=[Depends(_require_api_key)])
def list_pending_adjustments():
    entries: list[dict] = []
    if _db_ready():
        entries = db_load_pending_adjustments()
    else:
        pending_path = Path(_resolve_path(PENDING_FILE))
        if not pending_path.exists():
            return {"pending": []}
        with pending_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {"pending": entries}


@app.post("/clear-pending", dependencies=[Depends(_require_api_key)])
def clear_pending_adjustments():
    if _db_ready():
        db_clear_pending_adjustments()
    else:
        pending_path = Path(_resolve_path(PENDING_FILE))
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("")
    return {"cleared": True}


@app.delete("/pending-adjustments", dependencies=[Depends(_require_api_key)])
def delete_pending_adjustments():
    if _db_ready():
        db_clear_pending_adjustments()
    else:
        pending_path = Path(_resolve_path(PENDING_FILE))
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("")
    return {"cleared": True}


@app.post("/reset-production", dependencies=[Depends(_require_api_key)])
def reset_production(payload: ResetRequest):
    if payload.confirm != "RESET_PRODUCTION_DATA":
        raise HTTPException(status_code=400, detail="Confirmation token missing or invalid.")
    if not _db_ready():
        raise HTTPException(status_code=400, detail="Database not configured.")
    db_reset_production_data()
    return {"reset": True}
