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
    latest_summary as db_latest_summary,
    latest_trades as db_latest_trades,
    list_run_summaries as db_list_run_summaries,
    list_trades as db_list_trades,
    load_pending_adjustments as db_load_pending_adjustments,
    load_state as db_load_state,
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "runs/production")
PENDING_FILE = os.getenv("PENDING_FILE", "runs/production/pending_adjustments.jsonl")
STATE_FILE = os.getenv("STATE_FILE", "runs/production/state.json")
EXCHANGE_MAP_FILE = os.getenv("EXCHANGE_MAP_FILE", config.TRADINGVIEW_EXCHANGE_MAP_FILE)
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
        except Exception:
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
    mapping = _load_exchange_map(EXCHANGE_MAP_FILE)
    if not mapping:
        raise HTTPException(status_code=404, detail="Exchange map not found.")
    universe = [{"ticker": k, "exchange": v} for k, v in sorted(mapping.items())]
    return {"count": len(universe), "universe": universe}


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
