from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.production_db import (
    db_enabled,
    init_db as db_init,
    latest_universe_monitor_summary as db_latest_universe_monitor_summary,
    list_universe_monitor_candidates as db_list_universe_monitor_candidates,
    replace_universe_monitor_snapshot as db_replace_universe_monitor_snapshot,
)
from src.production_market_data import _fetch_tv_analyses_batch, _resolve_tv_interval
from src.universe_monitor import is_technology_sector, parse_tradingview_catalog, update_monitor_records
from src.universe_quality import compute_quality_exclusions

DEFAULT_TECH_KEYWORDS = (
    "technology",
    "software",
    "semiconductor",
    "information technology",
    "it services",
    "computer hardware",
    "communication equipment",
    "internet services",
    "electronic components",
    "cybersecurity",
    "cloud",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a broad TradingView symbol list, keep Technology candidates, "
            "apply universe-quality prechecks, and produce a manual review list."
        )
    )
    parser.add_argument("--catalog-file", default="data/tradingviewdata.txt", help="TradingView symbol catalog JSON.")
    parser.add_argument("--universe-file", default="data/universe_india.txt", help="Current universe ticker list.")
    parser.add_argument(
        "--allowed-exchanges",
        default="NSE,BSE",
        help="Comma-separated exchange allowlist when parsing TradingView symbols.",
    )
    parser.add_argument(
        "--exchange-priority",
        default="NSE,BSE",
        help="Comma-separated preferred order when a ticker appears on multiple exchanges.",
    )
    parser.add_argument("--max-candidates", type=int, default=3000, help="Max symbols to evaluate per run.")
    parser.add_argument("--screener", default="india", help="TradingView screener.")
    parser.add_argument("--interval", default="1d", help="TradingView interval.")
    parser.add_argument("--tv-timeout", type=float, default=None, help="TradingView request timeout in seconds.")
    parser.add_argument("--batch-size", type=int, default=200, help="TradingView batch size.")
    parser.add_argument("--max-batches", type=int, default=20, help="Max TradingView batches per run.")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum TradingView close price.")
    parser.add_argument("--min-volume", type=float, default=200_000, help="Minimum TradingView volume.")
    parser.add_argument("--min-dollar-volume", type=float, default=2_000_000.0, help="Minimum close*volume.")
    parser.add_argument("--min-market-cap", type=float, default=2_000_000_000.0, help="Minimum market cap.")
    parser.add_argument(
        "--min-recommend",
        type=float,
        default=-0.2,
        help="Minimum TradingView Recommend.All score.",
    )
    parser.add_argument(
        "--require-uptrend",
        action="store_true",
        help="Require close > SMA50 > SMA200 from TradingView indicators.",
    )
    parser.add_argument(
        "--sector-keywords",
        default=",".join(DEFAULT_TECH_KEYWORDS),
        help="Comma-separated keywords used to classify Technology sectors.",
    )
    parser.add_argument(
        "--max-sector-lookups",
        type=int,
        default=700,
        help="Maximum new yfinance metadata lookups (sector + market cap) per run (cached thereafter).",
    )
    parser.add_argument("--history-period", default="3y", help="yfinance history period for quality checks.")
    parser.add_argument("--history-interval", default="1d", help="yfinance interval for quality checks.")
    parser.add_argument(
        "--max-history-checks",
        type=int,
        default=400,
        help="Maximum Technology candidates to run through quality checks per run.",
    )
    parser.add_argument(
        "--min-pass-days",
        type=int,
        default=5,
        help="Consecutive passing runs required before flagging potential additions.",
    )
    parser.add_argument("--output-root", default="runs/universe_monitor", help="Output root directory.")
    parser.add_argument("--skip-file-output", action="store_true", help="Do not write CSV/JSON run artifacts.")
    return parser.parse_args()


def _resolve_path(path: str) -> Path:
    resolved = config.resolve_path(path)
    if os.path.isabs(resolved):
        return Path(resolved)
    return Path(PROJECT_ROOT) / resolved


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _load_universe_tickers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().upper() for line in path.read_text().splitlines() if line.strip()}


def _load_monitor_state_from_db() -> tuple[str | None, dict[str, dict], dict[str, str], dict[str, float | None]]:
    summary = db_latest_universe_monitor_summary() or {}
    prev_run_date_raw = summary.get("run_date")
    prev_run_date = str(prev_run_date_raw) if prev_run_date_raw else None

    records: dict[str, dict] = {}
    sector_cache: dict[str, str] = {}
    market_cap_cache: dict[str, float | None] = {}

    limit = 1000
    offset = 0
    while True:
        total, rows = db_list_universe_monitor_candidates(
            limit=limit,
            offset=offset,
            watchlist=False,
            potential=False,
        )
        if not rows:
            break
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            sector_val = str(row.get("sector") or "unknown").strip() or "unknown"
            sector_cache[ticker] = sector_val
            market_cap_cache[ticker] = _safe_float(row.get("yfinance_market_cap"))
            records[ticker] = {
                "pass_streak": int(row.get("pass_streak", 0) or 0),
                "total_pass_days": int(row.get("total_pass_days", 0) or 0),
                "last_status": "pass" if bool(row.get("monitor_pass", False)) else "fail",
                "last_seen": prev_run_date,
            }
        offset += len(rows)
        if offset >= total:
            break
    return prev_run_date, records, sector_cache, market_cap_cache


def _extract_tv_metrics(analysis) -> dict:
    indicators = getattr(analysis, "indicators", None) or {}
    close = _safe_float(indicators.get("close"))
    volume = _safe_float(indicators.get("volume"))
    recommend_all = _safe_float(indicators.get("Recommend.All"))
    sma50 = _safe_float(indicators.get("SMA50"))
    sma200 = _safe_float(indicators.get("SMA200"))
    dollar_volume = (close * volume) if (close is not None and volume is not None) else None
    trend_up = False
    if close is not None and sma50 is not None and sma200 is not None:
        trend_up = bool(close > sma50 > sma200)
    return {
        "close": close,
        "volume": volume,
        "recommend_all": recommend_all,
        "sma50": sma50,
        "sma200": sma200,
        "dollar_volume": dollar_volume,
        "trend_up": trend_up,
    }


def _passes_tv_precheck(metrics: dict, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    close = metrics.get("close")
    volume = metrics.get("volume")
    dollar_volume = metrics.get("dollar_volume")
    recommend_all = metrics.get("recommend_all")
    trend_up = bool(metrics.get("trend_up"))

    if close is None or close < args.min_price:
        reasons.append("price")
    if volume is None or volume < args.min_volume:
        reasons.append("volume")
    if dollar_volume is None or dollar_volume < args.min_dollar_volume:
        reasons.append("dollar_volume")
    if recommend_all is None or recommend_all < args.min_recommend:
        reasons.append("recommend")
    if args.require_uptrend and not trend_up:
        reasons.append("trend")
    return len(reasons) == 0, reasons


def _parse_keywords(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _fetch_yfinance_metadata(ticker: str) -> tuple[str, float | None]:
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        if hasattr(stock, "get_info"):
            info = stock.get_info()
        else:
            info = getattr(stock, "info", None)
    except Exception:  # noqa: BLE001
        return "unknown", None

    if not isinstance(info, dict):
        return "unknown", None

    sector = info.get("sector") or info.get("industry") or "unknown"
    if not isinstance(sector, str) or not sector.strip():
        sector = "unknown"
    sector = sector.strip()

    market_cap = _safe_float(info.get("marketCap"))
    return sector, market_cap


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["adv_21"] = out["Volume"].rolling(window=config.ADV_LOOKBACK, min_periods=config.ADV_LOOKBACK).mean()
    out["vol_21"] = out["log_return"].rolling(window=config.ADV_LOOKBACK, min_periods=config.ADV_LOOKBACK).std()
    return out


def _apply_quality_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if config.UNIVERSE_QUALITY_START_DATE:
        start = pd.to_datetime(config.UNIVERSE_QUALITY_START_DATE)
        out = out[out.index >= start]
    if config.UNIVERSE_QUALITY_END_DATE:
        end = pd.to_datetime(config.UNIVERSE_QUALITY_END_DATE)
        out = out[out.index < end]
    return out


def _quality_metrics(history: pd.DataFrame) -> dict:
    work = _apply_quality_window(history)
    if work.empty:
        return {
            "rows": 0,
            "median_adv_dollars": np.nan,
            "p20_adv_dollars": np.nan,
            "p05_price": np.nan,
            "median_vol21": np.nan,
            "p95_abs_log_return": np.nan,
            "quality_reasons": ["empty_window"],
        }

    rows = int(len(work))
    adv_dollars = work["adv_21"] * work["Close"] if {"adv_21", "Close"}.issubset(work.columns) else pd.Series(dtype=float)
    median_adv_dollars = float(adv_dollars.median()) if not adv_dollars.empty else np.nan
    p20_adv_dollars = float(adv_dollars.quantile(0.20)) if not adv_dollars.empty else np.nan
    p05_price = float(work["Close"].quantile(0.05)) if "Close" in work.columns else np.nan
    median_vol21 = float(work["vol_21"].median()) if "vol_21" in work.columns else np.nan
    p95_abs_log_return = (
        float(work["log_return"].abs().quantile(0.95)) if "log_return" in work.columns else np.nan
    )

    reasons: list[str] = []
    if rows < config.UNIVERSE_MIN_HISTORY_ROWS:
        reasons.append("min_history_rows")
    if np.isfinite(median_adv_dollars) and median_adv_dollars < config.UNIVERSE_MIN_MEDIAN_ADV_DOLLARS:
        reasons.append("min_median_adv_dollars")
    if np.isfinite(p20_adv_dollars) and p20_adv_dollars < config.UNIVERSE_MIN_P20_ADV_DOLLARS:
        reasons.append("min_p20_adv_dollars")
    if np.isfinite(p05_price) and p05_price < config.UNIVERSE_MIN_P05_PRICE:
        reasons.append("min_p05_price")
    if np.isfinite(median_vol21) and median_vol21 > config.UNIVERSE_MAX_MEDIAN_VOL21:
        reasons.append("max_median_vol21")
    if np.isfinite(p95_abs_log_return) and p95_abs_log_return > config.UNIVERSE_MAX_P95_ABS_LOG_RETURN:
        reasons.append("max_p95_abs_log_return")

    return {
        "rows": rows,
        "median_adv_dollars": median_adv_dollars,
        "p20_adv_dollars": p20_adv_dollars,
        "p05_price": p05_price,
        "median_vol21": median_vol21,
        "p95_abs_log_return": p95_abs_log_return,
        "quality_reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    if not db_enabled():
        raise ValueError(
            "Database is required for tech universe monitor streak/state tracking. "
            "Set DATABASE_URL or POSTGRES_URL."
        )
    db_init()
    allowed_exchanges = [item.strip().upper() for item in args.allowed_exchanges.split(",") if item.strip()]
    exchange_priority = [item.strip().upper() for item in args.exchange_priority.split(",") if item.strip()]
    sector_keywords = _parse_keywords(args.sector_keywords)
    run_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    output_root = _resolve_path(args.output_root)
    universe_path = _resolve_path(args.universe_file)
    catalog_path = _resolve_path(args.catalog_file)
    _, records, sector_cache, market_cap_cache = _load_monitor_state_from_db()
    universe_tickers = _load_universe_tickers(universe_path)

    catalog = parse_tradingview_catalog(
        catalog_path=catalog_path,
        allowed_exchanges=allowed_exchanges,
        exchange_priority=exchange_priority,
        max_candidates=None,
    )
    candidates = [item for item in catalog if item["ticker"] not in universe_tickers]
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    if not candidates:
        raise ValueError("No candidates available after parsing catalog and excluding current universe.")

    symbol_by_ticker = {item["ticker"]: item["symbol"] for item in candidates}
    interval = _resolve_tv_interval(args.interval)
    analysis_map = _fetch_tv_analyses_batch(
        symbol_by_ticker=symbol_by_ticker,
        screener=args.screener,
        interval=interval,
        timeout=args.tv_timeout,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    evaluation_rows: list[dict] = []
    tech_tv_candidates: list[str] = []
    for item in candidates:
        ticker = item["ticker"]
        analysis = analysis_map.get(ticker)
        row = {
            "ticker": ticker,
            "exchange": item["exchange"],
            "symbol": item["symbol"],
            "in_current_universe": False,
            "tv_data": analysis is not None,
            "tv_pass": False,
            "tv_fail_reasons": "",
            "sector": sector_cache.get(ticker, "unknown"),
            "tech_sector": False,
            "quality_pass": False,
            "quality_reasons": "",
            "monitor_pass": False,
            "pass_streak": 0,
            "total_pass_days": 0,
            "close": np.nan,
            "volume": np.nan,
            "dollar_volume": np.nan,
            "market_cap_basic": np.nan,
            "recommend_all": np.nan,
            "sma50": np.nan,
            "sma200": np.nan,
            "trend_up": False,
            "quality_rows": np.nan,
            "quality_median_adv_dollars": np.nan,
            "quality_p20_adv_dollars": np.nan,
            "quality_p05_price": np.nan,
            "quality_median_vol21": np.nan,
            "quality_p95_abs_log_return": np.nan,
            "metadata_fail_reasons": "",
            "market_cap_pass": False,
            "yfinance_market_cap": np.nan,
        }
        if analysis is None:
            row["tv_fail_reasons"] = "missing_analysis"
            evaluation_rows.append(row)
            continue

        metrics = _extract_tv_metrics(analysis)
        for key in ("close", "volume", "dollar_volume", "recommend_all", "sma50", "sma200"):
            row[key] = metrics.get(key)
        row["trend_up"] = bool(metrics.get("trend_up", False))
        tv_pass, tv_fail_reasons = _passes_tv_precheck(metrics, args)
        row["tv_pass"] = tv_pass
        row["tv_fail_reasons"] = ",".join(tv_fail_reasons)
        if tv_pass:
            tech_tv_candidates.append(ticker)
        evaluation_rows.append(row)

    uncached = [
        ticker
        for ticker in tech_tv_candidates
        if ticker not in sector_cache or ticker not in market_cap_cache
    ]
    if args.max_sector_lookups > 0:
        uncached = uncached[: args.max_sector_lookups]
    for ticker in tqdm(uncached, desc="Fetching metadata", unit="ticker"):
        sector, market_cap = _fetch_yfinance_metadata(ticker)
        sector_cache[ticker] = sector
        market_cap_cache[ticker] = market_cap

    for row in evaluation_rows:
        ticker = row["ticker"]
        sector = sector_cache.get(ticker, row.get("sector", "unknown"))
        row["sector"] = sector
        row["tech_sector"] = is_technology_sector(sector, sector_keywords)
        market_cap = _safe_float(market_cap_cache.get(ticker))
        row["yfinance_market_cap"] = market_cap if market_cap is not None else np.nan
        row["market_cap_basic"] = market_cap if market_cap is not None else np.nan
        if args.min_market_cap <= 0:
            row["market_cap_pass"] = True
        else:
            row["market_cap_pass"] = market_cap is not None and market_cap >= args.min_market_cap
            if row["tv_pass"] and not row["market_cap_pass"]:
                row["metadata_fail_reasons"] = "market_cap"

    tech_for_history = [
        row["ticker"]
        for row in evaluation_rows
        if row["tv_pass"] and row["tech_sector"] and row["market_cap_pass"]
    ]
    tech_for_history = list(dict.fromkeys(tech_for_history))
    if args.max_history_checks > 0:
        tech_for_history = tech_for_history[: args.max_history_checks]

    import yfinance as yf

    history_by_ticker: dict[str, pd.DataFrame] = {}
    history_frames: list[pd.DataFrame] = []
    history_failures: dict[str, str] = {}
    for ticker in tqdm(tech_for_history, desc="Downloading history", unit="ticker"):
        try:
            history = yf.Ticker(ticker).history(
                period=args.history_period,
                interval=args.history_interval,
                auto_adjust=True,
            )
        except Exception as exc:  # noqa: BLE001
            history_failures[ticker] = f"history_error:{type(exc).__name__}"
            continue
        if history.empty:
            history_failures[ticker] = "history_empty"
            continue
        if not {"Close", "Volume"}.issubset(history.columns):
            history_failures[ticker] = "history_missing_columns"
            continue
        normalized = _normalize_history(history[["Close", "Volume"]].copy())
        history_by_ticker[ticker] = normalized
        temp = normalized.copy()
        temp["ticker"] = ticker
        temp["date"] = temp.index
        history_frames.append(temp[["date", "ticker", "Close", "Volume", "log_return", "adv_21", "vol_21"]])

    quality_excluded: set[str] = set()
    if history_frames:
        combined_quality_df = pd.concat(history_frames, axis=0, ignore_index=True)
        combined_quality_df["date"] = pd.to_datetime(combined_quality_df["date"])
        combined_quality_df.set_index(["date", "ticker"], inplace=True)
        combined_quality_df.sort_index(inplace=True)
        quality_excluded = compute_quality_exclusions(combined_quality_df)

    metrics_by_ticker: dict[str, dict] = {}
    for ticker, history in history_by_ticker.items():
        metrics_by_ticker[ticker] = _quality_metrics(history)

    for row in evaluation_rows:
        ticker = row["ticker"]
        if not row["tv_pass"] or not row["tech_sector"] or not row["market_cap_pass"]:
            row["quality_reasons"] = "not_eligible"
            row["quality_pass"] = False
            continue
        if ticker in history_failures:
            row["quality_reasons"] = history_failures[ticker]
            row["quality_pass"] = False
            continue
        metrics = metrics_by_ticker.get(ticker)
        if metrics is None:
            row["quality_reasons"] = "quality_unavailable"
            row["quality_pass"] = False
            continue

        row["quality_rows"] = metrics["rows"]
        row["quality_median_adv_dollars"] = metrics["median_adv_dollars"]
        row["quality_p20_adv_dollars"] = metrics["p20_adv_dollars"]
        row["quality_p05_price"] = metrics["p05_price"]
        row["quality_median_vol21"] = metrics["median_vol21"]
        row["quality_p95_abs_log_return"] = metrics["p95_abs_log_return"]
        reasons = list(metrics["quality_reasons"])
        if ticker in quality_excluded and "filtered_by_quality_rules" not in reasons:
            reasons.append("filtered_by_quality_rules")
        row["quality_reasons"] = ",".join(reasons)
        row["quality_pass"] = ticker not in quality_excluded

    for row in evaluation_rows:
        row["monitor_pass"] = bool(
            row["tv_pass"] and row["tech_sector"] and row["market_cap_pass"] and row["quality_pass"]
        )

    updated_records = update_monitor_records(records=records, evaluations=evaluation_rows, run_date=run_date)
    for row in evaluation_rows:
        rec = updated_records.get(row["ticker"], {})
        row["pass_streak"] = int(rec.get("pass_streak", 0) or 0)
        row["total_pass_days"] = int(rec.get("total_pass_days", 0) or 0)

    all_df = pd.DataFrame(evaluation_rows)
    all_df.sort_values(
        by=["pass_streak", "monitor_pass", "recommend_all", "yfinance_market_cap", "dollar_volume"],
        ascending=[False, False, False, False, False],
        inplace=True,
        kind="stable",
    )

    watchlist_df = all_df[all_df["tv_pass"] & all_df["tech_sector"] & all_df["market_cap_pass"]].copy()
    potential_df = all_df[
        all_df["monitor_pass"] & (all_df["pass_streak"] >= int(args.min_pass_days))
    ].copy()

    summary = {
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_date": run_date,
        "catalog_file": str(catalog_path),
        "universe_file": str(universe_path),
        "candidates_evaluated": int(len(all_df)),
        "tv_pass_count": int(all_df["tv_pass"].sum()) if not all_df.empty else 0,
        "tech_tv_pass_count": int((all_df["tv_pass"] & all_df["tech_sector"]).sum()) if not all_df.empty else 0,
        "market_cap_pass_count": int((all_df["tv_pass"] & all_df["market_cap_pass"]).sum()) if not all_df.empty else 0,
        "tech_after_market_cap_count": int(
            (all_df["tv_pass"] & all_df["tech_sector"] & all_df["market_cap_pass"]).sum()
        )
        if not all_df.empty
        else 0,
        "quality_pass_count": int(all_df["quality_pass"].sum()) if not all_df.empty else 0,
        "potential_additions_count": int(len(potential_df)),
        "min_pass_days": int(args.min_pass_days),
        "outputs": {},
    }

    file_paths: dict[str, str] = {}
    if not args.skip_file_output:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / stamp
        run_dir.mkdir(parents=True, exist_ok=True)

        all_path = run_dir / "tech_candidates_all.csv"
        watchlist_path = run_dir / "tech_watchlist.csv"
        potential_path = run_dir / "potential_additions.csv"
        summary_path = run_dir / "summary.json"
        all_df.to_csv(all_path, index=False)
        watchlist_df.to_csv(watchlist_path, index=False)
        potential_df.to_csv(potential_path, index=False)
        file_paths = {
            "all_candidates_csv": str(all_path),
            "watchlist_csv": str(watchlist_path),
            "potential_additions_csv": str(potential_path),
            "summary_json": str(summary_path),
        }
        summary["outputs"] = file_paths
        summary_path.write_text(json.dumps(summary, indent=2))

    db_replace_universe_monitor_snapshot(summary, all_df.to_dict(orient="records"))
    db_written = True

    if file_paths:
        print(f"All candidates: {file_paths['all_candidates_csv']}")
        print(f"Watchlist: {file_paths['watchlist_csv']}")
        print(f"Potential additions: {file_paths['potential_additions_csv']}")
        print(f"Summary: {file_paths['summary_json']}")
    else:
        print("File output: skipped")
    print(f"DB write: {'yes' if db_written else 'no'}")
    print(
        "Counts | "
        f"evaluated={summary['candidates_evaluated']} "
        f"tv_pass={summary['tv_pass_count']} "
        f"tech_tv_pass={summary['tech_tv_pass_count']} "
        f"quality_pass={summary['quality_pass_count']} "
        f"potential={summary['potential_additions_count']}"
    )


if __name__ == "__main__":
    main()
