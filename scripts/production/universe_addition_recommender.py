from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.universe_monitor import parse_tradingview_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend new stocks to add to the tradable universe from TradingView candidate data. "
            "Existing universe members are never removed."
        )
    )
    parser.add_argument(
        "--source",
        choices=["db", "file"],
        default="db",
        help="db: read candidates from production_universe_monitor_candidates; file: read --input-file.",
    )
    parser.add_argument("--input-file", default="", help="Candidate input file (required when --source=file).")
    parser.add_argument(
        "--input-format",
        choices=["auto", "flat", "catalog"],
        default="auto",
        help="flat: tabular metrics input; catalog: TradingView symbol catalog.",
    )
    parser.add_argument(
        "--current-universe-source",
        choices=["parquet", "file"],
        default="parquet",
        help="parquet: use DATA_FILE index tickers; file: use --universe-file list.",
    )
    parser.add_argument("--universe-file", default="data/universe_india.txt", help="Universe ticker list when --current-universe-source=file.")

    parser.add_argument("--allowed-exchanges", default="NSE,BSE", help="Allowed exchanges for catalog parse.")
    parser.add_argument("--exchange-priority", default="NSE,BSE", help="Preferred exchange priority for duplicates.")
    parser.add_argument("--max-candidates", type=int, default=3000, help="Cap candidate count after parsing input.")

    parser.add_argument("--fetch-tv-metrics", action="store_true", help="For catalog input, fetch TradingView metrics via tradingview_ta.")
    parser.add_argument("--screener", default="india", help="TradingView screener for metric fetch.")
    parser.add_argument("--interval", default="1d", help="TradingView interval for metric fetch.")
    parser.add_argument("--tv-timeout", type=float, default=None, help="TradingView timeout in seconds.")
    parser.add_argument("--batch-size", type=int, default=200, help="TradingView batch size.")
    parser.add_argument("--max-batches", type=int, default=20, help="TradingView max batches.")

    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--min-volume", type=float, default=250_000.0)
    parser.add_argument("--min-dollar-volume", type=float, default=10_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=0.0, help="0 disables market-cap filter.")
    parser.add_argument("--min-recommend", type=float, default=0.0, help="Minimum TradingView Recommend.All.")
    parser.add_argument("--require-uptrend", dest="require_uptrend", action="store_true", help="Require close > SMA50 > SMA200.")
    parser.add_argument("--no-require-uptrend", dest="require_uptrend", action="store_false", help="Allow candidates without close > SMA50 > SMA200.")
    parser.set_defaults(require_uptrend=True)
    parser.add_argument("--max-additions", type=int, default=25, help="Maximum recommended additions to emit.")
    parser.add_argument("--db-watchlist", action="store_true", help="When source=db, restrict to DB watchlist filter.")
    parser.add_argument("--db-potential", action="store_true", help="When source=db, restrict to DB potential filter.")
    parser.add_argument("--db-min-pass-streak", type=int, default=5, help="Minimum pass_streak required for DB candidates.")
    parser.add_argument("--db-require-monitor-pass", dest="db_require_monitor_pass", action="store_true", help="Require monitor_pass=true for DB candidates.")
    parser.add_argument("--no-db-require-monitor-pass", dest="db_require_monitor_pass", action="store_false", help="Allow DB candidates even if monitor_pass=false.")
    parser.set_defaults(db_require_monitor_pass=True)
    parser.add_argument("--db-require-quality-pass", dest="db_require_quality_pass", action="store_true", help="Require quality_pass=true for DB candidates.")
    parser.add_argument("--no-db-require-quality-pass", dest="db_require_quality_pass", action="store_false", help="Allow DB candidates even if quality_pass=false.")
    parser.set_defaults(db_require_quality_pass=True)
    parser.add_argument("--db-min-quality-rows", type=float, default=250.0, help="Minimum quality_rows for DB candidates when available.")
    parser.add_argument("--db-min-quality-median-adv", type=float, default=2_000_000.0, help="Minimum quality_median_adv_dollars for DB candidates when available.")
    parser.add_argument("--enqueue-adjustments", action="store_true", help="Write recommended additions into production_pending_adjustments.")
    parser.add_argument(
        "--adjustment-note",
        default="Generated by universe_addition_recommender.py",
        help="Note attached when --enqueue-adjustments is enabled.",
    )

    parser.add_argument("--output-root", default="runs/universe_recommender")
    return parser.parse_args()


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


def _safe_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    if pd.isna(value):
        return None
    return bool(value)


def _parse_symbol(symbol: str | None) -> tuple[str | None, str | None]:
    if not symbol:
        return None, None
    token = str(symbol).strip().upper()
    if not token:
        return None, None
    if ":" in token:
        exchange, ticker = token.split(":", 1)
        return exchange.strip() or None, ticker.strip() or None
    return None, token


def _normalize_flat_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            raw = pd.DataFrame(payload["data"])
        elif isinstance(payload, list):
            raw = pd.DataFrame(payload)
        else:
            raise ValueError("Unsupported JSON structure for --input-format flat.")
    elif suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input extension for flat format: {suffix}")

    if raw.empty:
        return pd.DataFrame(columns=["ticker", "exchange", "symbol"])

    cols_lower = {c.lower(): c for c in raw.columns}

    def col(*names: str) -> str | None:
        for name in names:
            c = cols_lower.get(name.lower())
            if c:
                return c
        return None

    symbol_col = col("symbol", "s", "tv_symbol")
    ticker_col = col("ticker", "tradingsymbol", "stock")
    exchange_col = col("exchange")
    close_col = col("close", "last", "price")
    volume_col = col("volume", "vol")
    recommend_col = col("recommend_all", "recommend.all", "recommend")
    sma50_col = col("sma50", "sma_50")
    sma200_col = col("sma200", "sma_200")
    market_cap_col = col("market_cap", "marketcap", "marketCap")
    sector_col = col("sector", "industry")
    dollar_volume_col = col("dollar_volume", "dollarvolume", "turnover")

    rows: list[dict] = []
    for _, row in raw.iterrows():
        exchange = None
        ticker = None
        symbol = None
        if symbol_col:
            exchange, ticker = _parse_symbol(row.get(symbol_col))
            symbol = str(row.get(symbol_col) or "").strip().upper() or None
        if ticker_col and not ticker:
            ticker = str(row.get(ticker_col) or "").strip().upper() or None
        if exchange_col and not exchange:
            exchange = str(row.get(exchange_col) or "").strip().upper() or None
        if ticker and not symbol:
            symbol = f"{exchange}:{ticker}" if exchange else ticker
        if not ticker:
            continue

        close = _safe_float(row.get(close_col)) if close_col else None
        volume = _safe_float(row.get(volume_col)) if volume_col else None
        recommend_all = _safe_float(row.get(recommend_col)) if recommend_col else None
        sma50 = _safe_float(row.get(sma50_col)) if sma50_col else None
        sma200 = _safe_float(row.get(sma200_col)) if sma200_col else None
        market_cap = _safe_float(row.get(market_cap_col)) if market_cap_col else None
        dollar_volume = _safe_float(row.get(dollar_volume_col)) if dollar_volume_col else None
        if dollar_volume is None and close is not None and volume is not None:
            dollar_volume = close * volume
        trend_up = bool(close is not None and sma50 is not None and sma200 is not None and close > sma50 > sma200)
        sector = str(row.get(sector_col) or "").strip() if sector_col else ""

        rows.append(
            {
                "ticker": ticker,
                "exchange": exchange or "UNKNOWN",
                "symbol": symbol or ticker,
                "close": close,
                "volume": volume,
                "dollar_volume": dollar_volume,
                "recommend_all": recommend_all,
                "sma50": sma50,
                "sma200": sma200,
                "trend_up": trend_up,
                "market_cap": market_cap,
                "sector": sector,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["ticker"], keep="first")
    out = out.sort_values("ticker").reset_index(drop=True)
    return out


def _fetch_catalog_metrics(
    catalog_file: Path,
    allowed_exchanges: list[str],
    exchange_priority: list[str],
    max_candidates: int,
    screener: str,
    interval: str,
    timeout: float | None,
    batch_size: int,
    max_batches: int,
) -> pd.DataFrame:
    from src.production_market_data import _fetch_tv_analyses_batch, _resolve_tv_interval

    catalog = parse_tradingview_catalog(
        catalog_file,
        allowed_exchanges=allowed_exchanges,
        exchange_priority=exchange_priority,
        max_candidates=max_candidates,
    )
    if not catalog:
        return pd.DataFrame(
            columns=[
                "ticker",
                "exchange",
                "symbol",
                "close",
                "volume",
                "dollar_volume",
                "recommend_all",
                "sma50",
                "sma200",
                "trend_up",
                "market_cap",
                "sector",
            ]
        )
    symbol_by_ticker = {item["ticker"]: item["symbol"] for item in catalog}
    analyses = _fetch_tv_analyses_batch(
        symbol_by_ticker=symbol_by_ticker,
        screener=screener,
        interval=_resolve_tv_interval(interval),
        timeout=timeout,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    rows: list[dict] = []
    for item in catalog:
        ticker = item["ticker"]
        exchange = item["exchange"]
        symbol = item["symbol"]
        analysis = analyses.get(ticker)
        indicators = getattr(analysis, "indicators", None) or {}
        close = _safe_float(indicators.get("close"))
        volume = _safe_float(indicators.get("volume"))
        recommend_all = _safe_float(indicators.get("Recommend.All"))
        sma50 = _safe_float(indicators.get("SMA50"))
        sma200 = _safe_float(indicators.get("SMA200"))
        dollar_volume = (close * volume) if (close is not None and volume is not None) else None
        trend_up = bool(close is not None and sma50 is not None and sma200 is not None and close > sma50 > sma200)
        rows.append(
            {
                "ticker": ticker,
                "exchange": exchange,
                "symbol": symbol,
                "close": close,
                "volume": volume,
                "dollar_volume": dollar_volume,
                "recommend_all": recommend_all,
                "sma50": sma50,
                "sma200": sma200,
                "trend_up": trend_up,
                "market_cap": None,
                "sector": "",
            }
        )
    return pd.DataFrame(rows)


def _load_current_universe(args: argparse.Namespace) -> set[str]:
    if args.source == "db":
        from src.production_db import db_enabled, init_db, load_universe_map

        if not db_enabled():
            raise RuntimeError("Database is not enabled. Set DATABASE_URL/POSTGRES_URL.")
        init_db()
        mapping = load_universe_map()
        return {str(t).strip().upper() for t in mapping.keys() if str(t).strip()}
    if args.current_universe_source == "file":
        path = Path(config.resolve_path(args.universe_file))
        if not path.exists():
            return set()
        return {line.strip().upper() for line in path.read_text().splitlines() if line.strip()}
    data_path = Path(config.resolve_path(config.DATA_FILE))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found for current universe source: {data_path}")
    df = pd.read_parquet(data_path, columns=["Close"])
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"Expected MultiIndex parquet for current universe source: {data_path}")
    return {str(t).strip().upper() for t in df.index.get_level_values("ticker").unique() if str(t).strip()}


def _passes(candidate: dict, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    close = candidate.get("close")
    volume = candidate.get("volume")
    dollar_volume = candidate.get("dollar_volume")
    market_cap = candidate.get("market_cap")
    recommend_all = candidate.get("recommend_all")
    trend_up = bool(candidate.get("trend_up", False))

    if close is None or close < float(args.min_price):
        reasons.append("price")
    if volume is None or volume < float(args.min_volume):
        reasons.append("volume")
    if dollar_volume is None or dollar_volume < float(args.min_dollar_volume):
        reasons.append("dollar_volume")
    if recommend_all is None or recommend_all < float(args.min_recommend):
        reasons.append("recommend")
    if float(args.min_market_cap) > 0:
        if market_cap is None or market_cap < float(args.min_market_cap):
            reasons.append("market_cap")
    if bool(args.require_uptrend) and not trend_up:
        reasons.append("trend")
    if args.source == "db":
        pass_streak = _safe_float(candidate.get("pass_streak"))
        monitor_pass = candidate.get("monitor_pass")
        quality_pass = candidate.get("quality_pass")
        quality_rows = _safe_float(candidate.get("quality_rows"))
        quality_median_adv = _safe_float(candidate.get("quality_median_adv_dollars"))

        if bool(args.db_require_monitor_pass) and monitor_pass is not None and not bool(monitor_pass):
            reasons.append("monitor_pass")
        if pass_streak is not None and pass_streak < float(max(0, args.db_min_pass_streak)):
            reasons.append("pass_streak")
        if bool(args.db_require_quality_pass) and quality_pass is not None and not bool(quality_pass):
            reasons.append("quality_pass")
        if quality_rows is not None and quality_rows < float(max(0.0, args.db_min_quality_rows)):
            reasons.append("quality_rows")
        if (
            quality_median_adv is not None
            and quality_median_adv < float(max(0.0, args.db_min_quality_median_adv))
        ):
            reasons.append("quality_adv")
    return len(reasons) == 0, reasons


def _score(candidate: dict) -> float:
    dollar_volume = _safe_float(candidate.get("dollar_volume")) or 0.0
    recommend_all = _safe_float(candidate.get("recommend_all"))
    market_cap = _safe_float(candidate.get("market_cap"))
    trend_up = bool(candidate.get("trend_up", False))
    pass_streak = _safe_float(candidate.get("pass_streak"))
    quality_pass = candidate.get("quality_pass")

    liq_score = min(1.0, np.log1p(max(0.0, dollar_volume)) / np.log1p(1_000_000_000.0))
    rec_score = 0.50 if recommend_all is None else float(np.clip((recommend_all + 1.0) / 2.0, 0.0, 1.0))
    cap_score = 0.50 if market_cap is None else min(1.0, np.log1p(max(0.0, market_cap)) / np.log1p(1_000_000_000_000.0))
    trend_score = 1.0 if trend_up else 0.0
    streak_score = 0.50 if pass_streak is None else float(np.clip(pass_streak / 20.0, 0.0, 1.0))
    quality_score = 0.50 if quality_pass is None else (1.0 if bool(quality_pass) else 0.0)
    return float(
        0.40 * liq_score
        + 0.20 * rec_score
        + 0.15 * cap_score
        + 0.10 * trend_score
        + 0.10 * streak_score
        + 0.05 * quality_score
    )


def _resolve_input(path: Path, input_format: str, args: argparse.Namespace) -> pd.DataFrame:
    mode = input_format
    if mode == "auto":
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                if payload.get("data") and isinstance(payload["data"][0], dict) and "s" in payload["data"][0]:
                    mode = "catalog"
                else:
                    mode = "flat"
            else:
                mode = "flat"
        else:
            mode = "flat"
    if mode == "catalog":
        if not args.fetch_tv_metrics:
            catalog = parse_tradingview_catalog(
                path,
                allowed_exchanges=[x.strip().upper() for x in args.allowed_exchanges.split(",") if x.strip()],
                exchange_priority=[x.strip().upper() for x in args.exchange_priority.split(",") if x.strip()],
                max_candidates=args.max_candidates,
            )
            return pd.DataFrame(catalog)
        return _fetch_catalog_metrics(
            catalog_file=path,
            allowed_exchanges=[x.strip().upper() for x in args.allowed_exchanges.split(",") if x.strip()],
            exchange_priority=[x.strip().upper() for x in args.exchange_priority.split(",") if x.strip()],
            max_candidates=args.max_candidates,
            screener=args.screener,
            interval=args.interval,
            timeout=args.tv_timeout,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
    return _normalize_flat_input(path)


def _load_candidates_from_db(args: argparse.Namespace) -> pd.DataFrame:
    from src.production_db import db_enabled, init_db, list_universe_monitor_candidates

    if not db_enabled():
        raise RuntimeError("Database is not enabled. Set DATABASE_URL/POSTGRES_URL.")
    init_db()
    offset = 0
    limit = 1000
    rows: list[dict] = []
    while True:
        total, page = list_universe_monitor_candidates(
            limit=limit,
            offset=offset,
            watchlist=bool(args.db_watchlist),
            potential=bool(args.db_potential),
        )
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        if offset >= total:
            break
    if not rows:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for item in rows:
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        exchange = str(item.get("exchange") or "UNKNOWN").strip().upper() or "UNKNOWN"
        symbol = str(item.get("symbol") or (f"{exchange}:{ticker}" if exchange != "UNKNOWN" else ticker)).strip().upper()
        close = _safe_float(item.get("close"))
        volume = _safe_float(item.get("volume"))
        dollar_volume = _safe_float(item.get("dollar_volume"))
        recommend_all = _safe_float(item.get("recommend_all"))
        sma50 = _safe_float(item.get("sma50"))
        sma200 = _safe_float(item.get("sma200"))
        trend_up = bool(item.get("trend_up", False))
        market_cap = _safe_float(item.get("yfinance_market_cap"))
        if market_cap is None:
            market_cap = _safe_float(item.get("market_cap_basic"))
        if dollar_volume is None and close is not None and volume is not None:
            dollar_volume = close * volume
        if not trend_up and close is not None and sma50 is not None and sma200 is not None:
            trend_up = bool(close > sma50 > sma200)
        out_rows.append(
            {
                "ticker": ticker,
                "exchange": exchange,
                "symbol": symbol,
                "close": close,
                "volume": volume,
                "dollar_volume": dollar_volume,
                "recommend_all": recommend_all,
                "sma50": sma50,
                "sma200": sma200,
                "trend_up": trend_up,
                "market_cap": market_cap,
                "sector": str(item.get("sector") or "").strip(),
                "in_current_universe": bool(item.get("in_current_universe", False)),
                "monitor_pass": item.get("monitor_pass"),
                "pass_streak": item.get("pass_streak"),
                "quality_pass": item.get("quality_pass"),
                "quality_rows": _safe_float(item.get("quality_rows")),
                "quality_median_adv_dollars": _safe_float(item.get("quality_median_adv_dollars")),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()
    current_universe = _load_current_universe(args)
    if args.source == "db":
        input_path = None
        candidates_df = _load_candidates_from_db(args)
    else:
        if not str(args.input_file).strip():
            raise ValueError("--input-file is required when --source=file.")
        input_path = Path(config.resolve_path(args.input_file))
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        candidates_df = _resolve_input(input_path, args.input_format, args)
    if candidates_df.empty:
        raise ValueError("No candidates available from input.")

    if "ticker" not in candidates_df.columns:
        raise ValueError("Candidate input must contain ticker information.")
    candidates_df["ticker"] = candidates_df["ticker"].astype(str).str.strip().str.upper()
    candidates_df = candidates_df[candidates_df["ticker"] != ""]
    candidates_df = candidates_df.drop_duplicates(subset=["ticker"], keep="first")
    if args.max_candidates > 0:
        candidates_df = candidates_df.head(args.max_candidates)

    rows: list[dict] = []
    for _, row in candidates_df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        item = {
            "ticker": ticker,
            "exchange": str(row.get("exchange", "UNKNOWN") or "UNKNOWN").strip().upper(),
            "symbol": str(row.get("symbol", ticker) or ticker).strip().upper(),
            "close": _safe_float(row.get("close")),
            "volume": _safe_float(row.get("volume")),
            "dollar_volume": _safe_float(row.get("dollar_volume")),
            "recommend_all": _safe_float(row.get("recommend_all")),
            "sma50": _safe_float(row.get("sma50")),
            "sma200": _safe_float(row.get("sma200")),
            "trend_up": bool(row.get("trend_up", False)),
            "market_cap": _safe_float(row.get("market_cap")),
            "sector": str(row.get("sector", "") or "").strip(),
            "monitor_pass": row.get("monitor_pass"),
            "pass_streak": _safe_float(row.get("pass_streak")),
            "quality_pass": row.get("quality_pass"),
            "quality_rows": _safe_float(row.get("quality_rows")),
            "quality_median_adv_dollars": _safe_float(row.get("quality_median_adv_dollars")),
        }
        if item["dollar_volume"] is None and item["close"] is not None and item["volume"] is not None:
            item["dollar_volume"] = item["close"] * item["volume"]
        if not item["trend_up"] and item["close"] is not None and item["sma50"] is not None and item["sma200"] is not None:
            item["trend_up"] = bool(item["close"] > item["sma50"] > item["sma200"])

        in_current_universe = ticker in current_universe
        if "in_current_universe" in row and row.get("in_current_universe") is not None:
            in_current_universe = bool(row.get("in_current_universe"))
        passes, fail_reasons = _passes(item, args)
        score = _score(item)
        rows.append(
            {
                **item,
                "in_current_universe": in_current_universe,
                "passes_filters": bool(passes),
                "fail_reasons": ",".join(fail_reasons),
                "recommendation_score": score,
            }
        )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise ValueError("No valid candidates after normalization.")

    all_df = all_df.sort_values(
        by=["passes_filters", "recommendation_score", "dollar_volume"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    additions = all_df[
        (~all_df["in_current_universe"]) & (all_df["passes_filters"])
    ].copy()
    additions = additions.sort_values(
        by=["recommendation_score", "dollar_volume", "recommend_all"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if args.max_additions > 0:
        additions = additions.head(args.max_additions)

    run_started_utc = datetime.utcnow()
    run_id = run_started_utc.strftime("%Y%m%d_%H%M%S")
    run_date = run_started_utc.date().isoformat()
    generated_at_utc = run_started_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    run_dir = Path(args.output_root) / f"universe_addition_recommender_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_path = run_dir / "candidates_scored.csv"
    adds_path = run_dir / "recommended_additions.csv"
    queue_path = run_dir / "queue_adjustments_payload.json"
    summary_path = run_dir / "summary.json"

    all_df.to_csv(all_path, index=False)
    additions.to_csv(adds_path, index=False)
    queue_payload = {
        "type": "tickers",
        "tickers": additions["ticker"].tolist(),
        "exchanges": {
            str(t).strip().upper(): str(ex).strip().upper()
            for t, ex in zip(additions["ticker"], additions["exchange"])
            if str(t).strip()
        },
        "note": "Generated by universe_addition_recommender.py",
    }
    with queue_path.open("w", encoding="utf-8") as f:
        json.dump(queue_payload, f, indent=2)

    summary = {
        "run_id": run_id,
        "run_date": run_date,
        "generated_at_utc": generated_at_utc,
        "source": args.source,
        "input_file": str(input_path) if input_path is not None else None,
        "input_format": args.input_format,
        "current_universe_source": args.current_universe_source,
        "current_universe_size": int(len(current_universe)),
        "candidates_evaluated": int(len(all_df)),
        "already_in_universe": int(all_df["in_current_universe"].sum()),
        "passes_filters_total": int(all_df["passes_filters"].sum()),
        "recommended_additions": int(len(additions)),
        "thresholds": {
            "min_price": float(args.min_price),
            "min_volume": float(args.min_volume),
            "min_dollar_volume": float(args.min_dollar_volume),
            "min_market_cap": float(args.min_market_cap),
            "min_recommend": float(args.min_recommend),
            "require_uptrend": bool(args.require_uptrend),
            "db_min_pass_streak": int(args.db_min_pass_streak),
            "db_require_monitor_pass": bool(args.db_require_monitor_pass),
            "db_require_quality_pass": bool(args.db_require_quality_pass),
            "db_min_quality_rows": float(args.db_min_quality_rows),
            "db_min_quality_median_adv": float(args.db_min_quality_median_adv),
        },
        "outputs": {
            "all_candidates": str(all_path),
            "recommended_additions": str(adds_path),
            "queue_payload": str(queue_path),
        },
    }

    db_snapshot_persisted = False
    if args.source == "db":
        from src.production_db import (
            db_enabled,
            init_db,
            replace_universe_addition_recommendations_snapshot,
        )

        if not db_enabled():
            raise RuntimeError("Database is not enabled. Set DATABASE_URL/POSTGRES_URL.")
        init_db()

        recommended_tickers = {str(t).strip().upper() for t in additions["ticker"].tolist()}
        db_records: list[dict] = []
        for _, row in all_df.iterrows():
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            db_records.append(
                {
                    "ticker": ticker,
                    "exchange": str(row.get("exchange") or "").strip().upper() or None,
                    "symbol": str(row.get("symbol") or "").strip().upper() or None,
                    "in_current_universe": _safe_bool(row.get("in_current_universe")),
                    "passes_filters": _safe_bool(row.get("passes_filters")),
                    "fail_reasons": str(row.get("fail_reasons") or "").strip(),
                    "recommendation_score": _safe_float(row.get("recommendation_score")),
                    "is_recommended": ticker in recommended_tickers,
                    "close": _safe_float(row.get("close")),
                    "volume": _safe_float(row.get("volume")),
                    "dollar_volume": _safe_float(row.get("dollar_volume")),
                    "recommend_all": _safe_float(row.get("recommend_all")),
                    "sma50": _safe_float(row.get("sma50")),
                    "sma200": _safe_float(row.get("sma200")),
                    "trend_up": _safe_bool(row.get("trend_up")),
                    "market_cap": _safe_float(row.get("market_cap")),
                    "sector": str(row.get("sector") or "").strip(),
                    "monitor_pass": _safe_bool(row.get("monitor_pass")),
                    "pass_streak": _safe_float(row.get("pass_streak")),
                    "quality_pass": _safe_bool(row.get("quality_pass")),
                    "quality_rows": _safe_float(row.get("quality_rows")),
                    "quality_median_adv_dollars": _safe_float(row.get("quality_median_adv_dollars")),
                }
            )
        replace_universe_addition_recommendations_snapshot(summary=summary, records=db_records)
        db_snapshot_persisted = True
    summary["db_snapshot_persisted"] = db_snapshot_persisted

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.enqueue_adjustments and not additions.empty:
        from src.production_db import append_pending_adjustments, db_enabled, init_db

        if not db_enabled():
            raise RuntimeError("Database is not enabled. Set DATABASE_URL/POSTGRES_URL.")
        init_db()
        entry = {
            "type": "tickers",
            "tickers": additions["ticker"].tolist(),
            "exchanges": {
                str(t).strip().upper(): str(ex).strip().upper()
                for t, ex in zip(additions["ticker"], additions["exchange"])
                if str(t).strip()
            },
            "note": args.adjustment_note,
            "source": "universe_addition_recommender",
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        append_pending_adjustments([entry])
        print(f"Queued pending adjustment with {len(entry['tickers'])} tickers.")

    print(f"Run directory: {run_dir}")
    print(f"Candidates evaluated: {summary['candidates_evaluated']}")
    print(f"Already in current universe: {summary['already_in_universe']}")
    print(f"Passed filters: {summary['passes_filters_total']}")
    print(f"Recommended additions: {summary['recommended_additions']}")
    print(f"DB snapshot persisted: {summary['db_snapshot_persisted']}")
    print(f"Scored candidates: {all_path}")
    print(f"Recommended additions: {adds_path}")
    print(f"Queue payload: {queue_path}")


if __name__ == "__main__":
    main()
