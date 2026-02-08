from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .market_data_validation import validate_market_data_frame

logger = logging.getLogger(__name__)

REQUIRED_INDICATOR_COLS = [
    "RSI_14",
    "SMA_20",
    "SMA_50",
    "SMA_250",
    "ROC_10",
    "ROC_50",
    "MACD_12_26_9",
    "MACDs_12_26_9",
    "ATRr_14",
    "ADX_14",
    "SMA20_Volume",
]

_BASE_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def update_market_data(
    data_path: str | Path,
    lookback_days: int = 420,
    interval: str = "1d",
    rolling_window: int | None = None,
    vix_ticker: str = "CBOE:VIX",
    screener: str = "america",
    exchange_list: list[str] | None = None,
    timeout: float | None = None,
    require_all_tickers: bool = True,
    exchange_map_path: str | Path | None = None,
    exchange_map: dict[str, str] | None = None,
    batch_size: int = 200,
    max_batches: int = 3,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Incrementally update the daily parquet with latest TradingView bars."""
    import pandas_ta_classic as ta  # noqa: F401

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    existing = pd.read_parquet(data_path)
    validate_market_data_frame(
        existing,
        source=data_path,
        required_columns=_BASE_REQUIRED_COLUMNS,
    )
    existing = _coalesce_vix_columns(existing)

    tickers = sorted(existing.index.get_level_values("ticker").unique())
    rolling_window = rolling_window or config.ADV_LOOKBACK
    lookback_rows = max(lookback_days, 300, rolling_window + 50)
    exchange_list = exchange_list or ["NASDAQ", "NYSE", "AMEX"]
    interval_value = _resolve_tv_interval(interval)
    if exchange_map is None:
        exchange_map_path = exchange_map_path or config.TRADINGVIEW_EXCHANGE_MAP_FILE
        exchange_map = _load_exchange_map(exchange_map_path)
    else:
        normalized_map: dict[str, str] = {}
        for ticker, ex in exchange_map.items():
            t = str(ticker).strip().upper()
            if not t:
                continue
            exchange = str(ex or "").strip().upper()
            normalized_map[t] = exchange or "UNKNOWN"
        exchange_map = normalized_map

    sector_map = {}
    if "sector" in existing.columns:
        sector_map = (
            existing.reset_index()
            .drop_duplicates("ticker")
            .set_index("ticker")["sector"]
            .to_dict()
        )

    updated_rows: list[dict] = []
    updated_tickers: set[str] = set()
    resolved_tickers: set[str] = set()
    diagnostics: dict[str, dict] = {}
    symbol_by_ticker: dict[str, str] = {}
    fallback_exchange = exchange_list[0] if exchange_list else "NYSE"
    for ticker in tickers:
        mapped = exchange_map.get(ticker)
        exchange = mapped if mapped and mapped != "UNKNOWN" else fallback_exchange
        symbol_by_ticker[ticker] = f"{exchange}:{ticker}"

    analysis_map = _fetch_tv_analyses_batch(
        symbol_by_ticker=symbol_by_ticker,
        screener=screener,
        interval=interval_value,
        timeout=timeout,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    for ticker in tqdm(tickers, desc="Processing TradingView data", unit="ticker"):
        symbol = symbol_by_ticker.get(ticker, "")
        analysis = analysis_map.get(ticker)
        if analysis is None:
            resolved_tickers.add(ticker)
            diagnostics[ticker] = _diag_item(
                ticker=ticker,
                status="skipped",
                reason="missing_analysis",
                symbol=symbol,
            )
            continue

        bar_time = getattr(analysis, "time", None)
        bar_time = pd.to_datetime(bar_time) if bar_time is not None else pd.Timestamp.utcnow()
        bar_date = _to_naive_timestamp(bar_time).normalize()

        existing_ticker = existing.xs(ticker, level="ticker", drop_level=False)
        last_ticker_date = _to_naive_timestamp(existing_ticker.index.get_level_values("date").max())
        if bar_date <= last_ticker_date:
            resolved_tickers.add(ticker)
            diagnostics[ticker] = _diag_item(
                ticker=ticker,
                status="skipped",
                reason="no_new_bar",
                symbol=symbol,
                last_data_date=_fmt_date(last_ticker_date),
                bar_date=_fmt_date(bar_date),
            )
            continue

        indicators = analysis.indicators or {}
        open_price = indicators.get("open")
        high_price = indicators.get("high")
        low_price = indicators.get("low")
        close_price = indicators.get("close")
        volume = indicators.get("volume")
        if any(val is None for val in (open_price, high_price, low_price, close_price, volume)):
            diagnostics[ticker] = _diag_item(
                ticker=ticker,
                status="error",
                reason="invalid_ohlcv",
                symbol=symbol,
                last_data_date=_fmt_date(last_ticker_date),
                bar_date=_fmt_date(bar_date),
                message="TradingView returned missing OHLCV fields.",
            )
            continue

        row = _build_updated_row(
            ticker=ticker,
            existing_ticker=existing_ticker,
            bar_date=bar_date,
            open_price=float(open_price),
            high_price=float(high_price),
            low_price=float(low_price),
            close_price=float(close_price),
            volume=float(volume),
            sector=sector_map.get(ticker, "unknown") if sector_map else "unknown",
            rolling_window=rolling_window,
            lookback_rows=lookback_rows,
        )
        if row is None:
            diagnostics[ticker] = _diag_item(
                ticker=ticker,
                status="error",
                reason="feature_build_failed",
                symbol=symbol,
                last_data_date=_fmt_date(last_ticker_date),
                bar_date=_fmt_date(bar_date),
            )
            continue
        updated_rows.append(row)
        updated_tickers.add(ticker)
        resolved_tickers.add(ticker)
        diagnostics[ticker] = _diag_item(
            ticker=ticker,
            status="updated",
            reason="ok",
            symbol=symbol,
            last_data_date=_fmt_date(last_ticker_date),
            bar_date=_fmt_date(bar_date),
        )

    unresolved = sorted(set(tickers) - resolved_tickers)
    diagnostics_payload = _build_update_diagnostics_payload(
        diagnostics=diagnostics,
        total_tickers=len(tickers),
        updated_tickers=updated_tickers,
        unresolved_tickers=unresolved,
        batch_size=batch_size,
        max_batches=max_batches,
    )

    if not updated_rows:
        if return_diagnostics:
            return existing, diagnostics_payload
        return existing

    if require_all_tickers and unresolved:
        preview = ", ".join(unresolved[:10])
        reasons = []
        for ticker in unresolved[:5]:
            item = diagnostics.get(ticker) or {}
            reasons.append(f"{ticker}:{item.get('reason', 'unknown')}")
        reason_preview = ", ".join(reasons)
        raise ValueError(
            "Partial update detected from TradingView. "
            f"Updated {len(updated_tickers)}/{len(tickers)} tickers. "
            f"Missing (first 10): {preview}. "
            f"Reasons (first 5): {reason_preview}"
        )

    updates_df = pd.DataFrame(updated_rows)
    updates_df["date"] = pd.to_datetime(updates_df["date"])
    if getattr(updates_df["date"].dt, "tz", None) is not None:
        updates_df["date"] = updates_df["date"].dt.tz_convert(None)

    vix_close = _fetch_vix_close(
        vix_ticker,
        screener=screener,
        exchange_list=exchange_list,
        interval=interval_value,
        timeout=timeout,
        fallback_series=_extract_vix_series(existing),
    )
    vix_series = _extract_vix_series(existing)
    if vix_close is not None:
        vix_series = vix_series.copy()
        target_date = updates_df["date"].max()
        vix_series.loc[target_date] = vix_close
        vix_series = vix_series.sort_index()
        vix_stats = _compute_rolling_vix_stats(vix_series, rolling_window)
        updates_df = _drop_vix_merge_columns(updates_df)
        updates_df = updates_df.merge(vix_stats, how="left", left_on="date", right_index=True)

    if "vix_return" in updates_df.columns:
        vix_return_series = vix_series.pct_change()
        updates_df["vix_beta_sensitivity"] = updates_df.apply(
            lambda row: _compute_vix_beta(
                ticker=row["ticker"],
                new_date=row["date"],
                new_log_return=row.get("log_return"),
                existing=existing,
                vix_return_series=vix_return_series,
                rolling_window=rolling_window,
            ),
            axis=1,
        )

    updates_df = _normalize_features_cross_sectional(updates_df, REQUIRED_INDICATOR_COLS)
    updates_df = _normalize_additional_zscores(
        updates_df,
        ["dist_sma50", "dist_sma20", "rvol_20", "vix_beta_sensitivity"],
    )

    updates_df.set_index(["date", "ticker"], inplace=True)
    updates_df.sort_index(inplace=True)

    for col in existing.columns:
        if col not in updates_df.columns:
            updates_df[col] = np.nan
    for col in updates_df.columns:
        if col not in existing.columns:
            existing[col] = np.nan

    combined = pd.concat([existing, updates_df], axis=0, sort=False)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined = _prune_history(combined, config.RETENTION_TRADING_DAYS)
    combined.to_parquet(data_path, engine="pyarrow")
    validate_market_data_frame(
        combined,
        source=data_path,
        required_columns=_BASE_REQUIRED_COLUMNS,
    )
    if return_diagnostics:
        return combined, diagnostics_payload
    return combined


def add_universe_tickers(
    data_path: str | Path,
    tickers: Iterable[str],
    period: str = "20y",
    interval: str = "1d",
    min_trading_days: int = 50,
    rolling_window: int | None = None,
    vix_ticker: str = "^VIX",
    recompute_cross_sectional: bool = True,
) -> pd.DataFrame:
    """Add new tickers with yfinance history, then write back to parquet."""
    import yfinance as yf
    import pandas_ta_classic as ta  # noqa: F401

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    existing = pd.read_parquet(data_path)
    validate_market_data_frame(
        existing,
        source=data_path,
        required_columns=_BASE_REQUIRED_COLUMNS,
    )

    existing_tickers = set(existing.index.get_level_values("ticker").unique())
    requested = [str(t).strip().upper() for t in tickers if str(t).strip()]
    new_tickers = [t for t in requested if t not in existing_tickers]
    if not new_tickers:
        return existing

    rolling_window = rolling_window or config.ADV_LOOKBACK

    all_stock_data: dict[str, pd.DataFrame] = {}
    sector_by_ticker: dict[str, str] = {}
    fetch_errors: list[str] = []
    for ticker in tqdm(new_tickers, desc="Fetching yfinance history", unit="ticker"):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:  # noqa: BLE001
            msg = f"{ticker}: {type(exc).__name__}({exc})"
            fetch_errors.append(msg)
            logger.warning("Failed to fetch yfinance history for %s: %s", ticker, exc)
            continue
        if df.empty:
            fetch_errors.append(f"{ticker}: empty_history")
            logger.warning("Skipping %s: no yfinance history returned.", ticker)
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        for col in ("Dividends", "Stock Splits"):
            if col not in df.columns:
                df[col] = 0.0
        df = _calculate_indicators(df)
        for col in REQUIRED_INDICATOR_COLS:
            if col not in df.columns:
                df[col] = np.nan
        cleaned = df.dropna(subset=REQUIRED_INDICATOR_COLS)
        if len(cleaned) < min_trading_days:
            fetch_errors.append(f"{ticker}: insufficient_history({len(cleaned)})")
            continue
        all_stock_data[ticker] = cleaned
        sector_by_ticker[ticker] = _fetch_sector_yfinance(stock, ticker=ticker)

    if not all_stock_data:
        preview = ", ".join(fetch_errors[:10]) if fetch_errors else "no valid ticker data"
        raise ValueError(
            "No new tickers had sufficient data to add. "
            f"Sample failures: {preview}"
        )

    all_data_list: list[pd.DataFrame] = []
    for ticker, df in all_stock_data.items():
        temp_df = df.copy()
        temp_df["ticker"] = ticker
        all_data_list.append(temp_df)

    master_df = pd.concat(all_data_list)
    master_df.reset_index(inplace=True)
    if "Date" in master_df.columns:
        master_df.rename(columns={"Date": "date"}, inplace=True)
    elif "index" in master_df.columns:
        master_df.rename(columns={"index": "date"}, inplace=True)
    master_df["date"] = pd.to_datetime(master_df["date"]).dt.tz_localize(None)
    if sector_by_ticker:
        master_df["sector"] = master_df["ticker"].map(lambda x: sector_by_ticker.get(x, "unknown"))

    master_df.sort_values(by=["ticker", "date"], inplace=True)
    master_df["log_return"] = master_df.groupby("ticker")["Close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    master_df["adv_21"] = master_df.groupby("ticker")["Volume"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=rolling_window).mean()
    )
    master_df["vol_21"] = master_df.groupby("ticker")["log_return"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=rolling_window).std()
    )

    vix_series = _extract_vix_series(existing)
    if vix_series.empty:
        vix_df = yf.Ticker(vix_ticker).history(period=period, interval=interval, auto_adjust=True)
        if not vix_df.empty:
            vix_df = vix_df.copy()
            vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
            vix_series = vix_df["Close"].copy()

    if not vix_series.empty:
        vix_df = pd.DataFrame({"Close": vix_series}).sort_index()
        vix_stats = _compute_rolling_vix_z(vix_df, rolling_window)
        master_df = master_df.merge(vix_stats, how="left", left_on="date", right_index=True)
        if "vix_return" in master_df.columns:

            def _beta(group: pd.DataFrame) -> pd.Series:
                cov = group["log_return"].rolling(window=rolling_window, min_periods=rolling_window).cov(
                    group["vix_return"]
                )
                var = group["vix_return"].rolling(window=rolling_window, min_periods=rolling_window).var()
                return cov / (var + 1e-9)

            master_df["vix_beta_sensitivity"] = master_df.groupby("ticker", group_keys=False).apply(_beta)

    master_df = _normalize_features_cross_sectional(master_df, REQUIRED_INDICATOR_COLS)
    master_df = _add_swing_features(master_df)
    master_df = _normalize_additional_zscores(
        master_df,
        ["dist_sma50", "dist_sma20", "rvol_20", "vix_beta_sensitivity"],
    )

    master_df.set_index(["date", "ticker"], inplace=True)
    master_df.sort_index(inplace=True)

    for col in existing.columns:
        if col not in master_df.columns:
            master_df[col] = np.nan
    for col in master_df.columns:
        if col not in existing.columns:
            existing[col] = np.nan

    combined = pd.concat([existing, master_df], axis=0, sort=False)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    if recompute_cross_sectional:
        combined = _recompute_cross_sectional_z(
            combined,
            REQUIRED_INDICATOR_COLS + ["dist_sma50", "dist_sma20", "rvol_20", "vix_beta_sensitivity"],
        )

    combined = _prune_history(combined, config.RETENTION_TRADING_DAYS)
    combined.to_parquet(data_path, engine="pyarrow")
    validate_market_data_frame(
        combined,
        source=data_path,
        required_columns=_BASE_REQUIRED_COLUMNS,
    )
    return combined


def _diag_item(
    ticker: str,
    status: str,
    reason: str,
    symbol: str | None = None,
    last_data_date: str | None = None,
    bar_date: str | None = None,
    message: str | None = None,
) -> dict:
    item = {
        "ticker": ticker,
        "status": status,
        "reason": reason,
        "symbol": symbol,
        "last_data_date": last_data_date,
        "bar_date": bar_date,
    }
    if message:
        item["message"] = message
    return item


def _build_update_diagnostics_payload(
    diagnostics: dict[str, dict],
    total_tickers: int,
    updated_tickers: set[str],
    unresolved_tickers: list[str],
    batch_size: int,
    max_batches: int,
) -> dict:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for item in diagnostics.values():
        status = str(item.get("status") or "unknown")
        reason = str(item.get("reason") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tickers_total": total_tickers,
        "tickers_with_diagnostics": len(diagnostics),
        "updated_count": len(updated_tickers),
        "unresolved_count": len(unresolved_tickers),
        "unresolved_tickers": unresolved_tickers,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "request_config": {
            "batch_size": batch_size,
            "max_batches": max_batches,
        },
        "tickers": sorted(diagnostics.values(), key=lambda x: (x["ticker"])),
    }


def _fmt_date(value) -> str | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return str(value)


def _resolve_tv_interval(interval: str):
    from tradingview_ta import Interval

    if isinstance(interval, Interval):
        return interval
    key = str(interval).strip().lower()
    mapping = {
        "1d": Interval.INTERVAL_1_DAY,
        "1day": Interval.INTERVAL_1_DAY,
        "1w": Interval.INTERVAL_1_WEEK,
        "1week": Interval.INTERVAL_1_WEEK,
        "1h": Interval.INTERVAL_1_HOUR,
        "1hour": Interval.INTERVAL_1_HOUR,
        "4h": Interval.INTERVAL_4_HOURS,
    }
    return mapping.get(key, Interval.INTERVAL_1_DAY)


def _to_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert(None)
    return ts


def _load_exchange_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    resolved = config.resolve_path(str(path))
    path = Path(resolved)
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
        ticker = str(key).strip().upper()
        exchange = str(value).strip().upper()
        if not ticker:
            continue
        cleaned[ticker] = exchange or "UNKNOWN"
    return cleaned


def _chunked(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [values]
    return [values[i : i + size] for i in range(0, len(values), size)]


def _fetch_tv_analyses_batch(
    symbol_by_ticker: dict[str, str],
    screener: str,
    interval,
    timeout: float | None,
    batch_size: int,
    max_batches: int,
) -> dict[str, object]:
    from tradingview_ta import get_multiple_analysis

    symbols = list(dict.fromkeys(symbol_by_ticker.values()))
    batches = _chunked(symbols, batch_size)
    if max_batches > 0 and len(batches) > max_batches:
        raise ValueError(
            f"TradingView batch limit exceeded: {len(batches)} batches for {len(symbols)} symbols "
            f"(max {max_batches})."
        )

    analysis_by_symbol: dict[str, object] = {}
    for batch in tqdm(batches, desc="Fetching TradingView batches", unit="batch"):
        if not batch:
            continue
        try:
            try:
                result = get_multiple_analysis(
                    screener=screener,
                    interval=interval,
                    symbols=batch,
                    timeout=timeout,
                )
            except TypeError:
                result = get_multiple_analysis(
                    screener=screener,
                    interval=interval,
                    symbols=batch,
                )
        except Exception as exc:  # noqa: BLE001
            sample = ",".join(batch[:3])
            raise RuntimeError(
                f"TradingView batch request failed for {len(batch)} symbols (sample: {sample})."
            ) from exc
        if result:
            analysis_by_symbol.update(result)

    return {ticker: analysis_by_symbol.get(symbol) for ticker, symbol in symbol_by_ticker.items()}


def _fetch_tv_analysis(
    symbol: str,
    screener: str,
    exchange_list: list[str],
    interval,
    timeout: float | None,
):
    from tradingview_ta import TA_Handler

    errors: list[str] = []
    for exchange in exchange_list:
        try:
            if timeout is None:
                handler = TA_Handler(
                    symbol=symbol,
                    screener=screener,
                    exchange=exchange,
                    interval=interval,
                )
            else:
                handler = TA_Handler(
                    symbol=symbol,
                    screener=screener,
                    exchange=exchange,
                    interval=interval,
                    timeout=timeout,
                )
        except TypeError:
            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=interval,
            )
        try:
            return handler.get_analysis()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{exchange}:{type(exc).__name__}({exc})")
            continue
    if errors:
        preview = "; ".join(errors[:3])
        raise RuntimeError(
            f"TradingView analysis failed for {symbol} across exchanges. Sample errors: {preview}"
        )
    return None


def _build_updated_row(
    ticker: str,
    existing_ticker: pd.DataFrame,
    bar_date: pd.Timestamp,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float,
    sector: str,
    rolling_window: int,
    lookback_rows: int,
) -> dict | None:
    ticker_df = existing_ticker.copy()
    if isinstance(ticker_df.index, pd.MultiIndex):
        ticker_df = ticker_df.droplevel("ticker")
    ticker_df = ticker_df.sort_index()

    if bar_date in ticker_df.index:
        return None

    if "Dividends" not in ticker_df.columns:
        ticker_df["Dividends"] = 0.0
    if "Stock Splits" not in ticker_df.columns:
        ticker_df["Stock Splits"] = 0.0

    new_row = pd.DataFrame(
        {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close_price,
            "Volume": volume,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "sector": sector,
        },
        index=[bar_date],
    )
    ticker_df = pd.concat([ticker_df, new_row], axis=0, sort=False)
    ticker_df = ticker_df.tail(lookback_rows).copy()
    ticker_df.sort_index(inplace=True)

    ticker_df = _calculate_indicators(ticker_df)
    for col in REQUIRED_INDICATOR_COLS:
        if col not in ticker_df.columns:
            ticker_df[col] = np.nan

    ticker_df["log_return"] = np.log(ticker_df["Close"] / ticker_df["Close"].shift(1))
    ticker_df["adv_21"] = ticker_df["Volume"].rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).mean()
    ticker_df["vol_21"] = ticker_df["log_return"].rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).std()

    ticker_df["ticker"] = ticker
    ticker_df["date"] = ticker_df.index
    ticker_df = _add_swing_features(ticker_df)

    if bar_date not in ticker_df.index:
        return None
    row = ticker_df.loc[bar_date].copy()
    row["ticker"] = ticker
    row["date"] = bar_date
    return row.to_dict()


def _coalesce_vix_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for base in ("VIX", "VIX_z", "vix_return"):
        col_x = f"{base}_x"
        col_y = f"{base}_y"
        has_suffix = col_x in df.columns or col_y in df.columns
        if not has_suffix:
            continue
        if base not in df.columns:
            df[base] = np.nan
        if col_y in df.columns:
            df[base] = df[base].combine_first(df[col_y])
        if col_x in df.columns:
            df[base] = df[base].combine_first(df[col_x])
        drop_cols = [c for c in (col_x, col_y) if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
    return df


def _drop_vix_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols: list[str] = []
    for base in ("VIX", "VIX_z", "vix_return"):
        for col in (base, f"{base}_x", f"{base}_y"):
            if col in df.columns:
                drop_cols.append(col)
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def _extract_vix_series(existing: pd.DataFrame) -> pd.Series:
    if "VIX" not in existing.columns:
        return pd.Series(dtype=float)
    vix_series = existing.groupby(level="date")["VIX"].first()
    vix_series = vix_series.dropna()
    return vix_series


def _fetch_vix_close(
    vix_ticker: str,
    screener: str,
    exchange_list: list[str],
    interval,
    timeout: float | None,
    fallback_series: pd.Series,
) -> float | None:
    symbol = vix_ticker
    exchange_candidates = exchange_list
    if ":" in vix_ticker:
        exchange, symbol = vix_ticker.split(":", 1)
        exchange_candidates = [exchange]

    try:
        analysis = _fetch_tv_analysis(
            symbol,
            screener=screener,
            exchange_list=exchange_candidates,
            interval=interval,
            timeout=timeout,
        )
    except RuntimeError as exc:
        logger.warning("Falling back to historical VIX series for %s: %s", vix_ticker, exc)
        analysis = None

    if analysis is None:
        return float(fallback_series.iloc[-1]) if not fallback_series.empty else None
    indicators = analysis.indicators or {}
    close_price = indicators.get("close")
    if close_price is None:
        return float(fallback_series.iloc[-1]) if not fallback_series.empty else None
    return float(close_price)


def _compute_rolling_vix_stats(vix_series: pd.Series, rolling_window: int) -> pd.DataFrame:
    if vix_series.empty:
        return pd.DataFrame()
    vix_df = pd.DataFrame({"Close": vix_series})
    return _compute_rolling_vix_z(vix_df, rolling_window)


def _compute_vix_beta(
    ticker: str,
    new_date: pd.Timestamp,
    new_log_return: float | None,
    existing: pd.DataFrame,
    vix_return_series: pd.Series,
    rolling_window: int,
) -> float:
    if new_log_return is None or pd.isna(new_log_return) or vix_return_series.empty:
        return np.nan
    try:
        lr_series = existing.xs(ticker, level="ticker")["log_return"].copy()
    except KeyError:
        return np.nan
    lr_series.index = pd.to_datetime(lr_series.index).tz_localize(None)
    lr_series.loc[new_date] = float(new_log_return)
    aligned = pd.DataFrame(
        {
            "lr": lr_series,
            "vr": vix_return_series.reindex(lr_series.index),
        }
    ).dropna(subset=["lr", "vr"])
    if len(aligned) < rolling_window:
        return np.nan
    tail = aligned.tail(rolling_window)
    var = tail["vr"].var()
    if pd.isna(var) or var == 0:
        return np.nan
    cov = tail["lr"].cov(tail["vr"])
    return float(cov / (var + 1e-9))


def _normalize_additional_zscores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[f"{col}_z"] = df.groupby("date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
    return df


def _prune_history(df: pd.DataFrame, retention_days: int | None) -> pd.DataFrame:
    if retention_days is None or retention_days <= 0:
        return df
    if not isinstance(df.index, pd.MultiIndex):
        return df
    dates = pd.Index(df.index.get_level_values("date").unique()).sort_values()
    if len(dates) <= retention_days:
        return df
    keep_dates = set(dates[-retention_days:])
    mask = df.index.get_level_values("date").isin(keep_dates)
    pruned = df[mask]
    return pruned.sort_index()


def _recompute_cross_sectional_z(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[f"{col}_z"] = df.groupby(level="date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
    return df


def _fetch_sector_yfinance(stock, ticker: str = "") -> str:
    try:
        if hasattr(stock, "get_info"):
            info = stock.get_info()
        else:
            info = getattr(stock, "info", None)
        if isinstance(info, dict):
            sector = info.get("sector") or info.get("industry")
            if isinstance(sector, str) and sector.strip():
                return sector.strip()
    except Exception as exc:  # noqa: BLE001
        if ticker:
            logger.warning("Failed to fetch sector metadata for %s: %s", ticker, exc)
    return "unknown"


def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.rsi(close="Close", append=True)
    df.ta.sma(close="Close", length=20, append=True)
    df.ta.sma(close="Close", length=50, append=True)
    df.ta.sma(close="Close", length=250, append=True)
    df.ta.roc(close="Close", length=10, append=True)
    df.ta.roc(close="Close", length=50, append=True)
    df.ta.macd(close="Close", append=True)
    df.ta.atr(append=True)
    df.ta.adx(append=True)
    df.ta.sma(close="Volume", length=20, append=True, col_names=("SMA20_Volume",))
    return df


def _normalize_features_cross_sectional(df: pd.DataFrame, columns_to_normalize: list[str]) -> pd.DataFrame:
    for col in columns_to_normalize:
        if col in df.columns:
            df[f"{col}_z"] = df.groupby("date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
    return df


def _add_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "log_return" in df.columns:
        df["log_return_z"] = df.groupby("ticker")["log_return"].transform(
            lambda x: (x - x.rolling(window=21, min_periods=10).mean())
            / (x.rolling(window=21, min_periods=10).std() + 1e-9)
        )
    if "Close" in df.columns and "SMA_50" in df.columns:
        df["dist_sma50"] = (df["Close"] - df["SMA_50"]) / (df["SMA_50"] + 1e-9)
    if "Close" in df.columns and "SMA_20" in df.columns:
        df["dist_sma20"] = (df["Close"] - df["SMA_20"]) / (df["SMA_20"] + 1e-9)
    if "Volume" in df.columns and "SMA20_Volume" in df.columns:
        df["rvol_20"] = df["Volume"] / (df["SMA20_Volume"] + 1e-9)

    for col in ["dist_sma50", "dist_sma20", "rvol_20"]:
        if col in df.columns:
            df[f"{col}_z"] = df.groupby("date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )

    if "date" in df.columns:
        for col in ["dist_sma50", "dist_sma20", "rvol_20"]:
            if col in df.columns:
                df[f"{col}_z"] = df.groupby("date")[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-9)
                )
    else:
        for col in ["dist_sma50", "dist_sma20", "rvol_20"]:
            if col in df.columns:
                df[f"{col}_z"] = df.groupby(level="date")[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-9)
                )
    return df


def _compute_rolling_vix_z(vix_df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=vix_df.index)
    out["VIX"] = vix_df["Close"].astype(float)
    out["vix_return"] = out["VIX"].pct_change()
    roll_mean = out["VIX"].rolling(window=rolling_window, min_periods=rolling_window).mean()
    roll_std = out["VIX"].rolling(window=rolling_window, min_periods=rolling_window).std()
    out["VIX_z"] = (out["VIX"] - roll_mean) / (roll_std + 1e-9)
    return out
