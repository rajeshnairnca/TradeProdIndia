# data_extract_yfinance - v6_optimized.py - to be used in v57 or later
# Produces a single parquet with indicators, rolling stats, VIX features, and cross-sectional z-scores.

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from tqdm import tqdm

# --- Defaults ---
DEFAULT_DATA_FILENAME = os.path.join("data", "daily_data.parquet")
DEFAULT_SECTOR_MAP_FILE = os.path.join("data", "symbol_to_sector_map.json")
DEFAULT_STOCK_LIST = [
    "ZS", "WDAY", "TXN", "TRI", "TEAM", "SNPS", "SHOP", "ROP", "QCOM", "PLTR", "PDD", "PANW", "ON",
    "NXPI", "NVDA", "MU", "MSTR", "MSFT", "MRVL", "META", "MCHP", "LRCX", "KLAC", "INTU", "INTC",
    "GOOGL", "GOOG", "GFS", "FTNT", "DDOG", "CRWD", "CDW", "CDNS", "AVGO", "ASML", "ARM", "APP",
    "AMD", "AMAT", "ADSK", "ADI", "ADBE", "AAPL", "DASH", "MRNA", "ENPH", "REGN", "LCID", "ALGN",
    "LULU", "BIIB", "KHC", "FANG", "KDP", "FISV", "ILMN", "CHTR", "ODFL", "MDLZ", "PEP", "VRTX",
    "CMCSA", "SIRI", "AMGN", "SBUX", "CTSH", "CPRT", "ISRG", "PYPL", "VRSK", "ROST", "SMCI", "CTAS",
    "PAYX", "AEP", "PCAR", "RIVN", "HON", "ABNB", "ADP", "COST", "EXC", "CSGP", "XEL", "EA", "DXCM",
    "JD", "MELI", "MNST", "NFLX", "TMUS", "BKR", "IDXX", "ORLY", "FAST", "GILD", "BKNG", "DLTR", "BABA",
    "DISCA", "CEG",
]

DEFAULT_PERIOD = "20y"
DEFAULT_INTERVAL = "1d"
DEFAULT_MIN_TRADING_DAYS = 50
DEFAULT_ROLLING_WINDOW = 21
DEFAULT_VIX_TICKER = "^VIX"


def parse_args():
    parser = argparse.ArgumentParser(description="Download market data, engineer features, and save a consolidated parquet.")
    parser.add_argument("--stocks-file", help="Path to a newline-delimited list of tickers. If omitted, the built-in list is used.")
    parser.add_argument("--output-file", default=DEFAULT_DATA_FILENAME, help="Path to parquet output (default: data/daily_data.parquet)")
    parser.add_argument("--sector-map-file", default=DEFAULT_SECTOR_MAP_FILE, help="Path to sector map JSON (default: data/symbol_to_sector_map.json)")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance history period (default: 20y)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="yfinance interval (default: 1d)")
    parser.add_argument("--min-trading-days", type=int, default=DEFAULT_MIN_TRADING_DAYS, help="Minimum days per ticker to keep (default: 50)")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, help="Rolling window for ADV/vol/VIX z (default: 21)")
    parser.add_argument("--vix-ticker", default=DEFAULT_VIX_TICKER, help="VIX symbol (default: ^VIX)")
    return parser.parse_args()


def load_stock_list(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_STOCK_LIST
    with open(path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        raise ValueError("Stock list file is empty.")
    return tickers


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all necessary technical indicators for a given DataFrame."""
    df.ta.rsi(close='Close', append=True)
    df.ta.sma(close='Close', length=20, append=True)
    df.ta.sma(close='Close', length=50, append=True)
    df.ta.sma(close='Close', length=250, append=True)
    df.ta.roc(close='Close', length=10, append=True)
    df.ta.roc(close='Close', length=50, append=True)
    df.ta.macd(close='Close', append=True)
    df.ta.atr(append=True)
    df.ta.adx(append=True)
    df.ta.sma(close='Volume', length=20, append=True, col_names=('SMA20_Volume',))
    return df


def normalize_features_cross_sectional(df: pd.DataFrame, columns_to_normalize: list[str]) -> pd.DataFrame:
    """Calculates same-day cross-sectional z-scores to avoid lookahead."""
    for col in columns_to_normalize:
        if col in df.columns:
            df[f"{col}_z"] = df.groupby("date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
    return df


def compute_rolling_vix_z(vix_df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Compute VIX and its rolling z-score using only past data."""
    if vix_df.empty:
        return pd.DataFrame()
    vix_df = vix_df.copy()
    vix_df["VIX"] = vix_df["Close"]
    rolling_mean = vix_df["VIX"].rolling(window=rolling_window, min_periods=rolling_window).mean()
    rolling_std = vix_df["VIX"].rolling(window=rolling_window, min_periods=rolling_window).std()
    vix_df["VIX_z"] = (vix_df["VIX"] - rolling_mean) / (rolling_std + 1e-9)
    vix_df["vix_return"] = vix_df["VIX"].pct_change()
    return vix_df[["VIX", "VIX_z", "vix_return"]]


def validate_output(df: pd.DataFrame, expected_cols: list[str]):
    assert isinstance(df.index, pd.MultiIndex), "Output is not MultiIndex."
    assert df.index.names == ["date", "ticker"], f"Unexpected index names: {df.index.names}"
    duplicates = df.index.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate index entries."
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
        assert not df[col].isna().any(), f"NaNs found in {col}"


def main():
    args = parse_args()
    stock_list = load_stock_list(args.stocks_file)

    print("Starting optimized data preparation for RL model...")

    # --- 1. Fetch and Save Sector Information ---
    print(f"\nFetching sector data for {len(stock_list)} stocks...")
    symbol_to_sector_map = {}
    for ticker_str in tqdm(stock_list, desc="Fetching Sectors"):
        try:
            ticker_obj = yf.Ticker(ticker_str)
            info = ticker_obj.info
            sector = info.get('sector', 'Unknown')
            symbol_to_sector_map[ticker_str] = sector
            time.sleep(0.05)
        except Exception as e:
            print(f"\nCould not fetch info for {ticker_str}. Setting to 'Unknown'. Error: {e}")
            symbol_to_sector_map[ticker_str] = "Unknown"

    os.makedirs(os.path.dirname(args.sector_map_file), exist_ok=True)
    with open(args.sector_map_file, 'w') as f:
        json.dump(symbol_to_sector_map, f, indent=4)
    print(f"Sector map saved to '{args.sector_map_file}'")

    # --- 2. Data Download ---
    all_stock_data = {}
    print(f"\nDownloading data from yfinance for {len(stock_list)} stocks ({args.period} period)...")
    for ticker in tqdm(stock_list, desc="Downloading Stocks"):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=args.period, interval=args.interval, auto_adjust=True)
            if df.empty:
                continue
            all_stock_data[ticker] = df
        except Exception as e:
            print(f"\nError downloading {ticker}: {e}")

    print(f"\nDownloading VIX data for {args.vix_ticker}...")
    vix_df = yf.Ticker(args.vix_ticker).history(period=args.period, interval=args.interval, auto_adjust=True)
    if vix_df.empty:
        print(f"WARNING: No data for VIX {args.vix_ticker}. VIX features will be missing.")

    # --- 3. Indicator Calculation ---
    print("\nCalculating technical indicators...")
    processed_stock_data = {}
    for ticker, df in tqdm(all_stock_data.items(), desc="Calculating Indicators"):
        enriched_df = calculate_indicators(df.copy())
        processed_stock_data[ticker] = enriched_df

    vix_indicators = pd.DataFrame()
    if not vix_df.empty:
        vix_indicators = calculate_indicators(vix_df.copy())

    # --- 4. Stock History Analysis & Filtering ---
    print("\nFiltering stocks by history length and NaNs in required features...")
    required_cols = [
        'RSI_14', 'SMA_20', 'SMA_50', 'SMA_250', 'ROC_10', 'ROC_50',
        'MACD_12_26_9', 'MACDs_12_26_9', 'ATRr_14', 'ADX_14', 'SMA20_Volume'
    ]
    full_history_stocks = {}
    for ticker, df in processed_stock_data.items():
        cleaned = df.dropna(subset=required_cols)
        if len(cleaned) >= args.min_trading_days:
            full_history_stocks[ticker] = cleaned
    excluded_count = len(processed_stock_data) - len(full_history_stocks)
    print(f"Kept {len(full_history_stocks)} stocks with >= {args.min_trading_days} days.")
    print(f"Excluded {excluded_count} stocks with insufficient history.")
    if not full_history_stocks:
        print("\nFATAL: No stocks had sufficient data. Exiting."); sys.exit(1)

    # --- 5. Consolidate, Pre-calculate, and Merge ---
    print("\nConsolidating all data into a single DataFrame...")
    all_data_list = []
    for ticker, stock_df in tqdm(full_history_stocks.items(), desc="Consolidating"):
        temp_df = stock_df.copy()
        temp_df['ticker'] = ticker
        all_data_list.append(temp_df)

    master_df = pd.concat(all_data_list)
    master_df.reset_index(inplace=True)
    master_df.rename(columns={'Date': 'date'}, inplace=True)
    master_df['date'] = pd.to_datetime(master_df['date'])

    print(f"\nPre-calculating {args.rolling_window}-day rolling features (ADV and Volatility)...")
    master_df.sort_values(by=['ticker', 'date'], inplace=True)
    master_df['log_return'] = master_df.groupby('ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))
    master_df['adv_21'] = master_df.groupby('ticker')['Volume'].transform(
        lambda x: x.rolling(window=args.rolling_window, min_periods=args.rolling_window).mean()
    )
    master_df['vol_21'] = master_df.groupby('ticker')['log_return'].transform(
        lambda x: x.rolling(window=args.rolling_window, min_periods=args.rolling_window).std()
    )
    print("Rolling features calculated.")

    # Make both join keys timezone-naive to ensure a successful merge
    master_df['date'] = master_df['date'].dt.tz_localize(None)

    # Compute VIX z-score using rolling stats (no lookahead)
    vix_to_merge = compute_rolling_vix_z(vix_indicators, args.rolling_window) if not vix_indicators.empty else pd.DataFrame()
    if not vix_to_merge.empty:
        vix_to_merge.index = vix_to_merge.index.tz_localize(None)
        master_df = master_df.merge(
            vix_to_merge,
            how='left',
            left_on='date',
            right_index=True
        )

    # Compute rolling beta of each stock to VIX returns, then cross-sectional z per date
    if "vix_return" in master_df.columns:
        def _beta(group):
            cov = group['log_return'].rolling(window=args.rolling_window, min_periods=args.rolling_window).cov(group['vix_return'])
            var = group['vix_return'].rolling(window=args.rolling_window, min_periods=args.rolling_window).var()
            return cov / (var + 1e-9)
        master_df['vix_beta_sensitivity'] = master_df.groupby('ticker', group_keys=False).apply(_beta)
        master_df['vix_beta_sensitivity_z'] = master_df.groupby('date')['vix_beta_sensitivity'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    # Cross-sectional z-score per date (no lookahead)
    master_df = normalize_features_cross_sectional(master_df, required_cols)

    # Set final index
    master_df.set_index(['date', 'ticker'], inplace=True)
    master_df.sort_index(inplace=True)

    # --- 6. Save to a Single High-Performance File ---
    print(f"\nSaving consolidated data to '{args.output_file}'...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.path.exists('daily_data'):
        shutil.rmtree('daily_data')
        print("Removed old 'daily_data' directory.")
    master_df.to_parquet(args.output_file, engine='pyarrow')

    # Drop rows without full rolling context before validation
    z_cols = [f"{c}_z" for c in required_cols if c in master_df.columns]
    expected_cols = ['log_return', 'adv_21', 'vol_21'] + z_cols
    if "vix_beta_sensitivity_z" in master_df.columns:
        expected_cols.append("vix_beta_sensitivity_z")
    master_df.dropna(subset=expected_cols, inplace=True)
    validate_output(master_df, expected_cols)

    print(f"\nFinished! Consolidated data with pre-calculated features saved to '{args.output_file}'.")
    print("Data is now ready for the optimized V57 reinforcement learning script.")


if __name__ == "__main__":
    main()
