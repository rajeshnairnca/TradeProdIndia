# data_extract_yfinance - v6_optimized.py - to be used in v57 or later
# Produces a single parquet with indicators, rolling stats, VIX features, and cross-sectional z-scores.

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from tqdm import tqdm

# --- Defaults ---
DEFAULT_DATA_FILENAME = os.path.join("data", "daily_data.parquet")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# from src.universe import NASDAQ100_TICKERS

# DEFAULT_STOCK_LIST_NASDAQ100 = NASDAQ100_TICKERS

DEFAULT_STOCK_LIST_India = ["ACC.NS", "APLAPOLLO.NS", "AUBANK.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ATGL.NS", "AWL.NS", "ABCAPITAL.NS", "ABFRL.NS", "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "AUROPHARMA.NS", "DMART.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BATAINDIA.NS", "BERGEPAINT.NS", "BDL.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DLF.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", "DEVYANI.NS", "DIVISLAB.NS", "DIXON.NS", "LALPATHLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "NYKAA.NS", "FEDERALBNK.NS", "FACT.NS", "FORTIS.NS", "GAIL.NS", "GLAND.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRASIM.NS", "FLUOROCHEM.NS", "GUJGASLTD.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HAVELLS.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HAL.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDFCFIRSTB.NS", "ITC.NS", "INDIANB.NS", "INDHOTEL.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "IGL.NS", "INDUSTOWER.NS", "INDUSINDBK.NS", "NAUKRI.NS", "INFY.NS", "INDIGO.NS", "IPCALAB.NS", "JSWENERGY.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "KPITTECH.NS", "KOTAKBANK.NS", "LTTS.NS", "LICHSGFIN.NS", "LTIM.NS", "LT.NS", "LAURUSLABS.NS", "LICI.NS", "LUPIN.NS", "MRF.NS", "LODHA.NS", "M&MFIN.NS", "M&M.NS", "MANKIND.NS", "MARICO.NS", "MARUTI.NS", "MFSL.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "MSUMI.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NHPC.NS", "NMDC.NS", "NTPC.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", "OBEROIRLTY.NS", "ONGC.NS", "OIL.NS", "PAYTM.NS", "POLICYBZR.NS", "PIIND.NS", "PAGEIND.NS", "PATANJALI.NS", "PERSISTENT.NS", "PETRONET.NS", "PIDILITIND.NS", "PEL.NS", "POLYCAB.NS", "POONAWALLA.NS", "PFC.NS", "POWERGRID.NS", "PRESTIGE.NS", "PGHH.NS", "PNB.NS", "RECLTD.NS", "RVNL.NS", "RELIANCE.NS", "SBICARD.NS", "SBILIFE.NS", "SRF.NS", "MOTHERSON.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SONACOMS.NS", "SBIN.NS", "SAIL.NS", "SUNPHARMA.NS", "SUNTV.NS", "SYNGENE.NS", "TVSMOTOR.NS", "TATACHEM.NS", "TATACOMM.NS", "TCS.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TECHM.NS", "RAMCOCEM.NS", "TITAN.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TIINDIA.NS", "UPL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UBL.NS", "VBL.NS", "VEDL.NS", "IDEA.NS", "VOLTAS.NS", "WIPRO.NS", "YESBANK.NS", "ZEEL.NS", "ZYDUSLIFE.NS"]


DEFAULT_STOCK_LIST = ["NVDA",
"AAPL",
"MSFT",
"AMZN",
"GOOGL",
"GOOG",
"META",
"AVGO",
"TSLA",
"BRK.B",
"LLY",
"WMT",
"JPM",
"V",
"ORCL",
"MA",
"XOM",
"JNJ",
"PLTR",
"ABBV",
"BAC",
"NFLX",
"COST",
"AMD",
"HD",
"PG",
"GE",
"MU",
"CSCO",
"CVX",
"KO",
"UNH",
"WFC",
"MS",
"IBM",
"CAT",
"GS",
"MRK",
"AXP",
"PM",
"CRM",
"RTX",
"APP",
"TMUS",
"ABT",
"MCD",
"TMO",
"LRCX",
"C",
"AMAT",
"DIS",
"ISRG",
"LIN",
"PEP",
"INTU",
"QCOM",
"INTC",
"GEV",
"SCHW",
"AMGN",
"T",
"BKNG",
"TJX",
"VZ",
"BA",
"UBER",
"NEE",
"BLK",
"APH",
"ANET",
"ACN",
"KLAC",
"DHR",
"TXN",
"NOW",
"SPGI",
"COF",
"GILD",
"ADBE",
"BSX",
"PFE",
"UNP",
"LOW",
"SYK",
"PGR",
"ADI",
"PANW",
"WELL",
"DE",
"HON",
"ETN",
"MDT",
"CB",
"BX",
"CRWD",
"PLD",
"COP",
"VRTX",
"KKR",
"LMT",
"PH",
"CEG",
"BMY",
"NEM",
"CMCSA",
"HCA",
"ADP",
"HOOD",
"MCK",
"CVS",
"CME",
"DASH",
"MO",
"SBUX",
"SO",
"NKE",
"ICE",
"MMC",
"GD",
"DUK",
"MCO",
"SNPS",
"WM",
"TT",
"CDNS",
"MMM",
"DELL",
"UPS",
"APO",
"MAR",
"USB",
"CRH",
"HWM",
"PNC",
"AMT",
"ABNB",
"NOC",
"BK",
"REGN",
"SHW",
"ORLY",
"ELV",
"RCL",
"CTAS",
"GM",
"AON",
"GLW",
"EMR",
"EQIX",
"ECL",
"MNST",
"TDG",
"JCI",
"CI",
"WMB",
"FCX",
"ITW",
"WBD",
"CMI",
"MDLZ",
"TEL",
"FDX",
"HLT",
"CSX",
"AJG",
"RSG",
"COR",
"NSC",
"TRV",
"CL",
"TFC",
"MSI",
"PWR",
"ADSK",
"COIN",
"AEP",
"KMI",
"SPG",
"STX",
"CVNA",
"WDC",
"FTNT",
"ROST",
"SRE",
"AFL",
"PCAR",
"SLB",
"EOG",
"WDAY",
"AZO",
"NDAQ",
"BDX",
"ZTS",
"NXPI",
"APD",
"PYPL",
"LHX",
"VST",
"ALL",
"IDXX",
"DLR",
"F",
"MET",
"O",
"PSX",
"URI",
"EA",
"D",
"VLO",
"EW",
"CAH",
"MPC",
"CMG",
"GWW",
"CBRE",
"ROP",
"DDOG",
"TTWO",
"AME",
"FAST",
"OKE",
"AIG",
"AMP",
"PSA",
"BKR",
"CTVA",
"DAL",
"AXON",
"CARR",
"ROK",
"EXC",
"MPWR",
"TGT",
"XEL",
"MSCI",
"LVS",
"FANG",
"YUM",
"ETR",
"DHI",
"FICO",
"OXY",
"PAYX",
"CTSH",
"CCL",
"PEG",
"TRGP",
"PRU",
"XYZ",
"KR",
"GRMN",
"EBAY",
"A",
"IQV",
"HIG",
"CCI",
"KDP",
"MLM",
"EL",
"CPRT",
"VMC",
"GEHC",
"NUE",
"HSY",
"WAB",
"VTR",
"STT",
"FISV",
"ED",
"ARES",
"UAL",
"SYY",
"PCG",
"RMD",
"KEYS",
"SNDK",
"ACGL",
"EXPE",
"MCHP",
"FIS",
"WEC",
"OTIS",
"EQT",
"KMB",
"XYL",
"LYV",
"FIX",
"KVUE",
"ODFL",
"HPE",
"RJF",
"IR",
"WTW",
"HUM",
"MTB",
"VRSK",
"TER",
"FITB",
"NRG",
"SYF",
"VICI",
"DG",
"ROL",
"KHC",
"IBKR",
"MTD",
"CSGP",
"FSLR",
"EXR",
"ADM",
"EME",
"HBAN",
"BRO",
"AEE",
"ATO",
"CHTR",
"DOV",
"EFX",
"ULTA",
"DTE",
"WRB",
"EXE",
"CBOE",
"TSCO",
"TPR",
"NTRS",
"AVB",
"BR",
"PPL",
"DXCM",
"FE",
"LEN",
"CINF",
"AWK",
"ES",
"BIIB",
"OMC",
"CNP",
"CFG",
"VLTO",
"STE",
"GIS",
"STLD",
"LULU",
"IRM",
"JBL",
"DLTR",
"STZ",
"EQR",
"HAL",
"TDY",
"RF",
"HUBB",
"LDOS",
"EIX",
"PPG",
"DVN",
"PHM",
"WAT",
"VRSN",
"TROW",
"KEY",
"L",
"ON",
"RL",
"WSM",
"NTAP",
"CMS",
"DRI",
"LUV",
"CPAY",
"HPQ",
"LH",
"PTC",
"IP",
"TSN",
"SBAC",
"TPL",
"CHD",
"PODD",
"SW",
"CTRA",
"CNC",
"EXPD",
"NVR",
"NI",
"WST",
"TYL",
"INCY",
"PFG",
"DGX",
"AMCR",
"CHRW",
"PKG",
"TRMB",
"GPN",
"JBHT",
"TTD",
"IT",
"MKC",
"SNA",
"CDW",
"ZBH",
"FTV",
"SMCI",
"Q",
"BG",
"IFF",
"GPC",
"LII",
"PNR",
"WY",
"ESS",
"INVH",
"DD",
"GDDY",
"GEN",
"TKO",
"EVRG",
"ALB",
"DOW",
"LNT",
"HOLX",
"APTV",
"MAA",
"COO",
"J",
"TXT",
"FOX",
"FOXA",
"DECK",
"ERIE",
"FFIV",
"PSKY",
"VTRS",
"EG",
"BALL",
"AVY",
"DPZ",
"BBY",
"UHS",
"LYB",
"ALLE",
"KIM",
"SOLV",
"NDSN",
"HII",
"IEX",
"MAS",
"JKHY",
"HRL",
"REG",
"AKAM",
"WYNN",
"BEN",
"ZBRA",
"CLX",
"HST",
"UDR",
"BF.B",
"AIZ",
"CF",
"MRNA",
"CPT",
"IVZ",
"HAS",
"SWK",
"EPAM",
"BLDR",
"DOC",
"ALGN",
"GL",
"DAY",
"RVTY",
"FDS",
"BXP",
"PNW",
"SJM",
"AES",
"NCLH",
"MGM",
"BAX",
"CRL",
"NWSA",
"SWKS",
"AOS",
"TECH",
"TAP",
"HSIC",
"MOH",
"FRT",
"PAYC",
"APA",
"POOL",
"ARE",
"CPB",
"CAG",
"GNRC",
"DVA",
"MOS",
"MTCH",
"LW",
"NWS",
"ZS",
 "WDAY",
 "TXN",
 "TRI",
 "TEAM",
 "SNPS",
 "SHOP",
 "ROP",
 "QCOM",
 "PLTR",
 "PDD",
 "PANW",
 "ON",
 "NVDA",
 "MU",
 "MSTR",
 "MSFT",
 "MRVL",
 "META",
 "MCHP",
 "LRCX",
 "KLAC",
 "INTU",
 "INTC",
 "GOOG",
 "GFS",
 "FTNT",
 "DDOG",
 "CRWD",
 "CDW",
 "CDNS",
 "AVGO",
 "ASML",
 "ARM",
 "APP",
 "AMAT",
 "ADSK",
 "ADI",
 "ADBE",
 "AAPL",
 "DASH",
 "MRNA",
 "ENPH",
 "REGN",
 "LCID",
 "ALGN",
 "BIIB",
 "KHC",
 "FANG",
 "KDP",
 "FISV",
 "ILMN",
 "CHTR",
 "ODFL",
 "MDLZ",
 "PEP",
 "VRTX",
 "SIRI",
 "AMGN",
 "SBUX",
 "CTSH",
 "CPRT",
 "ISRG",
 "PYPL",
 "VRSK",
 "ROST",
 "SMCI",
 "CTAS",
 "AEP",
 "PCAR",
 "RIVN",
 "HON",
 "ABNB",
 "ADP",
 "COST",
 "EXC",
 "CSGP",
 "XEL",
 "EA",
 "DXCM",
"JD",
 "MELI",
 "MNST",
 "NFLX",
 "TMUS",
 "BKR",
 "IDXX",
 "ORLY",
 "FAST",
 "GILD",
 "BKNG",
 "DLTR",
 "BABA",
"DISCA",
 "CEG"]

DEFAULT_PERIOD = "20y"
DEFAULT_INTERVAL = "1d"
DEFAULT_MIN_TRADING_DAYS = 50
DEFAULT_ROLLING_WINDOW = 21
DEFAULT_VIX_TICKER = "^VIX"
DEFAULT_STOCKS_FILE = "data/universe_us.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Download market data, engineer features, and save a consolidated parquet.")
    parser.add_argument(
        "--stocks-file",
        default=DEFAULT_STOCKS_FILE,
        help=f"Path to a newline-delimited list of tickers (default: {DEFAULT_STOCKS_FILE}).",
    )
    parser.add_argument("--output-file", default=DEFAULT_DATA_FILENAME, help="Path to parquet output (default: data/daily_data.parquet)")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance history period (default: 20y)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="yfinance interval (default: 1d)")
    parser.add_argument("--min-trading-days", type=int, default=DEFAULT_MIN_TRADING_DAYS, help="Minimum days per ticker to keep (default: 50)")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, help="Rolling window for ADV/vol/VIX z (default: 21)")
    parser.add_argument("--vix-ticker", default=DEFAULT_VIX_TICKER, help="VIX symbol (default: ^VIX)")
    return parser.parse_args()


def load_stock_list(path: str | None) -> list[str]:
    if not path:
        path = DEFAULT_STOCKS_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stock list file not found: {path}")
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

def add_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds swing-friendly features: distance-from-MA ratios, relative volume, and time embeddings."""
    df = df.copy()
    # One-period log return z-score per ticker (rolling)
    if 'log_return' in df.columns:
        df['log_return_z'] = df.groupby('ticker')['log_return'].transform(
            lambda x: (x - x.rolling(window=21, min_periods=10).mean()) / (x.rolling(window=21, min_periods=10).std() + 1e-9)
        )
    if 'Close' in df.columns and 'SMA_50' in df.columns:
        df['dist_sma50'] = (df['Close'] - df['SMA_50']) / (df['SMA_50'] + 1e-9)
    if 'Close' in df.columns and 'SMA_20' in df.columns:
        df['dist_sma20'] = (df['Close'] - df['SMA_20']) / (df['SMA_20'] + 1e-9)
    if 'Volume' in df.columns and 'SMA20_Volume' in df.columns:
        df['rvol_20'] = df['Volume'] / (df['SMA20_Volume'] + 1e-9)

    # Cross-sectional z-scores for new ratios
    for col in ['dist_sma50', 'dist_sma20', 'rvol_20']:
        if col in df.columns:
            df[f"{col}_z"] = df.groupby("date")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    # Time embeddings
    if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names:
        dates = pd.to_datetime(df.index.get_level_values('date'))
    elif 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
    else:
        dates = None
    if dates is not None:
        dow = dates.dt.dayofweek
        month = dates.dt.month
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

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


def _fetch_sector(stock) -> str:
    """Best-effort sector lookup; returns 'unknown' if unavailable."""
    try:
        if hasattr(stock, "get_info"):
            info = stock.get_info()
        else:
            info = getattr(stock, "info", None)
        if isinstance(info, dict):
            sector = info.get("sector") or info.get("industry")
            if isinstance(sector, str) and sector.strip():
                return sector.strip()
    except Exception:
        pass
    return "unknown"


def main():
    args = parse_args()
    stock_list = load_stock_list(args.stocks_file)

    print("Starting optimized data preparation for RL model...")

    # --- 1. Data Download ---
    all_stock_data = {}
    sector_by_ticker = {}
    print(f"\nDownloading data from yfinance for {len(stock_list)} stocks ({args.period} period)...")
    for ticker in tqdm(stock_list, desc="Downloading Stocks"):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=args.period, interval=args.interval, auto_adjust=True)
            if df.empty:
                continue
            all_stock_data[ticker] = df
            sector_by_ticker[ticker] = _fetch_sector(stock)
        except Exception as e:
            print(f"\nError downloading {ticker}: {e}")

    print(f"\nDownloading VIX data for {args.vix_ticker}...")
    vix_df = yf.Ticker(args.vix_ticker).history(period=args.period, interval=args.interval, auto_adjust=True)
    if vix_df.empty:
        print(f"WARNING: No data for VIX {args.vix_ticker}. VIX features will be missing.")

    # --- 2. Indicator Calculation ---
    print("\nCalculating technical indicators...")
    processed_stock_data = {}
    for ticker, df in tqdm(all_stock_data.items(), desc="Calculating Indicators"):
        enriched_df = calculate_indicators(df.copy())
        processed_stock_data[ticker] = enriched_df

    vix_indicators = pd.DataFrame()
    if not vix_df.empty:
        vix_indicators = calculate_indicators(vix_df.copy())

    # --- 3. Stock History Analysis & Filtering ---
    print("\nFiltering stocks by history length and NaNs in required features...")
    required_cols = [
        'RSI_14', 'SMA_20', 'SMA_50', 'SMA_250', 'ROC_10', 'ROC_50',
        'MACD_12_26_9', 'MACDs_12_26_9', 'ATRr_14', 'ADX_14', 'SMA20_Volume'
    ]
    full_history_stocks = {}
    for ticker, df in processed_stock_data.items():
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        cleaned = df.dropna(subset=required_cols)
        if len(cleaned) >= args.min_trading_days:
            full_history_stocks[ticker] = cleaned
    excluded_count = len(processed_stock_data) - len(full_history_stocks)
    print(f"Kept {len(full_history_stocks)} stocks with >= {args.min_trading_days} days.")
    print(f"Excluded {excluded_count} stocks with insufficient history.")
    if not full_history_stocks:
        print("\nFATAL: No stocks had sufficient data. Exiting."); sys.exit(1)

    # --- 4. Consolidate, Pre-calculate, and Merge ---
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
    if sector_by_ticker:
        master_df['sector'] = master_df['ticker'].map(lambda x: sector_by_ticker.get(x, "unknown"))

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
    # Add swing-friendly features (distance from MAs, relative volume, time embeddings)
    master_df = add_swing_features(master_df)

    # Set final index
    master_df.set_index(['date', 'ticker'], inplace=True)
    master_df.sort_index(inplace=True)

    # --- 5. Save to a Single High-Performance File ---
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
