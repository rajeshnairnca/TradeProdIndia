"""Centralized configuration for the deterministic, rule-based trading system."""

from __future__ import annotations

import json
import os


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_optional_float(name: str) -> float | None:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _env_json_dict(name: str) -> dict | None:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        data = json.loads(val)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip()

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default

# ---- Portfolio Configuration ----
INITIAL_CAPITAL = _env_float("INITIAL_CAPITAL", 1_000_000.0)
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.15
TOP_K = _env_int("TOP_K", 4)
ADV_LOOKBACK = 21
ROLLING_WINDOW_FOR_VOL = _env_int("ROLLING_WINDOW_FOR_VOL", 14)
USE_VOL_PARITY = _env_bool("USE_VOL_PARITY", True)

USE_REGIME_SYSTEM = _env_bool("USE_REGIME_SYSTEM", False)
BACKTEST_USE_FULL_HISTORY = _env_bool("BACKTEST_USE_FULL_HISTORY", True)
REBALANCE_EVERY = max(1, _env_int("REBALANCE_EVERY", _env_int("REBALANCE_EVERY_N_DAYS", 28)))
# Backward-compatible alias used by older scripts/tests.
REBALANCE_EVERY_N_DAYS = REBALANCE_EVERY
BEAR_CASH_OUT = _env_bool("BEAR_CASH_OUT", False)
BEAR_GROSS_TARGET = _env_optional_float("BEAR_GROSS_TARGET")
REGIME_GROSS_TARGETS = _env_json_dict("REGIME_GROSS_TARGETS")

# ---- Market/Cost Configuration ----
TRADING_REGION = _env_str("TRADING_REGION", "india").lower()
US_COMMISSION_RATE = _env_float("US_COMMISSION_RATE", 0.0015)
US_FINRA_FEE_PER_SHARE = _env_float("US_FINRA_FEE_PER_SHARE", 0.000195)
US_SEC_FEE_RATE = _env_float("US_SEC_FEE_RATE", 0.0000278)
UNIVERSE_FILTER = os.getenv("UNIVERSE_FILTER", "all").strip().lower()
EXCLUDED_TICKERS_FILE = _env_str("EXCLUDED_TICKERS_FILE", "data/universe_excluded.txt")
ENABLE_UNIVERSE_QUALITY_FILTER = _env_bool("ENABLE_UNIVERSE_QUALITY_FILTER", False)
UNIVERSE_MIN_HISTORY_ROWS = _env_int("UNIVERSE_MIN_HISTORY_ROWS", 300)
UNIVERSE_MIN_MEDIAN_ADV_DOLLARS = _env_float("UNIVERSE_MIN_MEDIAN_ADV_DOLLARS", 1_000_000.0)
UNIVERSE_MIN_P20_ADV_DOLLARS = _env_float("UNIVERSE_MIN_P20_ADV_DOLLARS", 100_000.0)
UNIVERSE_MIN_P05_PRICE = _env_float("UNIVERSE_MIN_P05_PRICE", 0.10)
UNIVERSE_MAX_MEDIAN_VOL21 = _env_float("UNIVERSE_MAX_MEDIAN_VOL21", 0.20)
UNIVERSE_MAX_P95_ABS_LOG_RETURN = _env_float("UNIVERSE_MAX_P95_ABS_LOG_RETURN", 0.60)
UNIVERSE_QUALITY_START_DATE = _env_str("UNIVERSE_QUALITY_START_DATE", "")
UNIVERSE_QUALITY_END_DATE = _env_str("UNIVERSE_QUALITY_END_DATE", "")

# ---- Data Configuration ----
DATA_ROOT = _env_str("DATA_ROOT", "").strip()
REGION_DATA_FILES = {
    "us": "data/daily_data_us.parquet",
    "india": "data/daily_data_india.parquet",
}
DATA_FILE = _env_str("DATA_FILE", REGION_DATA_FILES.get(TRADING_REGION, "data/daily_data.parquet"))
MARKET_DATA_SOURCE = _env_str("MARKET_DATA_SOURCE", "auto").lower()
TRADINGVIEW_EXCHANGE_MAP_FILE = _env_str(
    "TRADINGVIEW_EXCHANGE_MAP_FILE",
    "data/universe_us_exchange_map.json",
)
DEFAULT_VIX_TICKER = _env_str(
    "DEFAULT_VIX_TICKER",
    "NSE:INDIAVIX" if TRADING_REGION == "india" else "CBOE:VIX",
)
DEFAULT_HISTORY_VIX_TICKER = _env_str(
    "DEFAULT_HISTORY_VIX_TICKER",
    "^INDIAVIX" if TRADING_REGION == "india" else "^VIX",
)
DEFAULT_STRATEGY_ROOTS = tuple(
    part.strip()
    for part in _env_str("DEFAULT_STRATEGY_ROOTS", "alphas_india").split(",")
    if part.strip()
) or ("alphas_india",)
RUN_CALENDAR_TIMEZONE = _env_str("RUN_CALENDAR_TIMEZONE", "America/New_York")
# Railway cron already skips weekends in this setup; keep weekend blocking opt-in.
RUN_CALENDAR_SKIP_WEEKENDS = _env_bool("RUN_CALENDAR_SKIP_WEEKENDS", False)
RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS = _env_bool("RUN_CALENDAR_SKIP_US_FEDERAL_HOLIDAYS", True)


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) or not DATA_ROOT:
        return path
    return os.path.join(DATA_ROOT, path)

# ---- Penalties & Costs ----
# From empirical analysis of Indian market brokerage charges
TURNOVER_PENALTY = 0.000718
LEVERAGE_PENALTY = 1e-4
RISK_PENALTY_COEFF = 0.5021
SLIPPAGE_COEFF = _env_float("SLIPPAGE_COEFF", 0.008)
CASH_DRAG_COEFF = 8e-5  # per-step penalty scaled by cash_weight to discourage idle cash
WEIGHT_CHANGE_PENALTY = 0.0005  # penalize large day-over-day weight changes (swing-friendly)
WEIGHT_SMOOTHING = _env_float("WEIGHT_SMOOTHING", 0.0)  # blend factor for previous weights vs new weights (higher = smoother)
CASH_RESERVE = _env_float("CASH_RESERVE", 0.02)  # keep this fraction in cash to absorb costs
MIN_WEIGHT_CHANGE_TO_TRADE = max(0.0, _env_float("MIN_WEIGHT_CHANGE_TO_TRADE", 0.01))
MIN_TRADE_DOLLARS = max(0.0, _env_float("MIN_TRADE_DOLLARS", 20_000.0))
MAX_DAILY_TURNOVER = _env_optional_float("MAX_DAILY_TURNOVER")
if MAX_DAILY_TURNOVER is None:
    MAX_DAILY_TURNOVER = 0.35
elif MAX_DAILY_TURNOVER <= 0:
    MAX_DAILY_TURNOVER = None
BACKTEST_ENFORCE_CASH_BALANCE = _env_bool("BACKTEST_ENFORCE_CASH_BALANCE", False)

# ---- Risk Overlays (Optional) ----
ADAPTIVE_TURNOVER_ENABLED = _env_bool("ADAPTIVE_TURNOVER_ENABLED", False)
ADAPTIVE_TURNOVER_RISK_CAP = max(0.0, _env_float("ADAPTIVE_TURNOVER_RISK_CAP", 0.10))
ADAPTIVE_TURNOVER_RISK_REGIMES = tuple(
    part.strip()
    for part in _env_str(
        "ADAPTIVE_TURNOVER_RISK_REGIMES",
        "bear_high_vol,bear_low_vol,bull_high_vol,sideways_high_vol,sideways_low_vol",
    ).split(",")
    if part.strip()
)

# ---- Regime Switch Confirmation Defaults ----
CONFIRM_DAYS = max(1, _env_int("CONFIRM_DAYS", 20))
CONFIRM_DAYS_SIDEWAYS = max(1, _env_int("CONFIRM_DAYS_SIDEWAYS", 30))
MIN_ADV_SHARES = 250_000.0  # minimum ADV shares required to trade
MIN_ADV_DOLLARS_FILTER = 20_000_000.0  # minimum ADV$ required to trade
MIN_ADV_DOLLARS_SLIPPAGE = 1_000_000.0  # minimum ADV$ used for slippage scaling

# ---- Data Retention ----
# Keep only the most recent N trading days in production data.
RETENTION_TRADING_DAYS = _env_int("RETENTION_TRADING_DAYS", 500)

# ---- Walk-Forward Validation Configuration ----
# If False, the system will use the simple TRAIN_RATIO split.
# If True, the orchestrator will use walk-forward validation to evaluate candidates.
ENABLE_WALK_FORWARD_VALIDATION = _env_bool("ENABLE_WALK_FORWARD_VALIDATION", True)
N_SPLITS = 1  # Number of chronological splits
TRAIN_YEARS = 3  # Number of years for each training set
VALIDATION_YEARS = 1  # Number of years for each validation set
SEED = 42

# ---- Regime Detection Configuration ----
REGIME_MODE = _env_str("REGIME_MODE", "heuristic").lower()
REGIME_DISPERSION_COL = _env_str("REGIME_DISPERSION_COL", "ROC_50")
REGIME_TREND_BAND = _env_float("REGIME_TREND_BAND", 0.02)

# ---- Broker Configuration ----
USE_TRADING212 = _env_bool("USE_TRADING212", False)
TRADING212_BASE_URL = _env_str("TRADING212_BASE_URL", "https://demo.trading212.com/api/v0")
TRADING212_TIMEOUT = _env_float("TRADING212_TIMEOUT", 20.0)
TRADING212_HTTP_MAX_RETRIES = _env_int("TRADING212_HTTP_MAX_RETRIES", 4)
TRADING212_HTTP_RETRY_BASE_SEC = _env_float("TRADING212_HTTP_RETRY_BASE_SEC", 1.0)
TRADING212_HTTP_RETRY_MAX_SEC = _env_float("TRADING212_HTTP_RETRY_MAX_SEC", 20.0)
TRADING212_ORDER_TIMEOUT = _env_float("TRADING212_ORDER_TIMEOUT", 300.0)
TRADING212_ORDER_POLL_SEC = _env_float("TRADING212_ORDER_POLL_SEC", 5.0)
TRADING212_POSITIONS_POLL_SEC = _env_float("TRADING212_POSITIONS_POLL_SEC", 1.0)
TRADING212_HISTORY_POLL_SEC = _env_float("TRADING212_HISTORY_POLL_SEC", 10.0)
TRADING212_HISTORY_PAGE_LIMIT = _env_int("TRADING212_HISTORY_PAGE_LIMIT", 50)
TRADING212_HISTORY_MAX_PAGES = _env_int("TRADING212_HISTORY_MAX_PAGES", 1)
TRADING212_EXTENDED_HOURS = _env_bool("TRADING212_EXTENDED_HOURS", False)
TRADING212_USE_BROKER_STATE_FOR_SIGNALS = _env_bool("TRADING212_USE_BROKER_STATE_FOR_SIGNALS", True)
TRADING212_PREFERRED_CURRENCY = _env_str("TRADING212_PREFERRED_CURRENCY", "USD").upper()
TRADING212_INSTRUMENTS_CACHE = _env_str("TRADING212_INSTRUMENTS_CACHE", "data/trading212_instruments.json")
TRADING212_TICKER_MAP_FILE = _env_str("TRADING212_TICKER_MAP_FILE", "data/trading212_ticker_map.json")
TRADING212_FX_RATE_USD_GBP = _env_optional_float("TRADING212_FX_RATE_USD_GBP")

USE_KITE = _env_bool("USE_KITE", True)
KITE_BASE_URL = _env_str("KITE_BASE_URL", "https://api.kite.trade")
KITE_TIMEOUT = _env_float("KITE_TIMEOUT", 20.0)
KITE_HTTP_MAX_RETRIES = _env_int("KITE_HTTP_MAX_RETRIES", 4)
KITE_HTTP_RETRY_BASE_SEC = _env_float("KITE_HTTP_RETRY_BASE_SEC", 1.0)
KITE_HTTP_RETRY_MAX_SEC = _env_float("KITE_HTTP_RETRY_MAX_SEC", 20.0)
KITE_ORDER_TIMEOUT = _env_float("KITE_ORDER_TIMEOUT", 300.0)
KITE_ORDER_POLL_SEC = _env_float("KITE_ORDER_POLL_SEC", 3.0)
KITE_PRODUCT = _env_str("KITE_PRODUCT", "CNC").upper()
KITE_ORDER_VARIETY = _env_str("KITE_ORDER_VARIETY", "regular").lower()
KITE_DEFAULT_EXCHANGE = _env_str("KITE_DEFAULT_EXCHANGE", "NSE").upper()
KITE_USE_BROKER_STATE_FOR_SIGNALS = _env_bool("KITE_USE_BROKER_STATE_FOR_SIGNALS", True)
KITE_INSTRUMENTS_CACHE = _env_str("KITE_INSTRUMENTS_CACHE", "data/kite_instruments.json")
KITE_TICKER_MAP_FILE = _env_str("KITE_TICKER_MAP_FILE", "data/kite_ticker_map.json")
KITE_INSTRUMENTS_EXCHANGE = _env_str("KITE_INSTRUMENTS_EXCHANGE", "NSE").upper()
KITE_ACCESS_TOKEN_FILE = _env_str("KITE_ACCESS_TOKEN_FILE", "data/kite_access_token.txt")
KITE_SESSION_GENERATE_ON_START = _env_bool("KITE_SESSION_GENERATE_ON_START", True)
KITE_QUOTE_BATCH_SIZE = max(1, _env_int("KITE_QUOTE_BATCH_SIZE", 1000))
KITE_QUOTE_MAX_BATCHES = _env_int("KITE_QUOTE_MAX_BATCHES", 0)
HMM_N_COMPONENTS = _env_int("HMM_N_COMPONENTS", 4)
HMM_WARMUP_PERIOD = _env_int("HMM_WARMUP_PERIOD", 2520)  # ~10 years
HMM_STEP_SIZE = _env_int("HMM_STEP_SIZE", 5)  # ~1 month
HMM_STATE_LABELS = _env_bool("HMM_STATE_LABELS", False)
HMM_COVARIANCE_TYPE = _env_str("HMM_COVARIANCE_TYPE", "full").lower()
HMM_MIN_COVAR = _env_float("HMM_MIN_COVAR", 1e-6)
