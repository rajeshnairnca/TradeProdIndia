# Trading Ensemble System (Rule-Based)

## Overview

This repository contains a deterministic, rule-based trading system. It loads strategy signals, tags them by market regime, and runs ensemble backtests or production trade generation. A production pipeline can refresh market data, generate daily trades, and expose results through a small API.

## Repository Layout

```
.
├── alphas_india/             # Promoted India strategies (each folder: strategy.py + description.txt)
├── data/                     # Local data: parquet, raw DB, universe lists
├── scripts/                  # CLI entry points
│   ├── backtesting/          # Backtester, sweeps, sector experiments
│   ├── data_extraction/      # yfinance extractor
│   └── production/           # Daily run, API server, queue tools
├── src/                      # Core engine (regimes, backtester, portfolio, production)
├── runs/                     # Runtime artifacts (ignored by git)
├── logs/                     # Logs (ignored by git)
├── requirements.txt
├── requirements.production.txt
└── Dockerfile
```

## Strategy Format

Each strategy lives at `alphas_india/<strategy_name>/strategy.py` and must define:

- `DESCRIPTION`: short text summary
- `REGIME_TAGS`: list of regime labels (any of: `bull_low_vol`, `bull_high_vol`, `bear_low_vol`, `bear_high_vol`, `sideways_low_vol`, `sideways_high_vol`)
- `generate_scores(df: pd.DataFrame) -> pd.Series`: per-row scores aligned to the input index

## Setup

- Use Python 3.11.
- Activate your virtual environment before running anything.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For production containers, use `requirements.production.txt` (see `Dockerfile`).

## Backtesting

```bash
# Single strategy
caffeinate -i python3 scripts/backtesting/backtester.py --strategies india_rule_trend_carry_slow

# Ensemble
caffeinate -i python3 scripts/backtesting/backtester.py --strategies india_rule_crash_resilient_slow india_rule_liquidity_momentum_core india_rule_pullback_reentry_slow

# Sweep combinations
caffeinate -i python3 scripts/backtesting/strategy_sweep.py --min-size 2 --max-size 4 --max-combos 200 --jobs 7
```

Sector experiments:

```bash
caffeinate -i python3 scripts/backtesting/sector_experiments.py --sectors Technology --regime-scope sector --plot
```

Outputs are written under `runs/_ensembles/india/...` (backtester, sweeps) or `runs/` (sector experiments).

## Production Pipeline

Daily run (market-data refresh + trade generation):

```bash
caffeinate -i python3 scripts/production/daily_run.py
```

Useful options:
- `--market-data-source kite_ohlc` to force Kite `/quote/ohlc` ingestion (default is `auto`)
- `--skip-update` to avoid market data calls (local dry runs)
- Backtest-alignment knobs are available in production too:
  `--confirm-days`, `--confirm-days-sideways`, `--rebalance-every`,
  `--min-weight-change`, `--min-trade-dollars`, `--max-daily-turnover`, `--weight-smoothing`
- `--print-trades` to emit trades to stdout
- `--sector` and `--regime-scope` for sector-focused production runs

`daily_run.py` is DB-backed and does not write production run artifacts to local `runs/production/`.

Queue adjustments (cash or tickers):

```bash
python3 scripts/production/queue_adjustments.py --add-cash 50000 --add-tickers RELIANCE.NS,TCS.NS
```

Backfill Postgres (requires `DATABASE_URL` / `POSTGRES_URL`):

```bash
python3 scripts/production/backfill_db.py
```

Sync universe exchange mapping JSON into Postgres:

```bash
python3 scripts/production/sync_universe_map.py --dry-run
python3 scripts/production/sync_universe_map.py
```

## Production API

Start the API server:

```bash
uvicorn scripts.production.api_server:app --host 0.0.0.0 --port 8000
```

Key endpoints:
- `GET /health`
- `GET /latest-run`, `GET /latest-summary`, `GET /latest-trades`
- `GET /trades`, `GET /cagr`, `GET /portfolio`, `GET /universe`
- `POST /queue-adjustments`, `GET /pending-adjustments`, `POST /clear-pending`, `DELETE /pending-adjustments`

Set `API_KEY` to protect non-health routes (supports `X-API-Key` or `Authorization: Bearer`).
The API is DB-only and requires `DATABASE_URL` / `POSTGRES_URL`.

## Data

- `data/daily_data_*.parquet` is the primary price/feature store.
- `data/market_data.db` is the raw SQLite store for yfinance ingestion.
- `data/universe_india.txt` controls the default India ticker universe for extraction.

Yfinance extraction:

```bash
python3 "scripts/data_extraction/data_extract_yfinance - days - v6.py" --output-file data/daily_data_india.parquet
```

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The suite includes deterministic unit tests and a fast CLI smoke backtest.

## Configuration (Environment)

Common env flags (see `src/config.py`):

- `DATA_FILE` (override parquet path)
- `DATA_ROOT` (prefix for relative paths)
- `UNIVERSE_FILTER` (`all`, `none`, or comma-separated tickers)
- `REGIME_MODE` (`heuristic`, `hmm`, `hmm_rolling`)
- `HMM_N_COMPONENTS`, `HMM_WARMUP_PERIOD`, `HMM_STEP_SIZE`, `HMM_STATE_LABELS`
- `BEAR_CASH_OUT`, `BEAR_GROSS_TARGET`, `REGIME_GROSS_TARGETS`
- `RETENTION_TRADING_DAYS` (production data retention)

Production storage + API:

- `DATABASE_URL` / `POSTGRES_URL` is required for production runs, queueing adjustments, and API endpoints
- `API_KEY` for API auth

Broker integration:

- `USE_TRADING212=true` enables Trading212 (requires `TRADING212_API_KEY`, `TRADING212_API_SECRET`)
- `USE_KITE=true` enables Zerodha Kite (default is `true`; requires `KITE_API_KEY` and either `KITE_ACCESS_TOKEN` or `KITE_REQUEST_TOKEN` + `KITE_API_SECRET`)
- Set only one broker integration at a time (`USE_TRADING212` or `USE_KITE`)
- Optional Kite routing/mapping knobs: `KITE_DEFAULT_EXCHANGE`, `KITE_PRODUCT`, `KITE_ORDER_VARIETY`, `KITE_TICKER_MAP_FILE`
- Market-data source knobs: `MARKET_DATA_SOURCE` (`auto|tradingview|kite_ohlc`), `KITE_QUOTE_BATCH_SIZE`, `KITE_QUOTE_MAX_BATCHES`
- VIX defaults can be overridden via `DEFAULT_VIX_TICKER` and `DEFAULT_HISTORY_VIX_TICKER`

## Notes

- `.env` holds secrets/keys and is ignored by git.
- Large artifacts belong in `runs/` or `logs/` (ignored by git).
- There are no formal tests; validate changes via short backtests before long runs.
