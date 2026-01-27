# Trading Ensemble System (Rule-Based)

## Overview

This repository contains a deterministic, rule-based trading system. It loads strategy signals, tags them by market regime, and runs ensemble backtests or production trade generation. A production pipeline can refresh market data, generate daily trades, and expose results through a small API.

## Repository Layout

```
.
├── alphas/                   # Promoted strategies (each folder: strategy.py + description.txt)
│   └── _ensembles/           # Backtest outputs by region
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

Each strategy lives at `alphas/<strategy_name>/strategy.py` and must define:

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
caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following

# Ensemble
caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following rule_mean_reversion

# Sweep combinations
caffeinate -i python3 scripts/backtesting/strategy_sweep.py --min-size 2 --max-size 4 --max-combos 200 --jobs 7
```

Sector experiments:

```bash
caffeinate -i python3 scripts/backtesting/sector_experiments.py --sectors Technology --regime-scope sector --plot
```

Outputs are written under `alphas/_ensembles/<region>/...` (backtester, sweeps) or `runs/` (sector experiments).

## Production Pipeline

Daily run (TradingView refresh + trade generation):

```bash
caffeinate -i python3 scripts/production/daily_run.py
```

Useful options:
- `--skip-update` to avoid TradingView calls (local dry runs)
- `--print-trades` to emit trades to stdout
- `--sector` and `--regime-scope` for sector-focused production runs

Queue adjustments (cash or tickers):

```bash
python3 scripts/production/queue_adjustments.py --add-cash 50000 --add-tickers NVDA,TSLA
```

Backfill Postgres (if enabled):

```bash
python3 scripts/production/backfill_db.py
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

## Data

- `data/daily_data_*.parquet` is the primary price/feature store.
- `data/market_data.db` is the raw SQLite store for yfinance ingestion.
- `data/universe_us.txt` and `data/universe_us_exchange_map.json` control the production universe and exchange mapping.

Yfinance extraction:

```bash
python3 "scripts/data_extraction/data_extract_yfinance - days - v6.py" --output-file data/daily_data_us.parquet
```

## Configuration (Environment)

Common env flags (see `src/config.py`):

- `TRADING_REGION` (`us` or `india`)
- `DATA_FILE` (override parquet path)
- `DATA_ROOT` (prefix for relative paths)
- `UNIVERSE_FILTER` (`all`, `none`, `nasdaq100`, or comma-separated tickers)
- `REGIME_MODE` (`heuristic`, `hmm`, `hmm_rolling`)
- `HMM_N_COMPONENTS`, `HMM_WARMUP_PERIOD`, `HMM_STEP_SIZE`, `HMM_STATE_LABELS`
- `BEAR_CASH_OUT`, `BEAR_GROSS_TARGET`, `REGIME_GROSS_TARGETS`
- `RETENTION_TRADING_DAYS` (production data retention)

Production storage + API:

- `DATABASE_URL` / `POSTGRES_URL` enables Postgres-backed runs/trades/state
- `API_KEY` for API auth
- `EXCHANGE_MAP_FILE`, `OUTPUT_DIR`, `STATE_FILE`, `PENDING_FILE` for API paths

## Notes

- `.env` holds secrets/keys and is ignored by git.
- Large artifacts belong in `runs/` or `logs/` (ignored by git).
- There are no formal tests; validate changes via short backtests before long runs.
