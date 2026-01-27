#!/usr/bin/env bash
set -euo pipefail

# Dry run checklist:
# 1) Queue pending adjustments (cash + tickers)
# 2) Run production with --skip-update (no TradingView calls)
# 3) Start API server locally

python3 scripts/production/queue_adjustments.py --add-cash 50000 --cash-note "dry-run" --add-tickers NVDA,TSLA

python3 scripts/production/daily_run.py --skip-update --print-trades --max-print 10 --force

uvicorn scripts.production.api_server:app --host 0.0.0.0 --port 8000
