India-specific strategy set focused on longer-hold, lower-churn scoring.

Use this folder as an isolated strategy root when running India experiments:

`TRADING_REGION=india DATA_FILE=data/daily_data_india.parquet python3 scripts/backtesting/sector_experiments.py --strategy-roots alphas_india --best-single-strategy --regime-scope global`

