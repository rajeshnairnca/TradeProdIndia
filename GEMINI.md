# Autonomous Trading Ensemble System (Rule-Based)

## Project Overview

This project runs a deterministic, regime-aware ensemble of rule-based trading strategies. A Gemini-powered LLM proposes new strategy scripts that are backtested and promoted if they improve performance.

Key technologies:

* **Python**
* **Pandas & NumPy**
* **Matplotlib**
* **Google Generative AI**

## Workflow

1. **Generate:** The orchestrator asks the LLM for a new rule-based strategy.
2. **Backtest:** The candidate strategy is backtested alongside the existing ensemble.
3. **Promote:** If performance improves, the strategy is moved to `alphas/`.

## Strategy Format

Each strategy lives in `alphas/<strategy_name>/strategy.py` and defines:

* `DESCRIPTION`: one-line summary
* `REGIME_TAGS`: list of target regimes
* `generate_scores(df: pd.DataFrame) -> pd.Series`: deterministic scores, indexed like `df`

## Key Commands

```bash
caffeinate -i python3 scripts/backtesting/backtester.py --strategies rule_trend_following rule_mean_reversion
caffeinate -i python3 src/walkforward.py --alpha-name rule_trend_following
caffeinate -i python3 scripts/orchestration/orchestrator.py
```
