import numpy as np
import pandas as pd

def calculate_cagr(initial_value, final_value, num_years):
    if initial_value <= 0 or final_value <= 0 or num_years <= 0: return 0.0
    return ((final_value / initial_value) ** (1 / num_years)) - 1

def calculate_performance_metrics(net_worth_series: pd.Series, num_days: int):
    """
    Calculates key performance metrics from a series of net worth values.
    """
    if net_worth_series.empty or num_days == 0:
        return {"CAGR": 0, "Sharpe Ratio": 0, "Max Drawdown": 0}

    # CAGR
    initial_value = net_worth_series.iloc[0]
    final_value = net_worth_series.iloc[-1]
    num_years = num_days / 252.0  # Assuming 252 trading days in a year
    cagr = ((final_value / initial_value) ** (1 / num_years)) - 1 if num_years > 0 else 0

    # Sharpe Ratio
    returns = net_worth_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Max Drawdown
    cumulative_max = net_worth_series.cummax()
    drawdown = (net_worth_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    return {
        "CAGR": cagr * 100,  # Return as percentage
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown * 100  # Return as percentage
    }