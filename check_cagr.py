
import pandas as pd

def calculate_cagr(initial_value, final_value, num_years):
    if initial_value <= 0 or final_value <= 0 or num_years <= 0:
        return 0.0
    return ((final_value / initial_value) ** (1 / num_years)) - 1

# Load the data
try:
    df = pd.read_parquet("data/daily_data.parquet")
    dates = df.index.get_level_values('date').unique().sort_values()

    # The backtest starts after a lookback period.
    # From config.py, ADV_LOOKBACK = 21 and ROLLING_WINDOW_FOR_VOL = 21.
    # The environment starts at max(21, 21) = 21st index.
    # The first date in the backtest is dates[21].
    # The loop for initial_date in train_alpha.py uses current_idx - 1, which is 20.
    start_date = dates[20]
    end_date = dates[-1]

    num_years = (end_date - start_date).days / 365.25

    initial_net_worth = 1_000_000
    final_net_worth = 1_141_885.94

    cagr = calculate_cagr(initial_net_worth, final_net_worth, num_years)

    print(f"Backtest Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"Backtest End Date: {end_date.strftime('%Y-%m-%d')}")
    print(f"Number of years in backtest: {num_years:.4f}")
    print(f"Calculated CAGR: {cagr:.2%}")

except FileNotFoundError:
    print("Error: data/daily_data.parquet not found.")
except Exception as e:
    print(f"An error occurred: {e}")
