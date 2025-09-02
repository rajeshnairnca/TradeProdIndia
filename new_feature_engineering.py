
import pandas as pd
import numpy as np
import json
import config

DESCRIPTION = "A volatility breakout strategy that buys into stocks showing price acceleration after a period of low volatility (Bollinger Band squeeze)."

def get_sector_mappings():
    """Loads the sector map and creates mappings between symbols, sectors, and integer IDs."""
    with open(config.SECTOR_MAP_FILE, 'r') as f:
        symbol_to_sector = json.load(f)
    
    unique_sectors = sorted(list(set(symbol_to_sector.values())))
    sector_to_id = {sector: i for i, sector in enumerate(unique_sectors)}
    symbol_to_sector_id = {symbol: sector_to_id[sector] for symbol, sector in symbol_to_sector.items()}
    
    return symbol_to_sector_id, sector_to_id

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the volatility breakout strategy.
    1. Bollinger Bands: To identify volatility contraction.
    2. Rate of Change (ROC): To measure price acceleration.
    3. Interaction Feature: Combines squeeze and acceleration.
    4. Sector ID: To incorporate sector information.
    """
    # Make a copy to avoid modifying the original DataFrame in place
    data = df.copy()

    # --- Sector Feature --- 
    symbol_to_sector_id, _ = get_sector_mappings()
    # Get the ticker from the MultiIndex
    tickers = data.index.get_level_values('ticker')
    data['sector_id'] = tickers.map(symbol_to_sector_id).fillna(-1).astype(int)

    # Group by ticker to apply calculations per stock
    grouped = data.groupby(level='ticker')

    # 1. Bollinger Bands (20-period)
    sma_20 = grouped['Close'].transform(lambda x: x.rolling(window=20).mean())
    std_20 = grouped['Close'].transform(lambda x: x.rolling(window=20).std())
    data['bb_upper'] = sma_20 + (std_20 * 2)
    data['bb_lower'] = sma_20 - (std_20 * 2)
    
    # Bollinger Bandwidth to measure the "squeeze"
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / (sma_20 + 1e-9)

    # 2. Rate of Change (Acceleration)
    data['roc_10'] = grouped['Close'].transform(lambda x: x.pct_change(periods=10))
    
    # 3. Interaction Feature: Acceleration during a squeeze
    # We want high ROC when BB width is low. A low BB width is a good thing.
    # So we can use (1 / bb_width) as a multiplier.
    data['breakout_strength'] = data['roc_10'] / (data['bb_width'] + 1e-9)

    # --- Z-Score Normalization ---
    # Select the new features to be normalized
    features_to_normalize = ['bb_width', 'roc_10', 'breakout_strength']
    
    for feature in features_to_normalize:
        # Group by date to normalize cross-sectionally
        data[f'{feature}_z'] = data.groupby(level='date')[feature].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    # Also include the existing volume feature as it's useful for confirmation
    if 'SMA20_Volume_z' not in data.columns:
        data['SMA20_Volume'] = grouped['Volume'].transform(lambda x: x.rolling(window=20).mean())
        data['SMA20_Volume_z'] = data.groupby(level='date')['SMA20_Volume'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    return data
