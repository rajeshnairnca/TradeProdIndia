import numpy as np


def _india_brokerage(trade_dollars: np.ndarray) -> np.ndarray:
    trade_value = np.abs(trade_dollars)
    buy_mask = trade_dollars > 1.0
    sell_mask = trade_dollars < -1.0
    stt = trade_value * 0.001
    nse_charges = trade_value * 0.0000322
    sebi_charges = trade_value * 1e-6
    stamp_charges = np.where(buy_mask, trade_value * 0.00015, 0.0)
    dp_charges = np.where(sell_mask, 15.34, 0.0)
    gst = (nse_charges + sebi_charges) * 0.18
    return stt + stamp_charges + nse_charges + sebi_charges + gst + dp_charges


def vectorized_brokerage_calculator(
    trade_dollars: np.ndarray, trade_shares: np.ndarray | None = None
) -> np.ndarray:
    _ = trade_shares
    return _india_brokerage(trade_dollars)
