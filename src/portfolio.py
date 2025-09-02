import numpy as np
from . import config

def get_target_weights(scores: np.ndarray, vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Converts raw model scores into a final portfolio allocation.

    Args:
        scores (np.ndarray): The raw output scores from the RL agent for each stock.
        vol (np.ndarray): The historical volatility for each stock.
        mask (np.ndarray): A boolean mask indicating which stocks are tradable.

    Returns:
        np.ndarray: The final target weights for the portfolio.
    """
    # Apply the mask to the scores, setting non-tradable stocks to negative infinity.
    masked_scores = np.where(mask > 0.5, scores, -np.inf)
    
    # Determine the number of stocks to select (top k).
    k = min(config.TOP_K, int(np.sum(mask)))
    
    target_weights = np.zeros_like(scores, dtype=np.float32)
    
    if k > 0:
        # Find the indices of the top k stocks based on their scores.
        topk_indices = np.argpartition(-masked_scores, kth=k-1)[:k]
        topk_scores = masked_scores[topk_indices]
        
        # Convert scores to weights using softmax.
        exp_scores = np.exp(topk_scores - np.max(topk_scores))
        weights_topk = exp_scores / np.sum(exp_scores)
        
        target_weights[topk_indices] = weights_topk

        # Apply volatility parity if enabled.
        if config.USE_VOL_PARITY:
            topk_vol = vol[topk_indices]
            valid_vols = topk_vol[~np.isnan(topk_vol) & (topk_vol > 0)]
            
            # Use the median volatility of the selected stocks as a fallback.
            median_vol = np.median(valid_vols) if len(valid_vols) > 0 else 1.0
            topk_vol[np.isnan(topk_vol) | (topk_vol <= 0)] = median_vol
            
            # Calculate inverse volatility and adjust weights.
            inv_vol = 1.0 / (topk_vol + 1e-6)
            adj_weights = target_weights[topk_indices] * inv_vol
            
            # Re-normalize the adjusted weights.
            if np.sum(adj_weights) > 0:
                target_weights[topk_indices] = adj_weights / np.sum(adj_weights)

    # Final normalization to ensure weights sum to 1.
    if np.sum(target_weights) > 1e-9:
        target_weights = target_weights / np.sum(target_weights)
        
    return target_weights
