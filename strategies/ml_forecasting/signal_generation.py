# strategies/ml_forecasting/signal_generation.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import MLConfig
from .models import SimpleModel

def generate_trading_signals(
    model: SimpleModel,
    X_features: pd.DataFrame,
    feature_names: list,
    config: MLConfig,
    thresholds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Generate trading signals based on model predictions."""
    model.eval()
    
    # Prepare data for prediction
    X_tensor = torch.tensor(X_features[feature_names].values, dtype=torch.float32).to(config.device)
    pred_dataset = TensorDataset(X_tensor)
    pred_loader = DataLoader(pred_dataset, batch_size=config.batch_size, shuffle=False)

    # Get model probabilities
    all_probabilities = []
    with torch.no_grad():
        for (batch_X,) in pred_loader:
            probs = model.predict_proba(batch_X)
            all_probabilities.append(probs.cpu().numpy())
    
    probabilities = np.vstack(all_probabilities)

    # Convert probabilities to signals using percentile method
    bottom_scores = probabilities[:, 0] + probabilities[:, 1]
    top_scores = probabilities[:, -2] + probabilities[:, -1]
    extreme_preference = top_scores - bottom_scores

    # Use fixed thresholds if provided (derived from training); otherwise fall back to per-slice percentiles
    if thresholds is not None:
        bottom_threshold, top_threshold = thresholds
    else:
        top_threshold = np.percentile(extreme_preference, config.signal_percentiles[1])
        bottom_threshold = np.percentile(extreme_preference, config.signal_percentiles[0])

    signals = np.zeros(len(probabilities), dtype=int)
    signals[extreme_preference > top_threshold] = 1   # Long
    signals[extreme_preference < bottom_threshold] = -1 # Short

    return signals