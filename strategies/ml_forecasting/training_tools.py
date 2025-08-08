# strategies/ml_forecasting/training_tools.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional

from .config import MLConfig
from .models import SimpleModel

def _bar_size_hours(interval: str) -> float:
    # Helper to convert interval string (e.g., '1h', '4h') to hours
    try:
        if 'm' in interval: return int(interval.replace('m', '')) / 60
        if 'h' in interval: return int(interval.replace('h', ''))
        if 'd' in interval: return int(interval.replace('d', '')) * 24
    except:
        return 1.0 # Default fallback
    return 1.0

# CHANGED: Added optional quantile_edges parameter
def generate_labels(df: pd.DataFrame, config: MLConfig, quantile_edges: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate quantile-based labels."""
    horizon_bars = int(config.forecast_horizon_hours / _bar_size_hours(config.interval))
    df['future_norm_ret'] = df['norm_return'].shift(-horizon_bars)
    df = df.dropna(subset=['future_norm_ret'])

    # If quantile_edges are not provided, calculate them from the given dataframe.
    # This should only be done on the training set.
    if quantile_edges is None:
        quantile_probs = np.linspace(0, 1, config.n_quantiles + 1)[1:-1]
        quantile_edges = np.quantile(df['future_norm_ret'], quantile_probs)

    labels = np.digitize(df['future_norm_ret'], quantile_edges, right=False)

    feature_df = df.drop(columns=['future_norm_ret'])
    return feature_df, labels, quantile_edges

class ModelTrainer:
    """Handles the model training loop."""
    def __init__(self, config: MLConfig):
        self.config = config

    def train_with_validation(self, model: SimpleModel, train_dataset: Dataset, val_dataset: Dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.n_epochs):
            model.train()
            # Training phase
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(self.config.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation phase for early stopping
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(self.config.device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                print(f"--- Early stopping at epoch {epoch+1} with best validation loss: {best_val_loss:.4f} ---")
                break