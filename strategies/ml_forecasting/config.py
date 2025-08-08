# strategies/ml_forecasting/config.py
from dataclasses import asdict, dataclass, field
from typing import Tuple
import torch

@dataclass
class MLConfig:
    """Unified configuration for the ML forecasting strategy."""
    price_column: str = "close"
    interval: str = "1h" # Needed for time-based calculations

    # Forecasting & Feature Engineering
    forecast_horizon_hours: int = 6
    vol_window_hours: int = 60
    vwap_window: int = 20 # if using VWAP price_column
    enable_regime_features: bool = True
    sma_windows: Tuple[int, ...] = (5, 10, 20, 30)
    volatility_windows: Tuple[int, ...] = (5, 10, 20)
    momentum_windows: Tuple[int, ...] = (7, 14, 21, 30)
    rsi_windows: Tuple[int, ...] = (7, 14, 21)
    volatility_regime_window: int = 60

    # Model Architecture
    n_quantiles: int = 5
    hidden_sizes: Tuple[int, ...] = (64, 32, 16)
    dropout_rate: float = 0.5

    # Training Parameters
    n_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    early_stopping_patience: int = 5

    # Signal Generation
    signal_percentiles: Tuple[int, int] = (10, 90)

    # Infrastructure
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    random_seed: int = 42

    def to_dict(self) -> dict:
        """Converts the dataclass to a dictionary."""
        return asdict(self)        