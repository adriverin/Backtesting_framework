# strategies/ml_forecasting/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings("ignore")

from .config import MLConfig

def _bar_size_hours(interval: str) -> float:
    """Converts interval string to hours."""
    val = int(interval[:-1])
    unit = interval[-1]
    if unit == 'm': return val / 60
    if unit == 'h': return val
    if unit == 'd': return val * 24
    return 1.0

class FeatureEngineer:
    """Unified feature engineering class."""
    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_names: List[str] = []
        self.price_column = config.price_column
        self.norm_mean = None
        self.norm_std = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on data and then transform it. For training data."""
        df_processed = self._preprocess(df.copy())
        df_features = self._add_all_features(df_processed)
        
        self.feature_names = [col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', self.price_column, 'return', 'norm_return', 'vol']]
        
        X = df_features[self.feature_names].copy()
        
        # Calculate and store normalization stats
        self.norm_mean = X.mean()
        self.norm_std = X.std().replace(0, 1e-8) # Avoid division by zero
        
        X_normalized = (X - self.norm_mean) / self.norm_std
        X_normalized = X_normalized.clip(-10, 10).fillna(0)
        
        df_features[self.feature_names] = X_normalized
        return df_features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using existing normalization stats. For OOS data."""
        if self.norm_mean is None or self.norm_std is None:
            raise RuntimeError("Must call fit_transform before transform.")
            
        df_processed = self._preprocess(df.copy())
        df_features = self._add_all_features(df_processed)
        
        X = df_features[self.feature_names].copy()

        # Apply stored normalization
        X_normalized = (X - self.norm_mean) / self.norm_std
        X_normalized = X_normalized.clip(-10, 10).fillna(0)

        df_features[self.feature_names] = X_normalized
        return df_features

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data preprocessing."""
        df['return'] = df[self.price_column].pct_change()
        vol_window_bars = max(5, int(self.config.vol_window_hours / _bar_size_hours(self.config.interval)))
        df['vol'] = df['return'].rolling(vol_window_bars, min_periods=3).std()
        df['norm_return'] = (df['return'] / (df['vol'] + 1e-8)).clip(-5, 5).fillna(0)
        return df

    def _add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical and regime features."""
        # SMA features
        for window in self.config.sma_windows:
            sma = df[self.price_column].rolling(window).mean()
            df[f'sma_{window}'] = (df[self.price_column] / sma - 1).fillna(0).clip(-1, 1)

        # Momentum features
        for window in self.config.momentum_windows:
            df[f'mom_{window}'] = df[self.price_column].pct_change(window).fillna(0).clip(-1, 1)

        # RSI features
        for window in self.config.rsi_windows:
            df[f'rsi_{window}'] = self._calculate_rsi(df[self.price_column], window)
        
        if self.config.enable_regime_features:
            self._add_regime_features(df)
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
        return df

    def _add_regime_features(self, df: pd.DataFrame):
        """Adds market regime features."""
        vol_window = self.config.volatility_regime_window
        rolling_vol = df['return'].rolling(vol_window).std()
        vol_threshold = rolling_vol.rolling(vol_window * 2).median()
        df['vol_regime'] = (rolling_vol > vol_threshold).astype(float).fillna(0)

    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return ((rsi - 50) / 50).fillna(0)