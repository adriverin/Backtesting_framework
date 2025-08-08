# strategies/ml_strategy.py
from __future__ import annotations
from typing import Any, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

from .base_strategy import BaseStrategy
from .ml_forecasting.config import MLConfig
from .ml_forecasting.feature_engineering import FeatureEngineer
from .ml_forecasting.models import create_model, SimpleModel
from .ml_forecasting.signal_generation import generate_trading_signals
# CHANGED: Import the corrected generate_labels
from .ml_forecasting.training_tools import generate_labels, ModelTrainer

class MLStrategy(BaseStrategy):
    """
    ML forecasting strategy that trains on the full period provided.
    It uses a simple MLP to predict future return quantiles.
    """
    def __init__(self, price_column: str = "close", **kwargs: Any):
        super().__init__(price_column, **kwargs)
        # 1. Start with a copy of all provided keyword arguments.
        config_params = self.extra_params.copy()

        # 2. Handle the 'timeframe' to 'interval' mapping.
        # The testing scripts pass 'timeframe', but MLConfig expects 'interval'.
        # We give 'interval' precedence if both are somehow provided.
        if 'timeframe' in config_params and 'interval' not in config_params:
            config_params['interval'] = config_params['timeframe']

        # 3. Remove 'timeframe' since MLConfig does not accept it.
        # This prevents an unexpected keyword argument error.
        if 'timeframe' in config_params:
            del config_params['timeframe']

        # 4. Now, create the MLConfig object safely.
        # 'price_column' is passed explicitly, and the prepared params are unpacked.
        self.config = MLConfig(price_column=price_column, **config_params)


        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        self.model: SimpleModel | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.quantile_edges: np.ndarray | None = None
        self.signal_thresholds: tuple[float, float] | None = None

    def _ensure_price_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Checks for and creates calculated price columns if needed."""
        df = df.copy()
        col = self.config.price_column
        
        if col in df.columns:
            return df # Column already exists
            
        if col in ["typical"]:
            if all(c in df.columns for c in ['high', 'low', 'close']):
                df[col] = (df['high'] + df['low'] + df['close']) / 3
                print(f"--- Created '{col}' column on-the-fly. ---")
            else:
                raise ValueError(f"Cannot create '{col}' column: missing high, low, or close.")
        elif col == "median":
            if all(c in df.columns for c in ['high', 'low']):
                df[col] = (df['high'] + df['low']) / 2
                print(f"--- Created '{col}' column on-the-fly. ---")
            else:
                raise ValueError(f"Cannot create '{col}' column: missing high or low.")
        elif col == "vwap":
            if 'volume' not in df.columns:
                raise ValueError("Cannot calculate VWAP: 'volume' column is missing.")
            
            # We calculate a rolling VWAP. A common period is 14 or 20 bars.
            # Let's make it a configurable parameter in MLConfig.
            vwap_window = getattr(self.config, 'vwap_window', 20)
            
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            tpv = typical_price * df['volume']
            
            cumulative_tpv = tpv.rolling(window=vwap_window, min_periods=1).sum()
            cumulative_volume = df['volume'].rolling(window=vwap_window, min_periods=1).sum()
            
            # Avoid division by zero for periods with no volume
            df[col] = (cumulative_tpv / cumulative_volume).fillna(method='ffill')
            
            print(f"--- Created rolling 'vwap' ({vwap_window}-bar) column on-the-fly. ---")            
        else:
            # If the column is not special and not present, it's an error.
            raise ValueError(f"Price column '{col}' not found in DataFrame.")
            
        return df



    def optimize(self, ohlc: pd.DataFrame) -> Tuple[Any, float]:
        """
        Trains the ML model on the entire provided in-sample 'ohlc' data.
        An internal validation set is used only for early stopping.
        """
        print(f"Optimizing MLStrategy on data from {ohlc.index.min()} to {ohlc.index.max()}...")

        ohlc = self._ensure_price_column(ohlc)

        # 1. Create an INTERNAL train/validation split on raw OHLC data
        n = len(ohlc)
        train_end_idx = int(n * self.config.train_ratio)
        ohlc_train = ohlc.iloc[:train_end_idx]
        ohlc_val = ohlc.iloc[train_end_idx:]

        # 2. Initialize and fit the feature engineer ONLY on the TRAIN split
        self.feature_engineer = FeatureEngineer(self.config)
        internal_train_df = self.feature_engineer.fit_transform(ohlc_train)
        # Transform validation split with the stats learned from training
        internal_val_df = self.feature_engineer.transform(ohlc_val)

        # 3. Generate labels.
        # CRITICAL: Calculate quantiles ONLY from the internal training part.
        train_df_labeled, y_train, self.quantile_edges = generate_labels(internal_train_df, self.config)

        # Use the training quantiles to label the internal validation set.
        val_df_labeled, y_val, _ = generate_labels(internal_val_df, self.config, quantile_edges=self.quantile_edges)
        
        X_train = train_df_labeled[self.feature_engineer.feature_names]
        X_val = val_df_labeled[self.feature_engineer.feature_names]

        # 4. Create Datasets and Model
        train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        
        self.model = create_model(len(self.feature_engineer.feature_names), self.config)
        
        # 5. Train the model
        trainer = ModelTrainer(self.config)
        trainer.train_with_validation(self.model, train_dataset, val_dataset)

        # 6. Derive fixed signal thresholds from TRAIN predictions only
        #    to avoid using information from the full sample when deciding thresholds.
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                probs = self.model.predict_proba(X_train_tensor).cpu().numpy()
            bottom_scores = probs[:, 0] + probs[:, 1]
            top_scores = probs[:, -2] + probs[:, -1]
            extreme_pref = top_scores - bottom_scores
            top_thr = np.percentile(extreme_pref, self.config.signal_percentiles[1])
            bot_thr = np.percentile(extreme_pref, self.config.signal_percentiles[0])
            self.signal_thresholds = (bot_thr, top_thr)
        
        print("MLStrategy optimization (training) complete.")

        # The "best_params" are the trained model itself. Return a placeholder metric.
        # The true performance is evaluated out-of-sample by the testing framework.
        return self.config.to_dict(), 1.0

    def generate_signals(self, ohlc: pd.DataFrame) -> pd.Series:
        """Generate signals for new data using the trained model."""
        ohlc = self._ensure_price_column(ohlc)

        if self.model is None or self.feature_engineer is None:
            print("Model not trained. Running optimization first.")
            self.optimize(ohlc)

        # 1. Transform features using the FITTED engineer
        df_features = self.feature_engineer.transform(ohlc)
        
        # 2. Generate signals from the features
        signals_array = generate_trading_signals(
            self.model,
            df_features,
            self.feature_engineer.feature_names,
            self.config,
            thresholds=self.signal_thresholds
        )
        
        # 3. Return as a pandas Series with the correct index
        signals = pd.Series(np.nan, index=ohlc.index)
        signals.loc[df_features.index] = signals_array
        signals = signals.fillna(0) 

        return signals

    # Helper function from BaseStrategy is used implicitly