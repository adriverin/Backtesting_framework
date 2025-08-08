"""Machine-learning based strategy that uses RSI, EMA and average Volume features.

This example implementation trains a simple Logistic Regression classifier
on in-sample data to predict the direction (positive/negative) of the next
log-return. The predicted sign is used as the trading signal (long/flat).

The implementation is intentionally lightweight and relies only on
pandas/numpy and scikit-learn – which is a common dependency in most
Python data science environments.

NOTE: This is merely a baseline skeleton showcasing *how* such a strategy
can be integrated into the modular framework. It is NOT meant for
production-grade trading.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .base_strategy import BaseStrategy


# --------------------------- Feature helpers --------------------------- #

def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period, min_periods=period).mean()
    loss = down.rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _compute_avg_volume(volume: pd.Series, period: int) -> pd.Series:
    return volume.rolling(window=period, min_periods=period).mean()


# --------------------------- Strategy class --------------------------- #

class MLRSIEMAVolumeStrategy(BaseStrategy):
    """Machine-learning strategy combining RSI, EMA and volume features."""

    DEFAULT_PARAM_GRID: Dict[str, List[int]] = {
        "rsi_period": [14, 21, 28],
        "ema_period": [20, 50, 100],
        "vol_period": [10, 20, 50],
    }

    def __init__(self, price_column: str = "close", param_grid: Dict[str, List[int]] | None = None, **kwargs: Any):
        super().__init__(price_column, **kwargs)
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.best_params: Dict[str, int] | None = None
        self.clf: LogisticRegression | None = None
        self.best_pf: float | None = None

    # ------------------------------------------------------------------ #
    # BaseStrategy interface                                             #
    # ------------------------------------------------------------------ #

    def optimize(self, ohlc: pd.DataFrame):
        # Prepare target variable: positive next log-return -> 1 else 0
        returns = np.log(ohlc[self.price_column]).diff().shift(-1)
        target = (returns > 0).astype(int)

        # Split into train/test avoiding look-ahead bias (time-series aware)
        cutoff = int(len(ohlc) * 0.7)
        train_df = ohlc.iloc[:cutoff].copy()
        test_df = ohlc.iloc[cutoff:].copy()
        y_train = target.iloc[:cutoff]
        y_test = target.iloc[cutoff:]

        best_pf = -np.inf
        best_params = None
        best_clf = None

        # Grid search over parameter combinations
        keys = list(self.param_grid.keys())
        for values in product(*[self.param_grid[k] for k in keys]):
            params = dict(zip(keys, values))

            # Feature engineering for train and test sets
            X_train = self._prepare_features(train_df, **params)
            X_test = self._prepare_features(test_df, **params)

            # Drop rows with NaNs created by rolling calculations
            valid_idx_train = X_train.dropna().index
            X_train = X_train.loc[valid_idx_train]
            y_train_clean = y_train.loc[valid_idx_train]

            valid_idx_test = X_test.dropna().index
            X_test = X_test.loc[valid_idx_test]
            y_test_clean = y_test.loc[valid_idx_test]

            if len(X_train) < 100:  # Require minimum samples
                continue

            clf = LogisticRegression(max_iter=1000, n_jobs=1)
            clf.fit(X_train, y_train_clean)

            # Evaluate on test set using accuracy AND profit factor
            preds_proba = clf.predict_proba(X_test)[:, 1]
            signals = (preds_proba > 0.5).astype(int)  # 1 = long, 0 = flat
            strat_returns = returns.loc[X_test.index] * signals
            pf = self._profit_factor(strat_returns)

            if pf > best_pf:
                best_pf = pf
                best_params = params
                best_clf = clf

        if best_params is None:
            raise RuntimeError("No valid parameter combination found during optimisation.")

        self.best_params = best_params
        self.clf = best_clf
        self.best_pf = best_pf
        return best_params, float(best_pf)

    def generate_signals(self, ohlc: pd.DataFrame):
        if self.clf is None or self.best_params is None:
            # Either optimise on full data or raise error
            self.optimize(ohlc)

        X = self._prepare_features(ohlc, **self.best_params)
        X = X.fillna(method="bfill").fillna(method="ffill")  # Basic imputation for initial NaNs
        preds_proba = self.clf.predict_proba(X)[:, 1]
        signals = (preds_proba > 0.5).astype(int)  # 1 for long else 0
        return pd.Series(signals, index=ohlc.index)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _prepare_features(self, df: pd.DataFrame, rsi_period: int, ema_period: int, vol_period: int):
        feats = pd.DataFrame(index=df.index)
        feats[f"RSI_{rsi_period}"] = _compute_rsi(df[self.price_column], rsi_period)
        feats[f"EMA_{ema_period}"] = _compute_ema(df[self.price_column], ema_period)
        if "volume" in df.columns:
            feats[f"VOL_AVG_{vol_period}"] = _compute_avg_volume(df["volume"], vol_period)
        else:
            # If volume is unavailable, fall back to constant to avoid NaNs
            feats[f"VOL_AVG_{vol_period}"] = 0.0
        return feats

