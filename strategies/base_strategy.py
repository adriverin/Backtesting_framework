"""Abstract base class defining the interface that every trading strategy must implement.

A concrete strategy must provide two main capabilities:
1. Parameter optimization on in-sample data to discover the best hyper-parameters.
2. Signal generation on an arbitrary DataFrame once the strategy is configured.

Signals should be encoded as numeric values:
    1  -> long / buy
    -1 -> short / sell (if the strategy supports it)
    0  -> flat / no position

The returned pd.Series must share the SAME index as the input OHLCV DataFrame.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict

import pandas as pd


class BaseStrategy(ABC):
    """Base class for all strategies."""

    def __init__(self, price_column: str = "close", **kwargs: Any) -> None:
        self.price_column = price_column
        # Store arbitrary keyword args on the instance for maximum flexibility
        self.extra_params: Dict[str, Any] = kwargs

    # ====================================================================== #
    # ------------------------------ API ----------------------------------- #
    # ====================================================================== #

    @abstractmethod
    def optimize(self, ohlc: pd.DataFrame) -> Tuple[Any, float]:
        """Find optimal hyper-parameters on the supplied DataFrame.

        Returns
        -------
        best_params : Any
            The discovered best parameter set. Type depends on the strategy implementation.
        best_metric : float
            The metric value associated with *best_params*. Often the profit factor.
        """

    @abstractmethod
    def generate_signals(self, ohlc: pd.DataFrame) -> pd.Series:
        """Generate trading signals for *ohlc* using internal parameters.

        If *optimize* has not been run yet the implementation SHOULD handle
        that gracefully – either by running a default optimisation or by
        raising a clear error message.
        """

    # ---------------------------------------------------------------------- #
    # Helper methods – optional to override                                 #
    # ---------------------------------------------------------------------- #

    def _profit_factor(self, returns: pd.Series) -> float:
        """Utility to compute the profit factor of a strategy."""
        positive_sum = returns[returns > 0].sum()
        negative_sum = returns[returns < 0].abs().sum()
        if negative_sum == 0:
            return float("inf") if positive_sum > 0 else 0.0
        return positive_sum / negative_sum

