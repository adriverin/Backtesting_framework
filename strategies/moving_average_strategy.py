"""Concrete implementation of a classic Moving Average crossover strategy."""
from __future__ import annotations

from typing import Any, Tuple

import pandas as pd

from .basic_strats import optimize_moving_average, moving_average

from .base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """Moving Average crossover using pre-existing optimisation utilities."""

    def __init__(self, price_column: str = "close", **kwargs: Any) -> None:
        super().__init__(price_column, **kwargs)
        self.best_params: Tuple[int, int] | None = None  # (short_window, long_window)
        self.best_pf: float | None = None

    # ------------------------------------------------------------------ #
    # BaseStrategy interface                                             #
    # ------------------------------------------------------------------ #

    def optimize(self, ohlc: pd.DataFrame):
        # Delegate to the already implemented helper
        best_params, best_pf = optimize_moving_average(ohlc, self.price_column)
        self.best_params = best_params
        self.best_pf = best_pf
        print(f"Best params: {best_params}, Best PF: {best_pf}")
        return best_params, best_pf

    def generate_signals(self, ohlc: pd.DataFrame):
        if self.best_params is None:
            # Fallback optimisation if the client forgot to call optimise first
            self.optimize(ohlc)
        short_win, long_win = self.best_params  # type: ignore
        return moving_average(ohlc, self.price_column, short_win, long_win)



    