from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


class DonchianStrategy(BaseStrategy):
    """Donchian breakout strategy.

    Optimises the lookback window and generates signals based on breakouts of
    the prior window's high/low using the configured price column.
    """

    def __init__(
        self,
        price_column: str = "close",
        lookback_min: int = 2,
        lookback_max: int = 168,
        **kwargs: Any,
    ) -> None:
        super().__init__(price_column, **kwargs)
        self.lookback_min = int(lookback_min)
        self.lookback_max = int(lookback_max)
        self.best_lookback: int | None = None
        self.best_pf: float | None = None

    # ------------------------------------------------------------------ #
    # BaseStrategy interface                                             #
    # ------------------------------------------------------------------ #

    def optimize(self, ohlc: pd.DataFrame) -> Tuple[int, float]:
        price = ohlc[self.price_column]
        r = np.log(price).diff().shift(-1)

        best_pf = 0.0
        best_lb = -1

        # inclusive range to allow selecting the upper bound
        for lb in range(max(2, self.lookback_min), max(2, self.lookback_max) + 1):
            signal = self._donchian_signals(ohlc, lb)
            sig_rets = signal * r
            pf = self._profit_factor(sig_rets.dropna())
            if pf > best_pf:
                best_pf = float(pf)
                best_lb = int(lb)

        if best_lb < 0:
            raise RuntimeError("Failed to find a valid lookback during optimisation.")

        self.best_lookback = best_lb
        self.best_pf = best_pf
        return best_lb, best_pf

    def generate_signals(self, ohlc: pd.DataFrame) -> pd.Series:
        if self.best_lookback is None:
            self.optimize(ohlc)
        return self._donchian_signals(ohlc, int(self.best_lookback))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _donchian_signals(self, ohlc: pd.DataFrame, lookback: int) -> pd.Series:
        price = ohlc[self.price_column]
        upper = price.rolling(lookback - 1).max().shift(1)
        lower = price.rolling(lookback - 1).min().shift(1)

        signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
        signal.loc[price > upper] = 1
        signal.loc[price < lower] = -1
        return signal.ffill()


