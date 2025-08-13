from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


def _annualisation_factor(timeframe: Optional[str]) -> float:
    if not timeframe:
        return 1.0
    tf = timeframe.lower().strip()
    minutes_per_unit = 1
    if tf.endswith("m"):
        minutes_per_unit = int(tf[:-1])
    elif tf.endswith("h"):
        minutes_per_unit = int(tf[:-1]) * 60
    elif tf.endswith("d"):
        minutes_per_unit = int(tf[:-1]) * 60 * 24
    else:
        # Fallback: hourly
        minutes_per_unit = 60
    periods_per_year = (365 * 24 * 60) / minutes_per_unit
    return float(np.sqrt(periods_per_year))


def compute_position_weights(
    signals: pd.Series,
    simple_returns: pd.Series,
    price: Optional[pd.Series] = None,
    timeframe: Optional[str] = None,
    mode: str = "full_notional",
    mode_params: Optional[Dict[str, Union[float, int, pd.Series]]] = None,
) -> pd.Series:
    """Compute portfolio weights from discrete signals using a sizing policy.

    Parameters
    ----------
    signals : pd.Series
        Discrete trading signal series in {-1, 0, 1} (or continuous sign proxy).
    simple_returns : pd.Series
        Simple return series of the asset (pct change shifted to next bar).
    price : Optional[pd.Series]
        Price series for ATR/stop calculations if needed by a mode.
    timeframe : Optional[str]
        Timeframe string (e.g. '1h', '4h') for annualisation in vol targeting.
    mode : str
        One of: 'full_notional' (default), 'fixed_notional', 'fixed_fraction',
        'vol_target', 'risk_stop'.
    mode_params : Optional[Dict]
        Additional parameters per mode (see below).

    Supported mode_params
    ---------------------
    - full_notional: none
    - fixed_notional:
        fixed_notional (float, default=1.0): Desired constant notional in units
            of initial equity (=1). Weight at time t is fixed_notional / equity_{t-1}.
    - fixed_fraction:
        fraction (float in [0,1], default=1.0): Constant fraction of equity.
    - vol_target:
        target_vol_annual (float, default=0.2): Annualised target volatility.
        lookback (int, default=50): Rolling window for realised vol estimate.
        max_leverage (float, default=3.0): Cap on absolute weight.
    - risk_stop:
        risk_per_trade_frac (float, default=0.01): Fraction of equity you risk per trade.
        stop_loss_pct (float or pd.Series): Stop size as percent of price (e.g., 0.02 = 2%).
        max_leverage (float, default=3.0): Cap on absolute weight.

    Returns
    -------
    pd.Series
        Weight series aligned with inputs.
    """
    mode_params = mode_params or {}
    signals = signals.astype(float).fillna(0.0)
    simple_returns = simple_returns.astype(float).fillna(0.0)

    if mode == "full_notional":
        # Existing behaviour: weight equals signal
        return signals

    if mode == "fixed_fraction":
        fraction = float(mode_params.get("fraction", 1.0))
        fraction = max(0.0, float(min(10.0, fraction)))  # safety cap
        return signals * fraction

    if mode == "vol_target":
        target_vol = float(mode_params.get("target_vol_annual", 0.2))
        lookback = int(mode_params.get("lookback", 50))
        max_leverage = float(mode_params.get("max_leverage", 3.0))
        ann = _annualisation_factor(timeframe)

        realized_vol_annual = simple_returns.rolling(lookback).std() * ann
        scale = target_vol / realized_vol_annual.replace(0, np.nan)
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weight = signals * scale
        weight = weight.clip(lower=-max_leverage, upper=max_leverage)
        weight = weight.fillna(0.0)
        return weight

    if mode == "fixed_notional":
        fixed_notional = float(mode_params.get("fixed_notional", 1.0))
        # Sequential sizing: maintain equity assuming starting equity 1.0
        equity = 1.0
        weights = []
        # Iterate in time order
        for sig, r in zip(signals.values, simple_returns.values):
            if equity <= 0:
                w = 0.0
            else:
                w = (fixed_notional / equity) * sig
            weights.append(w)
            # Update equity for next step using current period return and weight
            equity = float(equity * (1.0 + (w * r)))
        return pd.Series(weights, index=signals.index, dtype=float)

    if mode == "risk_stop":
        risk_frac = float(mode_params.get("risk_per_trade_frac", 0.01))
        max_leverage = float(mode_params.get("max_leverage", 3.0))
        stop_loss_pct = mode_params.get("stop_loss_pct", None)

        if isinstance(stop_loss_pct, pd.Series):
            stop_pct = stop_loss_pct.reindex(signals.index).astype(float)
        elif stop_loss_pct is None:
            # Fallback: use rolling volatility of returns as proxy for stop
            lookback = int(mode_params.get("lookback", 20))
            stop_pct = simple_returns.abs().rolling(lookback).mean().clip(lower=1e-6)
        else:
            stop_pct = pd.Series(float(stop_loss_pct), index=signals.index)

        raw_size = risk_frac / stop_pct.replace(0, np.nan)
        raw_size = raw_size.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weight = (signals * raw_size).clip(lower=-max_leverage, upper=max_leverage)
        return weight.fillna(0.0)

    # Unknown mode -> default to existing behaviour
    return signals


