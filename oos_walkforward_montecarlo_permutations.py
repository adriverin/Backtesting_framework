"""Generic walk-forward out-of-sample Monte-Carlo permutation tester.

Refactors *walkforward_test.py* to support any strategy that implements the
*BaseStrategy* interface.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from permutations import get_permutation

from strategies import aVAILABLE_STRATEGIES, BaseStrategy  # type: ignore
from position_sizing import compute_position_weights

# ---------------------------------------------------------------------- #
# Utility                                                                 #
# ---------------------------------------------------------------------- #

def _profit_factor(returns: pd.Series) -> float:
    positive_sum = returns[returns > 0].sum()
    negative_sum = returns[returns < 0].abs().sum()
    if negative_sum == 0:
        return float("inf") if positive_sum > 0 else 0.0
    return positive_sum / negative_sum


# ---------------------------------------------------------------------- #
# Helpers for parallel worker                                            #
# ---------------------------------------------------------------------- #

def _ensure_price_column_exists_oos(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """Ensure price column exists; support 'median', 'typical', and 'vwap' variants like 'vwap', 'vwap_10', 'vwap10'."""
    if price_column in df.columns:
        return df
    df = df.copy()
    col = price_column
    if col == "median":
        if all(c in df.columns for c in ["high", "low"]):
            df[col] = (df["high"] + df["low"]) / 2
            return df
        raise KeyError("Cannot create 'median' column: missing 'high' or 'low'.")
    if col == "typical":
        if all(c in df.columns for c in ["high", "low", "close"]):
            df[col] = (df["high"] + df["low"] + df["close"]) / 3
            return df
        raise KeyError("Cannot create 'typical' column: missing 'high', 'low' or 'close'.")
    if col.startswith("vwap"):
        required = ["high", "low", "close", "volume"]
        if all(c in df.columns for c in required):
            suffix = col[4:]
            if suffix.startswith("_") or suffix.startswith("-"):
                suffix = suffix[1:]
            if suffix.isdigit() and len(suffix) > 0:
                vwap_window = int(suffix)
            else:
                vwap_window = 20
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            tpv = typical_price * df["volume"]
            cumulative_tpv = tpv.rolling(window=vwap_window, min_periods=1).sum()
            cumulative_volume = df["volume"].rolling(window=vwap_window, min_periods=1).sum()
            df[col] = (cumulative_tpv / cumulative_volume).fillna(method="ffill")
            return df
        raise KeyError("Cannot create 'vwap' column: missing one of 'high','low','close','volume'.")
    raise KeyError(f"Price column '{price_column}' not found in DataFrame and cannot be derived.")


def _walkforward_signals_oos(
    ohlc: pd.DataFrame,
    price_column: str,
    strategy_name: str,
    strategy_kwargs: dict,
    train_lookback: int,
    train_step: int,
) -> np.ndarray:
    n = len(ohlc)
    wf_signals = np.full(n, np.nan)
    strategy_cls: Type[BaseStrategy] = aVAILABLE_STRATEGIES[strategy_name]

    for i in range(train_lookback, n, train_step):
        train_start = i - train_lookback
        train_end = i
        train_df = ohlc.iloc[train_start:train_end]
        train_df = _ensure_price_column_exists_oos(train_df, price_column)

        strategy: BaseStrategy = strategy_cls(price_column=price_column, **strategy_kwargs)
        _ = strategy.optimize(train_df)

        oos_end = min(i + train_step, n)
        long_lookback_ctx = train_lookback
        signal_calc_start = max(0, train_end - long_lookback_ctx)
        calc_df = ohlc.iloc[signal_calc_start:oos_end]
        calc_df = _ensure_price_column_exists_oos(calc_df, price_column)
        oos_signals_full = strategy.generate_signals(calc_df)
        oos_section_len = len(ohlc.iloc[train_end:oos_end])
        oos_signals_final = oos_signals_full.iloc[-oos_section_len:].values
        wf_signals[train_end:oos_end] = oos_signals_final

    return wf_signals


def _compute_single_perm_pf_net_oos(
    seed: int,
    asset: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    price_column: str,
    strategy_name: str,
    strategy_kwargs: dict,
    fee_bps: float,
    slippage_bps: float,
    train_lookback: int,
    train_step: int,
    perm_start_index: int,
    position_sizing_mode: str = "full_notional",
    position_sizing_params: dict | None = None,
    mode: str = "spot",
) -> float:
    base_dir = Path("data/futures") if (mode or "spot").strip().lower() == "futures" else Path("data/spot")
    full_fp = base_dir / f"ohlcv_{asset}_{timeframe}.parquet"
    full_df = pd.read_parquet(full_fp)
    # Normalize datetime index to UTC-naive for consistent comparisons
    if not isinstance(full_df.index, pd.DatetimeIndex):
        full_df.index = pd.to_datetime(full_df.index, errors="coerce")
    if getattr(full_df.index, "tz", None) is not None:
        full_df.index = full_df.index.tz_convert("UTC").tz_localize(None)
    from permutations import get_permutation  # local import for subprocess
    perm_df = get_permutation(full_df, start_index=perm_start_index, seed=seed)
    # Build a padded slice that includes pre-start lookback bars for training
    idx = perm_df.index
    _start_ts = pd.to_datetime(start_date)
    _end_ts = pd.to_datetime(end_date)
    analysis_start = int(idx.searchsorted(_start_ts, side="left"))
    analysis_end = int(idx.searchsorted(_end_ts, side="left"))
    slice_start = max(0, analysis_start - train_lookback)
    wf_df = perm_df.iloc[slice_start:analysis_end]
    wf_df = _ensure_price_column_exists_oos(wf_df, price_column)

    # Generate WF signals on the padded window
    perm_signals = _walkforward_signals_oos(
        wf_df,
        price_column,
        strategy_name,
        strategy_kwargs,
        train_lookback,
        train_step,
    )

    # Compute returns/weights on the padded window, then trim to analysis window
    wf_df["r"] = np.log(wf_df[price_column]).diff().shift(-1)
    wf_df["simple_r"] = wf_df[price_column].pct_change().shift(-1)
    wf_df["signal"] = perm_signals
    wf_df["weight"] = compute_position_weights(
        signals=pd.Series(perm_signals, index=wf_df.index),
        simple_returns=wf_df["simple_r"],
        price=wf_df[price_column],
        timeframe=timeframe,
        mode=position_sizing_mode,
        mode_params=position_sizing_params,
    )
    wf_df["strategy_r"] = wf_df["r"] * wf_df["weight"]
    wf_df["strategy_simple_r"] = wf_df["simple_r"] * wf_df["weight"]
    _mode = (mode or "spot").strip().lower()
    effective_fee_bps = float(fee_bps) if fee_bps and fee_bps > 0 else (4.0 if _mode == "futures" else 10.0)
    fee_rate = (effective_fee_bps + float(slippage_bps)) / 10000.0
    wf_df["turnover"] = (wf_df["weight"].diff().abs()).fillna(wf_df["weight"].abs())
    wf_df["cost_simple"] = fee_rate * wf_df["turnover"]
    wf_df["strategy_simple_r_net"] = wf_df["strategy_simple_r"] - wf_df["cost_simple"]
    wf_df["strategy_r_net"] = np.log((1.0 + wf_df["strategy_simple_r_net"]).clip(lower=1e-12))

    # Trim to analysis window for PF computation
    trimmed = wf_df[(wf_df.index >= _start_ts) & (wf_df.index < _end_ts)]
    trimmed = trimmed.dropna(subset=["r", "strategy_r_net"])  # avoid trailing NaNs impacting PF
    return _profit_factor(trimmed["strategy_r_net"].dropna())


# ---------------------------------------------------------------------- #
# Main tester class                                                       #
# ---------------------------------------------------------------------- #

class WalkForwardMCTester:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        strategy_name: str,
        asset: str,
        timeframe: str,
        train_lookback: int,
        train_step: int,
        n_perm: int = 10,
        price_column: str = "close",
        generate_plot: bool = False,
        strategy_kwargs: dict | None = None,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        position_sizing_mode: str = "full_notional",
        position_sizing_params: dict | None = None,
        mode: str = "spot",
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_name = strategy_name
        self.asset = asset
        self.timeframe = timeframe
        self.train_lookback = train_lookback
        self.train_step = train_step
        self.n_perm = n_perm
        self.price_column = price_column
        self.generate_plot = generate_plot
        self.strategy_kwargs = strategy_kwargs or {}

        if strategy_name not in aVAILABLE_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {list(aVAILABLE_STRATEGIES)}"
            )
        self.strategy_cls: Type[BaseStrategy] = aVAILABLE_STRATEGIES[strategy_name]
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.position_sizing_mode: str = position_sizing_mode
        self.position_sizing_params: dict = position_sizing_params or {}
        self.mode = (mode or "spot").strip().lower()

        self.perm_start_index = self._get_perm_start_index(start_date)
        print(f"Permutation start index: {self.perm_start_index}")

    # ------------------------------------------------------------------ #
    # Data helpers                                                       #
    # ------------------------------------------------------------------ #

    def _get_full_filepath(self) -> Path:
        base_dir = Path("data/futures") if (self.mode or "spot").strip().lower() == "futures" else Path("data/spot")
        return base_dir / f"ohlcv_{self.asset}_{self.timeframe}.parquet"

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_parquet(self._get_full_filepath())
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    def get_df(self) -> pd.DataFrame:
        full_df = self._load_raw()
        idx = full_df.index
        start_ts = pd.to_datetime(self.start_date)
        end_ts = pd.to_datetime(self.end_date)
        analysis_start = int(idx.searchsorted(start_ts, side="left"))
        analysis_end = int(idx.searchsorted(end_ts, side="left"))
        slice_start = max(0, analysis_start - self.train_lookback)
        return full_df.iloc[slice_start:analysis_end]

    def _ensure_price_column_exists(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the requested price column exists, creating common derived ones if needed.
        
        Supports:
        - 'median'   = (high + low) / 2
        - 'typical'  = (high + low + close) / 3
        - 'vwap' and variants like 'vwap_10' or 'vwap10' (requires volume)
        """
        price_column = self.price_column
        if price_column in df.columns:
            return df
        
        df = df.copy()
        col = price_column
        if col == "median":
            if all(c in df.columns for c in ["high", "low"]):
                df[col] = (df["high"] + df["low"]) / 2
                print(f"--- Created '{col}' column on-the-fly. ---")
                return df
            raise KeyError("Cannot create 'median' column: missing 'high' or 'low'.")
        if col == "typical":
            if all(c in df.columns for c in ["high", "low", "close"]):
                df[col] = (df["high"] + df["low"] + df["close"]) / 3
                print(f"--- Created '{col}' column on-the-fly. ---")
                return df
            raise KeyError("Cannot create 'typical' column: missing 'high', 'low' or 'close'.")
        if col.startswith("vwap"):
            required = ["high", "low", "close", "volume"]
            if all(c in df.columns for c in required):
                suffix = col[4:]
                if suffix.startswith("_") or suffix.startswith("-"):
                    suffix = suffix[1:]
                if suffix.isdigit() and len(suffix) > 0:
                    vwap_window = int(suffix)
                else:
                    vwap_window = 20
                typical_price = (df["high"] + df["low"] + df["close"]) / 3
                tpv = typical_price * df["volume"]
                cumulative_tpv = tpv.rolling(window=vwap_window, min_periods=1).sum()
                cumulative_volume = df["volume"].rolling(window=vwap_window, min_periods=1).sum()
                df[col] = (cumulative_tpv / cumulative_volume).fillna(method="ffill")
                print(f"--- Created rolling '{col}' ({vwap_window}-bar) column on-the-fly. ---")
                return df
            raise KeyError("Cannot create 'vwap' column: missing one of 'high','low','close','volume'.")
        raise KeyError(f"Price column '{price_column}' not found in DataFrame and cannot be derived.")

    def _get_perm_df_with_seed(self, seed: int) -> pd.DataFrame:
        full_df = self._load_raw()
        perm_df = get_permutation(full_df, start_index=self.perm_start_index, seed=seed)
        perm_df = perm_df[(perm_df.index >= self.start_date) & (perm_df.index < self.end_date)]
        return perm_df

    def _get_perm_start_index(self, start_date: str) -> int:
        full_df = self._load_raw()
        idx = full_df.index
        start_ts = pd.to_datetime(start_date)
        analysis_start_idx = int(idx.searchsorted(start_ts, side="left"))
        perm_start_idx = max(0, analysis_start_idx - self.train_lookback)
        return perm_start_idx

    # ------------------------------------------------------------------ #
    # Walk-forward signal generation                                     #
    # ------------------------------------------------------------------ #

    def _walkforward_signals(self, ohlc: pd.DataFrame) -> np.ndarray:
        n = len(ohlc)
        wf_signals = np.full(n, np.nan)
        price_col = self.price_column

        for i in range(self.train_lookback, n, self.train_step):
            train_start = i - self.train_lookback
            train_end = i
            train_df = ohlc.iloc[train_start:train_end]

            # Fresh strategy instance for each iteration to avoid leakage
            strategy: BaseStrategy = self.strategy_cls(price_column=price_col, **self.strategy_kwargs)
            _ = strategy.optimize(train_df)

            oos_end = min(i + self.train_step, n)
            long_lookback_ctx = self.train_lookback  # conservative
            signal_calc_start = max(0, train_end - long_lookback_ctx)
            calc_df = ohlc.iloc[signal_calc_start:oos_end]
            calc_df = self._ensure_price_column_exists(calc_df)
            oos_signals_full = strategy.generate_signals(calc_df)

            # Extract only new OOS section signals
            oos_section_len = len(ohlc.iloc[train_end:oos_end])
            oos_signals_final = oos_signals_full.iloc[-oos_section_len:].values
            wf_signals[train_end:oos_end] = oos_signals_final

        return wf_signals

    # ------------------------------------------------------------------ #
    # Parallel worker                                                    #
    # ------------------------------------------------------------------ #

    def _compute_single_perm_pf_net(self, seed: int) -> float:
        full_df = self._load_raw()
        from permutations import get_permutation  # local import for subprocess safety
        perm_df = get_permutation(full_df, start_index=self.perm_start_index, seed=seed)
        perm_df = perm_df[(perm_df.index >= self.start_date) & (perm_df.index < self.end_date)]
        perm_df = self._ensure_price_column_exists(perm_df)

        perm_signals = self._walkforward_signals(perm_df)
        perm_df["r"] = np.log(perm_df[self.price_column]).diff().shift(-1)
        perm_df["simple_r"] = perm_df[self.price_column].pct_change().shift(-1)
        perm_df["signal"] = perm_signals
        perm_df["weight"] = compute_position_weights(
            signals=pd.Series(perm_signals, index=perm_df.index),
            simple_returns=perm_df["simple_r"],
            price=perm_df[self.price_column],
            timeframe=self.timeframe,
            mode=self.position_sizing_mode,
            mode_params=self.position_sizing_params,
        )
        perm_df["strategy_r"] = perm_df["r"] * perm_df["weight"]
        perm_df["strategy_simple_r"] = perm_df["simple_r"] * perm_df["weight"]
        fee_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        turnover = (perm_df["weight"].diff().abs()).fillna(perm_df["weight"].abs())
        perm_df["cost_simple"] = fee_rate * turnover
        perm_df["strategy_simple_r_net"] = perm_df["strategy_simple_r"] - perm_df["cost_simple"]
        perm_df["strategy_r_net"] = np.log((1.0 + perm_df["strategy_simple_r_net"]).clip(lower=1e-12))
        return _profit_factor(perm_df["strategy_r_net"].dropna())

    # ------------------------------------------------------------------ #
    # Plot OOS walk-forward                                            #
    # ------------------------------------------------------------------ #

    def plot_oos_walkforward(self):
        real_df = self.get_df()
        real_df = self._ensure_price_column_exists(real_df)
        print(
            f"Calculating walk-forward signals for real data from {self.start_date} to {self.end_date} | Mode: {self.mode.upper()}"
        )

        real_signals = self._walkforward_signals(real_df)
        real_df["r"] = np.log(real_df[self.price_column]).diff().shift(-1)
        real_df["simple_r"] = real_df[self.price_column].pct_change().shift(-1)
        real_df["signal"] = real_signals
        real_df["weight"] = compute_position_weights(
            signals=pd.Series(real_signals, index=real_df.index),
            simple_returns=real_df["simple_r"],
            price=real_df[self.price_column],
            timeframe=self.timeframe,
            mode=self.position_sizing_mode,
            mode_params=self.position_sizing_params,
        )
        real_df["strategy_r"] = real_df["r"] * real_df["weight"]
        real_df["strategy_simple_r"] = real_df["simple_r"] * real_df["weight"]
        # Costs
        effective_fee_bps = float(self.fee_bps) if self.fee_bps and self.fee_bps > 0 else (4.0 if self.mode == "futures" else 10.0)
        fee_rate = (effective_fee_bps + float(self.slippage_bps)) / 10000.0
        turnover = (real_df["weight"].diff().abs()).fillna(real_df["weight"].abs())
        real_df["cost_simple"] = fee_rate * turnover
        real_df["strategy_simple_r_net"] = real_df["strategy_simple_r"] - real_df["cost_simple"]
        real_df["strategy_r_net"] = np.log((1.0 + real_df["strategy_simple_r_net"]).clip(lower=1e-12))
        # Trim to analysis window and drop trailing NaNs from shift/diff
        _start_ts = pd.to_datetime(self.start_date)
        _end_ts = pd.to_datetime(self.end_date)
        real_df = real_df[(real_df.index >= _start_ts) & (real_df.index < _end_ts)]
        real_df = real_df.dropna(subset=["r", "strategy_r"])        
        real_pf = _profit_factor(real_df["strategy_r"].dropna())
        real_pf_net = _profit_factor(real_df["strategy_r_net"].dropna())
        print(f"OOS real Profit Factor: {real_pf:.4f}")

        print("=" * 100)
        print(f"Real PF: {real_pf:.4f}")
        print("=" * 100)

        if self.generate_plot:
            plt.style.use("dark_background")
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(real_df["r"].cumsum(), label="Market Log Returns")
            ax1.plot(real_df["strategy_r"].cumsum(), label="Strategy Gross (log)")
            ax1.plot(real_df["strategy_r_net"].cumsum(), label="Strategy Net (log)")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Cumulative Log Return")
            ax1.set_title("Walk-Forward OOS (Gross vs Net)")
            ax2 = ax1.twinx()
            ax2.plot(real_df.index, real_df[self.price_column], color="gray", alpha=0.25, label="Asset Price")
            ax2.set_ylabel("Price")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.show()

    # ------------------------------------------------------------------ #
    # OOS MC permutation test                                            #
    # ------------------------------------------------------------------ #

    def run(self, save_json_dir: str | None = None):
        real_df = self.get_df()
        real_df = self._ensure_price_column_exists(real_df)
        print(
            f"Calculating walk-forward signals for real data from {self.start_date} to {self.end_date}"
        )

        real_signals = self._walkforward_signals(real_df)
        real_df["r"] = np.log(real_df[self.price_column]).diff().shift(-1)
        real_df["simple_r"] = real_df[self.price_column].pct_change().shift(-1)
        real_df["signal"] = real_signals
        real_df["weight"] = compute_position_weights(
            signals=pd.Series(real_signals, index=real_df.index),
            simple_returns=real_df["simple_r"],
            price=real_df[self.price_column],
            timeframe=self.timeframe,
            mode=self.position_sizing_mode,
            mode_params=self.position_sizing_params,
        )
        real_df["strategy_r"] = real_df["r"] * real_df["weight"]
        real_df["strategy_simple_r"] = real_df["simple_r"] * real_df["weight"]
        # Costs
        effective_fee_bps = float(self.fee_bps) if self.fee_bps and self.fee_bps > 0 else (4.0 if self.mode == "futures" else 10.0)
        fee_rate = (effective_fee_bps + float(self.slippage_bps)) / 10000.0
        turnover = (real_df["weight"].diff().abs()).fillna(real_df["weight"].abs())
        real_df["cost_simple"] = fee_rate * turnover
        real_df["strategy_simple_r_net"] = real_df["strategy_simple_r"] - real_df["cost_simple"]
        real_df["strategy_r_net"] = np.log((1.0 + real_df["strategy_simple_r_net"]).clip(lower=1e-12))
        # Trim to requested analysis window
        _start_ts = pd.to_datetime(self.start_date)
        _end_ts = pd.to_datetime(self.end_date)
        real_df = real_df[(real_df.index >= _start_ts) & (real_df.index < _end_ts)]
        real_df = real_df.dropna(subset=["r", "strategy_r"])        
        real_pf = _profit_factor(real_df["strategy_r"].dropna())
        real_pf_net = _profit_factor(real_df["strategy_r_net"].dropna())
        print(f"OOS real Profit Factor: {real_pf:.4f}")

        print("Starting OOS MC permutations (parallel)…")
        seeds = list(range(self.n_perm))
        with ProcessPoolExecutor() as executor:
            permuted_pfs_net = list(
                tqdm(
                    executor.map(
                        _compute_single_perm_pf_net_oos,
                        seeds,
                        repeat(self.asset),
                        repeat(self.timeframe),
                        repeat(self.start_date),
                        repeat(self.end_date),
                        repeat(self.price_column),
                        repeat(self.strategy_name),
                        repeat(self.strategy_kwargs),
                        repeat(self.fee_bps),
                        repeat(self.slippage_bps),
                        repeat(self.train_lookback),
                        repeat(self.train_step),
                        repeat(self.perm_start_index),
                        repeat(self.position_sizing_mode),
                        repeat(self.position_sizing_params),
                        repeat(self.mode),
                    ),
                    total=len(seeds),
                )
            )
        iperms_better = sum(1 for pf in permuted_pfs_net if pf >= real_pf_net)

        p_val = (iperms_better + 1) / (self.n_perm + 1)
        print("=" * 100)
        print(f"Walk-forward OOS MC p-value: {p_val:.4f}")
        print(f"Permutations better than real PF (net): {iperms_better}/{self.n_perm}")
        print(f"Real PF (gross): {real_pf:.4f}")
        print(f"Real PF (net)  : {real_pf_net:.4f}")
        print("=" * 100)

        if self.generate_plot:
            plt.style.use("dark_background")
            plt.figure(figsize=(10, 6))
            bins = max(10, len(permuted_pfs_net) // 3)
            plt.hist(permuted_pfs_net, bins=bins, color="blue", alpha=0.7, label="Permutations (net)", edgecolor="white")
            plt.axvline(real_pf_net, color="red", linewidth=2, label=f"Real Net (PF={real_pf_net:.2f})")
            plt.xlabel("Profit Factor")
            plt.ylabel("Frequency")
            plt.title(f"Walk-Forward OOS MC Permutations (net PF, p={p_val:.3f}, N={self.n_perm}) [{self.mode.upper()}]")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        # Save JSON report if requested
        if save_json_dir:
            os.makedirs(save_json_dir, exist_ok=True)
            report = {
                "run_type": "oos_walkforward_permutation",
                "params": {
                    "start": self.start_date,
                    "end": self.end_date,
                    "strategy": self.strategy_name,
                    "asset": self.asset,
                    "timeframe": self.timeframe,
                    "price_column": self.price_column,
                    "fee_bps": effective_fee_bps,
                    "slippage_bps": self.slippage_bps,
                    "train_lookback": self.train_lookback,
                    "train_step": self.train_step,
                    "n_perm": self.n_perm,
                    "strategy_kwargs": self.strategy_kwargs,
                    "position_sizing_mode": self.position_sizing_mode,
                    "position_sizing_params": self.position_sizing_params,
                    "instrument_mode": self.mode,
                },
                "metrics": {
                    "pf_gross": float(real_pf),
                    "pf_net": float(real_pf_net),
                    "p_value_net": float(p_val),
                },
                "series": {
                    "permutation_pfs_net": permuted_pfs_net,
                },
            }
            with open(os.path.join(save_json_dir, "oos_wf_mc.json"), "w") as f:
                json.dump(report, f)

# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Generic walk-forward MC tester.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument("--strategy", required=True, help="Strategy identifier")
    parser.add_argument("--asset", required=True, help="Asset symbol, e.g. BTCUSD")
    parser.add_argument("--tf", required=True, help="Timeframe, e.g. 1h")
    parser.add_argument("--lookback", type=int, default=24 * 365 * 4, help="Training lookback periods")
    parser.add_argument("--step", type=int, default=24 * 30, help="Training step size (periods)")
    parser.add_argument("--n_perm", type=int, default=10, help="Number of Monte Carlo permutations")
    parser.add_argument("--plot", action="store_true", help="Generate histogram plot")
    parser.add_argument("--fee_bps", type=float, default=0.0, help="Per-side fee in basis points")
    parser.add_argument("--slippage_bps", type=float, default=0.0, help="Per-side slippage in basis points")
    parser.add_argument("--save_json_dir", type=str, default=None, help="Directory to save JSON report")
    parser.add_argument("--mode", type=str, default="spot", choices=["spot","futures"], help="Instrument mode: spot or futures")
    args = parser.parse_args()

    tester = WalkForwardMCTester(
        start_date=args.start,
        end_date=args.end,
        strategy_name=args.strategy,
        asset=args.asset,
        timeframe=args.tf,
        train_lookback=args.lookback,
        train_step=args.step,
        n_perm=args.n_perm,
        generate_plot=args.plot,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        mode=args.mode,
    )
    tester.run(save_json_dir=args.save_json_dir)


if __name__ == "__main__":
    # main()

    # tester = WalkForwardMCTester(
    #     start_date="2019-01-01",
    #     end_date="2024-01-01",
    #     strategy_name="ma",
    #     asset="ETHUSD",
    #     timeframe="1h",
    #     train_lookback=24*365*4,
    #     train_step=24*30,
    #     n_perm=10,
    #     generate_plot=True,
    # )

    from is_results import ml_params

    # Derive sensible walk-forward window sizes in BARS for the chosen timeframe
    _tf = "4h"
    if _tf.endswith("m"):
        _bar_hours = int(_tf[:-1]) / 60.0
    elif _tf.endswith("h"):
        _bar_hours = float(int(_tf[:-1]))
    elif _tf.endswith("d"):
        _bar_hours = float(int(_tf[:-1]) * 24)
    else:
        _bar_hours = 1.0
    _bars_per_day = max(1, int(round(24.0 / _bar_hours)))

    _years_lb = 4
    _days_step = 30
    _lookback_bars = int(365 * _years_lb * _bars_per_day)   # e.g., 4 years of context
    _step_bars = int(_days_step * _bars_per_day)            # e.g., 30 days per step

    # ------------------------------------------------------------------ #
    # Run the tester                                                     #
    # ------------------------------------------------------------------ #

    tester = WalkForwardMCTester(
        start_date="2025-01-01",
        end_date="2025-08-01",
        strategy_name="ml",
        asset="VETUSD",
        timeframe=_tf,
        train_lookback=_lookback_bars,
        train_step=_step_bars,
        generate_plot=True,
        strategy_kwargs=ml_params,
        price_column="vwap_10",
        n_perm=20,
        fee_bps=10.0,
        slippage_bps=10.0,
        position_sizing_mode="fixed_fraction",
        position_sizing_params={
            "fraction": 0.1,
        },        
    )    
    tester.run(save_json_dir="reports/example_run")

