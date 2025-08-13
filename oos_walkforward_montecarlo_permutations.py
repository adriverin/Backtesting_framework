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
    if col == "vwap":
        required = ["high", "low", "close", "volume"]
        if all(c in df.columns for c in required):
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
) -> float:
    full_fp = Path(f"data/ohlcv_{asset}_{timeframe}.parquet")
    full_df = pd.read_parquet(full_fp)
    from permutations import get_permutation  # local import for subprocess
    perm_df = get_permutation(full_df, start_index=perm_start_index, seed=seed)
    perm_df = perm_df[(perm_df.index >= start_date) & (perm_df.index < end_date)]
    perm_df = _ensure_price_column_exists_oos(perm_df, price_column)

    perm_signals = _walkforward_signals_oos(
        perm_df,
        price_column,
        strategy_name,
        strategy_kwargs,
        train_lookback,
        train_step,
    )

    perm_df["r"] = np.log(perm_df[price_column]).diff().shift(-1)
    perm_df["simple_r"] = perm_df[price_column].pct_change().shift(-1)
    perm_df["signal"] = perm_signals
    perm_df["strategy_r"] = perm_df["r"] * perm_df["signal"]
    perm_df["strategy_simple_r"] = perm_df["simple_r"] * perm_df["signal"]
    fee_rate = (fee_bps + slippage_bps) / 10000.0
    turnover = (perm_df["signal"].diff().abs()).fillna(perm_df["signal"].abs())
    perm_df["cost_simple"] = fee_rate * turnover
    perm_df["strategy_simple_r_net"] = perm_df["strategy_simple_r"] - perm_df["cost_simple"]
    perm_df["strategy_r_net"] = np.log((1.0 + perm_df["strategy_simple_r_net"]).clip(lower=1e-12))
    return _profit_factor(perm_df["strategy_r_net"].dropna())


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

        self.perm_start_index = self._get_perm_start_index(start_date)
        print(f"Permutation start index: {self.perm_start_index}")

    # ------------------------------------------------------------------ #
    # Data helpers                                                       #
    # ------------------------------------------------------------------ #

    def _get_full_filepath(self) -> Path:
        return Path(f"data/ohlcv_{self.asset}_{self.timeframe}.parquet")

    def _load_raw(self) -> pd.DataFrame:
        return pd.read_parquet(self._get_full_filepath())

    def get_df(self) -> pd.DataFrame:
        df = self._load_raw()
        return df[(df.index >= self.start_date) & (df.index < self.end_date)]

    def _ensure_price_column_exists(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the requested price column exists, creating common derived ones if needed.

        Supports:
        - 'median'   = (high + low) / 2
        - 'typical'  = (high + low + close) / 3
        - 'vwap'     = rolling VWAP over 20 bars (requires volume)
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
        if col == "vwap":
            required = ["high", "low", "close", "volume"]
            if all(c in df.columns for c in required):
                vwap_window = 20
                typical_price = (df["high"] + df["low"] + df["close"]) / 3
                tpv = typical_price * df["volume"]
                cumulative_tpv = tpv.rolling(window=vwap_window, min_periods=1).sum()
                cumulative_volume = df["volume"].rolling(window=vwap_window, min_periods=1).sum()
                df[col] = (cumulative_tpv / cumulative_volume).fillna(method="ffill")
                print(f"--- Created rolling 'vwap' ({vwap_window}-bar) column on-the-fly. ---")
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
        start_date_parsed = pd.to_datetime(start_date, utc=True)
        analysis_start_idx = full_df.index.get_loc(start_date_parsed)
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
        perm_df["strategy_r"] = perm_df["r"] * perm_df["signal"]
        perm_df["strategy_simple_r"] = perm_df["simple_r"] * perm_df["signal"]
        fee_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        turnover = (perm_df["signal"].diff().abs()).fillna(perm_df["signal"].abs())
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
            f"Calculating walk-forward signals for real data from {self.start_date} to {self.end_date}"
        )

        real_signals = self._walkforward_signals(real_df)
        real_df["r"] = np.log(real_df[self.price_column]).diff().shift(-1)
        real_df["simple_r"] = real_df[self.price_column].pct_change().shift(-1)
        real_df["signal"] = real_signals
        real_df["strategy_r"] = real_df["r"] * real_df["signal"]
        real_df["strategy_simple_r"] = real_df["simple_r"] * real_df["signal"]
        # Costs
        fee_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        turnover = (real_df["signal"].diff().abs()).fillna(real_df["signal"].abs())
        real_df["cost_simple"] = fee_rate * turnover
        real_df["strategy_simple_r_net"] = real_df["strategy_simple_r"] - real_df["cost_simple"]
        real_df["strategy_r_net"] = np.log((1.0 + real_df["strategy_simple_r_net"]).clip(lower=1e-12))
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
        print(
            f"Calculating walk-forward signals for real data from {self.start_date} to {self.end_date}"
        )

        real_signals = self._walkforward_signals(real_df)
        real_df["r"] = np.log(real_df[self.price_column]).diff().shift(-1)
        real_df["simple_r"] = real_df[self.price_column].pct_change().shift(-1)
        real_df["signal"] = real_signals
        real_df["strategy_r"] = real_df["r"] * real_df["signal"]
        real_df["strategy_simple_r"] = real_df["simple_r"] * real_df["signal"]
        # Costs
        fee_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        turnover = (real_df["signal"].diff().abs()).fillna(real_df["signal"].abs())
        real_df["cost_simple"] = fee_rate * turnover
        real_df["strategy_simple_r_net"] = real_df["strategy_simple_r"] - real_df["cost_simple"]
        real_df["strategy_r_net"] = np.log((1.0 + real_df["strategy_simple_r_net"]).clip(lower=1e-12))
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
            plt.title(f"Walk-Forward OOS MC Permutations (net PF, p={p_val:.3f}, N={self.n_perm})")
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
                    "fee_bps": self.fee_bps,
                    "slippage_bps": self.slippage_bps,
                    "train_lookback": self.train_lookback,
                    "train_step": self.train_step,
                    "n_perm": self.n_perm,
                    "strategy_kwargs": self.strategy_kwargs,
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
    )
    tester.run(save_json_dir=args.save_json_dir)


if __name__ == "__main__":
    # main()

    tester = WalkForwardMCTester(
        start_date="2019-01-01",
        end_date="2024-01-01",
        strategy_name="ma",
        asset="ETHUSD",
        timeframe="1h",
        train_lookback=24*365*4,
        train_step=24*30,
        n_perm=10,
        generate_plot=True,
    )
    tester.plot_oos_walkforward()

