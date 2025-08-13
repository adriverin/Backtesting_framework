"""Generic in-sample Monte-Carlo permutation test that supports arbitrary strategies.

This is a refactor of *is_testing.py* allowing any strategy that conforms to
*BaseStrategy* to be plugged in.

Usage example from the CLI::

    python is_testing_general.py --start 2018-01-01 --end 2020-01-01 \
        --strategy ma --asset BTCUSD --tf 1h --n_perm 100 --plot

Available strategy identifiers are defined inside *strategies/__init__.py*.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
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
# Helper functions                                                       #
# ---------------------------------------------------------------------- #


def _profit_factor(returns: pd.Series) -> float:
    positive_sum = returns[returns > 0].sum()
    negative_sum = returns[returns < 0].abs().sum()
    if negative_sum == 0:
        return float("inf") if positive_sum > 0 else 0.0
    return positive_sum / negative_sum

def _annualisation_factor(timeframe: str) -> float:
    """Return the sqrt annualisation factor for the given timeframe string."""
    tf = timeframe.lower().strip()
    minutes_per_unit = 1
    if tf.endswith("m"):
        minutes_per_unit = int(tf[:-1])
    elif tf.endswith("h"):
        minutes_per_unit = int(tf[:-1]) * 60
    elif tf.endswith("d"):
        minutes_per_unit = int(tf[:-1]) * 60 * 24
    else:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    periods_per_year = (365 * 24 * 60) / minutes_per_unit
    return math.sqrt(periods_per_year)


def _ensure_price_column_exists(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """Ensure the requested price column exists, creating common derived ones if needed.

    Supports:
    - 'median'   = (high + low) / 2
    - 'typical'  = (high + low + close) / 3
    - 'vwap'     = rolling VWAP over 20 bars (requires volume)
    """
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

# ---------------------------------------------------------------------- #
# Parallel worker                                                         #
# ---------------------------------------------------------------------- #

def _compute_single_perm_pf_net(
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
    perm_start_index: int,
) -> float:
    full_fp = Path(f"data/ohlcv_{asset}_{timeframe}.parquet")
    full_df = pd.read_parquet(full_fp)
    from permutations import get_permutation  # local import for subprocess

    perm_df = get_permutation(full_df, start_index=perm_start_index, seed=seed)
    perm_df = perm_df[(perm_df.index >= start_date) & (perm_df.index < end_date)]
    perm_df = _ensure_price_column_exists(perm_df, price_column)

    strategy = aVAILABLE_STRATEGIES[strategy_name](price_column=price_column, **strategy_kwargs)
    _ = strategy.optimize(perm_df)
    perm_signals = strategy.generate_signals(perm_df)

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
# Config / Runner class                                                  #
# ---------------------------------------------------------------------- #


class InSampleMCTester:
    """Run in-sample Monte-Carlo permutation test for any given strategy."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        strategy_name: str,
        asset: str,
        timeframe: str,
        n_perm: int = 100,
        price_column: str = "close",
        generate_plot: bool = False,
        perm_start_index: int = 0,
        strategy_kwargs: dict | None = None,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_name = strategy_name
        self.asset = asset
        self.timeframe = timeframe
        self.n_perm = n_perm
        self.price_column = price_column
        self.generate_plot = generate_plot
        self.perm_start_index = perm_start_index
        self.strategy_kwargs = strategy_kwargs or {}
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.save_json_dir: str | None = None

        if strategy_name not in aVAILABLE_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {list(aVAILABLE_STRATEGIES)}"
            )

        strategy_cls: Type[BaseStrategy] = aVAILABLE_STRATEGIES[strategy_name]
        self.strategy: BaseStrategy = strategy_cls(price_column=price_column, **self.strategy_kwargs)

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

    def _get_perm_df(self) -> pd.DataFrame:
        full_df = self._load_raw()
        perm_df = get_permutation(full_df, start_index=self.perm_start_index)
        perm_df = perm_df[(perm_df.index >= self.start_date) & (perm_df.index < self.end_date)]
        return perm_df

    # ------------------------------------------------------------------ #
    # Main logic                                                         #
    # ------------------------------------------------------------------ #

    def run(self, save_json_dir: str | None = None):
        self.save_json_dir = save_json_dir
        train_df = self.get_df()
        train_df = _ensure_price_column_exists(train_df, self.price_column)
        print("=" * 100)
        print(f"Optimising strategy '{self.strategy_name}' on in-sample data")
        print(f"from {self.start_date} to {self.end_date}")
        print("=" * 100, "\n")

        # _, best_real_pf = self.strategy.optimize(train_df)
        # print(f"In-sample PF: {best_real_pf:.4f}")

        _, opt_metric = self.strategy.optimize(train_df)
        print(f"Strategy optimization complete. (Internal metric: {opt_metric:.4f})")
        
        # 2. NOW, generate signals on the training data using the optimized strategy.
        real_signals = self.strategy.generate_signals(train_df)

        # 3. Calculate the true in-sample profit factor (gross and net).
        train_df["r"] = np.log(train_df[self.price_column]).diff().shift(-1)
        train_df["simple_r"] = train_df[self.price_column].pct_change().shift(-1)
        train_df["signal"] = real_signals
        train_df["strategy_r"] = train_df["r"] * train_df["signal"]
        train_df["strategy_simple_r"] = train_df["simple_r"] * train_df["signal"]

        # Fees and slippage
        fee_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        turnover = (train_df["signal"].diff().abs()).fillna(train_df["signal"].abs())
        train_df["cost_simple"] = fee_rate * turnover
        train_df["strategy_simple_r_net"] = train_df["strategy_simple_r"] - train_df["cost_simple"]
        train_df["strategy_r_net"] = np.log((1.0 + train_df["strategy_simple_r_net"]).clip(lower=1e-12))

        best_real_pf_gross = _profit_factor(train_df["strategy_r"].dropna())
        best_real_pf_net = _profit_factor(train_df["strategy_r_net"].dropna())
        
        # This is the value we will compare against.
        print(f"Calculated In-Sample PF (gross): {best_real_pf_gross:.4f}")
        print(f"Calculated In-Sample PF (net)  : {best_real_pf_net:.4f}  (fees={self.fee_bps}bps, slip={self.slippage_bps}bps per side)")


        print("Running Monte-Carlo permutations (parallel)…")
        seeds = list(range(1, self.n_perm))
        with ProcessPoolExecutor() as executor:
            permuted_pfs_net = list(
                tqdm(
                    executor.map(
                        _compute_single_perm_pf_net,
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
                        repeat(self.perm_start_index),
                    ),
                    total=len(seeds),
                )
            )

        iperms_better = 1 + sum(1 for pf in permuted_pfs_net if pf >= best_real_pf_net)

        p_val = iperms_better / self.n_perm
        print(f"In-sample MC p-value: {p_val:.4f}")
        print(f"Number of permutations better than real PF (net): {iperms_better-1}/{self.n_perm}")

        if self.generate_plot:
            print("Generating histogram of permutation PFs (net) and time series plot …")
            plt.style.use("dark_background")
            # Histogram (net PF)
            plt.figure(figsize=(8, 4))
            pd.Series(permuted_pfs_net).hist(color="blue", label="Permutations (net)")
            plt.axvline(best_real_pf_net, color="red", label=f"Real (net) = {best_real_pf_net:.2f}")
            plt.xlabel("Profit Factor")
            plt.ylabel("Frequency")
            plt.title(f"In-sample MC Permutations (net PF, p-value: {p_val:.3f}, N={self.n_perm})")
            plt.legend()
            plt.tight_layout()

            # Time-series: gross vs net + price
            # Recompute cumulative series on the (already prepared) train_df
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(train_df["r"].cumsum(), label="Market Log Returns")
            ax1.plot(train_df["strategy_r"].cumsum(), label="Strategy Gross (log)")
            ax1.plot(train_df["strategy_r_net"].cumsum(), label="Strategy Net (log)")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Cumulative Log Return")
            ax1.set_title("In-Sample Cumulative Returns (Gross vs Net)")
            ax2 = ax1.twinx()
            ax2.plot(train_df.index, train_df[self.price_column], color="gray", alpha=0.25, label="Asset Price")
            ax2.set_ylabel("Price")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
            ax1.grid(alpha=0.3)
            fig.tight_layout()
            plt.show()

        # Save JSON report if requested
        if self.save_json_dir:
            os.makedirs(self.save_json_dir, exist_ok=True)
            report = {
                "run_type": "in_sample_permutation",
                "params": {
                    "start": self.start_date,
                    "end": self.end_date,
                    "strategy": self.strategy_name,
                    "asset": self.asset,
                    "timeframe": self.timeframe,
                    "price_column": self.price_column,
                    "fee_bps": self.fee_bps,
                    "slippage_bps": self.slippage_bps,
                    "n_perm": self.n_perm,
                    "strategy_kwargs": self.strategy_kwargs,
                },
                "metrics": {
                    "pf_gross": float(best_real_pf_gross),
                    "pf_net": float(best_real_pf_net),
                    "p_value_net": float(p_val),
                },
                "series": {
                    "timestamps": [ts.isoformat() for ts in train_df.index.to_pydatetime()],
                    "strategy": {
                        "gross": {
                            "ret_log": train_df["strategy_r"].fillna(0).tolist(),
                            "cum_log": train_df["strategy_r"].cumsum().fillna(0).tolist(),
                        },
                        "net": {
                            "ret_log": train_df["strategy_r_net"].fillna(0).tolist(),
                            "cum_log": train_df["strategy_r_net"].cumsum().fillna(0).tolist(),
                        },
                    },
                    "permutation_pfs_net": permuted_pfs_net,
                },
            }
            with open(os.path.join(self.save_json_dir, "is_mc.json"), "w") as f:
                json.dump(report, f)


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Generic in-sample MC tester.")
    parser.add_argument("--start", dest="start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", dest="end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", dest="strategy", required=True, help="Strategy identifier")
    parser.add_argument("--asset", dest="asset", required=True, help="Asset symbol, e.g. BTCUSD")
    parser.add_argument("--tf", dest="tf", required=True, help="Timeframe, e.g. 1h")
    parser.add_argument("--n_perm", dest="n_perm", type=int, default=100, help="Number of permutations")
    parser.add_argument("--plot", action="store_true", help="Generate histogram plot")
    parser.add_argument("--fee_bps", type=float, default=0.0, help="Per-side fee in basis points")
    parser.add_argument("--slippage_bps", type=float, default=0.0, help="Per-side slippage in basis points")
    parser.add_argument("--save_json_dir", type=str, default=None, help="Directory to save JSON report")
    args = parser.parse_args()

    tester = InSampleMCTester(
        start_date=args.start,
        end_date=args.end,
        strategy_name=args.strategy,
        asset=args.asset,
        timeframe=args.tf,
        n_perm=args.n_perm,
        generate_plot=args.plot,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    tester.run(save_json_dir=args.save_json_dir)


if __name__ == "__main__":
    # main()

    ml_params = {
        "interval": "4h",
        "forecast_horizon_hours": 4,
        "n_epochs": 150,
        "hidden_sizes": (256, 128, 64, 32),
        "signal_percentiles": (10, 90),
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "early_stopping_patience": 10,
        "lr": 5e-5,
        "weight_decay": 0.001,
        "batch_size": 128,
        "random_seed": 42,
        "hold_until_opposite": True,
        #
        "sma_windows": (4, 8, 16, 32),
        "volatility_windows": (4, 8, 16, 32),
        "momentum_windows": (4, 8, 16, 32, 64),
        "rsi_windows": (4, 8, 16, 32, 64),
        # "sma_windows": (5, 10, 20, 30),
        # "volatility_windows": (5, 10, 20, 30),
        # "momentum_windows": (7, 14, 21, 30),
        # "rsi_windows": (7, 14, 21, 30),        
    }

    tester = InSampleMCTester(
        start_date="2018-07-25",
        end_date="2024-10-31",
        strategy_name="ml",
        asset="VETUSD",
        timeframe="4h",
        n_perm=100,
        generate_plot=True,
        strategy_kwargs=ml_params,
        price_column="vwap",
        fee_bps=10.0,
        slippage_bps=10.0,
    )
    tester.run(save_json_dir="reports/example_run")
