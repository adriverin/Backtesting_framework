"""Generic walk-forward out-of-sample Monte-Carlo permutation tester.

Refactors *walkforward_test.py* to support any strategy that implements the
*BaseStrategy* interface.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

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

def _annualisation_factor(timeframe: str) -> float:
    """Return the sqrt annualisation factor for the given timeframe string."""
    tf = timeframe.lower().strip()
    units_per_year = 1
    if tf.endswith("m"):
        units_per_year = int(tf[:-1]) / (365 * 24 * 60)
    elif tf.endswith("h"):
        units_per_year = int(tf[:-1]) / (365 * 24)     
    elif tf.endswith("d"):
        units_per_year = int(tf[:-1]) / (365)        
    else:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    return units_per_year


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
# Main tester class                                                       #
# ---------------------------------------------------------------------- #

class WalkForward:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        strategy_name: str,
        asset: str,
        timeframe: str,
        train_lookback: int,
        train_step: int,
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
        self.price_column = price_column
        self.generate_plot = generate_plot
        self.strategy_kwargs = strategy_kwargs or {}
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

        if strategy_name not in aVAILABLE_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {list(aVAILABLE_STRATEGIES)}"
            )
        self.strategy_cls: Type[BaseStrategy] = aVAILABLE_STRATEGIES[strategy_name]




    # ------------------------------------------------------------------ #
    # Data helpers                                                       #
    # ------------------------------------------------------------------ #

    def _get_full_filepath(self) -> Path:
        return Path(f"data/ohlcv_{self.asset}_{self.timeframe}.parquet")

    def _load_raw(self) -> pd.DataFrame:
        return pd.read_parquet(self._get_full_filepath())

    def get_df(self) -> pd.DataFrame:
        # df = self._load_raw()
        # return df[(df.index >= self.start_date) & (df.index < self.end_date)]
        full_df = self._load_raw()
        analysis_start = full_df.index.get_loc(pd.to_datetime(self.start_date, utc=True))
        analysis_end = full_df.index.get_loc(pd.to_datetime(self.end_date, utc=True))
        slice_start = max(0, analysis_start - self.train_lookback)
        return full_df.iloc[slice_start:analysis_end]



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

            # Print the iteration date
            iteration_date = ohlc.index[train_end].strftime("%Y-%m-%d")
            # iteration_date_end = ohlc.index[train_end + self.train_step].strftime("%Y-%m-%d")
            print("")
            print(f"Processing iteration for date starting on {iteration_date}")

            # Fresh strategy instance for each iteration to avoid leakage
            strategy: BaseStrategy = self.strategy_cls(price_column=price_col, **self.strategy_kwargs)
            _ = strategy.optimize(train_df)

            oos_end = min(i + self.train_step, n)
            long_lookback_ctx = self.train_lookback  # conservative
            signal_calc_start = max(0, train_end - long_lookback_ctx)
            calc_df = ohlc.iloc[signal_calc_start:oos_end]
            calc_df = _ensure_price_column_exists(calc_df, price_col)
            oos_signals_full = strategy.generate_signals(calc_df)

            # Extract only new OOS section signals
            oos_section_len = len(ohlc.iloc[train_end:oos_end])
            oos_signals_final = oos_signals_full.iloc[-oos_section_len:].values
            wf_signals[train_end:oos_end] = oos_signals_final

        return wf_signals

    # ------------------------------------------------------------------ #
    # Plot OOS walk-forward                                            #
    # ------------------------------------------------------------------ #

    def run(self, save_json_dir: str | None = None):
        real_df = self.get_df()
        real_df = _ensure_price_column_exists(real_df, self.price_column)
        ann_factor = _annualisation_factor(self.timeframe)
        print("")
        print("="*100)
        print(
            f"Calculating walk-forward signals for real data from {self.start_date} to {self.end_date}"
        )
        print(f"Train lookback: {self.train_lookback} periods ({self.train_lookback * ann_factor} years)")
        print(f"Train step: {self.train_step} periods ({self.train_step / 24} days)")
        print("="*100)
        print("")
        
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

        real_df = real_df[(real_df.index >= self.start_date) & (real_df.index < self.end_date)]

        real_pf_gross = _profit_factor(real_df["strategy_r"].dropna())
        real_pf_net = _profit_factor(real_df["strategy_r_net"].dropna())

        # Win/loss stats per trade (aggregate contiguous non-zero positions)
        pos = pd.Series(real_df["signal"]).fillna(0)
        seg_id = (pos != pos.shift()).cumsum()
        mask = pos != 0
        if mask.any():
            trade_returns = (
                real_df.loc[mask]
                       .groupby(seg_id[mask])["strategy_simple_r_net"]
                       .apply(lambda g: float((1.0 + g).prod() - 1.0))
            )
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]
            denom = int(len(trade_returns))
            win_rate_net_pct = float(len(wins) / denom * 100.0) if denom > 0 else float("nan")
            avg_win_net_pct = float(wins.mean() * 100.0) if len(wins) > 0 else float("nan")
            avg_loss_net_pct = float(losses.mean() * 100.0) if len(losses) > 0 else float("nan")
        else:
            win_rate_net_pct = float("nan")
            avg_win_net_pct = float("nan")
            avg_loss_net_pct = float("nan")
        print(f"OOS real Profit Factor (gross): {real_pf_gross:.4f}")
        print(f"OOS real Profit Factor (net)  : {real_pf_net:.4f}  (fees={self.fee_bps}bps, slip={self.slippage_bps}bps per side)")

        print("=" * 100)
        print(f"Real PF (gross): {real_pf_gross:.4f}")
        print(f"Real PF (net)  : {real_pf_net:.4f}")
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

        # Save JSON report if requested
        if save_json_dir:
            os.makedirs(save_json_dir, exist_ok=True)

            # Rolling Sharpe (30-day equivalent) on log returns
            tf = self.timeframe.lower().strip()
            if tf.endswith("m"):
                bar_hours = int(tf[:-1]) / 60.0
            elif tf.endswith("h"):
                bar_hours = float(int(tf[:-1]))
            elif tf.endswith("d"):
                bar_hours = float(int(tf[:-1]) * 24)
            else:
                bar_hours = 1.0
            window_bars = max(10, int((24 * 30) / bar_hours))
            rolling_sharpe_gross = (real_df["strategy_r"].rolling(window_bars).mean() /
                                     real_df["strategy_r"].rolling(window_bars).std())
            rolling_sharpe_net = (real_df["strategy_r_net"].rolling(window_bars).mean() /
                                   real_df["strategy_r_net"].rolling(window_bars).std())

            # Drawdowns from simple equity
            equity_gross = (1.0 + real_df["strategy_simple_r"]).cumprod()
            equity_net = (1.0 + real_df["strategy_simple_r_net"]).cumprod()
            dd_gross = equity_gross / equity_gross.cummax() - 1.0
            dd_net = equity_net / equity_net.cummax() - 1.0

            equity_gross = (1.0 + real_df["strategy_simple_r"]).cumprod()
            equity_net = (1.0 + real_df["strategy_simple_r_net"]).cumprod()

            report = {
                "run_type": "oos_walkforward",
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
                    "strategy_kwargs": self.strategy_kwargs,
                },
                "metrics": {
                    "pf_gross": float(real_pf_gross),
                    "pf_net": float(real_pf_net),
                    "win_rate_net_pct": win_rate_net_pct,
                    "avg_win_net_pct": avg_win_net_pct,
                    "avg_loss_net_pct": avg_loss_net_pct,
                },
                "series": {
                    "timestamps": [ts.isoformat() for ts in real_df.index.to_pydatetime()],
                    "asset": {
                        "ret_log": real_df["r"].fillna(0).tolist(),
                        "price": real_df[self.price_column].fillna(0).tolist(),
                    },
                    "strategy": {
                        "gross": {
                            "ret_log": real_df["strategy_r"].fillna(0).tolist(),
                            "cum_log": real_df["strategy_r"].cumsum().fillna(0).tolist(),
                            "equity": equity_gross.fillna(1).clip(lower=0).tolist(),
                            "drawdown": dd_gross.fillna(0).tolist(),
                            "rolling_sharpe": rolling_sharpe_gross.fillna(method="bfill").fillna(0).tolist(),
                        },
                        "net": {
                            "ret_log": real_df["strategy_r_net"].fillna(0).tolist(),
                            "cum_log": real_df["strategy_r_net"].cumsum().fillna(0).tolist(),
                            "equity": equity_net.fillna(1).clip(lower=0).tolist(),
                            "drawdown": dd_net.fillna(0).tolist(),
                            "rolling_sharpe": rolling_sharpe_net.fillna(method="bfill").fillna(0).tolist(),
                        },
                    },
                },
            }
            with open(os.path.join(save_json_dir, "oos_wf.json"), "w") as f:
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
    parser.add_argument("--fee_bps", type=float, default=0.0, help="Per-side fee in basis points")
    parser.add_argument("--slippage_bps", type=float, default=0.0, help="Per-side slippage in basis points")
    parser.add_argument("--save_json_dir", type=str, default=None, help="Directory to save JSON report")
    args = parser.parse_args()

    tester = WalkForward(
        start_date=args.start,
        end_date=args.end,
        strategy_name=args.strategy,
        asset=args.asset,
        timeframe=args.tf,
        train_lookback=args.lookback,
        train_step=args.step,
        generate_plot=args.plot,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    tester.run(save_json_dir=args.save_json_dir)


if __name__ == "__main__":
    # main()

    ml_params = {
        # "interval": "1h",
        "forecast_horizon_hours": 1,
        "n_epochs": 300,
        "hidden_sizes": (128, 64, 32, 16),
        "signal_percentiles": (10, 90),
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "early_stopping_patience": 10,
        "lr": 5e-5,
        "weight_decay": 0.001,
        "batch_size": 128,
        "random_seed": 42,
    }

    tester = WalkForward(
        start_date="2025-01-01",
        end_date="2025-07-01",
        strategy_name="ml",
        asset="VETUSD",
        timeframe="1h",
        train_lookback=24*365*4,
        train_step=24*30,
        generate_plot=True,
        strategy_kwargs=ml_params,
        fee_bps=2.0,
        slippage_bps=1.0,
    )
    tester.run()

