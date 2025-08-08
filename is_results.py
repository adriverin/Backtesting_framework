"""Utility script to visualise cumulative returns for an asset and a strategy.

The script takes similar CLI parameters as the other general testers and will:

1. Load OHLCV data for the requested asset/time-frame.
2. Optimise the specified strategy on the in-sample slice (or full slice if no
   specific split is desired).
3. Generate the strategy's signals on that slice.
4. Compute log-returns of the asset and the strategy.
5. Plot both cumulative return curves in the same figure.
6. Print summary metrics: Profit Factor, Sharpe ratio and total % return.

Example
-------

python plot_cumulative_returns.py \
    --start 2018-01-01 --end 2020-01-01 \
    --strategy ml_rsi_ema_volume --asset BTCUSD --tf 1h --plot
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import json
import os
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies import aVAILABLE_STRATEGIES, BaseStrategy  # type: ignore

# ---------------------------------------------------------------------- #
# Helper utilities                                                       #
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


# ---------------------------------------------------------------------- #
# Core plotting logic                                                    #
# ---------------------------------------------------------------------- #


def plot_cumulative_returns(
    start_date: str,
    end_date: str,
    strategy_name: str,
    asset: str,
    timeframe: str,
    price_column: str = "close",
    strategy_kwargs: dict | None = None,
    show_plot: bool = True,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    save_json_dir: str | None = None,
):
    strategy_kwargs = strategy_kwargs or {}

    if strategy_name not in aVAILABLE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available: {list(aVAILABLE_STRATEGIES)}"
        )

    # ------------------------------------------------------------------ #
    # Load data                                                          #
    # ------------------------------------------------------------------ #
    filepath = Path(f"data/ohlcv_{asset}_{timeframe}.parquet")
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    df = pd.read_parquet(filepath)
    df = df[(df.index >= start_date) & (df.index < end_date)].copy()

    if len(df) == 0:
        raise ValueError("No data in the selected date range.")

    # ------------------------------------------------------------------ #
    # Run strategy                                                       #
    # ------------------------------------------------------------------ #
    strategy_cls: Type[BaseStrategy] = aVAILABLE_STRATEGIES[strategy_name]
    strategy = strategy_cls(price_column=price_column, **strategy_kwargs)
    strategy.optimize(df)  # optimise on same slice for simplicity
    signals = strategy.generate_signals(df)

    # ------------------------------------------------------------------ #
    # Compute returns                                                    #
    # ------------------------------------------------------------------ #
    df["r"] = np.log(df[price_column]).diff().shift(-1)
    df["simple_r"] = df[price_column].pct_change().shift(-1)
    df["signal"] = signals
    # Uncomment to shuffle the signals
    # Use to test if code is working and not always providing the same results
    # df["signal"] = np.random.permutation(df["signal"].to_numpy())
    df["strategy_r"] = df["r"] * df["signal"]
    df["strategy_simple_r"] = df["simple_r"] * df["signal"]

    # Fees/slippage model (per-side bps applied on position changes)
    fee_rate = (fee_bps + slippage_bps) / 10000.0
    turnover = (df["signal"].diff().abs()).fillna(df["signal"].abs())
    df["cost_simple"] = fee_rate * turnover
    # Net simple return after fees
    df["strategy_simple_r_net"] = df["strategy_simple_r"] - df["cost_simple"]
    # Convert to log-equivalent net return via log(1 + simple)
    df["strategy_r_net"] = np.log((1.0 + df["strategy_simple_r_net"]).clip(lower=1e-12))

    # Drop NaNs created by diff/shift and any initial NaNs from indicators
    df = df.dropna(subset=["r", "strategy_r"])

    # Cumulative log and simple returns (gross and net)
    asset_cum = df["r"].cumsum()
    strat_cum_gross = df["strategy_r"].cumsum()
    strat_cum_net = df["strategy_r_net"].cumsum()
    asset_cum_simple = (1.0 + df["simple_r"]).cumprod() - 1.0
    strat_cum_simple_gross = (1.0 + df["strategy_simple_r"]).cumprod() - 1.0
    strat_cum_simple_net = (1.0 + df["strategy_simple_r_net"]).cumprod() - 1.0

    # ------------------------------------------------------------------ #
    # Metrics                                                             #
    # ------------------------------------------------------------------ #
    pf_gross = _profit_factor(df["strategy_r"])          # on log-return stream
    pf_net = _profit_factor(df["strategy_r_net"])        # on net log-return stream
    ann_factor = _annualisation_factor(timeframe)
    sharpe_gross = df["strategy_r"].mean() / df["strategy_r"].std() * ann_factor
    sharpe_net = df["strategy_r_net"].mean() / df["strategy_r_net"].std() * ann_factor
    pct_return_asset_log = asset_cum.iloc[-1] * 100
    pct_return_log_gross = strat_cum_gross.iloc[-1] * 100
    pct_return_log_net = strat_cum_net.iloc[-1] * 100
    pct_return_asset_simple = asset_cum_simple.iloc[-1] * 100
    pct_return_simple_gross = strat_cum_simple_gross.iloc[-1] * 100
    pct_return_simple_net = strat_cum_simple_net.iloc[-1] * 100

    print("=" * 80)
    print(f"Results for strategy '{strategy_name}' on {asset} {timeframe}")
    print(f"Period: {start_date} -> {end_date} | Observations: {len(df)}")
    print("-" * 80)
    print(f"Gross  Profit Factor : {pf_gross:.4f}")
    print(f"Gross  Sharpe (log)  : {sharpe_gross:.4f}")
    print(f"Gross  % Log Return  : {pct_return_log_gross:.2f}%")
    print(f"Gross  % Simple Ret  : {pct_return_simple_gross:.2f}%")
    print("-")
    print(f"Net    Profit Factor : {pf_net:.4f}   (fees={fee_bps}bps, slip={slippage_bps}bps per side)")
    print(f"Net    Sharpe (log)  : {sharpe_net:.4f}")
    print(f"Net    % Log Return  : {pct_return_log_net:.2f}%")
    print(f"Net    % Simple Ret  : {pct_return_simple_net:.2f}%")
    print("-")
    print(f"Asset  % Log Return  : {pct_return_asset_log:.2f}%")
    print(f"Asset  % Simple Ret  : {pct_return_asset_simple:.2f}%")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # Plot                                                               #
    # ------------------------------------------------------------------ #
    if show_plot:
        plt.style.use("dark_background")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        # ax1.plot(asset_cum.index, asset_cum.values, label=f"{asset} {timeframe} (tot ret {pct_return_asset_log:.2f}%)")
        # ax1.plot(strat_cum_gross.index, strat_cum_gross.values, label=f"Strategy Gross ({strategy_name})")
        # ax1.plot(strat_cum_net.index, strat_cum_net.values, label=f"Strategy Net ({strategy_name})")
        # ax1.set_ylabel("Cumulative Log Return")
        # ax1.set_xlabel("Date")
        # ax1.set_title("Cumulative Returns (Gross vs Net)")

        ax1.plot(asset_cum_simple.index, asset_cum_simple.values, label=f"{asset} {timeframe} (tot ret {pct_return_asset_simple:.2f}%)")
        ax1.plot(strat_cum_simple_gross.index, strat_cum_simple_gross.values, label=f"Strategy Gross ({strategy_name})")
        ax1.plot(strat_cum_simple_net.index, strat_cum_simple_net.values, label=f"Strategy Net ({strategy_name})")
        # ax1.set_yscale("log")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Simple Return")
        ax1.set_title("Cumulative Returns (Gross vs Net)")

        # Secondary axis for actual asset price
        # ax2 = ax1.twinx()
        # ax2.plot(df.index, df[price_column], color="gray", alpha=0.25, label="Asset Price")
        # ax2.set_ylabel("Price")

        # Add text with metrics
        ax1.text(0.02, 0.95, f"Gross PF: {pf_gross:.4f}", transform=ax1.transAxes, fontsize=10, va='top', color="white")
        ax1.text(0.02, 0.90, f"Gross Sharpe: {sharpe_gross:.4f}", transform=ax1.transAxes, fontsize=10, va='top', color="white")
        ax1.text(0.02, 0.85, f"Gross Log Ret: {pct_return_log_gross:.2f}%", transform=ax1.transAxes, fontsize=10, va='top', color="white")
        ax1.text(0.02, 0.80, f"Gross Simple Ret: {pct_return_simple_gross:.2f}%", transform=ax1.transAxes, fontsize=10, va='top', color="white")
        ax1.text(0.02, 0.75, f"Net   PF: {pf_net:.4f}", transform=ax1.transAxes, fontsize=10, va='top', color="white")
        ax1.text(0.02, 0.70, f"Net   Sharpe: {sharpe_net:.4f}", transform=ax1.transAxes, fontsize=10, va='top', color="white")
        ax1.text(0.02, 0.65, f"Net   Log Ret: {pct_return_log_net:.2f}%", transform=ax1.transAxes, fontsize=10, va='top', color="white")
        ax1.text(0.02, 0.60, f"Net   Simple Ret: {pct_return_simple_net:.2f}%", transform=ax1.transAxes, fontsize=10, va='top', color="white")

        # Combined legend from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines1 + lines2, labels1 + labels2)
        ax1.legend(lines1, labels1)
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Save JSON report                                                   #
    # ------------------------------------------------------------------ #
    if save_json_dir:
        os.makedirs(save_json_dir, exist_ok=True)

        # Rolling Sharpe on log returns over ~30 days
        tf = timeframe.lower().strip()
        if tf.endswith("m"):
            bar_hours = int(tf[:-1]) / 60.0
        elif tf.endswith("h"):
            bar_hours = float(int(tf[:-1]))
        elif tf.endswith("d"):
            bar_hours = float(int(tf[:-1]) * 24)
        else:
            bar_hours = 1.0
        window_bars = max(10, int((24 * 30) / bar_hours))

        rolling_sharpe_gross = (df["strategy_r"].rolling(window_bars).mean() /
                                 df["strategy_r"].rolling(window_bars).std()) * ann_factor
        rolling_sharpe_net = (df["strategy_r_net"].rolling(window_bars).mean() /
                               df["strategy_r_net"].rolling(window_bars).std()) * ann_factor

        # Drawdowns from simple equity
        equity_gross = (1.0 + df["strategy_simple_r"]).cumprod()
        equity_net = (1.0 + df["strategy_simple_r_net"]).cumprod()
        dd_gross = equity_gross / equity_gross.cummax() - 1.0
        dd_net = equity_net / equity_net.cummax() - 1.0

        report = {
            "run_type": "in_sample",
            "params": {
                "start": start_date,
                "end": end_date,
                "strategy": strategy_name,
                "asset": asset,
                "timeframe": timeframe,
                "price_column": price_column,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "strategy_kwargs": strategy_kwargs,
                "rolling_window_bars": window_bars,
            },
            "metrics": {
                "pf_gross": pf_gross,
                "pf_net": pf_net,
                "sharpe_gross": sharpe_gross,
                "sharpe_net": sharpe_net,
                "pct_return_asset_log": pct_return_asset_log,
                "pct_return_log_gross": pct_return_log_gross,
                "pct_return_log_net": pct_return_log_net,
                "pct_return_asset_simple": pct_return_asset_simple,
                "pct_return_simple_gross": pct_return_simple_gross,
                "pct_return_simple_net": pct_return_simple_net,
            },
            "series": {
                "timestamps": [ts.isoformat() for ts in df.index.to_pydatetime()],
                "asset": {
                    "ret_log": df["r"].fillna(0).tolist(),
                    "ret_simple": df["simple_r"].fillna(0).tolist(),
                    "cum_log": asset_cum.fillna(0).tolist(),
                    "cum_simple": asset_cum_simple.fillna(0).tolist(),
                    "price": df[price_column].fillna(0).tolist(),
                },
                "strategy": {
                    "gross": {
                        "ret_log": df["strategy_r"].fillna(0).tolist(),
                        "ret_simple": df["strategy_simple_r"].fillna(0).tolist(),
                        "cum_log": strat_cum_gross.fillna(0).tolist(),
                        "cum_simple": strat_cum_simple_gross.fillna(0).tolist(),
                        "drawdown": dd_gross.fillna(0).tolist(),
                        "rolling_sharpe": rolling_sharpe_gross.fillna(method="bfill").fillna(0).tolist(),
                    },
                    "net": {
                        "ret_log": df["strategy_r_net"].fillna(0).tolist(),
                        "ret_simple": df["strategy_simple_r_net"].fillna(0).tolist(),
                        "cum_log": strat_cum_net.fillna(0).tolist(),
                        "cum_simple": strat_cum_simple_net.fillna(0).tolist(),
                        "drawdown": dd_net.fillna(0).tolist(),
                        "rolling_sharpe": rolling_sharpe_net.fillna(method="bfill").fillna(0).tolist(),
                    },
                },
            },
        }

        with open(os.path.join(save_json_dir, "is.json"), "w") as f:
            json.dump(report, f)


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Plot cumulative returns of asset and strategy.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--strategy", required=True, help="Strategy identifier")
    parser.add_argument("--asset", required=True, help="Asset symbol e.g. BTCUSD")
    parser.add_argument("--tf", required=True, help="Timeframe e.g. 1h")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting (only print metrics)")
    parser.add_argument("--fee_bps", type=float, default=0.0, help="Per-side fee in basis points")
    parser.add_argument("--slippage_bps", type=float, default=0.0, help="Per-side slippage in basis points")
    parser.add_argument("--save_json_dir", type=str, default=None, help="Directory to save JSON report")
    args = parser.parse_args()

    plot_cumulative_returns(
        start_date=args.start,
        end_date=args.end,
        strategy_name=args.strategy,
        asset=args.asset,
        timeframe=args.tf,
        show_plot=not args.no_plot,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        save_json_dir=args.save_json_dir,
    )


if __name__ == "__main__":
    # main()

    ml_params = {
        "interval": "1h",
        "forecast_horizon_hours": 1,
        "n_epochs": 300,
        "hidden_sizes": (128, 64, 32, 16),
        "signal_percentiles": (2, 98),
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "early_stopping_patience": 10,
        "lr": 5e-5,
        "weight_decay": 0.001,
        "batch_size": 128,
        "random_seed": 42,
    }


    plot_cumulative_returns(
        start_date="2019-01-01",
        end_date="2024-01-01",
        # strategy_name="ma",
        strategy_name="ml",
        asset="BTCUSD",
        timeframe="1h",
        show_plot=True,
        strategy_kwargs=ml_params,
        price_column="close",
        fee_bps=10.0,
        slippage_bps=2.0,
        save_json_dir="reports/example_run",
    )