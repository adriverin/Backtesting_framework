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
from position_sizing import compute_position_weights

# ---------------------------------------------------------------------- #
# Helper utilities                                                       #
# ---------------------------------------------------------------------- #

def _profit_factor(returns: pd.Series) -> float:
    positive_sum = returns[returns > 0].sum()
    negative_sum = returns[returns < 0].abs().sum()
    if negative_sum == 0:
        return float("inf") if positive_sum > 0 else 0.0
    return positive_sum / negative_sum


def _json_safe(obj):
    """Recursively convert an object so it can be serialized to strict JSON.

    - Replaces NaN/Infinity with None
    - Converts numpy/pandas scalars and arrays to native Python types
    - Leaves basic Python types intact
    """
    import numpy as _np  # local import to avoid polluting module namespace
    import math as _math

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, (float,)):
        return obj if _math.isfinite(obj) else None
    if isinstance(obj, (_np.floating,)):
        val = float(obj)
        return val if _math.isfinite(val) else None
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # pandas Series / numpy arrays
    if hasattr(obj, 'tolist'):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    # Fallback: stringify unknown types
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


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
    - 'median'     = (high + low) / 2
    - 'typical'    = (high + low + close) / 3
    - 'vwap'       = rolling VWAP over 20 bars (if volume available)
    - 'vwap_N'     = rolling VWAP over N bars (e.g. 'vwap_50', 'vwap50')
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

    # VWAP with optional custom window via names like 'vwap', 'vwap_50', or 'vwap50'
    if col.startswith("vwap"):
        required = ["high", "low", "close", "volume"]
        if all(c in df.columns for c in required):
            # Parse optional window suffix
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

    # If unrecognised derived column, leave as error to surface quickly
    raise KeyError(f"Price column '{price_column}' not found in DataFrame and cannot be derived.")


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
    position_sizing_mode: str = "full_notional",
    position_sizing_params: dict | None = None,
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
    # Normalize index to UTC-naive for robust comparisons
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[(df.index >= start_date) & (df.index < end_date)].copy()

    if len(df) == 0:
        raise ValueError("No data in the selected date range.")

    # Ensure the requested price column exists for downstream calculations
    df = _ensure_price_column_exists(df, price_column)

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
    # Position sizing -> portfolio weight
    df["weight"] = compute_position_weights(
        signals=df["signal"],
        simple_returns=df["simple_r"],
        price=df[price_column],
        timeframe=timeframe,
        mode=position_sizing_mode,
        mode_params=position_sizing_params,
    )
    # Uncomment to shuffle the signals
    # Use to test if code is working and not always providing the same results
    # df["signal"] = np.random.permutation(df["signal"].to_numpy())
    df["strategy_r"] = df["r"] * df["weight"]
    df["strategy_simple_r"] = df["simple_r"] * df["weight"]

    # Fees/slippage model (per-side bps applied on position changes)
    fee_rate = (fee_bps + slippage_bps) / 10000.0
    turnover = (df["weight"].diff().abs()).fillna(df["weight"].abs())
    df["turnover"] = turnover
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

    # Risk metrics (per-bar, on net simple return stream)
    net_simple_ser = df["strategy_simple_r_net"].dropna()
    net_simple = net_simple_ser.values
    if len(net_simple) > 0:
        q05 = float(np.quantile(net_simple, 0.05))
        var95_simple_net_pct = float(-q05 * 100.0)
        cvar95_simple_net_pct = float(-np.mean(net_simple[net_simple <= q05]) * 100.0)
    else:
        var95_simple_net_pct = float("nan")
        cvar95_simple_net_pct = float("nan")

    # Win/loss stats per trade (aggregate contiguous non-zero positions)
    pos = df["weight"].fillna(0)
    seg_id = (pos != pos.shift()).cumsum()
    mask = pos != 0
    if mask.any():
        trade_returns = (
            df.loc[mask]
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

    # Diagnostics for costs/turnover
    # Estimate bars per day from timeframe
    tf = timeframe.lower().strip()
    if tf.endswith("m"):
        bar_hours = int(tf[:-1]) / 60.0
    elif tf.endswith("h"):
        bar_hours = float(int(tf[:-1]))
    elif tf.endswith("d"):
        bar_hours = float(int(tf[:-1]) * 24)
    else:
        bar_hours = 1.0
    bars_per_day = max(1.0, 24.0 / bar_hours)
    avg_turnover_per_bar = float(df["turnover"].mean()) if len(df) else 0.0
    avg_turnover_per_day = avg_turnover_per_bar * bars_per_day
    total_cost_simple = float(df["cost_simple"].sum())
    total_turnover = float(df["turnover"].sum())
    breakeven_fee_rate = float(df["strategy_simple_r"].sum() / total_turnover) if total_turnover > 0 else float("nan")
    breakeven_fee_bps = breakeven_fee_rate * 10000.0

    print("=" * 80)
    print(f"Results for strategy '{strategy_name}' on {asset} {timeframe}")
    print(f"Period: {start_date} -> {end_date} | Observations: {len(df)}")
    print("-" * 80)
    print(f"Gross  Profit Factor : {pf_gross:.4f}")
    print(f"Gross  Sharpe (log)  : {sharpe_gross:.4f}")
    # print(f"Gross  % Log Return  : {pct_return_log_gross:.2f}%")
    print(f"Gross  % Simple Ret  : {pct_return_simple_gross:.2f}%")
    print("-")
    print(f"Net    Profit Factor : {pf_net:.4f}   (fees={fee_bps}bps, slip={slippage_bps}bps per side)")
    print(f"Net    Sharpe (log)  : {sharpe_net:.4f}")
    # print(f"Net    % Log Return  : {pct_return_log_net:.2f}%")
    print(f"Net    % Simple Ret  : {pct_return_simple_net:.2f}%")
    print("-")
    print(f"Diagnostics: avg turnover/bar={avg_turnover_per_bar:.3f}, ~turnover/day={avg_turnover_per_day:.3f}")
    print(f"Diagnostics: total fees (simple)={total_cost_simple:.6f} | breakeven fee per-side ≈ {breakeven_fee_bps:.2f} bps")
    print(f"Diagnostics: VaR95 (net, per-bar) ≈ {var95_simple_net_pct:.2f}% | CVaR95 ≈ {cvar95_simple_net_pct:.2f}%")
    print("-")
    # print(f"Asset  % Log Return  : {pct_return_asset_log:.2f}%")
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
        avg_net_dd = float(dd_net.mean()) if len(dd_net) else 0.0

        # Equity series (start at 1, min at 0)
        asset_equity = (1.0 + df["simple_r"]).cumprod()

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
                "avg_turnover_per_bar": avg_turnover_per_bar,
                "avg_turnover_per_day": avg_turnover_per_day,
                "total_cost_simple": total_cost_simple,
                "breakeven_fee_bps_per_side": breakeven_fee_bps,
                "avg_net_drawdown_pct": avg_net_dd * 100.0,
                "var95_net_pct": var95_simple_net_pct,
                "cvar95_net_pct": cvar95_simple_net_pct,
                "win_rate_net_pct": win_rate_net_pct,
                "avg_win_net_pct": avg_win_net_pct,
                "avg_loss_net_pct": avg_loss_net_pct,
            },
            "series": {
                "timestamps": [ts.isoformat() for ts in df.index.to_pydatetime()],
                "asset": {
                    "ret_log": df["r"].fillna(0).tolist(),
                    "ret_simple": df["simple_r"].fillna(0).tolist(),
                    "cum_log": asset_cum.fillna(0).tolist(),
                    "cum_simple": asset_cum_simple.fillna(0).tolist(),
                    "equity": asset_equity.fillna(1).clip(lower=0).tolist(),
                    "price": df[price_column].fillna(0).tolist(),
                },
                "strategy": {
                    "gross": {
                        "ret_log": df["strategy_r"].fillna(0).tolist(),
                        "ret_simple": df["strategy_simple_r"].fillna(0).tolist(),
                        "cum_log": strat_cum_gross.fillna(0).tolist(),
                        "cum_simple": strat_cum_simple_gross.fillna(0).tolist(),
                        "equity": equity_gross.fillna(1).clip(lower=0).tolist(),
                        "drawdown": dd_gross.fillna(0).tolist(),
                        "rolling_sharpe": rolling_sharpe_gross.fillna(method="bfill").fillna(0).tolist(),
                    },
                    "net": {
                        "ret_log": df["strategy_r_net"].fillna(0).tolist(),
                        "ret_simple": df["strategy_simple_r_net"].fillna(0).tolist(),
                        "cum_log": strat_cum_net.fillna(0).tolist(),
                        "cum_simple": strat_cum_simple_net.fillna(0).tolist(),
                        "equity": equity_net.fillna(1).clip(lower=0).tolist(),
                        "drawdown": dd_net.fillna(0).tolist(),
                        "rolling_sharpe": rolling_sharpe_net.fillna(method="bfill").fillna(0).tolist(),
                    },
                },
                "turnover": df["turnover"].fillna(0).tolist(),
            },
        }

        # Sanitize to strict JSON to avoid NaN/Infinity causing parse errors in the dashboard
        safe_report = _json_safe(report)
        with open(os.path.join(save_json_dir, "is.json"), "w") as f:
            json.dump(safe_report, f, allow_nan=False)


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







ml_params_deepseekR1 = { # pvalue 0.66, N=50
    "interval": "4h",
    "forecast_horizon_hours": 4,
    "n_epochs": 150,  # Reduced
    "hidden_sizes": (24, 12),  # Simpler architecture
    "signal_percentiles": (15, 85),  # Less extreme percentiles
    "train_ratio": 0.75,
    "val_ratio": 0.25,  # More validation data
    "early_stopping_patience": 15,  # More patience
    "lr": 1e-5,  # Smaller learning rate
    "dropout_rate": 0.6,  # Increased dropout
    "weight_decay": 0.05,  # Stronger L2 regularization
    "batch_size": 64,  # Smaller batch size
    "random_seed": 42,
    "hold_until_opposite": True,
    
    # Technical indicator windows - less overlapping
    "sma_windows": (6, 12, 24),  # Less frequent
    "volatility_windows": (6, 12),  # Fewer windows
    "momentum_windows": (6, 12),  # Fewer windows
    "rsi_windows": (6, 12),  # Fewer windows
    
    # Additional regularization
    # "use_batch_norm": True,  # Add batch normalization
    # "label_smoothing": 0.1  # Helps with overconfident predictions
}

ml_params_kimiK2 = { # pvalue 0.08, N=50
    # --- data & target -------------------------------------------------
    "interval": "4h",                       # unchanged
    "forecast_horizon_hours": 4,            # unchanged

    # --- model capacity (much smaller) ---------------------------------
    "hidden_sizes": (16, 8),                # drop the deepest layer
    "dropout_rate": 0.35,                   # slightly lower than 0.5 (works better with BN)

    # --- training length & early stopping ------------------------------
    "n_epochs": 300,                        # allow more epochs, but …
    "early_stopping_patience": 25,          # … be *much* stricter
    "lr": 1e-4,                             # a tad higher so it can escape sharp minima

    # --- regularisation -------------------------------------------------
    "weight_decay": 0.05,                   # stronger L2
    "batch_size": 64,                       # smaller → more noise = regulariser
    # "label_smoothing": 0.05,                # *new* aka invented by kimik2 – softens one-hot labels if you use them

    # --- data splits (unchanged) ---------------------------------------
    "train_ratio": 0.8,
    "val_ratio": 0.2,

    # --- reproducibility -----------------------------------------------
    "random_seed": 42,

    # --- trading logic (unchanged) -------------------------------------
    "hold_until_opposite": True,

    # --- feature engineering (shorter windows to avoid look-ahead bias) -
    "sma_windows": (3, 6, 12),              # shorter history
    "volatility_windows": (3, 6, 12),
    "momentum_windows": (3, 6, 12),
    "rsi_windows": (3, 6, 12),

    # --- new regularisation knobs --------------------------------------
    # "max_grad_norm": 1.0,                   # gradient clipping
    # "scheduler": "CosineAnnealingWarmRestarts",  # cyclic LR to escape sharp minima
    # "T_0": 25, "T_mult": 2,                # cosine-annealing hyper-params
}

ml_params_claudesonnet4 = { # pvalue 1, N=50
    # Data and timing parameters
    "interval": "4h",
    "forecast_horizon_hours": 4,
    "train_ratio": 0.7,  # Reduced to allocate more data for validation
    "val_ratio": 0.3,    # Increased validation set
    "random_seed": 42,
    "hold_until_opposite": True,
    
    # Model architecture - simplified
    "hidden_sizes": (16, 8),  # Reduced layers and neurons
    "dropout_rate": 0.3,      # Reduced but still meaningful
    
    # Training parameters - more conservative
    "n_epochs": 100,          # Fewer epochs
    "early_stopping_patience": 15,  # Increased patience
    "lr": 1e-4,              # Slightly higher learning rate
    "weight_decay": 0.001,    # Reduced regularization
    "batch_size": 64,         # Smaller batches
    
    # Feature parameters - reduced complexity
    "signal_percentiles": (15, 85),     # Less extreme percentiles
    "sma_windows": (8, 16),            # Fewer SMA periods
    "volatility_windows": (8, 16),     # Fewer volatility periods  
    "momentum_windows": (8, 16),       # Fewer momentum periods
    "rsi_windows": (8, 16),            # Fewer RSI periods
}

ml_params_geminipro25 = { # pvalue 0.62, N=50
    "interval": "4h",
    "forecast_horizon_hours": 4,
    
    # --- Model Simplicity ---
    "n_epochs": 100,                     # Reduced max epochs, rely on early stopping
    "hidden_sizes": (16, 8),             # SHALLOWER & NARROWER network
    
    # --- Data & Validation ---
    "signal_percentiles": (10, 90),
    "train_ratio": 0.7,                  # Smaller train set, larger val/test
    "val_ratio": 0.15,                   # IMPORTANT: Add a dedicated test set (remaining 0.15)
    "early_stopping_patience": 15,       # Slightly more patience for noisy validation loss
    
    # --- Regularization (Balanced Approach) ---
    "lr": 1e-4,                          # Slightly higher LR, often works better with smaller models
    "dropout_rate": 0.2,                 # REDUCED dropout, less aggressive
    "weight_decay": 0.001,               # REDUCED weight decay (L2 regularization)
    
    # --- Training Dynamics ---
    "batch_size": 64,                    # Smaller batch size can add regularizing noise
    "random_seed": 42,
    "hold_until_opposite": True,
    
    # --- Feature Simplification ---
    "sma_windows": (5, 10),              # Fewer, more distinct windows
    "volatility_windows": (10,),         # Only one, longer-term volatility measure
    "momentum_windows": (5, 10),         # Fewer, more distinct windows
    "rsi_windows": (14,),                # The standard RSI window
}


ml_params = {
    "interval": "4h",
    "forecast_horizon_hours": 4,
    "n_epochs": 200,
    "hidden_sizes": (32, 16, 8),
    "signal_percentiles": (10, 90),
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "early_stopping_patience": 10,
    "lr": 5e-5,
    "dropout_rate": 0.5,           # 0.4–0.6
    "weight_decay": 0.01,          # stronger L2 regularization
    "batch_size": 128,
    "random_seed": 42,
    "hold_until_opposite": True,
    #
    "sma_windows": (4, 8, 16),
    "volatility_windows": (4, 8, 16),
    "momentum_windows": (4, 8, 16),
    "rsi_windows": (4, 8, 16),
    # "sma_windows": (5, 10, 20, 30),
    # "volatility_windows": (5, 10, 20, 30),
    # "momentum_windows": (7, 14, 21, 30),
    # "rsi_windows": (7, 14, 21, 30),        
}



if __name__ == "__main__":
    # main()

    plot_cumulative_returns(
        start_date="2018-07-25",
        end_date="2024-10-31",
        # strategy_name="ma",
        strategy_name="ml",
        asset="VETUSD",
        timeframe="4h",
        show_plot=False,
        strategy_kwargs=ml_params,
        price_column="vwap_20",
        fee_bps=10.0,
        slippage_bps=10.0,
        save_json_dir="reports/example_run",
        position_sizing_mode="fixed_fraction",
        position_sizing_params={
            "fraction": 0.1,
        },
        # position_sizing_mode="fixed_notional",
        # position_sizing_params={
        #     "fixed_notional": 0.1,
        # },        
    )