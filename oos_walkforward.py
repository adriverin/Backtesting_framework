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
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def _annualisation_factor(timeframe: str) -> float:
    """Return sqrt(periods_per_year) consistent with IS results."""
    tf = timeframe.lower().strip()
    if tf.endswith("m"):
        minutes_per_bar = int(tf[:-1])
    elif tf.endswith("h"):
        minutes_per_bar = int(tf[:-1]) * 60
    elif tf.endswith("d"):
        minutes_per_bar = int(tf[:-1]) * 60 * 24
    else:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")
    periods_per_year = (365 * 24 * 60) / max(1, minutes_per_bar)
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

    raise KeyError(f"Price column '{price_column}' not found in DataFrame and cannot be derived.")

# ---------------------------------------------------------------------- #
# Parallel worker (module-level, pickle-safe)                            #
# ---------------------------------------------------------------------- #

def _wf_train_and_signal_worker_fn(
    train_df: pd.DataFrame,
    calc_df: pd.DataFrame,
    train_end: int,
    oos_end: int,
    oos_section_len: int,
    price_col: str,
    strategy_name: str,
    strategy_kwargs: dict,
):
    """Optimize on train_df, generate signals on calc_df, return OOS segment."""
    from strategies import aVAILABLE_STRATEGIES as _AVS  # type: ignore
    from strategies.base_strategy import BaseStrategy as _Base  # local import for subprocess
    strategy_cls: Type[_Base] = _AVS[strategy_name]
    strategy: _Base = strategy_cls(price_column=price_col, **strategy_kwargs)
    _ = strategy.optimize(train_df)
    oos_signals_full = strategy.generate_signals(calc_df)
    oos_signals_final = oos_signals_full.iloc[-oos_section_len:].values
    return train_end, oos_end, oos_signals_final


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
        position_sizing_mode: str = "full_notional",
        position_sizing_params: dict | None = None,
        parallel: bool = True,
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
        self.position_sizing_mode = position_sizing_mode
        self.position_sizing_params = position_sizing_params or {}
        self.parallel = parallel

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

        # Prepare tasks
        tasks: list[tuple[pd.DataFrame, pd.DataFrame, int, int, int, str, str, dict]] = []
        for i in range(self.train_lookback, n, self.train_step):
            train_start = i - self.train_lookback
            train_end = i
            oos_end = min(i + self.train_step, n)
            long_lookback_ctx = self.train_lookback
            signal_calc_start = max(0, train_end - long_lookback_ctx)
            train_df = ohlc.iloc[train_start:train_end]
            calc_df = ohlc.iloc[signal_calc_start:oos_end]
            calc_df = _ensure_price_column_exists(calc_df, price_col)
            oos_section_len = len(ohlc.iloc[train_end:oos_end])
            tasks.append((train_df, calc_df, train_end, oos_end, oos_section_len, price_col, self.strategy_name, self.strategy_kwargs))

        # Short-circuit if no tasks
        if not tasks:
            return wf_signals

        # Parallel execution (per-window training + signal generation)
        if self.parallel and len(tasks) > 1:
            futures = []
            with ProcessPoolExecutor() as ex:
                for args in tasks:
                    futures.append(ex.submit(_wf_train_and_signal_worker_fn, *args))
                for fut in as_completed(futures):
                    train_end, oos_end, oos_signals_final = fut.result()
                    wf_signals[train_end:oos_end] = oos_signals_final
        else:
            # Fallback: sequential (original behaviour)
            for train_df, calc_df, train_end, oos_end, oos_section_len, price_col, strategy_name, strategy_kwargs in tasks:
                # Fresh strategy instance for each iteration to avoid leakage
                strategy: BaseStrategy = aVAILABLE_STRATEGIES[strategy_name](price_column=price_col, **strategy_kwargs)
                _ = strategy.optimize(train_df)
                oos_signals_full = strategy.generate_signals(calc_df)
                oos_signals_final = oos_signals_full.iloc[-oos_section_len:].values
                wf_signals[train_end:oos_end] = oos_signals_final

        return wf_signals




    # ------------------------------------------------------------------ #
    # Plot OOS walk-forward                                            #
    # ------------------------------------------------------------------ #

    def run(self, save_json_dir: str | None = None):
        real_df = self.get_df()
        real_df = _ensure_price_column_exists(real_df, self.price_column)

        # Pretty-print helper for converting periods to human units based on timeframe
        tf = self.timeframe.lower().strip()
        if tf.endswith("m"):
            bar_hours = int(tf[:-1]) / 60.0
        elif tf.endswith("h"):
            bar_hours = float(int(tf[:-1]))
        elif tf.endswith("d"):
            bar_hours = float(int(tf[:-1]) * 24)
        else:
            bar_hours = 1.0
        bars_per_day = max(1.0, 24.0 / bar_hours)
        bars_per_year = bars_per_day * 365.0

        print("")
        print("="*100)
        print(
            f"Calculating walk-forward signals for real data from {self.start_date} to {self.end_date}"
        )
        print(f"Train lookback: {self.train_lookback} periods (~{self.train_lookback / bars_per_year:.2f} years)")
        print(f"Train step: {self.train_step} periods (~{self.train_step / bars_per_day:.1f} days)")
        print("="*100)
        print("")
        print(f"Loaded data slice: n={len(real_df)} | first={real_df.index.min()} | last={real_df.index.max()}")
        
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
        fee_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        real_df["turnover"] = (real_df["weight"].diff().abs()).fillna(real_df["weight"].abs())
        real_df["cost_simple"] = fee_rate * real_df["turnover"]
        real_df["strategy_simple_r_net"] = real_df["strategy_simple_r"] - real_df["cost_simple"]
        real_df["strategy_r_net"] = np.log((1.0 + real_df["strategy_simple_r_net"]).clip(lower=1e-12))

        # Trim to analysis window and drop trailing NaNs from shift/diff so plots don't jump at the end
        real_df = real_df[(real_df.index >= self.start_date) & (real_df.index < self.end_date)]
        real_df = real_df.dropna(subset=["r", "strategy_r"])  # match IS behaviour to avoid last-point jumps

        # Diagnostics: detect if no OOS iterations executed (all signals stayed NaN/0)
        num_signal_points = int(np.count_nonzero(~np.isnan(real_signals)))
        if num_signal_points == 0:
            n = len(real_df)
            est_iters = max(0, (n - self.train_lookback + self.train_step - 1) // self.train_step)
            print("WARNING: No walk-forward iterations executed.")
            print(f"- Length n={n}, train_lookback={self.train_lookback}, train_step={self.train_step}, estimated_iters={est_iters}")
            print("- Action: reduce 'train_lookback' and/or 'train_step', or extend the date range.")

        real_pf_gross = _profit_factor(real_df["strategy_r"].dropna())
        real_pf_net = _profit_factor(real_df["strategy_r_net"].dropna())
        ann_factor_run = _annualisation_factor(self.timeframe)
        sr_gross = (
            float(real_df["strategy_r"].mean() / real_df["strategy_r"].std() * ann_factor_run)
            if np.isfinite(real_df["strategy_r"].std()) and real_df["strategy_r"].std() > 0
            else float("nan")
        )
        sr_net = (
            float(real_df["strategy_r_net"].mean() / real_df["strategy_r_net"].std() * ann_factor_run)
            if np.isfinite(real_df["strategy_r_net"].std()) and real_df["strategy_r_net"].std() > 0
            else float("nan")
        )

        # Cumulative series (asset + strategy) for plotting
        asset_cum_log = real_df["r"].cumsum()
        asset_cum_simple = (1.0 + real_df["simple_r"]).cumprod() - 1.0
        asset_equity = (1.0 + real_df["simple_r"]).cumprod()

        # Win/loss stats per trade (aggregate contiguous non-zero positions)
        pos = pd.Series(real_df["weight"]).fillna(0)
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
        print("")
        print("")
        print("")
        print("=" * 100)
        print(f"PERFORMANCE ANALYSIS - {self.strategy_name} on {self.asset} {self.timeframe}")
        print("")
        print(f"Start Date: {self.start_date} | End Date: {self.end_date}")
        print(f"Train Lookback: {self.train_lookback} periods (~{self.train_lookback / bars_per_year:.2f} years)")
        print(f"Train Step: {self.train_step} periods (~{self.train_step / bars_per_day:.1f} days)")
        print(f"Position Sizing Mode: {self.position_sizing_mode}")
        print("")
        print("=" * 100)
        print(f"Initial Capital: $100,000.00")
        _ending_equity = float((1.0 + real_df["strategy_simple_r_net"]).cumprod().iloc[-1]) if len(real_df) else 1.0
        print(f"Final Capital: ${100000 * _ending_equity:.2f}")
        print("-" * 100)
        print(f"Win rate (net): {win_rate_net_pct:.2f}%")
        print(f"Avg win (net): {avg_win_net_pct:.2f}%")
        print(f"Avg loss (net): {avg_loss_net_pct:.2f}%")
        print(f"OOS real Profit Factor (gross): {real_pf_gross:.4f}")
        print(f"OOS real Profit Factor (net)  : {real_pf_net:.4f}  (fees={self.fee_bps}bps, slip={self.slippage_bps}bps per side)")
        print(f"OOS real Sharpe Ratio (gross)  : {sr_gross:.2f}")
        print(f"OOS real Sharpe Ratio (net)    : {sr_net:.2f}")
        print("=" * 100)
        print("")

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
            ann_factor = _annualisation_factor(self.timeframe)
            rolling_sharpe_gross = (real_df["strategy_r"].rolling(window_bars).mean() /
                                     real_df["strategy_r"].rolling(window_bars).std()) * ann_factor
            rolling_sharpe_net = (real_df["strategy_r_net"].rolling(window_bars).mean() /
                                   real_df["strategy_r_net"].rolling(window_bars).std()) * ann_factor

            # Drawdowns from simple equity
            equity_gross = (1.0 + real_df["strategy_simple_r"]).cumprod()
            equity_net = (1.0 + real_df["strategy_simple_r_net"]).cumprod()
            dd_gross = equity_gross / equity_gross.cummax() - 1.0
            dd_net = equity_net / equity_net.cummax() - 1.0
            avg_net_dd = float(dd_net.mean()) if len(dd_net) else 0.0
            max_net_dd = float(dd_net.min()) if len(dd_net) else 0.0

            # Diagnostics for costs/turnover
            bars_per_day = max(1.0, 24.0 / bar_hours)
            avg_turnover_per_bar = float(real_df["turnover"].mean()) if len(real_df) else 0.0
            avg_turnover_per_day = avg_turnover_per_bar * bars_per_day
            total_cost_simple = float(real_df["cost_simple"].sum())
            total_turnover = float(real_df["turnover"].sum())
            breakeven_fee_rate = float(real_df["strategy_simple_r"].sum() / total_turnover) if total_turnover > 0 else float("nan")
            breakeven_fee_bps = breakeven_fee_rate * 10000.0

            # Risk metrics (per-bar, on net simple return stream)
            net_simple_ser = real_df["strategy_simple_r_net"].dropna()
            net_simple = net_simple_ser.values
            if len(net_simple) > 0:
                q05 = float(np.quantile(net_simple, 0.05))
                var95_simple_net_pct = float(-q05 * 100.0)
                cvar95_simple_net_pct = float(-np.mean(net_simple[net_simple <= q05]) * 100.0)
            else:
                var95_simple_net_pct = float("nan")
                cvar95_simple_net_pct = float("nan")

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
                    "sharpe_gross": float(sr_gross),
                    "sharpe_net": float(sr_net),
                    "pct_return_log_gross": float(real_df["strategy_r"].cumsum().iloc[-1] * 100.0) if len(real_df) else 0.0,
                    "pct_return_log_net": float(real_df["strategy_r_net"].cumsum().iloc[-1] * 100.0) if len(real_df) else 0.0,
                    "pct_return_simple_gross": float(((1.0 + real_df["strategy_simple_r"]).cumprod().iloc[-1] - 1.0) * 100.0) if len(real_df) else 0.0,
                    "pct_return_simple_net": float(((1.0 + real_df["strategy_simple_r_net"]).cumprod().iloc[-1] - 1.0) * 100.0) if len(real_df) else 0.0,
                    "avg_turnover_per_bar": avg_turnover_per_bar,
                    "avg_turnover_per_day": avg_turnover_per_day,
                    "total_cost_simple": total_cost_simple,
                    "breakeven_fee_bps_per_side": breakeven_fee_bps,
                    "avg_net_drawdown_pct": avg_net_dd * 100.0,
                    "max_net_drawdown_pct": max_net_dd * 100.0,
                    "var95_net_pct": var95_simple_net_pct,
                    "cvar95_net_pct": cvar95_simple_net_pct,
                    "win_rate_net_pct": win_rate_net_pct,
                    "avg_win_net_pct": avg_win_net_pct,
                    "avg_loss_net_pct": avg_loss_net_pct,
                },
                "series": {
                    "timestamps": [ts.isoformat() for ts in real_df.index.to_pydatetime()],
                    "asset": {
                        "ret_log": real_df["r"].fillna(0).tolist(),
                        "ret_simple": real_df["simple_r"].fillna(0).tolist(),
                        "cum_log": asset_cum_log.fillna(0).tolist(),
                        "cum_simple": asset_cum_simple.fillna(0).tolist(),
                        "equity": asset_equity.fillna(1).clip(lower=0).tolist(),
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
                    "turnover": real_df["turnover"].fillna(0).tolist(),
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

    # ml_params = {
    #     # "interval": "1h",
    #     "forecast_horizon_hours": 1,
    #     "n_epochs": 300,
    #     "hidden_sizes": (128, 64, 32, 16),
    #     "signal_percentiles": (10, 90),
    #     "train_ratio": 0.8,
    #     "val_ratio": 0.2,
    #     "early_stopping_patience": 10,
    #     "lr": 5e-5,
    #     "weight_decay": 0.001,
    #     "batch_size": 128,
    #     "random_seed": 42,
    # }

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
    _lookback_bars = int(365 * _years_lb * _bars_per_day)   # e.g., 4 years of context
    _step_bars = int(30 * _bars_per_day)                    # e.g., 30 days per step

    tester = WalkForward(
        start_date="2025-01-01",
        end_date="2025-07-01",
        strategy_name="ml",
        asset="VETUSD",
        timeframe=_tf,
        train_lookback=_lookback_bars,
        train_step=_step_bars,
        generate_plot=True,
        strategy_kwargs=ml_params,
        price_column="vwap_20",
        fee_bps=10.0,
        slippage_bps=10.0,
        position_sizing_mode="fixed_fraction",
        position_sizing_params={
            "fraction": 0.5,
        },        
    )
    tester.run(save_json_dir="reports/example_run")

