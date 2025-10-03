from __future__ import annotations

from typing import Any, Tuple, Optional

import os
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy


def read_data(
    symbol: str,
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
    base_dir: str = "data/orderbook_depth",
    use_parquet_day_cache: bool = True,
    parquet_subdir: str = "_parquet",
) -> pd.DataFrame:
    """Load and combine per-day order book depth CSVs for a symbol.

    Expects files named: {SYMBOL}-bookDepth-YYYY-MM-DD.csv inside
    {base_dir}/{SYMBOL}/.
    """
    try:
        start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
        end = pd.Timestamp(year=end_year, month=end_month, day=end_day)
    except Exception:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    dates = pd.date_range(start=start, end=end, freq="D")
    symbol = str(symbol).upper()
    for d in dates:
        csv_path = os.path.join(
            base_dir, symbol, f"{symbol}-bookDepth-{d.year}-{d.month:02d}-{d.day:02d}.csv"
        )
        if use_parquet_day_cache:
            pq_dir = os.path.join(base_dir, symbol, parquet_subdir)
            pq_path = os.path.join(pq_dir, f"{symbol}-bookDepth-{d.year}-{d.month:02d}-{d.day:02d}.parquet")
        else:
            pq_dir = None
            pq_path = None

        df: pd.DataFrame | None = None
        # Prefer daily parquet cache if present
        if pq_path and os.path.exists(pq_path):
            try:
                df = pd.read_parquet(pq_path)
            except Exception:
                df = None

        if df is None:
            try:
                df = pd.read_csv(csv_path)
            except FileNotFoundError:
                continue
            # Normalize types
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # Optional light downcast to reduce memory
            for col in ("percentage", "depth", "notional"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Write parquet day cache for faster future loads
            if pq_path:
                try:
                    os.makedirs(pq_dir or "", exist_ok=True)
                    df.to_parquet(pq_path, index=False)
                except Exception:
                    pass
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, axis=0, ignore_index=True)
    return data


def orderbook_depth_strategy(
    data: pd.DataFrame,
    percentage: int = 1,
    lookback_mean: int = 100,
    lookback_current: int = 10,
    z_threshold: float = 2.0,
    persistence: int = 1,
    exit_band: float | None = None,
    shift_signal: bool = True,
    eps: float = 1e-9,
) -> pd.Series:
    """Imbalance z-score regime using notional at ±percentage buckets.

    Returns a pd.Series of signals indexed by timestamp.
    """
    if data is None or data.empty:
        return pd.Series(dtype=float)

    # Expect columns: ['timestamp','percentage','notional']
    df = data.pivot(index="timestamp", columns="percentage", values="notional").sort_index()
    if (percentage not in df.columns) or (-percentage not in df.columns):
        return pd.Series(dtype=float)

    ask = df[percentage]
    bid = df[-percentage]
    aligned = pd.concat({"ask": ask, "bid": bid}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    ratio = aligned["bid"] / (aligned["ask"] + aligned["bid"] + eps)

    # Support both time-based (string) and count-based (int) rolling windows
    # Time-based windows (e.g., "210D", "1min") automatically handle variable data density
    # Count-based windows (e.g., 599760 datapoints) assume fixed density
    if isinstance(lookback_mean, str):
        # Time-based rolling window
        r_ma = ratio.rolling(lookback_mean).mean()
        r_std = ratio.rolling(lookback_mean).std()
    else:
        # Count-based rolling window
        r_ma = ratio.rolling(window=lookback_mean, min_periods=lookback_mean).mean()
        r_std = ratio.rolling(window=lookback_mean, min_periods=lookback_mean).std()
    
    if isinstance(lookback_current, str):
        # Time-based rolling window
        r_current = ratio.rolling(lookback_current).mean()
    else:
        # Count-based rolling window
        r_current = ratio.rolling(window=lookback_current, min_periods=lookback_current).mean()    

    z_score = -(r_current - r_ma) / (r_std + eps)

    raw = pd.Series(
        np.where(z_score > z_threshold, 1, np.where(z_score < -z_threshold, -1, 0)),
        index=z_score.index,
    )

    if persistence > 1:
        pos_run = (raw == 1).astype(int).rolling(persistence).sum()
        neg_run = (raw == -1).astype(int).rolling(persistence).sum()
        filt = pd.Series(0, index=raw.index)
        filt[pos_run == persistence] = 1
        filt[neg_run == persistence] = -1
        raw = filt

    # Exit-band logic:
    #  - if exit_band > 0: symmetric neutral zone, go flat when |z| < exit_band
    #  - if exit_band <= 0 (including 0): directional hysteresis; exit longs when z <= -|exit_band|, exit shorts when z >= +|exit_band|
    #                       (i.e., exit on the opposite side of the entry threshold)
    #  - if exit_band is None: never exit (hold forever)
    if exit_band is not None and float(exit_band) > 0:
        intermediate = raw.copy()
        abs_z = z_score.abs()
        # Forward-fill positions only when we are outside the exit band
        # i.e., in neutral region but still outside exit band -> keep holding; inside band -> go flat (0)
        mask_ffill = (intermediate == 0) & (abs_z >= float(exit_band))
        intermediate[mask_ffill] = np.nan
        signal = intermediate.ffill().fillna(0)
    elif exit_band is not None:  # Handles both negative values AND zero
        band = abs(float(exit_band))
        intermediate = raw.copy()
        # Determine prior non-zero direction to apply opposite-side exit threshold
        prev_pos = raw.replace(0, np.nan).ffill().fillna(0)
        # For longs (entered at z >= z_threshold): hold while z > -band (exit when z <= -band)
        hold_long_mask = (intermediate == 0) & (prev_pos == 1) & (z_score > -band)
        # For shorts (entered at z <= -z_threshold): hold while z < +band (exit when z >= +band)
        hold_short_mask = (intermediate == 0) & (prev_pos == -1) & (z_score < band)
        mask_ffill = hold_long_mask | hold_short_mask
        intermediate[mask_ffill] = np.nan
        signal = intermediate.ffill().fillna(0)
    else:
        # exit_band is None: hold positions indefinitely (never exit based on z-score)
        signal = raw.replace(0, np.nan).ffill().fillna(0)
    if shift_signal:
        signal = signal.shift(1)
    return signal


def _profit_factor(returns: pd.Series) -> float:
    positive_sum = returns[returns > 0].sum()
    negative_sum = returns[returns < 0].abs().sum()
    if negative_sum == 0:
        return float("inf") if positive_sum > 0 else 0.0
    return float(positive_sum / negative_sum)


def optimize_orderbook_depth_strategy(
    orderbook_depth: pd.DataFrame,
    price_data: pd.DataFrame,
    price_column: str = "close",
    percentage: int = 1,
    lb_mean_min: int = 20,
    lb_mean_max: int = 300,
    lb_cur_min: int = 5,
    lb_cur_max: int = 60,
    z_threshold: float = 2.0,
    persistence: int = 1,
    exit_band: float | None = None,
    fee_bps: float = 4.0,
    slippage_bps: float = 0.0,
    use_time_based_windows: bool = True,
) -> Tuple[Tuple[str | int, str | int], float]:
    """Grid-search lookbacks to maximize net cumulative returns (accounting for fees).

    If use_time_based_windows=True, returns time-based window strings like ("210D", "1min").
    If False, returns datapoint counts like (599760, 2).
    
    Returns ((best_lookback_mean, best_lookback_current), best_net_cum_return).
    """
    if orderbook_depth is None or orderbook_depth.empty or price_data is None or price_data.empty:
        return ("0D", "0min") if use_time_based_windows else (0, 0), 0.0

    price = price_data[price_column]
    simple_r = price.pct_change().shift(-1)

    best_net_cum_return = float("-inf")
    best_params = ("0D", "0min") if use_time_based_windows else (0, 0)

    fee_rate = (fee_bps + slippage_bps) / 10000.0
    
    if use_time_based_windows:
        # TIME-BASED OPTIMIZATION: Use pandas time-aware rolling windows
        # Step sizes: finer granularity (30 days for mean, 1 minute for current)
        step_mean_days = 10
        step_cur_mins = 1
        
        print(f"\n🔍 Optimization search space (TIME-BASED):")
        print(f"   lb_mean: {lb_mean_min} to {lb_mean_max} days (step: {step_mean_days} days)")
        print(f"   lb_cur: {lb_cur_min} to {lb_cur_max} minutes (step: {step_cur_mins} minute)")
        print(f"   Fee rate: {fee_bps + slippage_bps:.1f} bps\n")
        
        iterations = 0
        for lb_mean_days in range(lb_mean_min, lb_mean_max + 1, step_mean_days):
            for lb_cur_mins in range(lb_cur_min, lb_cur_max + 1, step_cur_mins):
                iterations += 1
                lb_mean_str = f"{lb_mean_days}D"
                lb_cur_str = f"{lb_cur_mins}min"
                
                signal = orderbook_depth_strategy(
                    orderbook_depth,
                    percentage=percentage,
                    lookback_mean=lb_mean_str,
                    lookback_current=lb_cur_str,
                    z_threshold=z_threshold,
                    persistence=persistence,
                    exit_band=exit_band,
                    shift_signal=True,
                )
                if signal.empty:
                    continue
                
                # Align to price returns index
                sig = signal.reindex(simple_r.index).ffill().fillna(0)
                gross_rets = sig * simple_r
                
                # Calculate turnover and costs
                turnover = sig.diff().abs().fillna(sig.abs())
                costs = fee_rate * turnover
                net_rets = gross_rets - costs
                
                # Cumulative return (multiplicative for simple returns)
                cum_return = (1 + net_rets).prod() - 1
                
                if cum_return > best_net_cum_return:
                    best_net_cum_return = float(cum_return)
                    best_params = (lb_mean_str, lb_cur_str)
                    print(f"   ✨ New best: lb_mean={lb_mean_days}d, lb_cur={lb_cur_mins}m, "
                          f"net_cum_return={best_net_cum_return:.4f}")
        
        print(f"\n✅ Optimization complete ({iterations} iterations)")
        print(f"   Best params: {best_params[0]}, {best_params[1]}")
        print(f"   Best net cumulative return: {best_net_cum_return:.4f}\n")
        
    else:
        # COUNT-BASED OPTIMIZATION: Use fixed datapoint counts
        data_points_per_day = 2856
        data_points_per_1h = int(data_points_per_day / 24)
        data_points_per_1m = int(data_points_per_1h / 60)
        
        lb_mean_min_dp = lb_mean_min * data_points_per_day
        lb_mean_max_dp = lb_mean_max * data_points_per_day
        lb_cur_min_dp = lb_cur_min * data_points_per_1m
        lb_cur_max_dp = lb_cur_max * data_points_per_1m
        
        step_mean = 10 * data_points_per_day
        step_cur = 5 * data_points_per_1m
        
        print(f"\n🔍 Optimization search space (COUNT-BASED):")
        print(f"   lb_mean: {lb_mean_min} to {lb_mean_max} days (step: 10 days)")
        print(f"   lb_cur: {lb_cur_min} to {lb_cur_max} minutes (step: 5 minutes)")
        print(f"   Datapoints: lb_mean=[{lb_mean_min_dp}, {lb_mean_max_dp}], lb_cur=[{lb_cur_min_dp}, {lb_cur_max_dp}]")
        print(f"   Fee rate: {fee_bps + slippage_bps:.1f} bps\n")
        
        iterations = 0
        for lb_mean_dp in range(max(2, lb_mean_min_dp), max(2, lb_mean_max_dp) + 1, step_mean):
            lb_cur_hi_dp = min(lb_cur_max_dp, lb_mean_dp - 1) if lb_mean_dp > 1 else lb_cur_min_dp
            for lb_cur_dp in range(max(1, lb_cur_min_dp), max(1, lb_cur_hi_dp) + 1, step_cur):
                iterations += 1
                signal = orderbook_depth_strategy(
                    orderbook_depth,
                    percentage=percentage,
                    lookback_mean=lb_mean_dp,
                    lookback_current=lb_cur_dp,
                    z_threshold=z_threshold,
                    persistence=persistence,
                    exit_band=exit_band,
                    shift_signal=True,
                )
                if signal.empty:
                    continue
                
                sig = signal.reindex(simple_r.index).ffill().fillna(0)
                gross_rets = sig * simple_r
                turnover = sig.diff().abs().fillna(sig.abs())
                costs = fee_rate * turnover
                net_rets = gross_rets - costs
                cum_return = (1 + net_rets).prod() - 1
                
                if cum_return > best_net_cum_return:
                    best_net_cum_return = float(cum_return)
                    best_params = (int(lb_mean_dp), int(lb_cur_dp))
                    best_params_days = int(round(lb_mean_dp / data_points_per_day))
                    best_params_mins = int(round(lb_cur_dp / data_points_per_1m))
                    print(f"   ✨ New best: lb_mean={best_params_days}d, lb_cur={best_params_mins}m, "
                          f"net_cum_return={best_net_cum_return:.4f}")
        
        print(f"\n✅ Optimization complete ({iterations} iterations)")
        if isinstance(best_params[0], int):
            print(f"   Best params: {best_params[0]} datapoints, {best_params[1]} datapoints")
        print(f"   Best net cumulative return: {best_net_cum_return:.4f}\n")
    
    return best_params, best_net_cum_return


class OrderBookDepthStrategy(BaseStrategy):
    """Strategy using order book depth imbalance z-scores.

    Relies on daily CSVs under data/orderbook_depth/{SYMBOL}/ with filename
    pattern {SYMBOL}-bookDepth-YYYY-MM-DD.csv.
    """

    def __init__(
        self,
        price_column: str = "close",
        symbol: str = "BTCUSDT",
        percentage: int = 1,
        lb_mean_min: int = 20,
        lb_mean_max: int = 300,
        lb_cur_min: int = 5,
        lb_cur_max: int = 60,
        lookback_mean: str | int | None = None,  # If set, skip optimization and use this value
        lookback_current: str | int | None = None,  # If set, skip optimization and use this value
        z_threshold: float = 2.0,
        persistence: int = 1,
        exit_band: float | None = None,
        fee_bps: float = 4.0,
        slippage_bps: float = 0.0,
        base_dir: str = "data/orderbook_depth",
        use_parquet_day_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(price_column, **kwargs)
        self.symbol = str(symbol).upper()
        self.percentage = int(percentage)
        self.lb_mean_min = int(lb_mean_min)
        self.lb_mean_max = int(lb_mean_max)
        self.lb_cur_min = int(lb_cur_min)
        self.lb_cur_max = int(lb_cur_max)
        self.lookback_mean_fixed = lookback_mean  # Can be "210D" or int for fixed windows
        self.lookback_current_fixed = lookback_current  # Can be "1min" or int for fixed windows
        self.use_time_based_windows = True  # Prefer time-based windows for optimization
        self.z_threshold = float(z_threshold)
        self.persistence = int(persistence)
        self.exit_band = exit_band if exit_band is None else float(exit_band)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.base_dir = base_dir
        self.use_parquet_day_cache = bool(use_parquet_day_cache)

        self.best_params: Tuple[str | int, str | int] | None = None  # (lookback_mean, lookback_current)
        self.best_metric: float | None = None  # Net cumulative return

    def _load_depth_for_window(self, ohlc: pd.DataFrame, preload_days: Optional[int] = None) -> pd.DataFrame:
        if not isinstance(ohlc.index, pd.DatetimeIndex) or ohlc.empty:
            return pd.DataFrame()
        start = ohlc.index.min()
        # Hardcoded preload: extend the depth window backwards by N days, where N depends on lookback
        if preload_days and preload_days > 0:
            try:
                start = start - pd.Timedelta(days=int(preload_days))
            except Exception:
                pass
        end = ohlc.index.max()
        df = read_data(
            self.symbol,
            start.year,
            start.month,
            start.day,
            end.year,
            end.month,
            end.day,
            base_dir=self.base_dir,
            use_parquet_day_cache=self.use_parquet_day_cache,
        )
        return df

    def optimize(self, ohlc: pd.DataFrame):
        # If fixed lookback windows are provided, skip optimization
        if self.lookback_mean_fixed is not None and self.lookback_current_fixed is not None:
            print(f"📌 Using fixed lookback windows: {self.lookback_mean_fixed}, {self.lookback_current_fixed}")
            self.best_params = (self.lookback_mean_fixed, self.lookback_current_fixed)
            self.best_metric = 0.0  # Not computed when using fixed params
            return self.best_params, self.best_metric
        
        # During optimisation we may use up to lb_mean_max; preload that many days for depth warmup
        depth = self._load_depth_for_window(ohlc, preload_days=self.lb_mean_max)
        params, best_metric = optimize_orderbook_depth_strategy(
            orderbook_depth=depth,
            price_data=ohlc,
            price_column=self.price_column,
            percentage=self.percentage,
            lb_mean_min=self.lb_mean_min,
            lb_mean_max=self.lb_mean_max,
            lb_cur_min=self.lb_cur_min,
            lb_cur_max=self.lb_cur_max,
            z_threshold=self.z_threshold,
            persistence=self.persistence,
            exit_band=self.exit_band,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            use_time_based_windows=self.use_time_based_windows,
        )
        self.best_params = params
        self.best_metric = best_metric
        return params, best_metric

    def generate_signals(self, ohlc: pd.DataFrame) -> pd.Series:
        if self.best_params is None:
            self.optimize(ohlc)
        
        lb_mean, lb_cur = self.best_params or (self.lb_mean_min, self.lb_cur_min)
        
        # For live signal generation, preload depth data
        # Calculate preload days based on lookback window type
        if isinstance(lb_mean, str):
            # Time-based window like "210D" - extract days
            try:
                if lb_mean.endswith("D"):
                    preload_days = int(lb_mean[:-1])
                elif lb_mean.endswith("d"):
                    preload_days = int(lb_mean[:-1])
                else:
                    preload_days = self.lb_mean_max
            except ValueError:
                preload_days = self.lb_mean_max
        elif isinstance(lb_mean, int):
            # Convert datapoints back to days (rough approximation)
            preload_days = int(lb_mean / 2856) if lb_mean > 0 else self.lb_mean_min
        else:
            preload_days = self.lb_mean_max
        
        depth = self._load_depth_for_window(ohlc, preload_days=preload_days)
        signal = orderbook_depth_strategy(
            depth,
            percentage=self.percentage,
            lookback_mean=lb_mean,
            lookback_current=lb_cur,
            z_threshold=self.z_threshold,
            persistence=self.persistence,
            exit_band=self.exit_band,
            shift_signal=True,
        )
        # Align to OHLC index
        return signal.reindex(ohlc.index).ffill().fillna(0)
