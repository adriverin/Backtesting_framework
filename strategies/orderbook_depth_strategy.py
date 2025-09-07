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

    r_ma = ratio.rolling(window=lookback_mean, min_periods=lookback_mean).mean()
    r_std = ratio.rolling(window=lookback_mean, min_periods=lookback_mean).std()
    r_current = ratio.rolling(window=lookback_current, min_periods=lookback_current).mean()

    z_score = (r_current - r_ma) / (r_std + eps)

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
) -> Tuple[Tuple[int, int], float]:
    """Grid-search lookbacks to maximise Profit Factor.

    Returns ((best_lookback_mean, best_lookback_current), best_pf).
    """
    if orderbook_depth is None or orderbook_depth.empty or price_data is None or price_data.empty:
        return (0, 0), 0.0

    price = price_data[price_column]
    r = np.log(price).diff().shift(-1)

    best_pf = 0.0
    best_params = (0, 0)

    for lb_mean in range(max(2, lb_mean_min), max(2, lb_mean_max) + 1):
        lb_cur_hi = min(lb_cur_max, lb_mean - 1) if lb_mean > 1 else lb_cur_min
        for lb_cur in range(max(1, lb_cur_min), max(1, lb_cur_hi) + 1):
            signal = orderbook_depth_strategy(
                orderbook_depth,
                percentage=percentage,
                lookback_mean=lb_mean,
                lookback_current=lb_cur,
                z_threshold=z_threshold,
                persistence=persistence,
                shift_signal=True,
            )
            if signal.empty:
                continue
            # Align to price returns index
            sig = signal.reindex(r.index).ffill().fillna(0)
            sig_rets = sig * r
            pf = _profit_factor(sig_rets.dropna())
            if pf > best_pf:
                best_pf = float(pf)
                best_params = (int(lb_mean), int(lb_cur))
    print(f"Best params: {best_params}, Best PF: {best_pf}")
    return best_params, best_pf


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
        z_threshold: float = 2.0,
        persistence: int = 1,
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
        self.z_threshold = float(z_threshold)
        self.persistence = int(persistence)
        self.base_dir = base_dir
        self.use_parquet_day_cache = bool(use_parquet_day_cache)

        self.best_params: Tuple[int, int] | None = None  # (lookback_mean, lookback_current)
        self.best_pf: float | None = None

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
        # During optimisation we may use up to lb_mean_max; preload that many days for depth warmup
        depth = self._load_depth_for_window(ohlc, preload_days=self.lb_mean_max)
        params, best_pf = optimize_orderbook_depth_strategy(
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
        )
        self.best_params = params
        self.best_pf = best_pf
        return params, best_pf

    def generate_signals(self, ohlc: pd.DataFrame) -> pd.Series:
        if self.best_params is None:
            self.optimize(ohlc)
        lb_mean, lb_cur = self.best_params or (self.lb_mean_min, self.lb_cur_min)
        # For live signal generation, preload by the chosen lookback mean (hardcoded rule)
        depth = self._load_depth_for_window(ohlc, preload_days=int(lb_mean))
        signal = orderbook_depth_strategy(
            depth,
            percentage=self.percentage,
            lookback_mean=lb_mean,
            lookback_current=lb_cur,
            z_threshold=self.z_threshold,
            persistence=self.persistence,
            shift_signal=True,
        )
        # Align to OHLC index
        return signal.reindex(ohlc.index).ffill().fillna(0)
