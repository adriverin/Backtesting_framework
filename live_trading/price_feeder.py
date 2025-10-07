"""Live price data integration using existing CCXT feeder."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


class PriceFeeder:
    """Manages live price data using the existing live_feeder_ccxt.py infrastructure.
    
    This class monitors the parquet file being updated by live_feeder_ccxt.py
    and provides real-time price data access.
    """
    
    def __init__(
        self,
        symbol: str,
        interval: str,
        mode: str = "futures",
        data_dir: str = "data",
    ):
        """Initialize price feeder.
        
        Args:
            symbol: Trading symbol (e.g., 'VETUSD')
            interval: Timeframe (e.g., '1m', '5m', '1h')
            mode: 'spot' or 'futures'
            data_dir: Base directory for data files
        """
        self.symbol = symbol
        self.interval = interval
        self.mode = mode.lower()
        
        # Determine parquet file path
        # data_dir should already be absolute from config resolution
        data_path = Path(data_dir)
        base_dir = data_path / ("futures" if self.mode == "futures" else "spot")
        self.parquet_path = base_dir / f"ohlcv_{symbol}_{interval}.parquet"
        
        # Cache
        self.last_df: pd.DataFrame | None = None
        self.last_load_time: float = 0
        self.cache_duration_sec = 5  # Reload every 5 seconds
    
    def get_latest_data(self, lookback_bars: Optional[int] = None) -> pd.DataFrame:
        """Get latest price data from parquet file.
        
        Args:
            lookback_bars: Number of bars to return (None = all)
        
        Returns:
            DataFrame with OHLCV data
        """
        current_time = time.time()
        
        # Check cache
        if self.last_df is not None and (current_time - self.last_load_time) < self.cache_duration_sec:
            df = self.last_df
        else:
            # Reload from disk
            if not self.parquet_path.exists():
                print(f"[PriceFeeder] Warning: {self.parquet_path} not found")
                return pd.DataFrame()
            
            try:
                df = pd.read_parquet(self.parquet_path)
                
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')
                
                # Make UTC-naive
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                
                # Add derived columns if missing
                if 'typical' not in df.columns and all(c in df.columns for c in ['high', 'low', 'close']):
                    df['typical'] = (df['high'] + df['low'] + df['close']) / 3.0
                
                if 'median' not in df.columns and all(c in df.columns for c in ['high', 'low']):
                    df['median'] = (df['high'] + df['low']) / 2.0
                
                # VWAP columns (various windows)
                if 'volume' in df.columns:
                    for window in [10, 20, 30, 50, 100]:
                        col_name = f'vwap_{window}'
                        if col_name not in df.columns:
                            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
                            tpv = typical_price * df['volume']
                            cumulative_tpv = tpv.rolling(window=window, min_periods=1).sum()
                            cumulative_volume = df['volume'].rolling(window=window, min_periods=1).sum()
                            df[col_name] = (cumulative_tpv / cumulative_volume).ffill()
                
                self.last_df = df
                self.last_load_time = current_time
            
            except Exception as e:
                print(f"[PriceFeeder] Error loading data from {self.parquet_path}: {e}")
                import traceback
                traceback.print_exc()
                return pd.DataFrame()
        
        # Apply lookback
        if lookback_bars and len(df) > lookback_bars:
            return df.iloc[-lookback_bars:].copy()
        
        return df.copy()
    
    def get_latest_price(self, price_column: str = "close") -> float | None:
        """Get the most recent price value.
        
        Args:
            price_column: Column to use for price (e.g., 'close', 'vwap_30')
        
        Returns:
            Latest price or None if unavailable
        """
        df = self.get_latest_data(lookback_bars=1)
        if df.empty or price_column not in df.columns:
            return None
        
        return float(df[price_column].iloc[-1])
    
    def get_latest_timestamp(self) -> datetime | None:
        """Get the timestamp of the most recent bar."""
        df = self.get_latest_data(lookback_bars=1)
        if df.empty:
            return None
        
        return df.index[-1]
    
    async def wait_for_data(self, timeout_sec: int = 300) -> bool:
        """Wait for parquet file to exist and contain data.
        
        Args:
            timeout_sec: Maximum time to wait
        
        Returns:
            True if data is available, False if timeout
        """
        start_time = time.time()
        attempt = 0
        
        print(f"[PriceFeeder] Waiting for data at: {self.parquet_path.resolve()}")
        
        while (time.time() - start_time) < timeout_sec:
            attempt += 1
            exists = self.parquet_path.exists()
            
            if exists:
                print(f"[PriceFeeder] Attempt {attempt}: File exists, trying to load data...")
                df = self.get_latest_data()
                if not df.empty:
                    print(f"[PriceFeeder] Success! Loaded {len(df)} bars, latest: {df.index[-1]}")
                    return True
                else:
                    print(f"[PriceFeeder] Attempt {attempt}: File exists but data is empty")
            else:
                print(f"[PriceFeeder] Attempt {attempt}: File does not exist yet")
            
            await asyncio.sleep(5)
        
        print(f"[PriceFeeder] Timeout after {timeout_sec}s")
        return False

