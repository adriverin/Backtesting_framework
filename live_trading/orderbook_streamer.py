"""Real-time order book depth data streamer using Binance WebSocket.

This streamer computes percent-distance buckets, not raw level indices.

For each update, it emits rows for ±1..±N percent-of-price buckets where N is
`max_levels` (interpreted as number of percent buckets). For a given bucket p:
  - On the ask side (+p), we sum quantity and notional across asks needed to
    move the best ask up by p% (i.e., consume all asks strictly below
    best_ask * (1 + p/100)).
  - On the bid side (−p), we sum quantity and notional across bids needed to
    move the best bid down by p% (i.e., consume all bids strictly above
    best_bid * (1 − p/100)).

The resulting schema matches the historical orderbook depth files used during
backtests: columns [timestamp, percentage, depth, notional], where
`percentage` ∈ {−N, …, −1, 1, …, N} denotes ±percent buckets, not level index.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from collections import deque

import pandas as pd
import numpy as np


class OrderBookStreamer:
    """Stream and process live order book depth data from Binance.
    
    Transforms raw order book updates into percent-distance buckets compatible
    with the historical dataset used by the strategy:
    - timestamp: datetime
    - percentage: int (±1..±N interpreted as ±percent-of-price buckets)
    - depth: float (cumulative quantity needed to move to that ±percent)
    - notional: float (cumulative price*qty over consumed levels)
    """
    
    def __init__(
        self,
        symbol: str,
        max_levels: int = 5,
        update_interval_ms: int = 100,
        buffer_size: int = 10000,
    ):
        """Initialize orderbook streamer.
        
        Args:
            symbol: Trading symbol (e.g., 'VETUSDT')
            max_levels: Number of percent buckets to compute (±1%..±max_levels%)
            update_interval_ms: WebSocket update frequency (ms)
            buffer_size: Maximum number of snapshots to keep in memory
        """
        self.symbol = symbol.upper()
        self.max_levels = max_levels
        self.update_interval_ms = update_interval_ms
        self.buffer_size = buffer_size
        
        # Data buffer: stores processed orderbook snapshots
        self.data_buffer: deque = deque(maxlen=buffer_size)
        
        # WebSocket connection
        self.ws = None
        self.running = False
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable] = []
        
        # Last orderbook snapshot
        self.last_snapshot: Dict | None = None
        self.last_update_time: float = 0
    
    def add_update_callback(self, callback: Callable) -> None:
        """Register a callback to be called on each orderbook update."""
        self.update_callbacks.append(callback)
    
    async def start(self, mode: str = "futures", testnet: bool = False) -> None:
        """Start the WebSocket connection and begin streaming."""
        self.running = True
        
        # Determine WebSocket URL based on mode
        if mode.lower() == "futures":
            # Use testnet endpoint if requested
            if testnet:
                ws_url = f"wss://fstream.binancefuture.com/ws/{self.symbol.lower()}@depth@{self.update_interval_ms}ms"
            else:
                ws_url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@{self.update_interval_ms}ms"
        else:
            ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@depth@{self.update_interval_ms}ms"
        
        print(f"[OrderBookStreamer] Connecting to {ws_url}")
        
        try:
            import websockets  # type: ignore
            
            async with websockets.connect(ws_url) as websocket:
                self.ws = websocket
                print(f"[OrderBookStreamer] Connected! Streaming {self.symbol} orderbook depth...")
                
                while self.running:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        data = json.loads(msg)
                        await self._process_update(data)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                    except Exception as e:
                        print(f"[OrderBookStreamer] Error processing message: {e}")
                        await asyncio.sleep(1)
        
        except Exception as e:
            print(f"[OrderBookStreamer] WebSocket error: {e}")
            if self.running:
                # Attempt to reconnect after delay
                await asyncio.sleep(5)
                await self.start(mode)
    
    async def _process_update(self, data: Dict) -> None:
        """Process raw WebSocket orderbook update into ±percent buckets."""
        try:
            # Extract bids and asks
            bids = data.get('b', [])  # [[price, qty], ...]
            asks = data.get('a', [])  # [[price, qty], ...]
            
            if not bids or not asks:
                return
            
            # Convert to floats
            bids = [(float(p), float(q)) for p, q in bids]
            asks = [(float(p), float(q)) for p, q in asks]
            
            # Sort: bids descending (highest price first), asks ascending (lowest price first)
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])
            
            # Compute percent-of-price buckets relative to current best bid/ask
            timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
            
            processed_rows = []
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]

            # Helper: accumulate until price threshold is crossed
            def compute_ask_bucket(pct: int) -> tuple[float, float]:
                target = best_ask * (1.0 + (pct / 100.0))
                cum_qty = 0.0
                cum_notional = 0.0
                # Consume asks strictly below target; moving best ask up to target
                for price, qty in asks:
                    if price < target:
                        cum_qty += qty
                        cum_notional += price * qty
                    else:
                        break
                return cum_qty, cum_notional

            def compute_bid_bucket(pct: int) -> tuple[float, float]:
                target = best_bid * (1.0 - (pct / 100.0))
                cum_qty = 0.0
                cum_notional = 0.0
                # Consume bids strictly above target; moving best bid down to target
                for price, qty in bids:
                    if price > target:
                        cum_qty += qty
                        cum_notional += price * qty
                    else:
                        break
                return cum_qty, cum_notional

            max_pct = max(0, int(self.max_levels))
            for pct in range(1, max_pct + 1):
                # Bid side (negative percentage)
                b_qty, b_notional = compute_bid_bucket(pct)
                processed_rows.append({
                    'timestamp': timestamp,
                    'percentage': -pct,
                    'depth': b_qty,
                    'notional': b_notional,
                })

                # Ask side (positive percentage)
                a_qty, a_notional = compute_ask_bucket(pct)
                processed_rows.append({
                    'timestamp': timestamp,
                    'percentage': pct,
                    'depth': a_qty,
                    'notional': a_notional,
                })
            
            # Add to buffer
            self.data_buffer.append(processed_rows)
            self.last_snapshot = processed_rows
            self.last_update_time = time.time()
            
            # Trigger callbacks
            for callback in self.update_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(processed_rows)
                    else:
                        callback(processed_rows)
                except Exception as e:
                    print(f"[OrderBookStreamer] Callback error: {e}")
        
        except Exception as e:
            print(f"[OrderBookStreamer] Processing error: {e}")
    
    def get_dataframe(self, lookback_seconds: Optional[int] = None) -> pd.DataFrame:
        """Get orderbook data as DataFrame (matching strategy input format).
        
        Args:
            lookback_seconds: Only include data from last N seconds (None = all)
        
        Returns:
            DataFrame with columns: timestamp, percentage, depth, notional
        """
        if not self.data_buffer:
            return pd.DataFrame(columns=['timestamp', 'percentage', 'depth', 'notional'])
        
        # Flatten buffer (each element is a list of rows)
        all_rows = []
        cutoff_time = None
        if lookback_seconds:
            cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None).timestamp() - lookback_seconds
        
        for snapshot in self.data_buffer:
            for row in snapshot:
                if cutoff_time is None or row['timestamp'].timestamp() >= cutoff_time:
                    all_rows.append(row)
        
        if not all_rows:
            return pd.DataFrame(columns=['timestamp', 'percentage', 'depth', 'notional'])
        
        df = pd.DataFrame(all_rows)
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def get_level_data(self, percentage: int, lookback_seconds: Optional[int] = None) -> pd.DataFrame:
        """Get data for a specific orderbook level only.
        
        Args:
            percentage: Level to extract (e.g., 2 for ask level 2, -2 for bid level 2)
            lookback_seconds: Only include data from last N seconds
        
        Returns:
            DataFrame with columns: timestamp, notional (for the specified level)
        """
        df = self.get_dataframe(lookback_seconds)
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'notional'])
        
        level_df = df[df['percentage'] == percentage][['timestamp', 'notional']].copy()
        return level_df.sort_values('timestamp').reset_index(drop=True)
    
    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
    
    def get_current_snapshot(self) -> List[Dict] | None:
        """Get the most recent orderbook snapshot."""
        return self.last_snapshot
    
    def get_imbalance(self, percentage: int = 1) -> float | None:
        """Calculate current bid/ask imbalance at specified level.
        
        Returns:
            Ratio: bid / (bid + ask) at the specified percentage level
        """
        if not self.last_snapshot:
            return None
        
        bid_notional = None
        ask_notional = None
        
        for row in self.last_snapshot:
            if row['percentage'] == -percentage:
                bid_notional = row['notional']
            elif row['percentage'] == percentage:
                ask_notional = row['notional']
        
        if bid_notional is None or ask_notional is None:
            return None
        
        total = bid_notional + ask_notional
        if total == 0:
            return 0.5
        
        return bid_notional / total

