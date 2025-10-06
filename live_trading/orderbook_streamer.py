"""Real-time order book depth data streamer using Binance WebSocket."""
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
    
    Transforms raw order book snapshots into the format used by the strategy:
    - timestamp: datetime
    - percentage: int (1 to 5 for ask, -1 to -5 for bid)
    - depth: float (cumulative quantity at that level)
    - notional: float (cumulative value at that level)
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
            max_levels: Maximum depth levels to track (±1 to ±max_levels)
            update_interval_ms: WebSocket update frequency
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
    
    async def start(self, mode: str = "futures") -> None:
        """Start the WebSocket connection and begin streaming."""
        self.running = True
        
        # Determine WebSocket URL based on mode
        if mode.lower() == "futures":
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
        """Process raw WebSocket orderbook update."""
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
            
            # Calculate cumulative depth and notional for each level
            timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
            
            processed_rows = []
            
            # Process bid side (negative percentages: -1, -2, -3, ...)
            cumulative_bid_qty = 0.0
            cumulative_bid_notional = 0.0
            for i in range(min(self.max_levels, len(bids))):
                price, qty = bids[i]
                cumulative_bid_qty += qty
                cumulative_bid_notional += price * qty
                
                processed_rows.append({
                    'timestamp': timestamp,
                    'percentage': -(i + 1),
                    'depth': cumulative_bid_qty,
                    'notional': cumulative_bid_notional,
                })
            
            # Process ask side (positive percentages: +1, +2, +3, ...)
            cumulative_ask_qty = 0.0
            cumulative_ask_notional = 0.0
            for i in range(min(self.max_levels, len(asks))):
                price, qty = asks[i]
                cumulative_ask_qty += qty
                cumulative_ask_notional += price * qty
                
                processed_rows.append({
                    'timestamp': timestamp,
                    'percentage': i + 1,
                    'depth': cumulative_ask_qty,
                    'notional': cumulative_ask_notional,
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

