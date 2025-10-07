"""Main live trading engine with strategy execution and position management."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import json
from pathlib import Path

import pandas as pd
import numpy as np

from live_trading.config import TradingConfig
from live_trading.orderbook_streamer import OrderBookStreamer
from live_trading.price_feeder import PriceFeeder
from live_trading.performance_tracker import PerformanceTracker

# Import strategy
import sys
sys.path.append(str(Path(__file__).parent.parent))
from strategies import aVAILABLE_STRATEGIES  # type: ignore


class TradingEngine:
    """Core trading engine that executes the orderbook depth strategy in real-time."""
    
    def __init__(self, config: TradingConfig):
        """Initialize trading engine.
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        
        # Components
        self.orderbook_streamer: OrderBookStreamer | None = None
        self.price_feeder: PriceFeeder | None = None
        self.performance_tracker: PerformanceTracker | None = None
        
        # Strategy
        self.strategy = None
        self.strategy_params: Dict[str, Any] = {}
        
        # Position state
        self.current_position: float = 0.0  # -1, 0, or 1
        self.current_signal: float = 0.0
        self.entry_price: float | None = None
        self.position_size: float = 0.0  # In base currency
        
        # Capital management
        self.initial_capital: float = 1000.0  # Starting capital in USD
        self.current_capital: float = self.initial_capital
        self.unrealized_pnl: float = 0.0
        
        # Runtime state
        self.running = False
        self.last_signal_update: float = 0
        self.signal_update_interval: int = 60  # Update signals every 60 seconds
        
        # Binance client (for order execution)
        self.exchange = None
        # Telegram notifier (injected by run_live)
        # Optional attribute; present when Telegram is configured
        # self.telegram_notifier will be set externally

        # Persisted state
        log_dir = Path(self.config.get('logging.dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = log_dir / f"state_{self.config['symbol']}_{self.config['timeframe']}.json"
        self.prev_signal: float = 0.0
    
    async def initialize(self) -> None:
        """Initialize all components."""
        print("[TradingEngine] Initializing...")
        
        # Initialize orderbook streamer
        symbol = self.config['symbol']
        mode = self.config['mode']
        
        self.orderbook_streamer = OrderBookStreamer(
            symbol=symbol,
            max_levels=self.config.get('data.orderbook_depth_levels', 5),
            update_interval_ms=self.config.get('data.orderbook_update_interval_ms', 100),
        )
        
        # Initialize price feeder
        self.price_feeder = PriceFeeder(
            symbol=symbol,
            interval=self.config['timeframe'],
            mode=mode,
            data_dir=self.config.get('data.data_dir', 'data'),
        )
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(
            initial_capital=self.initial_capital,
            config=self.config,
        )
        
        # Initialize Binance client
        await self._init_exchange()
        
        # Initialize strategy
        await self._init_strategy()
        
        # Load persisted state
        self._load_state()
        
        print("[TradingEngine] Initialization complete!")
    
    async def _init_exchange(self) -> None:
        """Initialize Binance exchange connection."""
        try:
            import ccxt.async_support as ccxt  # type: ignore
            
            api_key = self.config.get('binance.api_key', '')
            api_secret = self.config.get('binance.api_secret', '')
            testnet = self.config.get('binance.testnet', False)
            
            if self.config['mode'] == 'futures':
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'},
                })
                if testnet:
                    self.exchange.set_sandbox_mode(True)
            else:
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })
            
            # Test connection
            if api_key and api_secret:
                balance = await self.exchange.fetch_balance()
                print(f"[TradingEngine] Connected to Binance. Balance: {balance.get('total', {})}")
            else:
                print("[TradingEngine] Warning: No API credentials provided (dry-run mode)")
        
        except Exception as e:
            print(f"[TradingEngine] Error initializing exchange: {e}")
            self.exchange = None
    
    async def _init_strategy(self) -> None:
        """Initialize the trading strategy."""
        strategy_name = self.config['strategy']['name']
        strategy_params = self.config['strategy']['params'].copy()
        
        # Add fee/slippage params
        strategy_params['fee_bps'] = self.config.get('fees.fee_bps', 4.0)
        strategy_params['slippage_bps'] = self.config.get('fees.slippage_bps', 0.0)
        
        # Get strategy class
        if strategy_name not in aVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_cls = aVAILABLE_STRATEGIES[strategy_name]
        price_column = self.config['price_column']
        
        # Special handling for orderbook depth strategy
        if strategy_name == 'orderbook_depth':
            strategy_params['symbol'] = self.config['symbol']
            strategy_params['base_dir'] = self.config.get('data.orderbook_depth_dir', 'data/orderbook_depth')
        
        self.strategy = strategy_cls(price_column=price_column, **strategy_params)
        self.strategy_params = strategy_params
        
        print(f"[TradingEngine] Strategy '{strategy_name}' initialized with params: {strategy_params}")
    
    async def start(self) -> None:
        """Start the trading engine."""
        self.running = True
        
        # Start orderbook streamer
        asyncio.create_task(self.orderbook_streamer.start(self.config['mode']))
        
        # Wait for initial data
        print("[TradingEngine] Waiting for price data...")
        has_data = await self.price_feeder.wait_for_data(timeout_sec=300)
        if not has_data:
            raise RuntimeError("Failed to get initial price data")
        
        print("[TradingEngine] Price data available. Starting trading loop...")
        
        # Main trading loop
        await self._trading_loop()
    
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        while self.running:
            try:
                current_time = time.time()
                
                # Update signals periodically
                if current_time - self.last_signal_update >= self.signal_update_interval:
                    await self._update_signals()
                    self.last_signal_update = current_time
                
                # Check for position changes
                await self._check_position_changes()
                
                # Update performance metrics
                await self._update_performance()
                
                # Sleep briefly
                await asyncio.sleep(1)
            
            except Exception as e:
                print(f"[TradingEngine] Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)
    
    async def _update_signals(self) -> None:
        """Update strategy signals based on latest data."""
        try:
            # Get latest price data
            price_df = self.price_feeder.get_latest_data()
            if price_df.empty:
                print("[TradingEngine] No price data available")
                return
            
            # Get orderbook data
            orderbook_df = self.orderbook_streamer.get_dataframe()
            
            # For orderbook depth strategy, we need to manually save orderbook data
            # to the expected location for the strategy to read
            if self.config['strategy']['name'] == 'orderbook_depth':
                await self._save_orderbook_snapshot(orderbook_df)
            
            # Generate signals
            signals = self.strategy.generate_signals(price_df)
            
            if signals.empty:
                print("[TradingEngine] No signals generated")
                return
            
            # Get latest signal
            latest_signal = float(signals.iloc[-1])
            self.current_signal = latest_signal
            # Save as previous for next comparison and persist state
            self.prev_signal = latest_signal
            self._save_state()
            
            print(f"[TradingEngine] Signal updated: {latest_signal} (position: {self.current_position})")
        
        except Exception as e:
            print(f"[TradingEngine] Error updating signals: {e}")
            import traceback
            traceback.print_exc()
    
    async def _save_orderbook_snapshot(self, orderbook_df: pd.DataFrame) -> None:
        """Save orderbook snapshot for strategy to read."""
        if orderbook_df.empty:
            return
        
        try:
            # Save to CSV in expected format
            symbol = self.config['symbol']
            ob_base = self.config.get('data.orderbook_depth_dir', 'data/orderbook_depth')
            base_dir = Path(ob_base) / symbol
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Get current date
            today = datetime.now(timezone.utc).date()
            filename = f"{symbol}-bookDepth-{today.year}-{today.month:02d}-{today.day:02d}.csv"
            filepath = base_dir / filename
            
            # Append or create
            if filepath.exists():
                existing = pd.read_csv(filepath)
                combined = pd.concat([existing, orderbook_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp', 'percentage'], keep='last')
                combined.to_csv(filepath, index=False)
            else:
                orderbook_df.to_csv(filepath, index=False)
        
        except Exception as e:
            print(f"[TradingEngine] Error saving orderbook snapshot: {e}")
    
    async def _check_position_changes(self) -> None:
        """Check if we need to open/close positions based on signals."""
        if self.current_signal == self.current_position:
            return  # No change needed
        
        # Determine action
        # Only open on signal change to avoid duplicate re-opens after restart
        if self.current_position == 0 and self.current_signal != 0 and self.current_signal != self.prev_signal:
            # Open new position
            await self._open_position(self.current_signal)
        elif self.current_position != 0 and self.current_signal == 0:
            # Close position
            await self._close_position()
        elif self.current_position != 0 and self.current_signal != 0 and self.current_signal != self.current_position:
            # Reverse position
            await self._close_position()
            await self._open_position(self.current_signal)
    
    async def _open_position(self, direction: float) -> None:
        """Open a new position.
        
        Args:
            direction: 1 for long, -1 for short
        """
        try:
            # Get current price
            price_column = self.config['price_column']
            current_price = self.price_feeder.get_latest_price(price_column)
            
            if current_price is None:
                print("[TradingEngine] Cannot open position: no price available")
                return
            
            # Calculate position size
            position_mode = self.config.get('position_sizing.mode', 'fixed_fraction')
            
            if position_mode == 'fixed_fraction':
                fraction = self.config.get('position_sizing.params.fraction', 1.0)
                position_value = self.current_capital * fraction
            else:
                position_value = self.current_capital
            
            # Convert to base currency quantity
            quantity = position_value / current_price
            
            # Execute order (if exchange is configured)
            if self.exchange and self.config.get('binance.api_key'):
                try:
                    symbol = self.config['symbol']
                    side = 'buy' if direction > 0 else 'sell'
                    
                    order = await self.exchange.create_market_order(
                        symbol=symbol,
                        side=side,
                        amount=quantity,
                    )
                    
                    print(f"[TradingEngine] Order executed: {order}")
                    executed_price = float(order.get('average', current_price))
                    executed_qty = float(order.get('filled', quantity))
                except Exception as e:
                    print(f"[TradingEngine] Order execution failed: {e}")
                    executed_price = current_price
                    executed_qty = quantity
            else:
                print(f"[TradingEngine] DRY RUN: Would open {direction} position at {current_price}")
                executed_price = current_price
                executed_qty = quantity
            
            # Update position state
            self.current_position = direction
            self.entry_price = executed_price
            self.position_size = executed_qty
            
            # Record trade
            self.performance_tracker.record_trade(
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                direction='long' if direction > 0 else 'short',
                action='open',
                price=executed_price,
                quantity=executed_qty,
                signal=self.current_signal,
            )

            # Telegram alert for trade open
            try:
                notifier = getattr(self, 'telegram_notifier', None)
                if notifier:
                    await notifier.send_trade_alert(
                        action='open',
                        direction='long' if direction > 0 else 'short',
                        price=executed_price,
                        quantity=executed_qty,
                        symbol=self.config['symbol'],
                    )
            except Exception as _e:
                pass

            # Persist state after opening
            self._save_state()
            
            print(f"[TradingEngine] Opened {'LONG' if direction > 0 else 'SHORT'} position: "
                  f"{executed_qty:.6f} @ {executed_price:.6f}")
        
        except Exception as e:
            print(f"[TradingEngine] Error opening position: {e}")
            import traceback
            traceback.print_exc()
    
    async def _close_position(self) -> None:
        """Close the current position."""
        if self.current_position == 0:
            return
        
        try:
            # Get current price
            price_column = self.config['price_column']
            current_price = self.price_feeder.get_latest_price(price_column)
            
            if current_price is None:
                print("[TradingEngine] Cannot close position: no price available")
                return
            
            # Calculate P&L
            if self.entry_price:
                if self.current_position > 0:
                    # Long position
                    pnl = (current_price - self.entry_price) * self.position_size
                else:
                    # Short position
                    pnl = (self.entry_price - current_price) * self.position_size
                
                pnl_pct = (pnl / (self.entry_price * self.position_size)) * 100
            else:
                pnl = 0
                pnl_pct = 0
            
            # Execute closing order
            if self.exchange and self.config.get('binance.api_key'):
                try:
                    symbol = self.config['symbol']
                    side = 'sell' if self.current_position > 0 else 'buy'
                    
                    order = await self.exchange.create_market_order(
                        symbol=symbol,
                        side=side,
                        amount=self.position_size,
                    )
                    
                    print(f"[TradingEngine] Close order executed: {order}")
                    executed_price = float(order.get('average', current_price))
                except Exception as e:
                    print(f"[TradingEngine] Close order failed: {e}")
                    executed_price = current_price
            else:
                print(f"[TradingEngine] DRY RUN: Would close position at {current_price}")
                executed_price = current_price
            
            # Update capital
            self.current_capital += pnl
            
            # Record trade
            self.performance_tracker.record_trade(
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                direction='long' if self.current_position > 0 else 'short',
                action='close',
                price=executed_price,
                quantity=self.position_size,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )

            # Telegram alert for trade close
            try:
                notifier = getattr(self, 'telegram_notifier', None)
                if notifier:
                    await notifier.send_trade_alert(
                        action='close',
                        direction='long' if self.current_position > 0 else 'short',
                        price=executed_price,
                        quantity=self.position_size,
                        symbol=self.config['symbol'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
            except Exception as _e:
                pass
            
            print(f"[TradingEngine] Closed {'LONG' if self.current_position > 0 else 'SHORT'} position: "
                  f"P&L = ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Reset position
            self.current_position = 0.0
            self.entry_price = None
            self.position_size = 0.0
            self._save_state()
        
        except Exception as e:
            print(f"[TradingEngine] Error closing position: {e}")
            import traceback
            traceback.print_exc()
    
    async def _update_performance(self) -> None:
        """Update performance metrics."""
        try:
            # Calculate unrealized P&L for open positions
            if self.current_position != 0 and self.entry_price:
                current_price = self.price_feeder.get_latest_price(self.config['price_column'])
                if current_price:
                    if self.current_position > 0:
                        self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
                    else:
                        self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
            else:
                self.unrealized_pnl = 0.0
            
            # Update tracker
            self.performance_tracker.update(
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                capital=self.current_capital,
                unrealized_pnl=self.unrealized_pnl,
                position=self.current_position,
                signal=self.current_signal,
            )
        
        except Exception as e:
            print(f"[TradingEngine] Error updating performance: {e}")
    
    async def stop(self) -> None:
        """Stop the trading engine."""
        print("[TradingEngine] Stopping...")
        self.running = False
        
        # Close any open positions
        if self.current_position != 0:
            await self._close_position()
        
        # Stop orderbook streamer
        if self.orderbook_streamer:
            await self.orderbook_streamer.stop()
        
        # Close exchange connection
        if self.exchange:
            await self.exchange.close()
        
        print("[TradingEngine] Stopped")

    def _load_state(self) -> None:
        try:
            if self.state_path.exists():
                with open(self.state_path, 'r') as f:
                    st = json.load(f)
                self.current_position = float(st.get('current_position', 0.0))
                self.current_signal = float(st.get('last_signal', 0.0))
                self.prev_signal = float(st.get('last_signal', 0.0))
                self.entry_price = st.get('entry_price', None)
                if self.entry_price is not None:
                    self.entry_price = float(self.entry_price)
                self.position_size = float(st.get('position_size', 0.0))
                # Keep current_capital as-is unless present
                if 'current_capital' in st:
                    self.current_capital = float(st['current_capital'])
                print(f"[TradingEngine] State loaded from {self.state_path}")
        except Exception as e:
            print(f"[TradingEngine] Failed to load state: {e}")

    def _save_state(self) -> None:
        try:
            st = {
                'current_position': self.current_position,
                'last_signal': self.current_signal,
                'entry_price': self.entry_price,
                'position_size': self.position_size,
                'current_capital': self.current_capital,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
            with open(self.state_path, 'w') as f:
                json.dump(st, f)
        except Exception:
            pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        return {
            'position': self.current_position,
            'signal': self.current_signal,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'capital': self.current_capital,
            'unrealized_pnl': self.unrealized_pnl,
            'total_value': self.current_capital + self.unrealized_pnl,
        }

