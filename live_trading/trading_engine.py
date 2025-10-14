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
        self.last_state_save: float = 0
        self.last_position_open: float = 0  # Track when we last opened a position
        # Update signals interval (configurable)
        self.signal_update_interval: int = int(config.get('runtime.signal_update_interval_sec', 60))
        self.state_save_interval: int = 300  # Save state every 5 minutes
        
        # Binance client (for order execution)
        self.exchange = None
        # Telegram notifier (injected by run_live)
        # Optional attribute; present when Telegram is configured
        # self.telegram_notifier will be set externally

        # Persisted state
        log_dir = Path(self.config.get('logging.dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = log_dir / f"state_{self.config['symbol']}_{self.config['timeframe']}.json"
        self.perf_state_path = log_dir / f"performance_{self.config['symbol']}_{self.config['timeframe']}.json"
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
        
        # Load or reset persisted state based on config/env
        import os
        start_fresh_cfg = bool(self.config.get('runtime.start_fresh', False))
        start_fresh_env = str(os.getenv('START_FRESH', '')).lower() in ('1','true','yes','y')
        if start_fresh_cfg or start_fresh_env:
            print("[TradingEngine] Starting fresh: resetting persisted state")
            self._reset_state()
        else:
            self._load_state()
            # Load performance tracker state (separate file)
            self.performance_tracker.load_state(self.perf_state_path)
        
        print("[TradingEngine] Initialization complete!")
    
    async def _init_exchange(self) -> None:
        """Initialize Binance exchange connection."""
        try:
            import ccxt.async_support as ccxt  # type: ignore

            api_key = self.config.get('binance.api_key', '')
            api_secret = self.config.get('binance.api_secret', '')
            testnet = self.config.get('binance.testnet', False)

            if self.config['mode'] == 'futures':
                # Prefer dedicated USDT-M futures exchange
                self.exchange = ccxt.binanceusdm({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })
            else:
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })

            # Enable testnet if requested
            if testnet:
                try:
                    self.exchange.set_sandbox_mode(True)
                except Exception:
                    pass
            
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
        asyncio.create_task(self.orderbook_streamer.start(self.config['mode'], bool(self.config.get('binance.testnet', False))))
        
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
                
                # Periodic state save (to persist equity curve and metrics)
                if current_time - self.last_state_save >= self.state_save_interval:
                    if self.performance_tracker:
                        self.performance_tracker.save_state(self.perf_state_path)
                    self.last_state_save = current_time
                
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
            
            # Debug: Show data being used (only on first run or every 10 minutes)
            current_time = time.time()
            if not hasattr(self, '_last_data_debug') or (current_time - self._last_data_debug > 600):
                self._last_data_debug = current_time
                # Count historical orderbook data
                if self.config['strategy']['name'] == 'orderbook_depth':
                    base_dir = self.config.get('data.orderbook_depth_dir', 'data/orderbook_depth')
                    symbol_dir = Path(base_dir) / self.config['symbol']
                    csv_files = list(symbol_dir.glob('*.csv'))
                    print(f"[TradingEngine] 📊 Strategy using {len(csv_files)} days of historical orderbook data")
                    print(f"[TradingEngine] 📊 Adding ~17,280 new datapoints per day (5s intervals)")
                    print(f"[TradingEngine] 📊 With 210 days = ~3.6M datapoints, new data has <0.0005% impact per update")
            
            # Get latest signal
            latest_signal = float(signals.iloc[-1])
            prev_signal = self.prev_signal
            
            # Compute z-score for debugging (orderbook depth strategy only)
            z_score_current = None
            if self.config['strategy']['name'] == 'orderbook_depth':
                z_score_current = self._compute_orderbook_z_score()
            
            # Debug info
            z_threshold = getattr(self.strategy, 'z_threshold', None)
            exit_band = getattr(self.strategy, 'exit_band', None)
            
            self.current_signal = latest_signal
            
            # Show signal change details with z-score
            z_str = f"{z_score_current:+.4f}" if z_score_current is not None else "N/A"
            if latest_signal != prev_signal:
                print(f"[TradingEngine] ⚡ SIGNAL CHANGE: {prev_signal} → {latest_signal} | Position: {self.current_position}")
                print(f"               z-score: {z_str} | entry: ±{z_threshold} | exit: {exit_band}")
            else:
                # Show detailed z-score info to diagnose why signal isn't changing
                if z_score_current is not None:
                    # Determine trading zone
                    if z_score_current > z_threshold:
                        zone = "LONG ENTRY"
                    elif z_score_current < -z_threshold:
                        zone = "SHORT ENTRY"
                    elif exit_band is not None and exit_band <= 0:
                        band = abs(exit_band)
                        if self.current_position > 0 and z_score_current > -band:
                            zone = "LONG HOLD"
                        elif self.current_position < 0 and z_score_current < band:
                            zone = "SHORT HOLD"
                        elif self.current_position > 0 and z_score_current <= -band:
                            zone = "LONG EXIT"
                        elif self.current_position < 0 and z_score_current >= band:
                            zone = "SHORT EXIT"
                        else:
                            zone = "FLAT ZONE"
                    else:
                        zone = "NEUTRAL"
                    
                    print(f"[TradingEngine] Signal: {latest_signal} | Pos: {self.current_position} | z: {z_str} [{zone}]")
                else:
                    print(f"[TradingEngine] Signal: {latest_signal} | Position: {self.current_position}")
        
        except Exception as e:
            print(f"[TradingEngine] Error updating signals: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_orderbook_z_score(self) -> float | None:
        """Compute current z-score from the orderbook data (same as strategy sees)."""
        try:
            # Load orderbook data the same way strategy does
            from pathlib import Path
            import pandas as pd
            
            base_dir = self.config.get('data.orderbook_depth_dir', 'data/orderbook_depth')
            symbol = self.config['symbol']
            symbol_dir = Path(base_dir) / symbol
            
            # Get today's file
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).date()
            filename = f"{symbol}-bookDepth-{today.year}-{today.month:02d}-{today.day:02d}.csv"
            filepath = symbol_dir / filename
            
            if not filepath.exists():
                return None
            
            # Read today's data
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Get strategy parameters
            percentage = getattr(self.strategy, 'percentage', 2)
            lookback_mean = getattr(self.strategy, 'lookback_mean_fixed', None)
            lookback_current = getattr(self.strategy, 'lookback_current_fixed', None)
            
            if not lookback_mean or not lookback_current:
                return None
            
            # Calculate z-score (same logic as strategy)
            eps = 1e-9
            pivoted = df.pivot(index="timestamp", columns="percentage", values="notional").sort_index()
            
            if percentage not in pivoted.columns or -percentage not in pivoted.columns:
                return None
            
            ask = pivoted[percentage]
            bid = pivoted[-percentage]
            aligned = pd.concat({"ask": ask, "bid": bid}, axis=1).dropna()
            
            if aligned.empty:
                return None
            
            ratio = aligned["bid"] / (aligned["ask"] + aligned["bid"] + eps)
            
            # Rolling windows (time-based)
            r_ma = ratio.rolling(lookback_mean).mean()
            r_std = ratio.rolling(lookback_mean).std()
            r_current = ratio.rolling(lookback_current).mean()
            
            z_score = -(r_current - r_ma) / (r_std + eps)
            
            if not z_score.empty:
                return float(z_score.iloc[-1])
            
            return None
            
        except Exception as e:
            # Don't spam logs with errors
            if not hasattr(self, '_z_score_error_logged'):
                print(f"[TradingEngine] Could not compute z-score: {e}")
                self._z_score_error_logged = True
            return None
    
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
        # Position matches signal - all good
        if self.current_signal == self.current_position:
            return
        
        # Position doesn't match signal - need to adjust
        current_time = time.time()
        min_time_between_opens = 5  # seconds - prevent rapid duplicate opens
        
        if self.current_position == 0 and self.current_signal != 0:
            # Need to open position
            # Prevent rapid duplicate opens (e.g., if signal just confirmed)
            time_since_last_open = current_time - self.last_position_open
            if time_since_last_open < min_time_between_opens:
                # Just opened recently, wait to avoid duplicate
                return
            
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
            # Get current price (use valuation price column to match UI/mark)
            valuation_col = self.config.get('valuation_price_column', 'close')
            price_column = self.config['price_column']
            current_price = self.price_feeder.get_latest_price(valuation_col) or self.price_feeder.get_latest_price(price_column)
            
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
            self.last_position_open = time.time()  # Track when we opened
            
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
            self.prev_signal = self.current_signal
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
            # Get current price (valuation)
            valuation_col = self.config.get('valuation_price_column', 'close')
            price_column = self.config['price_column']
            current_price = self.price_feeder.get_latest_price(valuation_col) or self.price_feeder.get_latest_price(price_column)
            
            if current_price is None:
                print("[TradingEngine] Cannot close position: no price available")
                return
            
            # Calculate P&L using valuation price (match charts/UI)
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
            # Allow re-entry on next non-zero signal
            self.prev_signal = 0.0
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
        # Prevent multiple stop attempts
        if not self.running:
            return
        
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
                
                # Handle both new and old state format for backward compatibility
                # New format: current_position, entry_price, position_size, last_signal
                # Old format: position_qty, position_entry_price, prev_signal
                
                # Load position - check both old and new keys
                if 'current_position' in st:
                    self.current_position = float(st['current_position'])
                elif 'position_qty' in st:
                    # Old format: infer position direction from prev_signal
                    pos_qty = float(st.get('position_qty', 0.0))
                    if pos_qty != 0:
                        # If we have a position, check prev_signal to determine direction
                        old_signal = float(st.get('prev_signal', 0.0))
                        if old_signal > 0:
                            self.current_position = 1.0  # Long
                        elif old_signal < 0:
                            self.current_position = -1.0  # Short
                        else:
                            self.current_position = 0.0
                    else:
                        self.current_position = 0.0
                else:
                    self.current_position = 0.0
                
                # Load entry price
                if 'entry_price' in st:
                    self.entry_price = st.get('entry_price')
                elif 'position_entry_price' in st:
                    self.entry_price = st.get('position_entry_price')
                else:
                    self.entry_price = None
                if self.entry_price is not None:
                    self.entry_price = float(self.entry_price)
                
                # Load position size
                if 'position_size' in st:
                    self.position_size = float(st['position_size'])
                elif 'position_qty' in st:
                    self.position_size = abs(float(st.get('position_qty', 0.0)))
                else:
                    self.position_size = 0.0
                
                # Load signal state
                if 'last_signal' in st:
                    self.current_signal = float(st['last_signal'])
                    self.prev_signal = float(st['last_signal'])
                elif 'prev_signal' in st:
                    # Old format - use prev_signal as last known signal
                    old_sig = float(st['prev_signal'])
                    self.current_signal = old_sig
                    self.prev_signal = old_sig
                else:
                    self.current_signal = 0.0
                    self.prev_signal = 0.0
                
                # Load capital
                if 'current_capital' in st:
                    self.current_capital = float(st['current_capital'])
                
                print(f"[TradingEngine] State loaded from {self.state_path}")
                print(f"[TradingEngine]   Position: {self.current_position}, Entry: {self.entry_price}, Size: {self.position_size}")
                print(f"[TradingEngine]   Signal: {self.current_signal}, Prev signal: {self.prev_signal}")
        except Exception as e:
            print(f"[TradingEngine] Failed to load state: {e}")
            import traceback
            traceback.print_exc()

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
            
            # Save performance tracker state
            if self.performance_tracker:
                self.performance_tracker.save_state(self.perf_state_path)
        except Exception:
            pass

    def _reset_state(self) -> None:
        try:
            if self.state_path.exists():
                self.state_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            if self.perf_state_path.exists():
                self.perf_state_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        self.current_position = 0.0
        self.current_signal = 0.0
        self.prev_signal = 0.0
        self.entry_price = None
        self.position_size = 0.0
        self.current_capital = self.initial_capital
        self._save_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        # Latest price for display (prefer valuation column if configured)
        latest_price: float | None = None
        try:
            if self.price_feeder is not None:
                valuation_col = self.config.get('valuation_price_column', self.config['price_column'])
                latest_price = self.price_feeder.get_latest_price(valuation_col) or self.price_feeder.get_latest_price(self.config['price_column'])
        except Exception:
            latest_price = None

        # Compute live z-score using current orderbook buffer and strategy params
        z_val: float | None = None
        try:
            z_val = self._compute_live_z_score()
        except Exception:
            z_val = None

        # Strategy thresholds (if available)
        z_threshold = None
        exit_band = None
        try:
            if self.strategy is not None and hasattr(self.strategy, 'z_threshold'):
                z_threshold = float(getattr(self.strategy, 'z_threshold'))
            if self.strategy is not None and hasattr(self.strategy, 'exit_band'):
                eb = getattr(self.strategy, 'exit_band')
                exit_band = None if eb is None else float(eb)
        except Exception:
            pass

        return {
            'position': self.current_position,
            'signal': self.current_signal,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'capital': self.current_capital,
            'unrealized_pnl': self.unrealized_pnl,
            'total_value': self.current_capital + self.unrealized_pnl,
            'price': latest_price,
            'z_score': z_val,
            'z_threshold': z_threshold,
            'exit_band': exit_band,
        }

    def _compute_live_z_score(self) -> float | None:
        """Compute the current z-score matching the orderbook depth strategy settings.

        Uses the in-memory orderbook buffer to approximate the rolling statistics.
        Returns None if insufficient data or parameters are unavailable.
        """
        # Ensure components exist
        if self.orderbook_streamer is None:
            return None
        if self.strategy is None:
            return None

        # Determine percentage and lookback windows
        try:
            percentage = int(self.strategy_params.get('percentage', 1))
        except Exception:
            percentage = 1

        lb_mean = None
        lb_cur = None
        # Prefer optimized params if available
        try:
            best_params = getattr(self.strategy, 'best_params', None)
            if best_params is not None:
                lb_mean, lb_cur = best_params
        except Exception:
            lb_mean = None
            lb_cur = None

        # Fallback to fixed params if provided
        if lb_mean is None or lb_cur is None:
            try:
                lb_mean = getattr(self.strategy, 'lookback_mean_fixed', None)
                lb_cur = getattr(self.strategy, 'lookback_current_fixed', None)
            except Exception:
                pass

        # If still unavailable, cannot compute
        if lb_mean is None or lb_cur is None:
            return None

        # Build DataFrame from buffer
        df = self.orderbook_streamer.get_dataframe()
        if df.empty:
            return None

        # Pivot to get notional at ±percentage
        try:
            pivot = df.pivot(index='timestamp', columns='percentage', values='notional').sort_index()
        except Exception:
            return None

        if (percentage not in pivot.columns) or (-percentage not in pivot.columns):
            return None

        ask = pivot[percentage]
        bid = pivot[-percentage]
        aligned = pd.concat({'ask': ask, 'bid': bid}, axis=1).dropna()
        if aligned.empty:
            return None

        # Ratio series
        eps = 1e-9
        ratio = aligned['bid'] / (aligned['ask'] + aligned['bid'] + eps)

        # Rolling stats: support both time-based (str) and count-based (int)
        def _rolling_mean_std(series: pd.Series, window: str | int) -> tuple[pd.Series, pd.Series]:
            if isinstance(window, str):
                r_mean = series.rolling(window).mean()
                r_std = series.rolling(window).std()
            else:
                r_mean = series.rolling(window=window, min_periods=window).mean()
                r_std = series.rolling(window=window, min_periods=window).std()
            return r_mean, r_std

        def _rolling_current(series: pd.Series, window: str | int) -> pd.Series:
            if isinstance(window, str):
                return series.rolling(window).mean()
            return series.rolling(window=window, min_periods=window).mean()

        r_ma, r_std = _rolling_mean_std(ratio, lb_mean)
        r_current = _rolling_current(ratio, lb_cur)

        z_series = -(r_current - r_ma) / (r_std + eps)
        if z_series.empty:
            return None

        z_latest = float(z_series.dropna().iloc[-1]) if not z_series.dropna().empty else None
        return z_latest

