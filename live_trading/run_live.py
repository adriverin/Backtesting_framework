"""Main entry point for live trading system."""
from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.config import TradingConfig
from live_trading.trading_engine import TradingEngine
from live_trading.telegram_notifier import TelegramNotifier
from live_trading.reoptimizer import StrategyReoptimizer
from live_trading.api_server import DashboardServer


class LiveTradingSystem:
    """Main live trading system coordinator."""
    
    def __init__(self, config_path: str | None = None):
        """Initialize live trading system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = TradingConfig(config_path)
        
        # Components
        self.trading_engine: TradingEngine | None = None
        self.telegram_notifier: TelegramNotifier | None = None
        self.reoptimizer: StrategyReoptimizer | None = None
        self.dashboard_server: DashboardServer | None = None
        
        # Runtime state
        self.running = False
        self.restart_count = 0
        self.max_restart_attempts = self.config.get('runtime.max_restart_attempts', 10)
    
    async def initialize(self) -> None:
        """Initialize all system components."""
        print("=" * 80)
        print("🚀 LIVE TRADING SYSTEM - INITIALIZATION")
        print("=" * 80)
        print(f"Symbol: {self.config['symbol']}")
        print(f"Mode: {self.config['mode'].upper()}")
        print(f"Timeframe: {self.config['timeframe']}")
        print(f"Strategy: {self.config['strategy']['name']}")
        print(f"Price Column: {self.config['price_column']}")
        print("=" * 80)
        
        # Initialize Telegram notifier
        telegram_config = self.config['telegram']
        self.telegram_notifier = TelegramNotifier(
            bot_token=telegram_config['bot_token'],
            chat_id=telegram_config['chat_id'],
            enabled=telegram_config['enabled'],
        )
        
        # Send startup notification
        await self.telegram_notifier.send_message(
            f"🚀 *Live Trading System Started*\n\n"
            f"Symbol: `{self.config['symbol']}`\n"
            f"Mode: `{self.config['mode'].upper()}`\n"
            f"Strategy: `{self.config['strategy']['name']}`"
        )
        
        # Initialize trading engine
        self.trading_engine = TradingEngine(self.config)
        
        # Add telegram notifier to engine
        self.trading_engine.telegram_notifier = self.telegram_notifier
        
        await self.trading_engine.initialize()
        
        # Initialize re-optimizer
        self.reoptimizer = StrategyReoptimizer(
            config=self.config,
            trading_engine=self.trading_engine,
        )
        
        # Initialize dashboard server
        self.dashboard_server = DashboardServer(
            trading_engine=self.trading_engine,
            config=self.config,
        )
        
        print("✅ Initialization complete!")
        print("=" * 80)
    
    async def start(self) -> None:
        """Start the live trading system."""
        self.running = True
        
        try:
            # Start all components
            tasks = [
                self.trading_engine.start(),
                self.reoptimizer.start(),
                self.dashboard_server.run(),
            ]
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
        
        except Exception as e:
            print(f"❌ Error in live trading system: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error notification
            await self.telegram_notifier.send_error_alert(
                error_type="System Error",
                error_msg=str(e),
            )
            
            # Attempt restart if enabled
            if self.config.get('runtime.auto_restart', True):
                await self._attempt_restart()
    
    async def _attempt_restart(self) -> None:
        """Attempt to restart the system after an error."""
        if self.restart_count >= self.max_restart_attempts:
            print(f"❌ Maximum restart attempts ({self.max_restart_attempts}) reached. Exiting.")
            await self.telegram_notifier.send_message(
                f"❌ *System Shutdown*\n\nMaximum restart attempts reached. Manual intervention required."
            )
            await self.stop()
            return
        
        self.restart_count += 1
        restart_delay = self.config.get('runtime.restart_delay_sec', 60)
        
        print(f"🔄 Attempting restart ({self.restart_count}/{self.max_restart_attempts}) in {restart_delay}s...")
        
        await self.telegram_notifier.send_message(
            f"🔄 *System Restarting*\n\nAttempt {self.restart_count}/{self.max_restart_attempts}"
        )
        
        # Stop current components
        await self._stop_components()
        
        # Wait before restart
        await asyncio.sleep(restart_delay)
        
        # Re-initialize and start
        await self.initialize()
        await self.start()
    
    async def _stop_components(self) -> None:
        """Stop all system components gracefully."""
        print("🛑 Stopping system components...")
        
        if self.trading_engine:
            await self.trading_engine.stop()
        
        if self.reoptimizer:
            await self.reoptimizer.stop()
        
        if self.dashboard_server:
            await self.dashboard_server.stop()
    
    async def stop(self) -> None:
        """Stop the live trading system."""
        print("=" * 80)
        print("🛑 SHUTTING DOWN LIVE TRADING SYSTEM")
        print("=" * 80)
        
        self.running = False
        
        # Stop all components
        await self._stop_components()
        
        # Send shutdown notification with performance summary
        try:
            te = self.trading_engine
            if te and te.performance_tracker:
                pt = te.performance_tracker
                metrics = pt.get_metrics()
                trades_df = pt.get_trades_df()
                start_cap = pt.initial_capital
                final_equity = te.current_capital + te.unrealized_pnl
                # Report round-trips (closed trades) to avoid counting open+close as two
                num_trades = int(len(trades_df[trades_df['action'] == 'close'])) if not trades_df.empty else int(metrics.get('num_trades', 0))
                win_rate = float(metrics.get('win_rate', 0.0))
                avg_win = float(metrics.get('avg_win', 0.0))
                avg_loss = float(metrics.get('avg_loss', 0.0))
                # Compute return from final vs starting equity to avoid staleness
                cum_return_pct = ((final_equity - start_cap) / start_cap) * 100 if start_cap > 0 else 0.0

                msg = (
                    "🛑 *Live Trading System Stopped*\n\n"
                    f"Starting Balance: `${start_cap:.2f}` USDT\n"
                    f"Final Equity: `${final_equity:.2f}` USDT\n"
                    f"Cumulative Simple Return: `{cum_return_pct:.2f}%`\n"
                    f"Number of Trades: `{num_trades}`\n"
                    f"Win Rate: `{win_rate:.1f}%`\n"
                    f"Average Win: `${avg_win:.2f}`\n"
                    f"Average Loss: `${avg_loss:.2f}`"
                )
            else:
                final_equity = self.trading_engine.current_capital + self.trading_engine.unrealized_pnl if self.trading_engine else 0.0
                msg = (
                    "🛑 *Live Trading System Stopped*\n\n"
                    f"Final Equity: `${final_equity:.2f}` USDT"
                )
            await self.telegram_notifier.send_message(msg)
        except Exception as _e:
            # Fallback minimal message if anything goes wrong building summary
            final_equity = self.trading_engine.current_capital + self.trading_engine.unrealized_pnl if self.trading_engine else 0.0
            await self.telegram_notifier.send_message(
                "🛑 *Live Trading System Stopped*\n\n"
                f"Final Equity: `${final_equity:.2f}` USDT"
            )
        
        print("✅ Shutdown complete")
        print("=" * 80)


async def main():
    """Main entry point."""
    # Create system
    system = LiveTradingSystem()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        print("\n⚠️  Shutdown signal received...")
        asyncio.create_task(system.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Initialize and start
        await system.initialize()
        await system.start()
    
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received...")
    
    finally:
        await system.stop()


if __name__ == "__main__":
    # Run the system
    asyncio.run(main())

