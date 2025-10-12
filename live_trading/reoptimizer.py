"""Periodic strategy re-optimization service."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd

from live_trading.config import TradingConfig
from live_trading.price_feeder import PriceFeeder

import sys
sys.path.append(str(Path(__file__).parent.parent))
from strategies import aVAILABLE_STRATEGIES  # type: ignore


class StrategyReoptimizer:
    """Periodically re-optimize strategy parameters based on recent performance."""
    
    def __init__(self, config: TradingConfig, trading_engine: Any):
        """Initialize reoptimizer.
        
        Args:
            config: Trading configuration
            trading_engine: Reference to the trading engine
        """
        self.config = config
        self.trading_engine = trading_engine
        
        self.enabled = config.get('reoptimization.enabled', True)
        self.interval_hours = config.get('reoptimization.interval_hours', 24)
        self.lookback_days = config.get('reoptimization.lookback_days', 365)
        self.min_data_points = config.get('reoptimization.min_data_points', 10000)
        
        self.running = False
        self.last_optimization_time: Optional[datetime] = None
        self.optimization_history: list[Dict[str, Any]] = []
    
    async def start(self) -> None:
        """Start the re-optimization loop."""
        if not self.enabled:
            print("[Reoptimizer] Disabled in config")
            return
        
        self.running = True
        print(f"[Reoptimizer] Started (interval: {self.interval_hours}h, lookback: {self.lookback_days}d)")
        
        # Run optimization loop
        await self._optimization_loop()
    
    async def _optimization_loop(self) -> None:
        """Main re-optimization loop."""
        while self.running:
            try:
                # Wait for interval
                await asyncio.sleep(self.interval_hours * 3600)
                
                # Run optimization
                await self._run_optimization()
            
            except Exception as e:
                print(f"[Reoptimizer] Error in optimization loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _run_optimization(self) -> None:
        """Execute strategy optimization."""
        print("[Reoptimizer] Running strategy re-optimization...")
        
        try:
            # Get historical data
            price_feeder = PriceFeeder(
                symbol=self.config['symbol'],
                interval=self.config['timeframe'],
                mode=self.config['mode'],
                data_dir=self.config.get('data.data_dir', 'data'),
            )
            
            # Load data with lookback
            price_data = price_feeder.get_latest_data()
            
            if price_data.empty:
                print("[Reoptimizer] No price data available")
                return
            
            # Filter to lookback period
            cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=self.lookback_days)
            price_data = price_data[price_data.index >= cutoff_date]
            
            if len(price_data) < self.min_data_points:
                print(f"[Reoptimizer] Insufficient data ({len(price_data)} < {self.min_data_points})")
                return
            
            # Get current strategy parameters
            old_params = self.trading_engine.strategy_params.copy()
            
            # Create new strategy instance for optimization
            strategy_name = self.config['strategy']['name']
            strategy_cls = aVAILABLE_STRATEGIES[strategy_name]
            
            # Prepare strategy kwargs
            strategy_kwargs = old_params.copy()
            
            # For orderbook depth strategy, we need to load orderbook data
            if strategy_name == 'orderbook_depth':
                orderbook_data = await self._load_orderbook_data(price_data)
                
                if orderbook_data.empty:
                    print("[Reoptimizer] No orderbook data available")
                    return
            
            # Create strategy and optimize
            price_column = self.config['price_column']
            strategy = strategy_cls(price_column=price_column, **strategy_kwargs)
            
            # Run optimization
            new_params, best_metric = strategy.optimize(price_data)
            
            print(f"[Reoptimizer] Optimization complete:")
            print(f"  Old params: {old_params}")
            print(f"  New params: {new_params}")
            print(f"  Best metric: {best_metric}")
            
            # Calculate improvement
            if hasattr(strategy, 'best_metric') and strategy.best_metric:
                old_metric = strategy.best_metric
                improvement_pct = ((best_metric - old_metric) / abs(old_metric)) * 100 if old_metric != 0 else 0
            else:
                improvement_pct = None
            
            # Update strategy in trading engine
            # Note: We update best_params which will be used for next signal generation
            self.trading_engine.strategy.best_params = new_params
            self.trading_engine.strategy_params.update({
                'lookback_mean': new_params[0] if isinstance(new_params, tuple) else new_params,
                'lookback_current': new_params[1] if isinstance(new_params, tuple) and len(new_params) > 1 else None,
            })
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': datetime.now(timezone.utc).replace(tzinfo=None),
                'old_params': old_params,
                'new_params': dict(new_params) if isinstance(new_params, dict) else new_params,
                'best_metric': best_metric,
                'improvement_pct': improvement_pct,
            })
            
            self.last_optimization_time = datetime.now(timezone.utc).replace(tzinfo=None)
            
            # Send Telegram notification
            if hasattr(self.trading_engine, 'telegram_notifier'):
                await self.trading_engine.telegram_notifier.send_reoptimization_alert(
                    old_params=old_params,
                    new_params=dict(new_params) if isinstance(new_params, dict) else new_params,
                    improvement=improvement_pct,
                )
            
            print("[Reoptimizer] Strategy parameters updated successfully")
        
        except Exception as e:
            print(f"[Reoptimizer] Optimization failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _load_orderbook_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Load orderbook data for the optimization period."""
        try:
            from strategies.orderbook_depth_strategy import read_data  # type: ignore
            
            symbol = self.config['symbol']
            start_date = price_data.index.min()
            end_date = price_data.index.max()
            
            ob_base = self.config.get('data.orderbook_depth_dir', 'data/orderbook_depth')
            orderbook_data = read_data(
                symbol=symbol,
                start_year=start_date.year,
                start_month=start_date.month,
                start_day=start_date.day,
                end_year=end_date.year,
                end_month=end_date.month,
                end_day=end_date.day,
                base_dir=ob_base,
                use_parquet_day_cache=True,
            )
            
            return orderbook_data
        
        except Exception as e:
            print(f"[Reoptimizer] Error loading orderbook data: {e}")
            return pd.DataFrame()
    
    async def stop(self) -> None:
        """Stop the re-optimization loop."""
        # Prevent multiple stop attempts
        if not self.running:
            return
        
        self.running = False
        print("[Reoptimizer] Stopped")
    
    def get_history(self) -> list[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history.copy()

