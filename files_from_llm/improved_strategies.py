"""
Improved trading strategies module with better performance and structure.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategyResult:
    """Results from strategy optimization."""
    best_params: Tuple
    best_profit_factor: float
    signal: pd.Series


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def generate_signal(self, ohlc: pd.DataFrame, **params) -> pd.Series:
        """Generate trading signals for given parameters."""
        pass
    
    @abstractmethod
    def optimize(self, ohlc: pd.DataFrame, price_column: str = 'close') -> StrategyResult:
        """Optimize strategy parameters."""
        pass
    
    @staticmethod
    def calculate_profit_factor(signal: pd.Series, returns: pd.Series) -> float:
        """Calculate profit factor for strategy returns."""
        strategy_returns = signal * returns
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 0
        
        return positive_returns.sum() / negative_returns.abs().sum()


class DonchianStrategy(BaseStrategy):
    """Donchian Channel Breakout Strategy with improved performance."""
    
    def generate_signal(self, ohlc: pd.DataFrame, lookback: int) -> pd.Series:
        """Generate Donchian breakout signals."""
        close = ohlc['close']
        upper = close.rolling(lookback - 1).max().shift(1)
        lower = close.rolling(lookback - 1).min().shift(1)
        
        # Vectorized signal generation
        signal = pd.Series(np.nan, index=ohlc.index, dtype=float)
        signal = np.where(close > upper, 1, signal)
        signal = np.where(close < lower, -1, signal)
        
        return pd.Series(signal, index=ohlc.index).ffill()
    
    def optimize(self, ohlc: pd.DataFrame, price_column: str = 'close') -> StrategyResult:
        """Optimize Donchian lookback period."""
        returns = np.log(ohlc[price_column]).diff().shift(-1)
        best_profit_factor = 0
        best_lookback = 2
        
        # Vectorized optimization
        lookback_range = range(2, 169)
        for lookback in lookback_range:
            signal = self.generate_signal(ohlc, lookback)
            profit_factor = self.calculate_profit_factor(signal, returns)
            
            if profit_factor > best_profit_factor:
                best_profit_factor = profit_factor
                best_lookback = lookback
        
        final_signal = self.generate_signal(ohlc, best_lookback)
        return StrategyResult((best_lookback,), best_profit_factor, final_signal)


class MovingAverageStrategy(BaseStrategy):
    """Moving Average Crossover Strategy with optimized performance."""
    
    def generate_signal(self, ohlc: pd.DataFrame, price_column: str, 
                       fast_period: int, slow_period: int) -> pd.Series:
        """Generate MA crossover signals."""
        price = ohlc[price_column]
        fast_ma = price.rolling(window=fast_period).mean()
        slow_ma = price.rolling(window=slow_period).mean()
        
        # Vectorized signal generation
        signal = np.where(fast_ma > slow_ma, 1, -1)
        return pd.Series(signal, index=ohlc.index).ffill()
    
    def optimize(self, ohlc: pd.DataFrame, price_column: str = 'close') -> StrategyResult:
        """Optimize MA periods with improved efficiency."""
        returns = np.log(ohlc[price_column]).diff().shift(-1)
        price = ohlc[price_column]
        
        best_profit_factor = 0
        best_params = (1, 2)
        
        # Pre-calculate all moving averages to avoid redundant calculations
        max_period = 100
        mas = {}
        for period in range(1, max_period):
            mas[period] = price.rolling(window=period).mean()
        
        # Optimized nested loop
        for fast in range(1, max_period):
            fast_ma = mas[fast]
            for slow in range(fast + 1, max_period):
                slow_ma = mas[slow]
                
                # Vectorized signal and profit factor calculation
                signal = np.where(fast_ma > slow_ma, 1, 0)
                strategy_returns = signal * returns
                
                positive_sum = strategy_returns[strategy_returns > 0].sum()
                negative_sum = strategy_returns[strategy_returns < 0].abs().sum()
                
                if negative_sum > 0:
                    profit_factor = positive_sum / negative_sum
                    if profit_factor > best_profit_factor:
                        best_profit_factor = profit_factor
                        best_params = (fast, slow)
        
        final_signal = self.generate_signal(ohlc, price_column, best_params[0], best_params[1])
        return StrategyResult(best_params, best_profit_factor, final_signal)


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    _strategies = {
        'donchian': DonchianStrategy,
        'ma': MovingAverageStrategy,
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str) -> BaseStrategy:
        """Create strategy instance by name."""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(cls._strategies.keys())}")
        
        return cls._strategies[strategy_name]()
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())