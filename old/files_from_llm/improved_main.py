"""
Improved main application with better structure, performance, and usability.

Key improvements over original:
1. Dataclass-based configuration with validation
2. Separation of concerns with dedicated classes
3. Better error handling and type hints
4. Cached data loading for performance
5. Flexible strategy selection and parameter management
6. Enhanced plotting and analysis capabilities
7. Simplified and intuitive API
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import warnings

from improved_strategies import StrategyFactory, BaseStrategy, StrategyResult
from improved_permutations import PermutationGenerator, PermutationConfig
from improved_monte_carlo import MonteCarloTester, MonteCarloConfig, MonteCarloResult

warnings.filterwarnings('ignore')


@dataclass
class TradingConfig:
    """Configuration for trading analysis with validation."""
    # Data parameters
    start_date: str = '2018-01-01'
    end_date: str = '2020-01-01'
    data_symbol: str = 'BTCUSD'
    timeframe: str = '1h'
    price_column: str = 'close'
    data_path: str = 'data'
    
    # Strategy parameters
    strategy: str = 'ma'
    
    # Monte Carlo parameters
    n_permutations: int = 100
    perm_start_index: int = 0
    confidence_level: float = 0.05
    mc_seed: Optional[int] = None
    
    # Analysis parameters
    generate_plots: bool = False
    plot_style: str = 'dark_background'
    run_monte_carlo: bool = True
    compare_strategies: bool = False
    
    # Performance parameters
    cache_data: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_dates()
        self._validate_strategy()
        self._validate_parameters()
    
    def _validate_dates(self):
        """Validate date format and order."""
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            if start >= end:
                raise ValueError("Start date must be before end date")
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def _validate_strategy(self):
        """Validate strategy name."""
        available = StrategyFactory.get_available_strategies()
        if self.strategy not in available:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Available: {available}")
    
    def _validate_parameters(self):
        """Validate other parameters."""
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be positive")
        if not 0 <= self.confidence_level <= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.perm_start_index < 0:
            raise ValueError("perm_start_index must be non-negative")


class DataManager:
    """Efficient data loading and caching manager."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, symbol: Optional[str] = None, 
                  timeframe: Optional[str] = None) -> pd.DataFrame:
        """Load OHLCV data with caching."""
        symbol = symbol or self.config.data_symbol
        timeframe = timeframe or self.config.timeframe
        
        cache_key = f"{symbol}_{timeframe}"
        
        if self.config.cache_data and cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            file_path = Path(self.config.data_path) / f"ohlcv_{symbol}_{timeframe}.parquet"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_parquet(file_path)
            
            if self.config.cache_data:
                self._cache[cache_key] = df
        
        # Filter by date range
        filtered_df = df[
            (df.index >= self.config.start_date) & 
            (df.index < self.config.end_date)
        ].copy()
        
        if len(filtered_df) == 0:
            raise ValueError(f"No data found for date range {self.config.start_date} to {self.config.end_date}")
        
        return filtered_df
    
    def get_returns(self, symbol: Optional[str] = None) -> pd.Series:
        """Calculate log returns for given symbol."""
        df = self.load_data(symbol)
        returns = np.log(df[self.config.price_column]).diff().shift(-1)
        return returns.dropna()
    
    def clear_cache(self):
        """Clear data cache."""
        self._cache.clear()


class TradingAnalyzer:
    """Main trading analysis engine with improved performance and features."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.strategy = StrategyFactory.create_strategy(config.strategy)
        
        # Initialize Monte Carlo components
        mc_config = MonteCarloConfig(
            n_permutations=config.n_permutations,
            start_index=config.perm_start_index,
            confidence_level=config.confidence_level,
            generate_plots=config.generate_plots,
            plot_style=config.plot_style,
            seed=config.mc_seed
        )
        self.mc_tester = MonteCarloTester(mc_config)
    
    def run_strategy_analysis(self) -> StrategyResult:
        """Run strategy optimization and analysis."""
        print(f"Loading data for {self.config.data_symbol} ({self.config.timeframe})")
        print(f"Date range: {self.config.start_date} to {self.config.end_date}")
        
        df = self.data_manager.load_data()
        print(f"Loaded {len(df)} data points")
        
        print(f"\nOptimizing {self.config.strategy.upper()} strategy...")
        result = self.strategy.optimize(df, self.config.price_column)
        
        print(f"Best parameters: {result.best_params}")
        print(f"Best profit factor: {result.best_profit_factor:.4f}")
        
        return result
    
    def run_monte_carlo_test(self) -> MonteCarloResult:
        """Run Monte Carlo significance test."""
        df = self.data_manager.load_data()
        
        print(f"\n{'='*60}")
        print(f"MONTE CARLO ANALYSIS - {self.config.strategy.upper()} STRATEGY")
        print(f"{'='*60}")
        
        result = self.mc_tester.run_test(df, self.strategy, self.config.price_column)
        
        return result
    
    def compare_strategies(self, strategies: Optional[List[str]] = None) -> Dict[str, MonteCarloResult]:
        """Compare multiple strategies using Monte Carlo testing."""
        if strategies is None:
            strategies = StrategyFactory.get_available_strategies()
        
        df = self.data_manager.load_data()
        
        print(f"\n{'='*60}")
        print(f"STRATEGY COMPARISON ANALYSIS")
        print(f"{'='*60}")
        
        return self.mc_tester.run_strategy_comparison(df, strategies, self.config.price_column)
    
    def generate_performance_plots(self, strategy_result: StrategyResult):
        """Generate comprehensive performance visualization."""
        df = self.data_manager.load_data()
        returns = self.data_manager.get_returns()
        
        strategy_returns = strategy_result.signal * returns
        
        plt.style.use(self.config.plot_style)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.config.strategy.upper()} Strategy Performance Analysis', fontsize=16)
        
        # Cumulative returns comparison
        axes[0, 0].plot(returns.cumsum(), label='Market Returns', alpha=0.8)
        axes[0, 0].plot(strategy_returns.cumsum(), label='Strategy Returns', alpha=0.8)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Log Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Strategy signals
        axes[0, 1].plot(df[self.config.price_column], alpha=0.7, label='Price')
        signal_changes = strategy_result.signal.diff().dropna()
        buy_signals = signal_changes[signal_changes > 0].index
        sell_signals = signal_changes[signal_changes < 0].index
        
        axes[0, 1].scatter(buy_signals, df.loc[buy_signals, self.config.price_column], 
                          color='green', marker='^', s=50, label='Buy', alpha=0.8)
        axes[0, 1].scatter(sell_signals, df.loc[sell_signals, self.config.price_column], 
                          color='red', marker='v', s=50, label='Sell', alpha=0.8)
        axes[0, 1].set_title('Trading Signals')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1, 0].hist(returns.dropna(), bins=50, alpha=0.7, label='Market', density=True)
        axes[1, 0].hist(strategy_returns.dropna(), bins=50, alpha=0.7, label='Strategy', density=True)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics
        metrics = self._calculate_performance_metrics(returns, strategy_returns)
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_performance_metrics(self, market_returns: pd.Series, 
                                     strategy_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        strategy_clean = strategy_returns.dropna()
        market_clean = market_returns.dropna()
        
        # Align series
        aligned_market, aligned_strategy = market_clean.align(strategy_clean, join='inner')
        
        metrics = {
            'Strategy Total Return': aligned_strategy.sum(),
            'Market Total Return': aligned_market.sum(),
            'Strategy Sharpe Ratio': aligned_strategy.mean() / aligned_strategy.std() if aligned_strategy.std() > 0 else 0,
            'Market Sharpe Ratio': aligned_market.mean() / aligned_market.std() if aligned_market.std() > 0 else 0,
            'Strategy Volatility': aligned_strategy.std(),
            'Market Volatility': aligned_market.std(),
            'Max Drawdown (Strategy)': self._calculate_max_drawdown(aligned_strategy),
            'Max Drawdown (Market)': self._calculate_max_drawdown(aligned_market),
        }
        
        # Calculate profit factor
        positive_returns = strategy_clean[strategy_clean > 0]
        negative_returns = strategy_clean[strategy_clean < 0]
        
        if len(negative_returns) > 0:
            metrics['Profit Factor'] = positive_returns.sum() / negative_returns.abs().sum()
        else:
            metrics['Profit Factor'] = np.inf if len(positive_returns) > 0 else 0
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = returns.cumsum()
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        return drawdown.min()
    
    def run_full_analysis(self):
        """Run complete trading analysis pipeline."""
        print(f"Starting comprehensive trading analysis...")
        print(f"Configuration: {self.config.data_symbol} | {self.config.strategy} | {self.config.timeframe}")
        
        # Strategy analysis
        strategy_result = self.run_strategy_analysis()
        
        # Monte Carlo test (if enabled)
        mc_result = None
        if self.config.run_monte_carlo:
            mc_result = self.run_monte_carlo_test()
        
        # Strategy comparison (if enabled)
        comparison_results = None
        if self.config.compare_strategies:
            comparison_results = self.compare_strategies()
        
        # Generate plots (if enabled)
        if self.config.generate_plots:
            self.generate_performance_plots(strategy_result)
        
        return {
            'strategy_result': strategy_result,
            'monte_carlo_result': mc_result,
            'comparison_results': comparison_results
        }


def create_default_config(**overrides) -> TradingConfig:
    """Create default configuration with optional overrides."""
    defaults = {
        'start_date': '2018-04-01',
        'end_date': '2024-12-31',
        'strategy': 'ma',
        'data_symbol': 'BTCUSD',
        'timeframe': '1h',
        'n_permutations': 100,
        'perm_start_index': 0,
        'generate_plots': True,
        'run_monte_carlo': True,
    }
    defaults.update(overrides)
    return TradingConfig(**defaults)


def main():
    """Main application entry point with example usage."""
    
    # Example 1: Basic usage matching original functionality
    print("=" * 80)
    print("EXAMPLE 1: Basic Monte Carlo Analysis (Original Functionality)")
    print("=" * 80)
    
    config = create_default_config()
    analyzer = TradingAnalyzer(config)
    
    # Run the same test as original code
    mc_result = analyzer.run_monte_carlo_test()
    
    # Example 2: Strategy comparison
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Enhanced Strategy Comparison")
    print("=" * 80)
    
    config2 = create_default_config(
        compare_strategies=True,
        n_permutations=50,  # Reduced for faster demo
        generate_plots=True
    )
    analyzer2 = TradingAnalyzer(config2)
    comparison_results = analyzer2.compare_strategies(['ma', 'donchian'])
    
    # Example 3: Full analysis pipeline
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Complete Analysis Pipeline")
    print("=" * 80)
    
    config3 = create_default_config(
        strategy='donchian',
        n_permutations=30,  # Reduced for demo
        generate_plots=True,
        run_monte_carlo=True
    )
    analyzer3 = TradingAnalyzer(config3)
    full_results = analyzer3.run_full_analysis()


if __name__ == "__main__":
    main()