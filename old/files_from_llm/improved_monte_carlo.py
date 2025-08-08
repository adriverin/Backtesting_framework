"""
Monte Carlo testing framework for trading strategies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from improved_strategies import BaseStrategy, StrategyFactory
from improved_permutations import PermutationGenerator, PermutationConfig


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo analysis."""
    n_permutations: int = 100
    start_index: int = 0
    confidence_level: float = 0.05
    generate_plots: bool = False
    plot_style: str = 'dark_background'
    seed: Optional[int] = None


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo analysis."""
    real_profit_factor: float
    permuted_profit_factors: List[float]
    p_value: float
    n_better_permutations: int
    confidence_level: float
    is_significant: bool


class MonteCarloTester:
    """High-performance Monte Carlo testing for trading strategies."""
    
    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        self.permutation_generator = PermutationGenerator(
            PermutationConfig(
                start_index=self.config.start_index,
                seed=self.config.seed
            )
        )
    
    def run_test(self, ohlc: pd.DataFrame, strategy: BaseStrategy, 
                 price_column: str = 'close') -> MonteCarloResult:
        """
        Run Monte Carlo test for a strategy.
        
        Args:
            ohlc: OHLC price data
            strategy: Strategy instance to test
            price_column: Price column to use for signals
            
        Returns:
            Monte Carlo test results
        """
        # Optimize strategy on real data
        print("Optimizing strategy on real data...")
        real_result = strategy.optimize(ohlc, price_column)
        real_pf = real_result.best_profit_factor
        
        print(f"Real profit factor: {real_pf:.4f}")
        print(f"Best parameters: {real_result.best_params}")
        
        # Run permutation tests
        print(f"Running {self.config.n_permutations} permutation tests...")
        permuted_pfs = []
        n_better = 1  # Include real data in count
        
        for i in tqdm(range(1, self.config.n_permutations), desc="MC Permutations"):
            # Generate permuted data
            perm_ohlc = self.permutation_generator.generate(ohlc)
            
            # Optimize strategy on permuted data
            perm_result = strategy.optimize(perm_ohlc, price_column)
            perm_pf = perm_result.best_profit_factor
            
            permuted_pfs.append(perm_pf)
            
            if perm_pf >= real_pf:
                n_better += 1
        
        # Calculate p-value
        p_value = n_better / self.config.n_permutations
        is_significant = p_value <= self.config.confidence_level
        
        result = MonteCarloResult(
            real_profit_factor=real_pf,
            permuted_profit_factors=permuted_pfs,
            p_value=p_value,
            n_better_permutations=n_better,
            confidence_level=self.config.confidence_level,
            is_significant=is_significant
        )
        
        print(f"Monte Carlo p-value: {p_value:.4f}")
        print(f"Significant at {self.config.confidence_level*100}% level: {is_significant}")
        
        if self.config.generate_plots:
            self._plot_results(result)
        
        return result
    
    def _plot_results(self, result: MonteCarloResult):
        """Generate visualization of Monte Carlo results."""
        plt.style.use(self.config.plot_style)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of permuted profit factors
        ax1.hist(result.permuted_profit_factors, bins=50, alpha=0.7, 
                color='blue', label='Permutations', density=True)
        ax1.axvline(result.real_profit_factor, color='red', linewidth=2, 
                   label=f'Real (PF={result.real_profit_factor:.3f})')
        ax1.set_xlabel('Profit Factor')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Monte Carlo Distribution (p-value: {result.p_value:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # P-value visualization
        sorted_pfs = sorted(result.permuted_profit_factors + [result.real_profit_factor])
        percentiles = np.arange(len(sorted_pfs)) / len(sorted_pfs)
        
        ax2.plot(sorted_pfs, percentiles, 'b-', alpha=0.7, label='Empirical CDF')
        real_percentile = sorted_pfs.index(result.real_profit_factor) / len(sorted_pfs)
        ax2.axvline(result.real_profit_factor, color='red', linewidth=2,
                   label=f'Real ({real_percentile:.1%} percentile)')
        ax2.axhline(1 - result.p_value, color='orange', linestyle='--',
                   label=f'p-value = {result.p_value:.4f}')
        ax2.set_xlabel('Profit Factor')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_strategy_comparison(self, ohlc: pd.DataFrame, strategy_names: List[str],
                              price_column: str = 'close') -> dict:
        """Compare multiple strategies using Monte Carlo testing."""
        results = {}
        
        for strategy_name in strategy_names:
            print(f"\n{'='*50}")
            print(f"Testing {strategy_name.upper()} strategy")
            print(f"{'='*50}")
            
            strategy = StrategyFactory.create_strategy(strategy_name)
            result = self.run_test(ohlc, strategy, price_column)
            results[strategy_name] = result
        
        # Print comparison summary
        print(f"\n{'='*50}")
        print("STRATEGY COMPARISON SUMMARY")
        print(f"{'='*50}")
        
        for name, result in results.items():
            significance = "SIGNIFICANT" if result.is_significant else "NOT SIGNIFICANT"
            print(f"{name.upper():15} | PF: {result.real_profit_factor:.4f} | "
                  f"p-value: {result.p_value:.4f} | {significance}")
        
        return results


def run_monte_carlo_analysis(ohlc: pd.DataFrame, strategy_name: str, 
                           config: MonteCarloConfig = None,
                           price_column: str = 'close') -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo analysis.
    
    Args:
        ohlc: OHLC price data
        strategy_name: Name of strategy to test
        config: Monte Carlo configuration
        price_column: Price column for analysis
        
    Returns:
        Monte Carlo test results
    """
    config = config or MonteCarloConfig()
    tester = MonteCarloTester(config)
    strategy = StrategyFactory.create_strategy(strategy_name)
    
    return tester.run_test(ohlc, strategy, price_column)