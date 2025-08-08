"""
Demo script showing the improved codebase functionality.
This script demonstrates the exact same functionality as your original main.py 
but with better performance, structure, and additional features.
"""

from improved_main import TradingAnalyzer, create_default_config


def demo_original_functionality():
    """Demonstrate the exact same functionality as original main.py"""
    
    print("=" * 80)
    print("DEMO: Reproducing Original Functionality with Improved Code")
    print("=" * 80)
    
    # Create configuration matching your original settings
    config = create_default_config(
        start_date='2018-04-01',
        end_date='2024-12-31',
        strategy='ma',
        data_symbol='BTCUSD',
        timeframe='1h',
        n_permutations=10,
        perm_start_index=0,
        generate_plots=True,
        run_monte_carlo=True
    )
    
    # Create analyzer instance
    analyzer = TradingAnalyzer(config)
    
    # This reproduces exactly what your original code does:
    # c.get_insample_mc() 
    mc_result = analyzer.run_monte_carlo_test()
    
    print(f"\nResults Summary:")
    print(f"Real Profit Factor: {mc_result.real_profit_factor:.4f}")
    print(f"Monte Carlo p-value: {mc_result.p_value:.4f}")
    print(f"Number of permutations with better/equal performance: {mc_result.n_better_permutations}")
    print(f"Statistically significant: {mc_result.is_significant}")


def demo_enhanced_features():
    """Demonstrate additional features not in original code"""
    
    print("\n" + "=" * 80)
    print("DEMO: Enhanced Features (Beyond Original)")
    print("=" * 80)
    
    # Enhanced configuration with additional features
    config = create_default_config(
        strategy='donchian',  # Try different strategy
        n_permutations=50,    # Reduced for faster demo
        generate_plots=True,
        compare_strategies=True,
        confidence_level=0.05
    )
    
    analyzer = TradingAnalyzer(config)
    
    # Full analysis pipeline
    results = analyzer.run_full_analysis()
    
    print("\nEnhanced Analysis Complete!")
    print("Features added:")
    print("- Better performance through vectorization")
    print("- Type hints and validation")
    print("- Modular architecture")
    print("- Strategy comparison")
    print("- Enhanced plotting")
    print("- Comprehensive metrics")


if __name__ == "__main__":
    # Run demo of original functionality
    demo_original_functionality()
    
    # Uncomment to see enhanced features
    # demo_enhanced_features()