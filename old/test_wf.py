import pandas as pd
import numpy as np
from is_mc_testing import optimize_moving_average, moving_average
from tqdm import tqdm
from permutations import get_permutation
import matplotlib.pyplot as plt

def walk_forward_ma(ohlc: pd.DataFrame, train_lookback: int = 24*365*4, train_step: int = 24*30, price_column: str = 'close'):
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_lookback, _ = optimize_moving_average(ohlc.iloc[i-train_lookback:i], price_column)
            tmp_signal = moving_average(ohlc, price_column, best_lookback[0], best_lookback[1])
            next_train += train_step
            print(f"Optimized on {i-train_lookback} - {i}")
            print(f"Data: {ohlc.iloc[i-train_lookback:i]}")
        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal



def walk_forward_ma_optimized(ohlc: pd.DataFrame, train_lookback: int = 24*365*4, train_step: int = 24*30, price_column: str = 'close'):
    """
    Performs a walk-forward test for a moving average strategy in an optimized manner.

    This function operates in chunks (steps) rather than iterating row-by-row,
    leading to a massive performance improvement.

    Args:
        ohlc (pd.DataFrame): DataFrame with OHLC data and a datetime index.
        train_lookback (int): The number of periods to use for each training window.
        train_step (int): The number of periods in the out-of-sample "forward" test.
                          The model is re-optimized after each step.
        price_column (str): The name of the column to use for calculations.

    Returns:
        np.ndarray: An array containing the generated signals for the entire timeline.
    """
    n = len(ohlc)
    wf_signals = np.full(n, np.nan)

    # Use tqdm for a progress bar, wrapping the range iterator
    for i in range(train_lookback, n, train_step):
        # 1. Define the training period
        train_start = i - train_lookback
        train_end = i
        train_data = ohlc.iloc[train_start:train_end]

        # 2. Optimize parameters on the training data
        best_lookback, _ = optimize_moving_average(train_data, price_column)
        short_win = best_lookback[0]
        long_win = best_lookback[1]
        
        # 3. Define the out-of-sample (OOS) period for signal generation
        oos_end = min(i + train_step, n)
        
        # We need to include some prior data to correctly calculate the first MAs in the OOS period
        # The context should be at least as long as the largest lookback window
        signal_calc_start = max(0, train_end - long_win)
        signal_calc_data = ohlc.iloc[signal_calc_start:oos_end]
        
        # 4. Generate signals only for the data needed
        oos_signals_full = moving_average(signal_calc_data, price_column, short_win, long_win)
        
        # 5. Extract only the signals corresponding to the OOS period
        # and assign them to the final array in one operation
        oos_signals_final = oos_signals_full.iloc[-len(ohlc.iloc[train_end:oos_end]):].values
        wf_signals[train_end:oos_end] = oos_signals_final

    return wf_signals




def run_walk_forward_monte_carlo(
        ohlc: pd.DataFrame,
        n_permutations: int,
        train_lookback: int = 24*365*4, train_step: int = 24*30, price_column: str = 'close'
    ) -> pd.DataFrame:
    """
    Runs the full walk-forward analysis on multiple permuted price series.

    Args:
        ohlc (pd.DataFrame): The original, true OHLC data.
        n_permutations (int): The number of Monte Carlo simulations to run.
        train_lookback (int): The lookback period for training.
        train_step (int): The forward step for out-of-sample testing.
        price_column (str): The price column to use.

    Returns:
        pd.DataFrame: A DataFrame where each column is the cumulative equity curve
                      of one complete walk-forward test on a permuted series.
    """
    all_permutation_returns = {}

    # The permutation should not touch the first training period, so it remains a
    # consistent basis for the first optimization across all runs.
    # The shuffling of subsequent returns is what tests the strategy's robustness.
    perm_start_index = train_lookback

    for i in tqdm(range(n_permutations), desc="Running Monte Carlo Permutations"):
        # 1. Create a permuted OHLC series for this run
        permuted_ohlc = get_permutation(ohlc.copy(), start_index=perm_start_index, seed=i)
        
        # 2. Run the entire walk-forward test on this single permuted series
        permuted_signals = walk_forward_ma_optimized(
            permuted_ohlc, train_lookback, train_step, price_column
        )
        
        # 3. Calculate the returns for this permuted path
        permuted_cumulative_returns = permuted_signals.cumsum()
        all_permutation_returns[f'perm_{i}'] = permuted_cumulative_returns
        
    return pd.DataFrame(all_permutation_returns)








if __name__ == "__main__":
    btc_real = pd.read_parquet('data/ohlcv_BTCUSD_1h.parquet')
    # btc_real = btc_real[(btc_real.index.year >= 2018) & (btc_real.index.year < 2020)]
    btc_real = btc_real[(btc_real.index.year >= 2019) ]

    wf_signal = walk_forward_ma_optimized(btc_real, train_lookback=24*365*4, train_step=24*30)

    btc_real['r'] = np.log(btc_real['close']).diff().shift(-1)
    btc_real['strategy_r'] = wf_signal * btc_real['r']

    res_to_plot = btc_real[(btc_real.index.year >= 2024) & (btc_real.index.year < 2025)]

    sig_rets = res_to_plot['strategy_r']
    
    positive_sum = sig_rets[sig_rets > 0].sum()
    negative_sum = sig_rets[sig_rets < 0].abs().sum()
    
    if negative_sum > 0:
        sig_profit_factor = positive_sum / negative_sum

    plt.figure(figsize=(12, 6))
    plt.title(f"OOS-walk-forward (Profit Factor: {sig_profit_factor:.2f})")
    res_to_plot['strategy_r'].cumsum().plot(label='Strategy Returns')
    res_to_plot['r'].cumsum().plot(label='Market Returns')
    plt.legend()
    plt.show()

    