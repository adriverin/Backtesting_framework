from permutations import get_permutation
from strategies.basic_strats import donchian_breakout, optimize_donchian, optimize_moving_average, moving_average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




class config_walkforward:
    start_date = '2018-01-01'
    end_date = '2020-01-01'
    strategy = 'ma'
    data = 'BTCUSD'
    timeframe = '1h'
    price_column = "close"  # Options: "open", "high", "low", "close" -> not for now: "vwap", "typical", "median"
    n_perm = 2
    train_lookback = 24*365*4
    train_step = 24*30
    generate_plot = False

    def __init__(self, start_date, end_date, strategy, data, timeframe, n_perm, generate_plot, train_lookback, train_step):
        self.start_date = start_date
        self.end_date = end_date
        self.train_lookback = train_lookback
        self.train_step = train_step
        self.strategy = strategy
        self.data = data
        self.timeframe = timeframe
        self.perm_start_index = self._get_perm_start_index(self.start_date)
        self.n_perm = n_perm
        self.generate_plot = generate_plot
        print(f"Permutation start index: {self.perm_start_index}")

    def get_df(self):
        df = pd.read_parquet(f'data/ohlcv_{self.data}_{self.timeframe}.parquet')
        df = df[(df.index >= self.start_date) & (df.index < self.end_date)]
        return df
    
    def _get_perm_df(self):
        # Get full dataset, apply permutation, then filter to date range
        full_df = pd.read_parquet(f'data/ohlcv_{self.data}_{self.timeframe}.parquet')
        perm_df = get_permutation(full_df, start_index=self.perm_start_index)
        # Filter to the same date range as the real data
        perm_df = perm_df[(perm_df.index >= self.start_date) & (perm_df.index < self.end_date)]
        return perm_df
    
    def _get_perm_df_with_seed(self, seed):
        # Get full dataset, apply permutation with seed, then filter to date range
        full_df = pd.read_parquet(f'data/ohlcv_{self.data}_{self.timeframe}.parquet')
        perm_df = get_permutation(full_df, start_index=self.perm_start_index, seed=seed)
        # Filter to the same date range as the real data
        perm_df = perm_df[(perm_df.index >= self.start_date) & (perm_df.index < self.end_date)]
        return perm_df
    
    def _get_perm_start_index(self, start_date: str) -> int:
        # Get the full dataset to find the correct permutation start
        full_df = pd.read_parquet(f'data/ohlcv_{self.data}_{self.timeframe}.parquet')
        start_date_parsed = pd.to_datetime(start_date, utc=True)
        
        # Find the index of our analysis start date in the full dataset
        analysis_start_idx = full_df.index.get_loc(start_date_parsed)
        
        # We need to start permutation early enough to have training data
        # Start permutation from analysis_start - train_lookback
        perm_start_idx = max(0, analysis_start_idx - self.train_lookback)
        
        return perm_start_idx
    
    def get_returns(self):
        df = self.get_df()
        r = np.log(df['close']).diff().shift(-1)
        return r
    
    def get_perm_returns(self):
        df = self.get_perm_df()
        r = np.log(df['close']).diff().shift(-1)
        return r
    
    def get_all_perms(self):
        perm_dfs = []
        for i in range(self.n_perm):
            perm_df = self._get_perm_df()
            perm_dfs.append(perm_df)
        return perm_dfs




    def walkforward_ma(self, ohlc_data=None):
    # def walkforward_ma(self):
        """
        Performs a walk-forward test for a moving average strategy in an optimized manner.

        This function operates in chunks (steps) rather than iterating row-by-row,
        leading to a massive performance improvement.

        Args:
            ohlc_data (pd.DataFrame, optional): DataFrame with OHLC data and a datetime index.
                                               If None, uses self.get_df().

        Returns:
            np.ndarray: An array containing the generated signals for the entire timeline.
        """
        # ohlc = self.get_df()
        ohlc = ohlc_data if ohlc_data is not None else self.get_df()
        train_lookback = self.train_lookback
        train_step = self.train_step
        price_column = self.price_column
        
        print(f"Train lookback: {train_lookback}, Train step: {train_step}")

        n = len(ohlc)
        wf_signals = np.full(n, np.nan)
        best_lookback = None

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



    def walkforward_oos_mc_permutations(self):
        # Get the real data for the specified date range
        real_df = self.get_df()
        
        print(f"Calculating walk-forward signals for real data from {self.start_date} to {self.end_date}")
        
        # Generate walk-forward signals for real data
        real_signals = self.walkforward_ma(real_df)
        
        # Calculate returns and strategy returns for real data
        real_df['r'] = np.log(real_df[self.price_column]).diff().shift(-1)
        real_df['signal'] = real_signals
        real_df['strategy_r'] = real_df['r'] * real_df['signal']
        
        # Debug: Check signal statistics
        signal_counts = pd.Series(real_signals).value_counts()
        print(f"Real signal distribution: {signal_counts.to_dict()}")
        print(f"Number of valid signals: {pd.Series(real_signals).notna().sum()}")
        print(f"Strategy returns range: {real_df['strategy_r'].min():.6f} to {real_df['strategy_r'].max():.6f}")

        # Calculate real profit factor
        valid_strategy_returns = real_df['strategy_r'].dropna()
        
        positive_sum = valid_strategy_returns[valid_strategy_returns > 0].sum()
        negative_sum = valid_strategy_returns[valid_strategy_returns < 0].abs().sum()
        
        real_pf = 0  
        if negative_sum > 0:
            real_pf = positive_sum / negative_sum
        else:
            real_pf = float('inf') if positive_sum > 0 else 0

        print(f"OOS real Profit Factor: {real_pf}")
        print(f"Real data shape: {real_df.shape}, date range: {real_df.index.min()} to {real_df.index.max()}")
        print(f"Train lookback: {self.train_lookback} periods, Train step: {self.train_step} periods")
        print(f"Data length: {len(real_df)} periods, Sufficient for training: {len(real_df) > self.train_lookback}")

        n_permutations = self.n_perm
        iperms_better = 0
        permuted_pfs = []
        print("Starting Walk-Forward OOS MC permutations")

        for perm_i in tqdm(range(n_permutations)):
            # Get permuted data with a unique seed for each permutation
            perm_df = self._get_perm_df_with_seed(perm_i)
            
            # Generate walk-forward signals for permuted data
            perm_signals = self.walkforward_ma(perm_df)
            
            # Calculate returns and strategy returns for permuted data
            perm_df['r'] = np.log(perm_df[self.price_column]).diff().shift(-1)
            perm_df['signal'] = perm_signals
            perm_df['strategy_r'] = perm_df['r'] * perm_df['signal']

            # Calculate permuted profit factor
            valid_perm_returns = perm_df['strategy_r'].dropna()
            perm_positive_sum = valid_perm_returns[valid_perm_returns > 0].sum()
            perm_negative_sum = valid_perm_returns[valid_perm_returns < 0].abs().sum()
            
            perm_pf = 0 
            if perm_negative_sum > 0:
                perm_pf = perm_positive_sum / perm_negative_sum
            elif perm_positive_sum > 0:
                perm_pf = float('inf')
            
            if perm_pf >= real_pf:
                iperms_better += 1

            permuted_pfs.append(perm_pf)
            print(f"Permutation {perm_i}: PF = {perm_pf:.4f}")
            
        # Calculate Monte Carlo p-value
        oos_mcpt_pval = (iperms_better + 1) / (n_permutations + 1)
        print(f"Walk-Forward OOS MC p-value: {oos_mcpt_pval:.4f}")
        print(f"Number of permutations: {n_permutations}")
        print(f"Number of permutations better than real profit factor: {iperms_better}")
        print(f"Real Profit Factor: {real_pf:.4f}")
        print("="*100)
        print("")
        
        if self.generate_plot:
            print(f"Generating histogram of the profit factor of the walk-forward OOS vs MC permutations")
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 6))
            plt.hist(permuted_pfs, bins=max(10, len(permuted_pfs)//3), color='blue', alpha=0.7, label='Permutations', edgecolor='white')
            plt.axvline(real_pf, color='red', linewidth=2, label=f'Real (PF={real_pf:.2f})')
            plt.xlabel('Profit Factor')
            plt.ylabel('Frequency')
            plt.title(f'Walk-Forward OOS MC Permutations (p-value: {oos_mcpt_pval:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()            

    # strategy signals

    def get_signal_donchian(self):
        df = self.get_df()
        best_lookback, best_profit_factor = optimize_donchian(df)
        print(f"Best Lookback: {best_lookback}, Best Profit Factor: {best_profit_factor}")
        signal = donchian_breakout(df, best_lookback)
        return signal
    
    def get_signal_ma(self):
        df = self.get_df()
        best_lookback, best_profit_factor = optimize_moving_average(df, self.price_column)
        print(f"Best Lookback: {best_lookback}, Best Profit Factor: {best_profit_factor}")
        signal = moving_average(df, self.price_column, best_lookback[0], best_lookback[1])
        return signal
    






if __name__ == "__main__":


    c = config_walkforward(
        start_date='2018-01-01',
        end_date='2020-01-01', 
        strategy='ma', 
        data='BTCUSD', 
        timeframe='1h',
        n_perm=5,
        generate_plot=True,
        train_lookback=24*365*4,  
        train_step=24*30*3        
        )

    df = c.get_df()

    if c.strategy == 'don':
        signal = c.get_signal_donchian()
    elif c.strategy == 'ma':
        c.walkforward_oos_mc_permutations()


    else:
        print("Invalid strategy")        


