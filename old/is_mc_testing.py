from permutations import get_permutation
from basic_strats import donchian_breakout, optimize_donchian, optimize_moving_average, moving_average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class config_insample:
    start_date = '2018-01-01'
    end_date = '2020-01-01'
    strategy = 'ma'
    data = 'BTCUSD'
    timeframe = '1h'
    price_column = "close"  # Options: "open", "high", "low", "close" -> not for now: "vwap", "typical", "median"
    perm_start_index = 0
    n_perm = 100
    generate_plot = False

    def __init__(self, start_date, end_date, strategy, data, timeframe, n_perm, generate_plot, perm_start_index=0):
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.data = data
        self.timeframe = timeframe
        self.perm_start_index = perm_start_index
        self.n_perm = n_perm
        self.generate_plot = generate_plot

    def get_df(self):
        df = pd.read_parquet(f'data/ohlcv_{self.data}_{self.timeframe}.parquet')
        df = df[(df.index >= self.start_date) & (df.index < self.end_date)]
        return df
    
    def _get_perm_df(self):
        df = self.get_df()
        perm_df = get_permutation(df, start_index=self.perm_start_index)
        return perm_df
    

    
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




    def get_insample_mc_permutations(self):
        train_df = self.get_df()
        print("="*100)
        print(f"Optimizing strategy ({self.strategy}) on train data")
        print(f"from {self.start_date} to {self.end_date}")
        print("="*100)
        print("")
        best_lookback, best_real_pf = optimize_moving_average(train_df, self.price_column)
        print(f"In-sample PF: {best_real_pf:.4f}, Best Lookback: {best_lookback}")

        n_permutations = self.n_perm
        iperms_better = 1
        permuted_pfs = []
        print("Running In-Sample MC permutations")
        for perm_i in tqdm(range(1, n_permutations)):
            train_perm = self._get_perm_df()
            _, best_perm_pf = optimize_moving_average(train_perm, self.price_column)
            
            if best_perm_pf >= best_real_pf:
                iperms_better += 1

            permuted_pfs.append(best_perm_pf)
        insample_mcpt_pval = iperms_better / n_permutations # Rough estimation of p-value
        print(f"In-sample MC p-value: {insample_mcpt_pval:.4f}")
        print(f"Number of permutations: {n_permutations}")
        print(f"Number of permutations better than real profit factor: {iperms_better}")
        print(f"Real Profit Factor: {best_real_pf:.4f}")
        print("="*100)
        print("")

        if self.generate_plot:
            print(f"Generating histogram of the profit factor of the in-sample MC permutations")
            plt.style.use('dark_background')
            pd.Series(permuted_pfs).hist(color='blue', label='Permutations')
            plt.axvline(best_real_pf, color='red', label='Real')
            plt.xlabel('Profit Factor')
            plt.ylabel('Frequency')
            plt.title(f'In-sample MC Permutations (p-value: {insample_mcpt_pval})')
            plt.legend()
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

    c = config_insample(
        start_date='2019-01-01',
        end_date='2024-01-01', 
        strategy='ma', 
        data='ETHUSD', 
        timeframe='1h',
        n_perm=100,
        generate_plot=True
        )

    df = c.get_df()

    if c.strategy == 'don':
        signal = c.get_signal_donchian()
    elif c.strategy == 'ma':
        c.get_insample_mc_permutations()


    else:
        print("Invalid strategy")




