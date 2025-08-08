import numpy as np
import pandas as pd
from typing import List, Union

def get_permutation(
        ohlc: Union[pd.DataFrame, List[pd.DataFrame]], #Union -> can be a pd.DataFrame or a List of pd.DataFrame
        start_index: int = 0, 
        seed=None
        ):
    assert start_index >= 0 #larger than one accommodates walk forward tests

    np.random.seed(seed)

    if isinstance(ohlc, list): #if ohlc is a list of pd.DataFrame use the first one to get the index and check if all the other have the same index
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Index do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc] #make it a list of one pd.DataFrame

    n_bars = len(ohlc[0]) #number of periods, i.e. bars

    perm_index = start_index + 1 #index of the first bar of the permutation
    perm_n = n_bars - perm_index #number of bars in the permutation

    start_bar = np.empty((n_markets, 4)) #empty array to store the start bar of the permutation; 4 is for (open, high, low, close), to change if needed
    relative_open = np.empty((n_markets, perm_n)) 
    relative_high = np.empty((n_markets, perm_n)) 
    relative_low = np.empty((n_markets, perm_n)) 
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])

        #get start bar 
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # open relative to last close
        r_open = (log_bars['open'] - log_bars['close'].shift()).to_numpy()

        # get prices relative to this bars open
        r_high = (log_bars['high'] - log_bars['open']).to_numpy()
        r_low = (log_bars['low'] - log_bars['open']).to_numpy()
        r_close = (log_bars['close'] - log_bars['open']).to_numpy()

        relative_open[mkt_i] = r_open[perm_index:]
        relative_high[mkt_i] = r_high[perm_index:]
        relative_low[mkt_i] = r_low[perm_index:]
        relative_close[mkt_i] = r_close[perm_index:]

    idx = np.arange(perm_n)

    # shuffle intrabar relative values (high, low, close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    #shuffle last close to open (gaps) separately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # copy over real data before start index
        log_bars= np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]

        # copy start bar
        perm_bars[start_index] = start_bar[mkt_i]
        
        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i-1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])

        perm_ohlc.append(perm_bars)

        if n_markets > 1:
            return perm_ohlc
        else:   
            return perm_ohlc[0]



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    btc_real = pd.read_parquet('data/ohlcv_BTCUSD_1h.parquet')
    btc_real = btc_real[(btc_real.index.year >= 2018) & (btc_real.index.year < 2020)]

    btc_perm = get_permutation(btc_real)


    plt.figure(figsize=(12, 6))
    plt.plot(btc_real['close'], label='Real')
    plt.plot(btc_perm['close'], label='Permuted')
    plt.legend()
    plt.show()