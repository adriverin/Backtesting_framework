import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def donchian_breakout(ohlc: pd.DataFrame, lookback:int):
    upper = ohlc['close'].rolling(lookback-1).max().shift(1)
    lower = ohlc['close'].rolling(lookback-1).min().shift(1)

    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc['close'] > upper] = 1
    signal.loc[ohlc['close'] < lower] = -1

    signal = signal.ffill()
    return signal

def optimize_donchian(ohlc: pd.DataFrame):

    best_profit_factor = 0
    best_lookback = -1
    r = np.log(ohlc['close']).diff().shift(-1)
    for lookback in range(2, 169):
        signal = donchian_breakout(ohlc, lookback)
        sig_rets = signal * r 
        positive_sum = sig_rets[sig_rets > 0].sum()
        negative_sum = sig_rets[sig_rets < 0].abs().sum()
        
        if negative_sum > 0:
            sig_profit_factor = positive_sum / negative_sum
            if sig_profit_factor > best_profit_factor:
                best_profit_factor = sig_profit_factor
                best_lookback = lookback

    return best_lookback, best_profit_factor





def moving_average(ohlc: pd.DataFrame, data='close', fast_lookback=10, slow_lookback=30):
    fast_ma = ohlc[data].rolling(window=fast_lookback).mean()
    slow_ma = ohlc[data].rolling(window=slow_lookback).mean()

    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[fast_ma > slow_ma] = 1
    signal.loc[fast_ma < slow_ma] = -1

    signal = signal.ffill()
    return signal


def optimize_moving_average(ohlc: pd.DataFrame, data='close'):
    best_profit_factor = 0
    best_lookback = [-1,-1]
    r = np.log(ohlc[data]).diff().shift(-1)
    for i in range(100):
        fast_ma = ohlc[data].rolling(window=i).mean()
        for j in range(i,100):
            slow_ma = ohlc[data].rolling(window=j).mean()
            signal = np.where(fast_ma > slow_ma, 1, 0)
            sig_rets = signal * r 
            
            positive_sum = sig_rets[sig_rets > 0].sum()
            negative_sum = sig_rets[sig_rets < 0].abs().sum()
            
            if negative_sum > 0:
                sig_profit_factor = positive_sum / negative_sum
                if sig_profit_factor > best_profit_factor:
                    best_profit_factor = sig_profit_factor
                    best_lookback = [i,j]
    return best_lookback, best_profit_factor



if __name__ == "__main__":
    df = pd.read_parquet('data/ohlcv_BTCUSD_1h.parquet')

    df = df[(df.index.year >= 2019) & (df.index.year < 2020)]

    strategy = input("Enter the strategy you want to use (don, ma): ")
    if strategy == 'don':
        best_lookback, best_profit_factor = optimize_donchian(df)
        print(f"Best Lookback: {best_lookback}, Best Profit Factor: {best_profit_factor}")
        signal = donchian_breakout(df, best_lookback)
    elif strategy == 'ma':
        best_lookback, best_profit_factor = optimize_moving_average(df, 'close')
        print(f"Best Lookback: {best_lookback}, Best Profit Factor: {best_profit_factor}")
        signal = moving_average(df, 'close', best_lookback[0], best_lookback[1])
    else:
        print("Invalid strategy")

    df['r'] = np.log(df['close']).diff().shift(-1)

    df['strategy_r'] = df['r'] * signal
    sharpe_ratio = df['strategy_r'].mean() / df['strategy_r'].std() * np.sqrt(8760) # 8760 is the number of hours in a year to annualize the sharpe ratio


    plt.figure(figsize=(12, 6))
    plt.title(f"{strategy} Strategy (PF: {best_profit_factor:.4f}, Sharpe: {sharpe_ratio:.4f})")
    df['strategy_r'].cumsum().plot(label='Strategy Returns')
    df['r'].cumsum().plot(label='Market Returns') 
    plt.ylabel('Cumulative Log Returns')  # Added y-axis label
    plt.legend()  # Added legend
    plt.show()