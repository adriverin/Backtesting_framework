import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet('data/ohlcv_BTCUSD_1h.parquet')
# df.index = df.index.astype('datetime64[s]')

fast_ma = df['close'].rolling(window=10).mean()
slow_ma = df['close'].rolling(window=30).mean()

#signal
df['signal'] = np.where(fast_ma > slow_ma, 1, 0)
# df['signal'] = np.where(fast_ma > slow_ma, 1, np.where(fast_ma < slow_ma, -1, 0))

df['return'] = np.log(df['close']).diff().shift(-1)
df['strategy_return'] = df['signal'] * df['return']

r = df['strategy_return']
profit_factor = r[r > 0].sum() / r[r < 0].abs().sum()
sharpe_ratio = r.mean() / r.std()

# print(f"Profit Factor: {profit_factor:.2f}")
# print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# plt.figure(figsize=(12, 6))
# plt.plot(df['close'], label='Close Price')
# plt.plot(fast_ma, label='Fast MA')
# plt.plot(slow_ma, label='Slow MA')
# plt.legend()
# plt.show()