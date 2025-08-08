### Backtesting Framework (Crypto OHLCV)

A lightweight research toolkit to download OHLCV data, run in-sample Monte Carlo permutation tests, and perform out-of-sample walk-forward evaluations for modular trading strategies.

### Features
- **Data caching** from Binance into Parquet under `data/ohlcv_{ASSET}_{TF}.parquet`
- **In-sample MC permutations** to assess overfitting risk
- **Out-of-sample walk-forward** evaluation
- **Walk-forward MC permutations** for robustness
- **JSON reports** and optional plots for fast inspection

### Install
Requires Python 3.10+.

```bash
pip install pandas numpy matplotlib ccxt pyarrow tqdm scikit-learn
# Optional (only for strategy "ml")
pip install torch
```

### Prepare data
Downloads OHLCV caches for multiple assets and timeframes from Binance (via ccxt) into `data/`.

```bash
python data/download_data.py
```

Inspect what’s cached:

```bash
python data/info_avail_data.py
```

Data files are named `data/ohlcv_{ASSET}_{TF}.parquet` (e.g., `ohlcv_BTCUSD_1h.parquet`). Supported `TF`: `1m, 5m, 15m, 1h, 4h, 1d`.

### Available strategies
- `ma` — moving-average crossover
- `ml_rsi_ema_volume` — lightweight sklearn-based classifier
- `ml` — forecasting MLP (PyTorch)

### Quickstart
- In-sample MC permutations (prints p-value, can save JSON):

```bash
python is_montecarlo_permutations.py \
  --start 2019-01-01 --end 2021-01-01 \
  --strategy ma --asset BTCUSD --tf 1h --n_perm 50 --plot \
  --save_json_dir reports/example_run
```

- Out-of-sample walk-forward:

```bash
python oos_walkforward.py \
  --start 2021-01-01 --end 2023-01-01 \
  --strategy ma --asset BTCUSD --tf 1h \
  --lookback 35000 --step 720 --plot \
  --save_json_dir reports/example_run
```

- Walk-forward MC permutations:

```bash
python oos_walkforward_montecarlo_permutations.py \
  --start 2019-01-01 --end 2021-01-01 \
  --strategy ma --asset ETHUSD --tf 1h \
  --lookback 35000 --step 720 --n_perm 20 --plot 
```

- One-file visual check (cumulative returns + JSON series):

```bash
python is_results.py \
  --start 2019-01-01 --end 2024-01-01 \
  --strategy ml --asset BTCUSD --tf 1h \
  --fee_bps 10 --slippage_bps 2 \
  --save_json_dir reports/example_run
```

### Notes
- If you don’t install PyTorch, use `ma` or `ml_rsi_ema_volume` instead of `ml`.
- Price columns supported include `close` (default) and for ML: `typical`, `median`, `vwap` (created on-the-fly when possible).
- Symbols map to Binance spot pairs. Inputs like `BTCUSD`, `BTC-USD`, `BTC/USDT` are accepted; they are fetched as `BTC/USDT` and saved under `BTCUSD` in filenames.
- No API key is required; public Binance kline endpoints are used via ccxt with rate limiting enabled.
- If Binance is unavailable in your region, swap the exchange in `data/download_data.py` (ccxt supports many exchanges).

### Folder structure
- `data/` — cached OHLCV, cache summaries, helpers
- `strategies/` — strategy interface and implementations
- `reports/` — JSON outputs (when enabled)


