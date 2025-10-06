# Live Trading System

A comprehensive 24/7 live trading system for Binance using order book depth strategy.

## 🚀 Features

- **Real-time Order Book Data Streaming**: WebSocket-based order book depth data collection
- **Live Price Data Integration**: Integrates with existing `live_feeder_ccxt.py` infrastructure
- **Strategy Execution**: Automated trading based on order book depth signals
- **Performance Tracking**: Real-time calculation of Sharpe ratio, profit factor, win rate, etc.
- **Web Dashboard**: Beautiful HTML dashboard with real-time charts and metrics
- **Telegram Alerts**: Instant notifications for trades and system errors
- **Periodic Re-optimization**: Automatic strategy parameter tuning based on recent performance
- **Robust Error Handling**: Auto-restart mechanisms and comprehensive logging
- **Cloud-Ready**: Designed for 24/7 operation on cloud servers

## 📁 Project Structure

```
live_trading/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration management
├── orderbook_streamer.py       # WebSocket order book data streamer
├── price_feeder.py             # Live price data integration
├── trading_engine.py           # Main trading engine with position management
├── performance_tracker.py      # Real-time performance metrics
├── api_server.py               # FastAPI server with WebSocket/SSE
├── telegram_notifier.py        # Telegram alert integration
├── reoptimizer.py              # Periodic strategy re-optimization
├── dashboard.html              # Web dashboard
├── run_live.py                 # Main entry point
└── README.md                   # This file
```

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install ccxt websockets fastapi uvicorn aiohttp python-telegram-bot
```

### 2. Configure Binance API

Create a `.env` file or set environment variables:

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export BINANCE_TESTNET="false"
```

### 3. Configure Telegram (Optional)

```bash
export TELEGRAM_ENABLED="true"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

## 🎯 Configuration

The system uses a JSON configuration file or environment variables. Key settings:

```json
{
  "symbol": "VETUSD",
  "mode": "futures",
  "timeframe": "1m",
  "price_column": "vwap_30",
  
  "strategy": {
    "name": "orderbook_depth",
    "params": {
      "percentage": 2,
      "lookback_mean": "210D",
      "lookback_current": "1min",
      "z_threshold": 1.5,
      "exit_band": -1.5,
      "persistence": 1
    }
  },
  
  "fees": {
    "fee_bps": 4.0,
    "slippage_bps": 5.0
  },
  
  "reoptimization": {
    "enabled": true,
    "interval_hours": 24,
    "lookback_days": 365
  }
}
```

## 🚦 Usage

### Start the System

```bash
cd live_trading
python run_live.py
```

### Access Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

### Monitor Logs

Logs are saved to `logs/live_trading/` in JSON format.

## 📊 Dashboard Features

The web dashboard displays:

- **Portfolio Metrics**: Total equity, capital, unrealized P&L, total return
- **Current Position**: Position status, signal, entry price, position size
- **Performance Metrics**: Sharpe ratio, profit factor, max drawdown
- **Trading Stats**: Number of trades, win rate, average win/loss
- **Cumulative Returns Chart**: Real-time equity curve visualization
- **Trade Log**: Recent trade history with timestamps and P&L

## 🔄 How It Works

### 1. Data Collection

- **Order Book Streamer**: Connects to Binance WebSocket and streams real-time order book depth data
- **Price Feeder**: Monitors the parquet file updated by `live_feeder_ccxt.py` for OHLCV data

### 2. Signal Generation

- The trading engine periodically generates signals using the order book depth strategy
- Signals are based on bid/ask imbalance z-scores at specified depth levels

### 3. Trade Execution

- When signals change, the engine opens/closes positions via Binance API
- Supports both paper trading (dry-run) and live trading modes

### 4. Performance Tracking

- Real-time calculation of metrics (Sharpe, profit factor, drawdown, etc.)
- Equity curve and trade history maintained in memory

### 5. Re-optimization

- Periodically re-optimizes strategy parameters using recent data
- Updates strategy parameters automatically if improvement is found

## 📱 Telegram Alerts

The system sends notifications for:

- **Trade Executions**: Open/close positions with P&L details
- **System Errors**: Critical errors and exceptions
- **Status Updates**: Periodic performance summaries
- **Re-optimization**: When strategy parameters are updated

## 🌐 Cloud Deployment

### Recommended Setup

1. **Cloud Provider**: AWS EC2, Google Cloud, DigitalOcean, or similar
2. **Instance Type**: Small instance with 1-2 vCPU, 2-4 GB RAM
3. **OS**: Ubuntu 20.04+ or similar
4. **Process Manager**: Use `systemd` or `supervisor` for auto-restart

### Example systemd Service

Create `/etc/systemd/system/live-trading.service`:

```ini
[Unit]
Description=Live Trading System
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/Backtesting_framework/live_trading
Environment="BINANCE_API_KEY=your_key"
Environment="BINANCE_API_SECRET=your_secret"
Environment="TELEGRAM_BOT_TOKEN=your_token"
Environment="TELEGRAM_CHAT_ID=your_chat_id"
ExecStart=/usr/bin/python3 run_live.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable live-trading
sudo systemctl start live-trading
sudo systemctl status live-trading
```

### Using Screen or Tmux

```bash
# Start in screen session
screen -S live_trading
cd /path/to/Backtesting_framework/live_trading
python run_live.py

# Detach: Ctrl+A, D
# Reattach: screen -r live_trading
```

## 🔧 Troubleshooting

### Common Issues

1. **No order book data**
   - Ensure WebSocket connection is established
   - Check symbol format (e.g., VETUSDT for futures)

2. **No price data**
   - Verify `live_feeder_ccxt.py` is running
   - Check parquet file path and permissions

3. **API errors**
   - Verify API credentials
   - Check API permissions (trading, futures)
   - Ensure sufficient balance

4. **Dashboard not loading**
   - Check if port 8000 is available
   - Verify FastAPI server is running
   - Check firewall settings

## 🔐 Security Best Practices

1. **API Keys**: Never commit API keys to version control
2. **IP Whitelist**: Enable IP whitelist on Binance API
3. **Read-Only Keys**: Use read-only keys for testing
4. **VPN**: Consider using VPN for cloud servers
5. **Monitoring**: Set up external monitoring/alerts

## 📈 Performance Monitoring

The system tracks:

- **Returns**: Simple and log returns
- **Sharpe Ratio**: Annualized risk-adjusted return
- **Profit Factor**: Ratio of gross profits to gross losses
- **Win Rate**: Percentage of winning trades
- **Drawdown**: Maximum peak-to-trough decline
- **Trade Frequency**: Number of trades over time

## 🛡️ Risk Management

Built-in risk controls:

- **Position Sizing**: Fixed fraction or fixed notional
- **Stop Loss**: Optional per-trade stop loss (configurable)
- **Take Profit**: Optional take profit targets
- **Max Position Size**: Limit maximum position exposure

## 📝 Logging

Logs include:

- System startup/shutdown events
- Trade executions (open/close)
- Performance snapshots
- Errors and exceptions
- Re-optimization events

Log files: `logs/live_trading/live_trading_YYYY-MM-DD.log`

## 🔄 Updates and Maintenance

1. **Regular Updates**: Pull latest code periodically
2. **Dependency Updates**: Keep packages up to date
3. **Data Cleanup**: Archive old order book data
4. **Backup Config**: Backup configuration and strategy parameters
5. **Monitor Performance**: Review metrics and adjust as needed

## 📞 Support

For issues or questions:
1. Check this README
2. Review system logs
3. Check Telegram alerts
4. Examine dashboard metrics

## ⚠️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk. Always test thoroughly with paper trading before live deployment.

