# Live Trading System - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Install live trading dependencies
pip install -r live_trading/requirements.txt

# Or install all project dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

Copy the example environment file and add your credentials:

```bash
cd live_trading
cp env_example.txt .env
nano .env  # or use your preferred editor
```

**Minimum required settings:**
```
BINANCE_API_KEY=your_actual_api_key
BINANCE_API_SECRET=your_actual_secret
TRADING_SYMBOL=VETUSD
TRADING_MODE=futures
```

### Step 3: Start the Price Feeder

The system needs live price data. Start the CCXT price feeder:

```bash
# From project root
python data/live_feeder_ccxt.py \
    --symbol VETUSD \
    --interval 1m \
    --poll 30 \
    --mode futures
```

Leave this running in a separate terminal/screen session.

### Step 4: Start Live Trading System

```bash
# Option 1: Use the startup script (recommended)
./start_live_trading.sh

# Option 2: Manual start
cd live_trading
python run_live.py
```

### Step 5: Access Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

You should see the live trading dashboard with real-time metrics!

---

## 📝 Configuration Options

### Basic Configuration (config.json)

Edit `live_trading/config.json` to customize:

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
      "z_threshold": 1.5,
      "exit_band": -1.5
    }
  }
}
```

### Environment Variables

Alternatively, use environment variables (see `env_example.txt`):

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export TRADING_SYMBOL="VETUSD"
export TRADING_MODE="futures"
```

---

## 🔧 Testing Before Live Trading

### 1. Paper Trading Mode

Test without real money by not providing API credentials:

```bash
# Remove or comment out in .env:
# BINANCE_API_KEY=
# BINANCE_API_SECRET=
```

The system will run in "dry-run" mode, simulating trades.

### 2. Binance Testnet

Use Binance testnet for safe testing:

```bash
export BINANCE_TESTNET=true
```

Get testnet API keys from: https://testnet.binancefuture.com/

---

## 📱 Enable Telegram Alerts

### 1. Create Telegram Bot

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the bot token

### 2. Get Your Chat ID

1. Message your bot anything
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Find your `chat_id` in the response

### 3. Configure

```bash
export TELEGRAM_ENABLED=true
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

You'll now receive trade alerts and system notifications!

---

## 🌐 Cloud Deployment

### Using Screen (Simple)

```bash
# SSH to your server
ssh user@your-server.com

# Start price feeder
screen -S price_feeder
python data/live_feeder_ccxt.py --symbol VETUSD --interval 1m --mode futures
# Press Ctrl+A, D to detach

# Start trading system
screen -S live_trading
cd live_trading
python run_live.py
# Press Ctrl+A, D to detach

# Reattach anytime:
screen -r live_trading
```

### Using systemd (Production)

Create `/etc/systemd/system/live-trading.service`:

```ini
[Unit]
Description=Live Trading System
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/Backtesting_framework
EnvironmentFile=/path/to/live_trading/.env
ExecStart=/usr/bin/python3 live_trading/run_live.py
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

---

## 🔍 Monitoring

### View Logs

```bash
# Live trading logs
tail -f logs/live_trading/*.log

# Price feeder logs
tail -f logs/price_feeder.log
```

### Check Dashboard

Monitor real-time at: `http://your-server:8000`

### Telegram Alerts

Receive instant notifications for:
- Trade executions
- System errors
- Daily performance summaries

---

## ❓ Troubleshooting

### Issue: "No price data available"

**Solution:** Ensure price feeder is running:
```bash
ps aux | grep live_feeder_ccxt
```

### Issue: "WebSocket connection failed"

**Solution:** Check symbol format (e.g., `VETUSDT` not `VETUSD` for WebSocket)

### Issue: "API authentication failed"

**Solution:** Verify API credentials and permissions:
```bash
# Test credentials
python -c "import ccxt; exchange = ccxt.binance({'apiKey': 'YOUR_KEY', 'secret': 'YOUR_SECRET'}); print(exchange.fetch_balance())"
```

### Issue: "Dashboard not loading"

**Solution:** Check if port 8000 is available:
```bash
lsof -i :8000
```

---

## 🛡️ Safety Checklist

Before going live:

- [ ] Tested in paper trading mode
- [ ] Tested on Binance testnet
- [ ] Verified strategy parameters
- [ ] Set appropriate position sizing
- [ ] Enabled Telegram alerts
- [ ] Configured stop loss (optional)
- [ ] Tested on small capital first
- [ ] Monitored for 24 hours
- [ ] Reviewed dashboard metrics
- [ ] Set up cloud monitoring

---

## 📞 Quick Commands

```bash
# Start everything
./start_live_trading.sh

# Stop gracefully
# Press Ctrl+C in the terminal

# View status
curl http://localhost:8000/api/status | jq

# Get current metrics
curl http://localhost:8000/api/metrics | jq

# View recent trades
curl http://localhost:8000/api/trades | jq
```

---

## 🎯 Next Steps

1. **Monitor Performance**: Watch the dashboard for 24-48 hours
2. **Optimize Parameters**: Use the re-optimization feature
3. **Scale Up**: Gradually increase position size
4. **Diversify**: Add more symbols/strategies
5. **Automate**: Set up cloud deployment with auto-restart

---

**Happy Trading! 🚀**

*Remember: Always start small and test thoroughly before deploying significant capital.*

