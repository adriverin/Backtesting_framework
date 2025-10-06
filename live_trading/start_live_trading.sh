#!/bin/bash

# Live Trading System Startup Script
# This script starts both the price feeder and the live trading system

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Live Trading System - Startup Script ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo ""

# Configuration
SYMBOL="${TRADING_SYMBOL:-VETUSD}"
INTERVAL="${TRADING_TIMEFRAME:-1m}"
MODE="${TRADING_MODE:-futures}"
POLL_INTERVAL="${PRICE_POLL_INTERVAL:-30}"

# Check if price feeder is already running
PRICE_FEEDER_PID=$(pgrep -f "live_feeder_ccxt.py.*${SYMBOL}.*${INTERVAL}" || true)

if [ -n "$PRICE_FEEDER_PID" ]; then
    echo -e "${GREEN}✓${NC} Price feeder already running (PID: ${PRICE_FEEDER_PID})"
else
    # Start price feeder in background
    echo -e "${BLUE}▶${NC} Starting price feeder for ${SYMBOL} ${INTERVAL} (${MODE})..."
    cd ..
    nohup python data/live_feeder_ccxt.py \
        --symbol "${SYMBOL}" \
        --interval "${INTERVAL}" \
        --poll "${POLL_INTERVAL}" \
        --mode "${MODE}" \
        > logs/price_feeder.log 2>&1 &
    
    PRICE_FEEDER_PID=$!
    echo -e "${GREEN}✓${NC} Price feeder started (PID: ${PRICE_FEEDER_PID})"
    echo "${PRICE_FEEDER_PID}" > /tmp/price_feeder.pid
    
    # Wait for initial data
    echo -e "${BLUE}⏳${NC} Waiting for initial price data (30s)..."
    sleep 30
fi

# Start live trading system
echo ""
echo -e "${BLUE}▶${NC} Starting live trading system..."
cd live_trading

# Load environment variables if .env exists
if [ -f .env ]; then
    echo -e "${GREEN}✓${NC} Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the trading system
python run_live.py

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${RED}⏹${NC} Shutting down..."
    
    # Stop price feeder if we started it
    if [ -f /tmp/price_feeder.pid ]; then
        PID=$(cat /tmp/price_feeder.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${BLUE}⏹${NC} Stopping price feeder (PID: ${PID})..."
            kill $PID
            rm /tmp/price_feeder.pid
        fi
    fi
    
    echo -e "${GREEN}✓${NC} Shutdown complete"
}

trap cleanup EXIT INT TERM

