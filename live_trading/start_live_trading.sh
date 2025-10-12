#!/bin/bash

# Live Trading System Startup Script
# This script starts both the price feeder and the live trading system


set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Cleanup function (must be defined before trap)
cleanup() {
    # Disable exit-on-error during cleanup
    set +e
    
    echo ""
    echo -e "${RED}⏹${NC} Shutting down..."
    
    # Stop price feeder if we started it
    if [ -f /tmp/price_feeder.pid ]; then
        PID=$(cat /tmp/price_feeder.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${BLUE}⏹${NC} Stopping price feeder (PID: ${PID})..."
            kill $PID 2>/dev/null || true
            # Wait briefly for graceful shutdown
            sleep 1
            # Force kill if still running
            if ps -p $PID > /dev/null 2>&1; then
                kill -9 $PID 2>/dev/null || true
            fi
        fi
        rm -f /tmp/price_feeder.pid
    fi
    
    echo -e "${GREEN}✓${NC} Shutdown complete"
    exit 0
}

# Set trap BEFORE any long-running commands
trap cleanup EXIT INT TERM

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Live Trading System - Startup Script  ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo ""

# Configuration
SYMBOL="${TRADING_SYMBOL:-VETUSDT}"
INTERVAL="${TRADING_TIMEFRAME:-1m}"
MODE="${TRADING_MODE:-futures}"
POLL_INTERVAL="${PRICE_POLL_INTERVAL:-30}"

# Resolve project root (parent of this script's directory)
PROJECT_ROOT="$(cd .. && pwd)"

# Normalize symbol: for UM futures/spot, prefer USDT suffix if user passed USD
if [[ "${SYMBOL}" =~ ^[A-Z0-9]+USD$ ]] && [[ ! "${SYMBOL}" =~ USDT$ ]]; then
    SYMBOL="${SYMBOL}T"
fi

# Check if price feeder is already running
PRICE_FEEDER_PID=$(pgrep -f "live_feeder_ccxt.py.*${SYMBOL}.*${INTERVAL}" || true)

if [ -n "$PRICE_FEEDER_PID" ]; then
    echo -e "${GREEN}✓${NC} Price feeder already running (PID: ${PRICE_FEEDER_PID})"
else
    # Start price feeder in background (must run from project root)
    echo -e "${BLUE}▶${NC} Starting price feeder for ${SYMBOL} ${INTERVAL} (${MODE})..."
    cd "${PROJECT_ROOT}"
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
    
    cd "${PROJECT_ROOT}/live_trading"
fi

# Backfill orderbook depth history before starting live engine
echo ""
echo -e "${BLUE}▶${NC} Backfilling orderbook depth history (to populate lookback)..."

CONFIG_PATH="${PROJECT_ROOT}/live_trading/config.json"
OB_SYMBOL="${SYMBOL}"
OB_MODE="${MODE}"
LOOKBACK_DAYS="210"

if command -v jq >/dev/null 2>&1 && [ -f "${CONFIG_PATH}" ]; then
    CFG_SYMBOL=$(jq -r '.symbol // empty' "${CONFIG_PATH}")
    CFG_MODE=$(jq -r '.mode // empty' "${CONFIG_PATH}")
    CFG_LB=$(jq -r '.strategy.params.lookback_mean // empty' "${CONFIG_PATH}")
    if [ -n "${CFG_SYMBOL}" ]; then OB_SYMBOL="${CFG_SYMBOL}"; fi
    if [ -n "${CFG_MODE}" ]; then OB_MODE="${CFG_MODE}"; fi
    # Parse values like "210D" or "210d" into days
    if [[ "${CFG_LB}" =~ ^([0-9]+)[Dd]$ ]]; then
        LOOKBACK_DAYS="${BASH_REMATCH[1]}"
    fi
fi

# Normalize OB_SYMBOL as well
if [[ "${OB_SYMBOL}" =~ ^[A-Z0-9]+USD$ ]] && [[ ! "${OB_SYMBOL}" =~ USDT$ ]]; then
    OB_SYMBOL="${OB_SYMBOL}T"
fi

echo -e "${BLUE}⏳${NC} Downloading last ${LOOKBACK_DAYS} days for ${OB_SYMBOL} (${OB_MODE})..."
set +e
cd "${PROJECT_ROOT}"
python live_trading/backfill_orderbook_depth.py --symbol "${OB_SYMBOL}" --mode "${OB_MODE}" --days "${LOOKBACK_DAYS}"
BACKFILL_RC=$?
set -e
cd "${PROJECT_ROOT}/live_trading"
if [ ${BACKFILL_RC} -ne 0 ]; then
    echo -e "${RED}⚠${NC} Backfill encountered errors (code ${BACKFILL_RC}). Continuing startup."
else
    echo -e "${GREEN}✓${NC} Backfill complete."
fi

# Start live trading system
echo ""
echo -e "${BLUE}▶${NC} Starting live trading system..."

# Load environment variables if .env exists
if [ -f .env ]; then
    echo -e "${GREEN}✓${NC} Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the trading system (runs in foreground, blocks until Ctrl+C or exit)
python run_live.py

# Script exits here, triggering the EXIT trap which runs cleanup()

