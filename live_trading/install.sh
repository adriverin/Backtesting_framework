#!/bin/bash

# Live Trading System - Installation Script
# This script helps set up the live trading system

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Live Trading System - Installation & Setup          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo -e "${BLUE}▶${NC} Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python version: ${python_version}"
echo ""

# Install dependencies
echo -e "${BLUE}▶${NC} Installing dependencies..."
echo ""

if [ -f "requirements.txt" ]; then
    echo "Installing from local requirements.txt..."
    pip install -r requirements.txt
else
    echo "Installing from main project requirements.txt..."
    pip install -r ../requirements.txt
fi

echo ""
echo -e "${GREEN}✓${NC} Dependencies installed successfully!"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${BLUE}▶${NC} Creating .env configuration file..."
    cp env_example.txt .env
    echo -e "${GREEN}✓${NC} Created .env file from template"
    echo -e "${YELLOW}⚠${NC}  Please edit .env and add your Binance API credentials:"
    echo "      nano .env"
    echo ""
else
    echo -e "${GREEN}✓${NC} .env file already exists"
    echo ""
fi

# Create logs directory
echo -e "${BLUE}▶${NC} Creating log directories..."
mkdir -p logs/live_trading
mkdir -p ../logs/live_trading
echo -e "${GREEN}✓${NC} Log directories created"
echo ""

# Create data directories
echo -e "${BLUE}▶${NC} Creating data directories..."
mkdir -p ../data/spot
mkdir -p ../data/futures
mkdir -p ../data/orderbook_depth
echo -e "${GREEN}✓${NC} Data directories created"
echo ""

# Make scripts executable
echo -e "${BLUE}▶${NC} Making scripts executable..."
chmod +x start_live_trading.sh
chmod +x health_check.py
chmod +x install.sh
echo -e "${GREEN}✓${NC} Scripts are now executable"
echo ""

# Run health check
echo -e "${BLUE}▶${NC} Running system health check..."
echo ""
python3 health_check.py || true
echo ""

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║             Installation Complete!                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo "1. Configure your API credentials:"
echo -e "   ${YELLOW}nano .env${NC}"
echo ""
echo "2. Edit configuration (optional):"
echo -e "   ${YELLOW}nano config.json${NC}"
echo ""
echo "3. Start the price feeder (in a separate terminal):"
echo -e "   ${YELLOW}python ../data/live_feeder_ccxt.py --symbol VETUSD --interval 1m --mode futures${NC}"
echo ""
echo "4. Start the live trading system:"
echo -e "   ${YELLOW}./start_live_trading.sh${NC}"
echo ""
echo "5. Open the dashboard in your browser:"
echo -e "   ${YELLOW}http://localhost:8000${NC}"
echo ""
echo -e "${RED}⚠  IMPORTANT SECURITY REMINDERS:${NC}"
echo "   • Never commit .env to version control"
echo "   • Test with paper trading first (no API keys)"
echo "   • Test on Binance testnet before live trading"
echo "   • Start with small amounts"
echo "   • Monitor regularly"
echo ""
echo -e "${GREEN}Happy Trading! 🚀${NC}"
echo ""

