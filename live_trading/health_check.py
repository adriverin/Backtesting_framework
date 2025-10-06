"""System health check utility for live trading setup verification."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class HealthCheck:
    """Verify live trading system setup and configuration."""
    
    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []
    
    def add_check(self, name: str, passed: bool, message: str) -> None:
        """Add a check result."""
        self.checks.append((name, passed, message))
    
    def check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        dependencies = [
            ('ccxt', 'pip install ccxt'),
            ('websockets', 'pip install websockets'),
            ('fastapi', 'pip install fastapi'),
            ('uvicorn', 'pip install uvicorn[standard]'),
            ('aiohttp', 'pip install aiohttp'),
            ('pandas', 'pip install pandas'),
            ('numpy', 'pip install numpy'),
        ]
        
        for dep, install_cmd in dependencies:
            try:
                __import__(dep)
                self.add_check(f"Dependency: {dep}", True, f"✓ {dep} installed")
            except ImportError:
                self.add_check(f"Dependency: {dep}", False, f"✗ {dep} missing. Install: {install_cmd}")
    
    def check_configuration(self) -> None:
        """Check configuration files."""
        config_file = Path(__file__).parent / "config.json"
        env_file = Path(__file__).parent / ".env"
        
        if config_file.exists():
            self.add_check("Config file", True, "✓ config.json found")
            
            # Try to load config
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check critical fields
                if 'symbol' in config:
                    self.add_check("Config: symbol", True, f"✓ Symbol: {config['symbol']}")
                else:
                    self.add_check("Config: symbol", False, "✗ Symbol not configured")
                
                if 'mode' in config:
                    self.add_check("Config: mode", True, f"✓ Mode: {config['mode']}")
                else:
                    self.add_check("Config: mode", False, "✗ Mode not configured")
            
            except Exception as e:
                self.add_check("Config parsing", False, f"✗ Error loading config: {e}")
        else:
            self.add_check("Config file", False, "✗ config.json not found")
        
        if env_file.exists():
            self.add_check("Environment file", True, "✓ .env file found")
        else:
            self.add_check("Environment file", False, "⚠ .env file not found (optional)")
    
    def check_binance_credentials(self) -> None:
        """Check Binance API credentials."""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if api_key and api_secret:
            self.add_check("Binance API Key", True, "✓ API key configured")
            self.add_check("Binance API Secret", True, "✓ API secret configured")
            
            # Try to connect
            try:
                import ccxt
                exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })
                # Test connection (synchronous for health check)
                # Note: This will fail if called from async context
                # balance = exchange.fetch_balance()
                # self.add_check("Binance Connection", True, "✓ API connection successful")
                self.add_check("Binance Connection", True, "✓ Credentials look valid (connection not tested)")
            except Exception as e:
                self.add_check("Binance Connection", False, f"✗ Connection error: {str(e)[:50]}")
        else:
            self.add_check("Binance API Credentials", False, "⚠ No API credentials (paper trading mode)")
    
    def check_telegram(self) -> None:
        """Check Telegram configuration."""
        enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if enabled:
            if bot_token and chat_id:
                self.add_check("Telegram", True, f"✓ Telegram enabled with chat ID: {chat_id}")
            else:
                self.add_check("Telegram", False, "✗ Telegram enabled but missing bot_token or chat_id")
        else:
            self.add_check("Telegram", True, "✓ Telegram disabled (notifications off)")
    
    def check_data_directory(self) -> None:
        """Check data directories."""
        data_dir = Path(__file__).parent.parent / "data"
        
        if data_dir.exists():
            self.add_check("Data directory", True, f"✓ Data directory found: {data_dir}")
            
            # Check for spot/futures subdirs
            spot_dir = data_dir / "spot"
            futures_dir = data_dir / "futures"
            
            if spot_dir.exists():
                self.add_check("Spot data dir", True, "✓ Spot data directory exists")
            else:
                self.add_check("Spot data dir", False, "⚠ Spot data directory not found")
            
            if futures_dir.exists():
                self.add_check("Futures data dir", True, "✓ Futures data directory exists")
            else:
                self.add_check("Futures data dir", False, "⚠ Futures data directory not found")
        else:
            self.add_check("Data directory", False, "✗ Data directory not found")
    
    def check_orderbook_data(self) -> None:
        """Check order book data directory."""
        ob_dir = Path(__file__).parent.parent / "data" / "orderbook_depth"
        
        if ob_dir.exists():
            self.add_check("Order book directory", True, f"✓ Order book directory found")
            
            # Check if any symbol directories exist
            symbol_dirs = [d for d in ob_dir.iterdir() if d.is_dir()]
            if symbol_dirs:
                self.add_check("Order book data", True, f"✓ {len(symbol_dirs)} symbol(s) with orderbook data")
            else:
                self.add_check("Order book data", False, "⚠ No order book data found (will be created on first run)")
        else:
            self.add_check("Order book directory", False, "⚠ Order book directory not found (will be created)")
    
    def check_strategy_files(self) -> None:
        """Check if strategy files exist."""
        strategies_dir = Path(__file__).parent.parent / "strategies"
        
        if strategies_dir.exists():
            self.add_check("Strategies directory", True, "✓ Strategies directory found")
            
            # Check for order book depth strategy
            ob_strategy = strategies_dir / "orderbook_depth_strategy.py"
            if ob_strategy.exists():
                self.add_check("OrderBook strategy", True, "✓ Order book depth strategy found")
            else:
                self.add_check("OrderBook strategy", False, "✗ Order book depth strategy not found")
        else:
            self.add_check("Strategies directory", False, "✗ Strategies directory not found")
    
    def run_all_checks(self) -> bool:
        """Run all health checks."""
        print("=" * 70)
        print("🏥 LIVE TRADING SYSTEM - HEALTH CHECK")
        print("=" * 70)
        print()
        
        self.check_dependencies()
        self.check_configuration()
        self.check_binance_credentials()
        self.check_telegram()
        self.check_data_directory()
        self.check_orderbook_data()
        self.check_strategy_files()
        
        # Print results
        passed = 0
        failed = 0
        warnings = 0
        
        for name, success, message in self.checks:
            if '⚠' in message:
                warnings += 1
                print(f"⚠️  {message}")
            elif success:
                passed += 1
                print(f"✅ {message}")
            else:
                failed += 1
                print(f"❌ {message}")
        
        print()
        print("=" * 70)
        print(f"Summary: {passed} passed, {failed} failed, {warnings} warnings")
        print("=" * 70)
        
        if failed > 0:
            print()
            print("⚠️  Some checks failed. Please address the issues above before starting live trading.")
            return False
        elif warnings > 0:
            print()
            print("✓  All critical checks passed! Some optional components are missing.")
            return True
        else:
            print()
            print("✅ All checks passed! System is ready for live trading.")
            return True
    
    def print_next_steps(self) -> None:
        """Print recommended next steps."""
        print()
        print("=" * 70)
        print("📋 NEXT STEPS")
        print("=" * 70)
        print()
        print("1. Start the price feeder:")
        print("   python data/live_feeder_ccxt.py --symbol VETUSD --interval 1m --mode futures")
        print()
        print("2. Start the live trading system:")
        print("   cd live_trading")
        print("   python run_live.py")
        print()
        print("3. Open the dashboard:")
        print("   http://localhost:8000")
        print()
        print("4. Monitor Telegram alerts (if enabled)")
        print()
        print("=" * 70)


def main():
    """Run health check."""
    # Load .env if exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            print(f"Warning: Error loading .env: {e}")
    
    # Run health check
    health_check = HealthCheck()
    all_passed = health_check.run_all_checks()
    
    if all_passed:
        health_check.print_next_steps()
        sys.exit(0)
    else:
        print()
        print("Please fix the issues and run this health check again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

