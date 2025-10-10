"""Configuration management for live trading system."""
from __future__ import annotations

import os
from typing import Any, Dict
from pathlib import Path
import json


class TradingConfig:
    """Central configuration for live trading system."""
    
    def __init__(self, config_path: str | None = None):
        """Initialize config from file or environment variables."""
        # Resolve config path robustly: prefer provided path, then env, else file-relative default
        default_config_path = (Path(__file__).parent / "config.json").resolve()
        env_path = os.getenv("LIVE_TRADING_CONFIG")
        if config_path:
            cp = Path(config_path)
            self.config_path = str(cp if cp.is_absolute() else (Path(__file__).parent / cp).resolve())
        elif env_path:
            ep = Path(env_path)
            self.config_path = str(ep if ep.is_absolute() else (Path(__file__).parent / ep).resolve())
        else:
            self.config_path = str(default_config_path)
        self.config: Dict[str, Any] = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file or use defaults."""
        default_config = {
            # Trading parameters
            "symbol": os.getenv("TRADING_SYMBOL", "VETUSD"),
            "mode": os.getenv("TRADING_MODE", "futures"),  # "spot" or "futures"
            "timeframe": os.getenv("TRADING_TIMEFRAME", "1m"),
            "price_column": os.getenv("PRICE_COLUMN", "vwap_30"),
            
            # Strategy parameters (orderbook depth)
            "strategy": {
                "name": "orderbook_depth",
                "params": {
                    "percentage": int(os.getenv("OB_PERCENTAGE", "2")),
                    "lookback_mean": os.getenv("OB_LOOKBACK_MEAN", "210D"),  # Or set to None for optimization
                    "lookback_current": os.getenv("OB_LOOKBACK_CURRENT", "1min"),  # Or set to None for optimization
                    "z_threshold": float(os.getenv("OB_Z_THRESHOLD", "1.5")),
                    "exit_band": float(os.getenv("OB_EXIT_BAND", "-1.5")),
                    "persistence": int(os.getenv("OB_PERSISTENCE", "1")),
                }
            },
            
            # Binance API
            "binance": {
                "api_key": os.getenv("BINANCE_API_KEY", ""),
                "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                "testnet": os.getenv("BINANCE_TESTNET", "false").lower() == "true",
            },
            
            # Position sizing
            "position_sizing": {
                "mode": os.getenv("POSITION_MODE", "fixed_fraction"),
                "params": {
                    "fraction": float(os.getenv("POSITION_FRACTION", "1.0")),
                }
            },
            
            # Fees and slippage
            "fees": {
                "fee_bps": float(os.getenv("FEE_BPS", "4.0")),
                "slippage_bps": float(os.getenv("SLIPPAGE_BPS", "5.0")),
            },
            
            # Data collection
            "data": {
                # Number of percent-of-price buckets to compute per side (±1%..±N%)
                # This drives live transformation to match historical orderbook depth files
                "orderbook_depth_levels": 5,
                "orderbook_update_interval_ms": 100,  # WebSocket update frequency
                "price_update_interval_sec": 30,  # Price data polling interval
                "data_dir": "data",
                "orderbook_depth_dir": "data/orderbook_depth",  # Directory for orderbook depth data
            },
            
            # Re-optimization
            "reoptimization": {
                "enabled": True,
                "interval_hours": 24,  # Re-optimize every 24 hours
                "lookback_days": 365,  # Use 1 year of data for re-optimization
                "min_data_points": 10000,  # Minimum data points required
            },
            
            # Risk management
            "risk": {
                "max_position_size": 1.0,  # Maximum position size (fraction of capital)
                "stop_loss_pct": None,  # Optional stop loss percentage
                "take_profit_pct": None,  # Optional take profit percentage
            },
            
            # Telegram notifications
            "telegram": {
                "enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
                "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
            },
            
            # Dashboard / API server
            "server": {
                "host": os.getenv("SERVER_HOST", "0.0.0.0"),
                "port": int(os.getenv("SERVER_PORT", "8000")),
            },
            
            # Logging
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "dir": "logs/live_trading",
                "format": "json",  # "json" or "text"
            },
            
            # Runtime
            "runtime": {
                "auto_restart": True,
                "max_restart_attempts": 10,
                "restart_delay_sec": 60,
                "signal_update_interval_sec": int(os.getenv("SIGNAL_UPDATE_INTERVAL_SEC", "5")),
            }
        }
        
        # Try to load from file if exists
        config_file_path = Path(self.config_path)
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    file_config = json.load(f)
                    # Deep merge file config with defaults
                    self._deep_merge(default_config, file_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
        
        # Resolve relative paths to absolute paths
        # Paths in config are relative to the config file's directory
        config_dir = config_file_path.parent if config_file_path.exists() else Path(__file__).parent.resolve()
        
        # Resolve data_dir if it's relative
        if 'data' in default_config and 'data_dir' in default_config['data']:
            data_dir = default_config['data']['data_dir']
            data_path = Path(data_dir)
            if not data_path.is_absolute():
                # Resolve relative to config directory
                resolved_path = (config_dir / data_path).resolve()
                default_config['data']['data_dir'] = str(resolved_path)
        
        # Resolve orderbook_depth_dir if it's relative
        if 'data' in default_config and 'orderbook_depth_dir' in default_config['data']:
            ob_dir = default_config['data']['orderbook_depth_dir']
            ob_path = Path(ob_dir)
            if not ob_path.is_absolute():
                resolved_path = (config_dir / ob_path).resolve()
                default_config['data']['orderbook_depth_dir'] = str(resolved_path)
        
        # Resolve logging dir if it's relative
        if 'logging' in default_config and 'dir' in default_config['logging']:
            log_dir = default_config['logging']['dir']
            log_path = Path(log_dir)
            if not log_path.is_absolute():
                resolved_path = (config_dir / log_path).resolve()
                default_config['logging']['dir'] = str(resolved_path)
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge update dict into base dict (in-place)."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def save(self, path: str | None = None) -> None:
        """Save current config to file."""
        save_path = path or self.config_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation key (e.g., 'strategy.params.percentage')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access to top-level config."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment to top-level config."""
        self.config[key] = value

