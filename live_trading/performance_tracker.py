"""Real-time performance tracking and metrics calculation."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque

import pandas as pd
import numpy as np


class PerformanceTracker:
    """Tracks and calculates real-time trading performance metrics."""
    
    def __init__(self, initial_capital: float = 1000.0, config: Any = None):
        """Initialize performance tracker.
        
        Args:
            initial_capital: Starting capital
            config: Trading configuration
        """
        self.initial_capital = initial_capital
        self.config = config
        
        # Equity curve
        self.equity_history: deque = deque(maxlen=100000)
        
        # Trade history
        self.trades: List[Dict[str, Any]] = []
        
        # Current metrics
        self.metrics: Dict[str, float] = {
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0,
            'num_wins': 0,
            'num_losses': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
        }
        
        # Returns for Sharpe calculation
        self.returns: deque = deque(maxlen=10000)
        
        # Peak equity for drawdown calculation
        self.peak_equity: float = initial_capital
    
    def record_trade(
        self,
        timestamp: datetime,
        direction: str,
        action: str,
        price: float,
        quantity: float,
        signal: float = 0.0,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
    ) -> None:
        """Record a trade execution.
        
        Args:
            timestamp: Trade timestamp
            direction: 'long' or 'short'
            action: 'open' or 'close'
            price: Execution price
            quantity: Trade quantity
            signal: Strategy signal value
            pnl: Profit/loss (for close trades)
            pnl_pct: P&L percentage (for close trades)
        """
        trade = {
            'timestamp': timestamp,
            'direction': direction,
            'action': action,
            'price': price,
            'quantity': quantity,
            'signal': signal,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
        }
        
        self.trades.append(trade)
        
        # Update metrics if this is a closing trade
        if action == 'close' and pnl is not None:
            self._update_trade_metrics(pnl, pnl_pct or 0.0)
    
    def _update_trade_metrics(self, pnl: float, pnl_pct: float) -> None:
        """Update trade-based metrics."""
        self.metrics['num_trades'] += 1
        
        if pnl > 0:
            self.metrics['num_wins'] += 1
        elif pnl < 0:
            self.metrics['num_losses'] += 1
        
        # Win rate
        if self.metrics['num_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['num_wins'] / self.metrics['num_trades']) * 100
        
        # Average win/loss
        wins = [t['pnl'] for t in self.trades if t.get('pnl') and t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t.get('pnl') and t['pnl'] < 0]
        
        self.metrics['avg_win'] = np.mean(wins) if wins else 0.0
        self.metrics['avg_loss'] = np.mean(losses) if losses else 0.0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        
        if total_losses > 0:
            self.metrics['profit_factor'] = total_wins / total_losses
        else:
            self.metrics['profit_factor'] = float('inf') if total_wins > 0 else 0.0
    
    def update(
        self,
        timestamp: datetime,
        capital: float,
        unrealized_pnl: float = 0.0,
        position: float = 0.0,
        signal: float = 0.0,
    ) -> None:
        """Update performance metrics with current state.
        
        Args:
            timestamp: Current timestamp
            capital: Realized capital
            unrealized_pnl: Unrealized P&L from open positions
            position: Current position (-1, 0, 1)
            signal: Current strategy signal
        """
        total_equity = capital + unrealized_pnl
        
        # Record equity
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'capital': capital,
            'unrealized_pnl': unrealized_pnl,
            'position': position,
            'signal': signal,
        })
        
        # Calculate return
        if len(self.equity_history) > 1:
            prev_equity = self.equity_history[-2]['equity']
            if prev_equity > 0:
                ret = (total_equity - prev_equity) / prev_equity
                self.returns.append(ret)
        
        # Total return
        self.metrics['total_return'] = total_equity - self.initial_capital
        self.metrics['total_return_pct'] = (self.metrics['total_return'] / self.initial_capital) * 100
        
        # Sharpe ratio (annualized)
        if len(self.returns) > 30:  # Need minimum data
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns)
            
            if std_return > 0:
                # Annualize based on update frequency (assume ~1 update per minute for live trading)
                periods_per_year = 365 * 24 * 60  # Minutes in a year
                self.metrics['sharpe_ratio'] = (mean_return / std_return) * math.sqrt(periods_per_year)
            else:
                self.metrics['sharpe_ratio'] = 0.0
        
        # Drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        if self.peak_equity > 0:
            current_dd = self.peak_equity - total_equity
            current_dd_pct = (current_dd / self.peak_equity) * 100
            
            if current_dd > self.metrics['max_drawdown']:
                self.metrics['max_drawdown'] = current_dd
            
            if current_dd_pct > self.metrics['max_drawdown_pct']:
                self.metrics['max_drawdown_pct'] = current_dd_pct
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_history:
            return pd.DataFrame()
        
        return pd.DataFrame(list(self.equity_history))
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_cumulative_returns(self) -> List[float]:
        """Get cumulative returns series."""
        if not self.equity_history:
            return []
        
        equities = [e['equity'] for e in self.equity_history]
        cum_returns = [(e - self.initial_capital) / self.initial_capital for e in equities]
        return cum_returns
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        equity_df = self.get_equity_curve()
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_equity': equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital,
            'metrics': self.get_metrics(),
            'num_data_points': len(self.equity_history),
            'num_trades': len(self.trades),
        }
        
        return summary

