"""Telegram notification service for trade alerts and system status."""
from __future__ import annotations

import asyncio
from typing import Optional
from datetime import datetime


class TelegramNotifier:
    """Send notifications via Telegram bot."""
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        
        if self.enabled:
            print(f"[TelegramNotifier] Enabled for chat {chat_id}")
        else:
            print("[TelegramNotifier] Disabled (no credentials or explicitly disabled)")
    
    async def send_message(self, message: str) -> bool:
        """Send a message to Telegram.
        
        Args:
            message: Message text to send
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            import aiohttp  # type: ignore
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        print(f"[TelegramNotifier] Failed to send: {resp.status}")
                        return False
        
        except Exception as e:
            print(f"[TelegramNotifier] Error sending message: {e}")
            return False
    
    async def send_trade_alert(
        self,
        action: str,
        direction: str,
        price: float,
        quantity: float,
        symbol: str,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
    ) -> bool:
        """Send trade execution alert.
        
        Args:
            action: 'open' or 'close'
            direction: 'long' or 'short'
            price: Execution price
            quantity: Trade quantity
            symbol: Trading symbol
            pnl: Profit/loss (for close trades)
            pnl_pct: P&L percentage (for close trades)
        
        Returns:
            True if sent successfully
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        if action == 'open':
            emoji = '🟢' if direction == 'long' else '🔴'
            message = f"{emoji} *Trade Opened*\n\n"
            message += f"Symbol: `{symbol}`\n"
            message += f"Direction: *{direction.upper()}*\n"
            message += f"Price: `{price:.6f}`\n"
            message += f"Quantity: `{quantity:.6f}`\n"
            message += f"Time: `{timestamp}`"
        
        else:  # close
            if pnl and pnl > 0:
                emoji = '✅'
                status = 'PROFIT'
            elif pnl and pnl < 0:
                emoji = '❌'
                status = 'LOSS'
            else:
                emoji = '➖'
                status = 'BREAKEVEN'
            
            message = f"{emoji} *Trade Closed - {status}*\n\n"
            message += f"Symbol: `{symbol}`\n"
            message += f"Direction: *{direction.upper()}*\n"
            message += f"Price: `{price:.6f}`\n"
            message += f"Quantity: `{quantity:.6f}`\n"
            
            if pnl is not None:
                message += f"P&L: `${pnl:.2f}` ({pnl_pct:.2f}%)\n"
            
            message += f"Time: `{timestamp}`"
        
        return await self.send_message(message)
    
    async def send_error_alert(self, error_type: str, error_msg: str) -> bool:
        """Send error alert.
        
        Args:
            error_type: Type of error
            error_msg: Error message
        
        Returns:
            True if sent successfully
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        message = f"⚠️ *System Error*\n\n"
        message += f"Type: `{error_type}`\n"
        message += f"Message: `{error_msg}`\n"
        message += f"Time: `{timestamp}`"
        
        return await self.send_message(message)
    
    async def send_status_update(
        self,
        capital: float,
        unrealized_pnl: float,
        position: float,
        num_trades: int,
        win_rate: float,
        total_return_pct: float,
    ) -> bool:
        """Send periodic status update.
        
        Args:
            capital: Current realized capital
            unrealized_pnl: Unrealized P&L
            position: Current position
            num_trades: Total number of trades
            win_rate: Win rate percentage
            total_return_pct: Total return percentage
        
        Returns:
            True if sent successfully
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        total_equity = capital + unrealized_pnl
        
        pos_emoji = '🟢' if position > 0 else ('🔴' if position < 0 else '⚪')
        pos_text = 'LONG' if position > 0 else ('SHORT' if position < 0 else 'FLAT')
        
        message = f"📊 *Status Update*\n\n"
        message += f"Equity: `${total_equity:.2f}`\n"
        message += f"Capital: `${capital:.2f}`\n"
        message += f"Unrealized P&L: `${unrealized_pnl:.2f}`\n"
        message += f"Position: {pos_emoji} *{pos_text}*\n"
        message += f"Total Return: `{total_return_pct:.2f}%`\n"
        message += f"Trades: `{num_trades}`\n"
        message += f"Win Rate: `{win_rate:.1f}%`\n"
        message += f"Time: `{timestamp}`"
        
        return await self.send_message(message)
    
    async def send_reoptimization_alert(
        self,
        old_params: dict,
        new_params: dict,
        improvement: Optional[float] = None,
    ) -> bool:
        """Send strategy re-optimization alert.
        
        Args:
            old_params: Previous strategy parameters
            new_params: New optimized parameters
            improvement: Performance improvement metric
        
        Returns:
            True if sent successfully
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        message = f"🔄 *Strategy Re-optimized*\n\n"
        message += f"Old params: `{old_params}`\n"
        message += f"New params: `{new_params}`\n"
        
        if improvement is not None:
            message += f"Improvement: `{improvement:.2f}%`\n"
        
        message += f"Time: `{timestamp}`"
        
        return await self.send_message(message)

