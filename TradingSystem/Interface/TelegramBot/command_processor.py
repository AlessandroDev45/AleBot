import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CommandProcessor:
    def __init__(self, trading_system):
        """Initialize command processor"""
        self.trading_system = trading_system
        self.commands = {
            'start': self.start_command,
            'help': self.help_command,
            'status': self.status_command,
            'balance': self.balance_command,
            'positions': self.positions_command,
            'performance': self.performance_command,
            'trades': self.trades_command,
            'settings': self.settings_command
        }
        
    def process_command(self, command: str, args: list = None) -> str:
        """Process incoming command and return response"""
        try:
            if command in self.commands:
                return self.commands[command](args)
            return "Unknown command. Type /help for available commands."
            
        except Exception as e:
            logger.error(f"Error processing command {command}: {e}")
            return f"Error processing command: {str(e)}"
            
    def start_command(self, args=None) -> str:
        """Handle /start command"""
        return (
            "Welcome to the Trading Bot!\n\n"
            "Available commands:\n"
            "/status - Get system status\n"
            "/balance - Get account balance\n"
            "/positions - View open positions\n"
            "/performance - Get performance metrics\n"
            "/trades - View recent trades\n"
            "/settings - View/modify settings\n"
            "/help - Show this help message"
        )
        
    def help_command(self, args=None) -> str:
        """Handle /help command"""
        return self.start_command()
        
    def status_command(self, args=None) -> str:
        """Handle /status command"""
        try:
            status = self.trading_system.get_status()
            return (
                f"System Status\n\n"
                f"Trading: {'Active' if status['trading_active'] else 'Inactive'}\n"
                f"Strategy: {status['strategy']}\n"
                f"Active Since: {status['start_time']}\n"
                f"Open Positions: {status['open_positions']}\n"
                f"Pending Orders: {status['pending_orders']}\n"
                f"CPU Usage: {status['cpu_usage']}%\n"
                f"Memory Usage: {status['memory_usage']}%"
            )
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            return "Error getting system status"
            
    def balance_command(self, args=None) -> str:
        """Handle /balance command"""
        try:
            balance = self.trading_system.get_account_balance()
            return (
                f"Account Balance\n\n"
                f"Total: {balance['total']:.2f} USDT\n"
                f"Available: {balance['available']:.2f} USDT\n"
                f"In Position: {balance['in_position']:.2f} USDT\n"
                f"On Order: {balance['on_order']:.2f} USDT\n"
                f"P&L Today: {balance['daily_pnl']:.2f}%"
            )
        except Exception as e:
            logger.error(f"Error in balance command: {e}")
            return "Error getting account balance"
            
    def positions_command(self, args=None) -> str:
        """Handle /positions command"""
        try:
            positions = self.trading_system.get_positions()
            if not positions:
                return "No open positions"
                
            response = "Open Positions:\n\n"
            for pos in positions:
                response += (
                    f"Symbol: {pos['symbol']}\n"
                    f"Side: {pos['side']}\n"
                    f"Size: {pos['size']:.8f}\n"
                    f"Entry: {pos['entry_price']:.8f}\n"
                    f"Current: {pos['current_price']:.8f}\n"
                    f"P&L: {pos['pnl']:.2f}%\n"
                    f"Duration: {pos['duration']}\n\n"
                )
            return response
            
        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            return "Error getting positions"
            
    def performance_command(self, args=None) -> str:
        """Handle /performance command"""
        try:
            perf = self.trading_system.get_performance_metrics()
            return (
                f"Performance Metrics\n\n"
                f"Daily P&L: {perf['daily_pnl']:.2f}%\n"
                f"Weekly P&L: {perf['weekly_pnl']:.2f}%\n"
                f"Monthly P&L: {perf['monthly_pnl']:.2f}%\n"
                f"Total Trades: {perf['total_trades']}\n"
                f"Win Rate: {perf['win_rate']:.1f}%\n"
                f"Avg Trade: {perf['avg_trade']:.2f}%\n"
                f"Best Trade: {perf['best_trade']:.2f}%\n"
                f"Worst Trade: {perf['worst_trade']:.2f}%\n"
                f"Profit Factor: {perf['profit_factor']:.2f}\n"
                f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}"
            )
        except Exception as e:
            logger.error(f"Error in performance command: {e}")
            return "Error getting performance metrics"
            
    def trades_command(self, args=None) -> str:
        """Handle /trades command"""
        try:
            trades = self.trading_system.get_recent_trades()
            if not trades:
                return "No recent trades"
                
            response = "Recent Trades:\n\n"
            for trade in trades:
                response += (
                    f"Symbol: {trade['symbol']}\n"
                    f"Side: {trade['side']}\n"
                    f"Entry: {trade['entry_price']:.8f}\n"
                    f"Exit: {trade['exit_price']:.8f}\n"
                    f"P&L: {trade['pnl']:.2f}%\n"
                    f"Time: {trade['timestamp']}\n\n"
                )
            return response
            
        except Exception as e:
            logger.error(f"Error in trades command: {e}")
            return "Error getting recent trades"
            
    def settings_command(self, args=None) -> str:
        """Handle /settings command"""
        try:
            settings = self.trading_system.get_settings()
            return (
                f"Current Settings\n\n"
                f"Strategy: {settings['strategy']}\n"
                f"Risk Per Trade: {settings['risk_per_trade']}%\n"
                f"Max Positions: {settings['max_positions']}\n"
                f"Trading Pairs: {', '.join(settings['trading_pairs'])}\n"
                f"Notifications: {'Enabled' if settings['notifications'] else 'Disabled'}"
            )
        except Exception as e:
            logger.error(f"Error in settings command: {e}")
            return "Error getting settings"
