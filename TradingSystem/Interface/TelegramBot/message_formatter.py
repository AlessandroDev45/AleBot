import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MessageFormatter:
    @staticmethod
    def format_trade_alert(trade_data):
        """Format trade entry/exit alert message"""
        try:
            emoji = "🟢" if trade_data['side'] == "BUY" else "🔴"
            return (
                f"{emoji} Trade {trade_data['type'].title()}\n\n"
                f"Symbol: {trade_data['symbol']}\n"
                f"Side: {trade_data['side']}\n"
                f"Entry: {trade_data['price']:.8f}\n"
                f"Size: {trade_data['size']:.8f}\n"
                f"Value: {trade_data['value']:.2f} USDT\n"
                f"Stop Loss: {trade_data['stop_loss']:.8f}\n"
                f"Take Profit:\n"
                f"- TP1 (30%): {trade_data['tp1']:.8f}\n"
                f"- TP2 (30%): {trade_data['tp2']:.8f}\n"
                f"- TP3 (40%): {trade_data['tp3']:.8f}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            logger.error(f"Error formatting trade alert: {e}")
            return "Error formatting trade alert"

    @staticmethod
    def format_daily_report(performance_data):
        """Format daily performance report"""
        try:
            return (
                f"📊 Daily Performance Report\n\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"P&L: {performance_data['daily_pnl']:.2f}%\n"
                f"Trades: {performance_data['total_trades']}\n"
                f"Win Rate: {performance_data['win_rate']:.1f}%\n"
                f"Best Trade: {performance_data['best_trade']:.2f}%\n"
                f"Worst Trade: {performance_data['worst_trade']:.2f}%\n"
                f"Current Drawdown: {performance_data['drawdown']:.2f}%\n"
                f"Volume: {performance_data['volume']:.2f} USDT"
            )
        except Exception as e:
            logger.error(f"Error formatting daily report: {e}")
            return "Error formatting daily report"

    @staticmethod
    def format_weekly_report(performance_data):
        """Format weekly performance report"""
        try:
            return (
                f"📈 Weekly Performance Report\n\n"
                f"Week: {datetime.now().strftime('%Y-W%W')}\n"
                f"P&L: {performance_data['weekly_pnl']:.2f}%\n"
                f"Total Trades: {performance_data['total_trades']}\n"
                f"Win Rate: {performance_data['win_rate']:.1f}%\n"
                f"Average Trade: {performance_data['avg_trade']:.2f}%\n"
                f"Profit Factor: {performance_data['profit_factor']:.2f}\n"
                f"Max Drawdown: {performance_data['max_drawdown']:.2f}%\n"
                f"Sharpe Ratio: {performance_data['sharpe_ratio']:.2f}"
            )
        except Exception as e:
            logger.error(f"Error formatting weekly report: {e}")
            return "Error formatting weekly report"

    @staticmethod
    def format_monthly_report(performance_data):
        """Format monthly performance report"""
        try:
            return (
                f"📊 Monthly Performance Report\n\n"
                f"Month: {datetime.now().strftime('%Y-%m')}\n"
                f"P&L: {performance_data['monthly_pnl']:.2f}%\n"
                f"Total Trades: {performance_data['total_trades']}\n"
                f"Win Rate: {performance_data['win_rate']:.1f}%\n"
                f"Best Day: {performance_data['best_day']:.2f}%\n"
                f"Worst Day: {performance_data['worst_day']:.2f}%\n"
                f"Average Daily P&L: {performance_data['avg_daily_pnl']:.2f}%\n"
                f"Profit Factor: {performance_data['profit_factor']:.2f}\n"
                f"Sharpe Ratio: {performance_data['sharpe_ratio']:.2f}"
            )
        except Exception as e:
            logger.error(f"Error formatting monthly report: {e}")
            return "Error formatting monthly report"

    @staticmethod
    def format_error_alert(error_data):
        """Format error alert message"""
        try:
            return (
                f"⚠️ System Alert\n\n"
                f"Type: {error_data['type']}\n"
                f"Component: {error_data['component']}\n"
                f"Message: {error_data['message']}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            logger.error(f"Error formatting error alert: {e}")
            return "Error formatting error alert"
