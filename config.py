import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Dict, Any

class ConfigManager:
    def __init__(self):
        # Load environment variables
        load_dotenv('BTC.env')

        # API Credentials
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
        self.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

        if not all([self.BINANCE_API_KEY, self.BINANCE_API_SECRET, self.TELEGRAM_BOT_TOKEN]):
            raise ValueError('Missing required API credentials in BTC.env')

        # Paths
        self.BASE_DIR = Path(__file__).parent
        self.DATA_DIR = self.BASE_DIR / 'data'
        self.MODELS_DIR = self.BASE_DIR / 'models'
        self.LOGS_DIR = self.BASE_DIR / 'logs'

        # Create necessary directories
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(exist_ok=True)

        # Database Configuration
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')

        # Trading Configuration
        self.TRADING_CONFIG = {
            'symbols': ['BTCUSDT'],
            'timeframes': ['1m', '5m', '15m', '1h', '4h'],
            'base_timeframe': '1m',
            'update_interval': 1,  # seconds
            'max_positions': 3,
            'min_position_size': 0.001,  # BTC
            'max_position_size': 0.1,    # BTC
            'price_precision': 2,
            'quantity_precision': 6
        }

        # Risk Management Configuration
        self.RISK_CONFIG = {
            'max_risk_per_trade': 0.01,   # 1% per trade
            'max_daily_risk': 0.03,       # 3% per day
            'max_position_size': 0.1,     # 10% of portfolio
            'stop_loss_margin': 0.02,     # 2% default stop loss
            'trailing_stop': 0.01,        # 1% trailing stop
            'take_profit_margin': 0.05    # 5% take profit
        }

    def get_database_config(self) -> Dict[str, Any]:
        return {
            'url': self.DATABASE_URL,
            'echo': False,
            'pool_size': 20,
            'max_overflow': 10
        }

    def get_exchange_config(self) -> Dict[str, Any]:
        return {
            'api_key': self.BINANCE_API_KEY,
            'api_secret': self.BINANCE_API_SECRET,
            'symbols': self.TRADING_CONFIG['symbols'],
            'update_interval': self.TRADING_CONFIG['update_interval']
        }
