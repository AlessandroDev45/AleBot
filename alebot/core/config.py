import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from decimal import Decimal
import logging
import json

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        self._load_environment()
        self._initialize_paths()
        self._load_trading_config()
        self._validate_config()

    def _load_environment(self):
        """Load environment variables from .env file"""
        env_path = Path('.env')
        if not env_path.exists():
            raise FileNotFoundError('No .env file found. Please create one based on .env.example')

        load_dotenv(env_path)
        
        # Required API credentials
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
        self.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.BINANCE_API_KEY, self.BINANCE_API_SECRET, self.TELEGRAM_BOT_TOKEN]):
            raise ValueError('Missing required API credentials in .env file')

    def _initialize_paths(self):
        """Initialize project paths"""
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / 'data'
        self.LOGS_DIR = self.BASE_DIR / 'logs'
        self.MODELS_DIR = self.BASE_DIR / 'models'
        
        # Create directories
        for directory in [self.DATA_DIR, self.LOGS_DIR, self.MODELS_DIR]:
            directory.mkdir(exist_ok=True)

    def _load_trading_config(self):
        """Load trading configuration"""
        self.TRADING_CONFIG = {
            'symbols': ['BTCUSDT'],  # Trading pairs
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'base_timeframe': '1m',
            'max_positions': 3,
            'min_position_size': Decimal('0.001'),  # BTC
            'max_position_size': Decimal('0.1'),    # BTC
            'price_precision': 2,
            'quantity_precision': 6
        }
        
        self.RISK_CONFIG = {
            'max_risk_per_trade': Decimal('0.01'),    # 1% per trade
            'max_daily_risk': Decimal('0.03'),        # 3% per day
            'max_drawdown': Decimal('0.2'),           # 20% max drawdown
            'trailing_stop': Decimal('0.01'),         # 1% trailing stop
            'min_risk_reward': Decimal('2.5'),        # 2.5:1 minimum R:R
            'initial_capital': Decimal('10000'),      # USDT
            'risk_free_rate': Decimal('0.03')         # 3% annual risk-free rate
        }
        
        self.ML_CONFIG = {
            'model_update_interval': 3600,  # 1 hour
            'training_lookback': 30,        # 30 days
            'prediction_horizon': 12,        # 12 periods ahead
            'min_accuracy': 0.6,            # 60% minimum accuracy
            'ensemble_models': ['lstm', 'xgboost', 'lightgbm']
        }

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate trading pairs
        if not self.TRADING_CONFIG['symbols']:
            raise ValueError('No trading symbols configured')
            
        # Validate risk parameters
        if self.RISK_CONFIG['max_risk_per_trade'] > Decimal('0.02'):
            raise ValueError('Maximum risk per trade cannot exceed 2%')
            
        if self.RISK_CONFIG['max_daily_risk'] > Decimal('0.05'):
            raise ValueError('Maximum daily risk cannot exceed 5%')
            
        # Validate position sizes
        if self.TRADING_CONFIG['max_position_size'] <= self.TRADING_CONFIG['min_position_size']:
            raise ValueError('Invalid position size configuration')

    def save_to_file(self, filename: str = 'config_backup.json'):
        """Save current configuration to file"""
        config = {
            'trading_config': self.TRADING_CONFIG,
            'risk_config': self.RISK_CONFIG,
            'ml_config': self.ML_CONFIG
        }
        
        with open(self.DATA_DIR / filename, 'w') as f:
            json.dump(config, f, indent=4, default=str)

    @classmethod
    def load_from_file(cls, filename: str) -> 'ConfigManager':
        """Load configuration from file"""
        instance = cls()
        
        with open(Path(filename), 'r') as f:
            config = json.load(f)
            
        instance.TRADING_CONFIG.update(config['trading_config'])
        instance.RISK_CONFIG.update(config['risk_config'])
        instance.ML_CONFIG.update(config['ml_config'])
        
        return instance

    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get configuration for specific symbol"""
        return {
            'price_precision': self.TRADING_CONFIG['price_precision'],
            'quantity_precision': self.TRADING_CONFIG['quantity_precision'],
            'min_position_size': float(self.TRADING_CONFIG['min_position_size']),
            'max_position_size': float(self.TRADING_CONFIG['max_position_size'])
        }