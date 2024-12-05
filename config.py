import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        load_dotenv('BTC.env')
        
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
        self.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.BINANCE_API_KEY, self.BINANCE_API_SECRET, self.TELEGRAM_BOT_TOKEN]):
            raise ValueError('Missing required API credentials')