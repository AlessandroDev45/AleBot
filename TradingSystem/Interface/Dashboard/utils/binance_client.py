import os
import sys
import pandas as pd

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from Core.DataManager.data_manager import DataManager

class BinanceConnection:
    def __init__(self):
        try:
            # Initialize DataManager
            self.data_manager = DataManager()
            self.trading_pair = os.getenv('TRADING_PAIR', 'BTCUSDT')
            self.interval = os.getenv('INTERVAL', '1h')
            
        except Exception as e:
            print(f"Error initializing Binance connection: {e}")
            raise

    def get_historical_data(self, limit=1000):
        try:
            # Use DataManager to get historical data
            df = self.data_manager.get_market_data(
                symbol=self.trading_pair,
                timeframe=self.interval,
                limit=limit
            )
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def get_account_balance(self):
        try:
            # Use DataManager to get account balance
            account_info = self.data_manager.get_account_info()
            if account_info and 'balances' in account_info:
                balances = {asset['asset']: float(asset['free']) 
                          for asset in account_info['balances'] 
                          if float(asset['free']) > 0}
                return balances
            return None
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return None 