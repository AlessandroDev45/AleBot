import unittest
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
import datetime

class TestBinanceRealConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load environment variables and setup client"""
        load_dotenv('TradingSystem/Config/BTC.env')
        cls.api_key = os.getenv('BINANCE_API_KEY')
        cls.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not cls.api_key or not cls.api_secret:
            raise ValueError("API credentials not found in environment variables")
        
        cls.client = Client(cls.api_key, cls.api_secret)

    def test_real_connection(self):
        """Test real connection to Binance API"""
        try:
            status = self.client.get_system_status()
            self.assertIn('status', status)
            print(f"Connection successful! System status: {status}")
        except BinanceAPIException as e:
            self.fail(f"Connection failed: {str(e)}")

    def test_server_time(self):
        """Test server time endpoint"""
        try:
            server_time = self.client.get_server_time()
            self.assertIn('serverTime', server_time)
            self.assertIsInstance(server_time['serverTime'], int)
            print(f"Server time: {datetime.datetime.fromtimestamp(server_time['serverTime']/1000)}")
        except BinanceAPIException as e:
            self.fail(f"Server time request failed: {str(e)}")

    def test_account_info(self):
        """Test account information endpoint"""
        try:
            account = self.client.get_account()
            self.assertIn('balances', account)
            self.assertIsInstance(account['balances'], list)
            print(f"Account type: {account.get('accountType', 'Unknown')}")
            
            # Print non-zero balances
            balances = [
                b for b in account['balances']
                if float(b['free']) > 0 or float(b['locked']) > 0
            ]
            print(f"Non-zero balances: {balances}")
        except BinanceAPIException as e:
            self.fail(f"Account info request failed: {str(e)}")

    def test_market_data(self):
        """Test market data endpoint"""
        try:
            klines = self.client.get_klines(
                symbol="BTCUSDT",
                interval=self.client.KLINE_INTERVAL_1MINUTE,
                limit=5
            )
            self.assertTrue(len(klines) > 0)
            print(f"Recent BTC/USDT price: {float(klines[-1][4])}")  # Latest close price
        except BinanceAPIException as e:
            self.fail(f"Market data request failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()