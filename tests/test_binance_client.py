import unittest
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from TradingSystem.Core.DataManager.data_manager import DataManager

class TestBinanceClient(unittest.TestCase):
    """Test cases for Binance client access through DataManager"""

    @classmethod
    def setUpClass(cls):
        """Initialize test environment"""
        cls.data_manager = DataManager()
        cls.symbol = "BTCUSDT"
        cls.timeframe = "1m"

    def setUp(self):
        """Setup before each test"""
        self.data_manager.clear_cache()

    def tearDown(self):
        """Cleanup after each test"""
        self.data_manager.clear_cache()

    def test_market_data_retrieval(self):
        """Test market data retrieval through DataManager"""
        try:
            # Get current market data
            df = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            self.assertFalse(df.empty, "Market data should not be empty")
            self.assertTrue(all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']))
            logging.info(f"Retrieved {len(df)} {self.timeframe} candles")
            
            # Validate data
            self.assertTrue(df['high'].gt(df['low']).all(), "High should be greater than low")
            self.assertTrue(df['volume'].ge(0).all(), "Volume should be non-negative")
            
        except Exception as e:
            self.fail(f"Market data retrieval failed: {str(e)}")

    def test_order_book_data(self):
        """Test order book data retrieval through DataManager"""
        try:
            order_book = self.data_manager.get_order_book(symbol=self.symbol)
            
            self.assertFalse(order_book.empty, "Order book should not be empty")
            self.assertTrue(len(order_book) > 0, "Order book should have entries")
            
            logging.info(f"Order book entries retrieved successfully")
            
        except Exception as e:
            self.fail(f"Order book retrieval failed: {str(e)}")

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test invalid symbol
        df = self.data_manager.get_market_data(
            symbol="INVALID",
            timeframe=self.timeframe,
            limit=100
        )
        self.assertTrue(df.empty, "Should return empty DataFrame for invalid symbol")
        
        # Test invalid timeframe
        df = self.data_manager.get_market_data(
            symbol=self.symbol,
            timeframe="INVALID",
            limit=100
        )
        self.assertTrue(df.empty, "Should return empty DataFrame for invalid timeframe")

    def test_data_consistency(self):
        """Test data consistency across different requests"""
        try:
            # Get data with different limits
            data1 = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=50
            )
            
            data2 = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            # Verify that data2 contains data1's most recent data
            common_timestamps = data1.index.intersection(data2.index)
            data1_common = data1.loc[common_timestamps]
            data2_common = data2.loc[common_timestamps]
            
            self.assertTrue(
                data1_common.equals(data2_common),
                "Data should be consistent across different requests"
            )
            
        except Exception as e:
            self.fail(f"Data consistency test failed: {str(e)}")

    def test_get_server_time(self):
        """Test server time retrieval"""
        try:
            server_time = self.data_manager.binance_client.get_server_time()
            
            self.assertIsInstance(server_time, dict, "Server time should be a dictionary")
            self.assertIn('serverTime', server_time, "Missing serverTime field")
            self.assertIsInstance(server_time['serverTime'], int, "Server time should be an integer")
            
            # Check if server time is reasonable (within a day of current time)
            current_time = int(time.time() * 1000)
            time_diff = abs(server_time['serverTime'] - current_time)
            self.assertLess(time_diff, 24 * 60 * 60 * 1000, "Server time should be within 24 hours of current time")
            
        except Exception as e:
            self.fail(f"Server time retrieval failed: {str(e)}")

    def test_get_system_status(self):
        """Test system status retrieval"""
        try:
            status = self.data_manager.binance_client.get_system_status()
            
            self.assertIsInstance(status, dict, "System status should be a dictionary")
            self.assertIn('status', status, "Missing status field")
            self.assertIn('msg', status, "Missing msg field")
            
            # Status should be 0 (normal) or 1 (maintenance)
            self.assertIn(status['status'], [0, 1], "Invalid status value")
            
        except Exception as e:
            self.fail(f"System status retrieval failed: {str(e)}")

    def test_get_symbol_ticker(self):
        """Test symbol ticker retrieval"""
        try:
            ticker = self.data_manager.binance_client.get_symbol_ticker(symbol=self.symbol)
            
            self.assertIsInstance(ticker, dict, "Symbol ticker should be a dictionary")
            self.assertIn('symbol', ticker, "Missing symbol field")
            self.assertIn('price', ticker, "Missing price field")
            
            # Check symbol and price
            self.assertEqual(ticker['symbol'], self.symbol, "Symbol mismatch")
            self.assertIsInstance(float(ticker['price']), float, "Price should be convertible to float")
            self.assertGreater(float(ticker['price']), 0, "Price should be positive")
            
        except Exception as e:
            self.fail(f"Symbol ticker retrieval failed: {str(e)}")

    def test_get_account_trades(self):
        """Test account trades retrieval"""
        try:
            trades = self.data_manager.binance_client.get_account_trades(symbol=self.symbol)
            
            self.assertIsInstance(trades, list, "Trades should be a list")
            
            # If there are trades, check their structure
            if trades:
                trade = trades[0]
                required_fields = ['symbol', 'id', 'price', 'qty', 'time']
                for field in required_fields:
                    self.assertIn(field, trade, f"Missing required field: {field}")
                
                # Check data types
                self.assertEqual(trade['symbol'], self.symbol, "Symbol mismatch")
                self.assertIsInstance(float(trade['price']), float, "Price should be convertible to float")
                self.assertIsInstance(float(trade['qty']), float, "Quantity should be convertible to float")
                self.assertIsInstance(trade['time'], int, "Time should be integer")
            
        except Exception as e:
            self.fail(f"Account trades retrieval failed: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main() 