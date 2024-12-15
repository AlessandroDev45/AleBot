import unittest
import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from TradingSystem.Core.DataManager.data_manager import DataManager

class TestBinanceAPI(unittest.TestCase):
    """Test cases for Binance API access through DataManager"""

    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.data_manager = DataManager()
        cls.symbol = "BTCUSDT"
        cls.timeframe = "1m"

    def setUp(self):
        """Setup before each test"""
        self.data_manager.clear_cache()
        
    def tearDown(self):
        """Cleanup after each test"""
        self.data_manager.clear_cache()

    def test_market_data_access(self):
        """Test market data access through DataManager"""
        try:
            # Get market data
            df = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            # Verify data structure
            self.assertFalse(df.empty, "Market data should not be empty")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, df.columns, f"Required column {col} missing")
            
            # Verify data types
            self.assertTrue(df['volume'].dtype.kind in 'if', "Volume should be numeric")
            self.assertTrue(df['close'].dtype.kind in 'if', "Close price should be numeric")
            
            logging.info(f"Market data retrieved successfully: {len(df)} candles")
            
        except Exception as e:
            self.fail(f"Market data access failed: {str(e)}")

    def test_order_book_access(self):
        """Test order book access through DataManager"""
        try:
            # Get order book
            order_book = self.data_manager.get_order_book(symbol=self.symbol)
            
            # Verify order book structure
            self.assertFalse(order_book.empty, "Order book should not be empty")
            self.assertTrue(len(order_book) > 0, "Order book should contain entries")
            
            logging.info(f"Order book retrieved successfully")
            
        except Exception as e:
            self.fail(f"Order book access failed: {str(e)}")

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

    def test_data_caching(self):
        """Test data caching mechanism"""
        # First request
        data1 = self.data_manager.get_market_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=100
        )
        
        # Second request (should use cache)
        data2 = self.data_manager.get_market_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=100
        )
        
        self.assertTrue(data1.equals(data2), "Cached data should match original data")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main() 