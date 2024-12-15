import unittest
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import patch, Mock
import time

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from TradingSystem.Core.DataManager.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    """Test cases for DataManager class"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment before all tests"""
        cls.env_file = os.path.join(root_dir, 'TradingSystem', 'Config', 'BTC.env')
        cls.data_manager = DataManager(env_file=cls.env_file)
        cls.symbol = "BTCUSDT"
        cls.timeframe = "1m"

    def setUp(self):
        """Set up before each test"""
        self.data_manager.clear_cache()
        time.sleep(1)  # Prevent rate limiting

    def tearDown(self):
        """Clean up after each test"""
        self.data_manager.clear_cache()

    def test_btc_default_symbol(self):
        """Test that BTCUSDT is the default and primary symbol"""
        active_symbols = self.data_manager.get_active_symbols()
        self.assertEqual(len(active_symbols), 1, "Should only have one default symbol")
        self.assertEqual(active_symbols[0], "BTCUSDT", "Default symbol should be BTCUSDT")

    def test_cannot_remove_btc(self):
        """Test that BTCUSDT cannot be removed"""
        result = self.data_manager.remove_symbol("BTCUSDT")
        self.assertFalse(result, "Should not be able to remove BTCUSDT")
        active_symbols = self.data_manager.get_active_symbols()
        self.assertIn("BTCUSDT", active_symbols, "BTCUSDT should still be in active symbols")

    def test_add_and_remove_symbol(self):
        """Test adding and removing a secondary symbol"""
        # Add a new symbol
        result = self.data_manager.add_symbol("ETHUSDT")
        self.assertTrue(result, "Should successfully add ETHUSDT")
        active_symbols = self.data_manager.get_active_symbols()
        self.assertIn("ETHUSDT", active_symbols, "ETHUSDT should be in active symbols")
        
        # Remove the symbol
        result = self.data_manager.remove_symbol("ETHUSDT")
        self.assertTrue(result, "Should successfully remove ETHUSDT")
        active_symbols = self.data_manager.get_active_symbols()
        self.assertNotIn("ETHUSDT", active_symbols, "ETHUSDT should not be in active symbols")

    def test_market_data_cache(self):
        """Test market data caching mechanism"""
        # First request - should fetch from API
        data1 = self.data_manager.get_market_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=100
        )
        self.assertFalse(data1.empty, "Should receive market data")
        
        # Wait for a moment to ensure cache is set
        time.sleep(1)
        
        # Second request - should use cache
        data2 = self.data_manager.get_market_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=100
        )
        
        # Compare the data frames
        pd.testing.assert_frame_equal(data1, data2, check_exact=True)

    def test_order_book_cache(self):
        """Test order book caching mechanism"""
        # First request - should fetch from API
        book1 = self.data_manager.get_order_book(symbol=self.symbol)
        self.assertFalse(book1.empty, "Should receive order book data")
        
        # Wait for a moment to ensure cache is set
        time.sleep(1)
        
        # Second request - should use cache
        book2 = self.data_manager.get_order_book(symbol=self.symbol)
        
        # Compare the data frames
        pd.testing.assert_frame_equal(book1, book2, check_exact=True)

    def test_data_validation(self):
        """Test market data validation"""
        data = self.data_manager.get_market_data(symbol=self.symbol, timeframe=self.timeframe, limit=100)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in required_columns:
            self.assertIn(column, data.columns, f"Missing required column: {column}")
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(data[column]), 
                          f"Column {column} should be numeric")
        
        # Check price consistency
        self.assertTrue(all(data['high'] >= data['low']), 
                       "High prices should be greater than or equal to low prices")
        self.assertTrue(all(data['volume'] >= 0), 
                       "Volume should be non-negative")

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test invalid symbol
        data = self.data_manager.get_market_data(symbol="INVALID", timeframe=self.timeframe)
        self.assertTrue(data.empty, "Should return empty DataFrame for invalid symbol")
        
        # Test invalid timeframe
        data = self.data_manager.get_market_data(symbol=self.symbol, timeframe="INVALID")
        self.assertTrue(data.empty, "Should return empty DataFrame for invalid timeframe")

    def test_stop_and_cleanup(self):
        """Test stopping the DataManager and cleanup"""
        # Ensure threads are running
        self.assertTrue(hasattr(self.data_manager, 'running'), 
                       "DataManager should have 'running' attribute")
        self.assertTrue(self.data_manager.running, 
                       "DataManager should be running before stop")
        
        # Stop the manager
        self.data_manager.stop()
        time.sleep(2)  # Give time for threads to stop
        
        # Verify stopped state
        self.assertFalse(self.data_manager.running,
                        "DataManager should not be running after stop")
        self.assertEqual(len(self.data_manager.cache), 0,
                        "Cache should be empty after cleanup")

    def test_account_data(self):
        """Test account data retrieval"""
        account_data = self.data_manager.get_account_data()
        
        # Check required fields
        required_fields = ['total_balance', 'available_balance', 'position_value']
        for field in required_fields:
            self.assertIn(field, account_data, f"Missing required field: {field}")
        
        # Check data types
        for field in required_fields:
            self.assertIsInstance(account_data[field], (int, float), 
                                f"Field {field} should be numeric")
        
        # Check value consistency
        self.assertGreaterEqual(account_data['total_balance'], 0, 
                              "Total balance should be non-negative")
        self.assertGreaterEqual(account_data['available_balance'], 0, 
                              "Available balance should be non-negative")
        self.assertGreaterEqual(account_data['position_value'], 0, 
                              "Position value should be non-negative")

    def test_api_status(self):
        """Test API status retrieval"""
        status = self.data_manager.get_api_status()
        
        # Check required fields
        required_fields = ['status', 'server_time', 'last_update']
        for field in required_fields:
            self.assertIn(field, status, f"Missing required field: {field}")
        
        # Check status value
        self.assertIn(status['status'], ['connected', 'maintenance', 'error', 'disconnected'],
                     "Invalid status value")
        
        # Check timestamp formats
        try:
            datetime.strptime(status['server_time'], '%Y-%m-%d %H:%M:%S')
            datetime.strptime(status['last_update'], '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            self.fail(f"Invalid timestamp format: {e}")

    def test_daily_trading_stats(self):
        """Test daily trading statistics"""
        stats = self.data_manager.get_daily_trading_stats()
        
        # Check required fields
        required_fields = ['total_trades', 'profitable_trades', 'win_rate', 'total_pnl', 'last_update']
        for field in required_fields:
            self.assertIn(field, stats, f"Missing required field: {field}")
        
        # Check data types and value ranges
        self.assertIsInstance(stats['total_trades'], int, "Total trades should be integer")
        self.assertIsInstance(stats['profitable_trades'], int, "Profitable trades should be integer")
        self.assertIsInstance(stats['win_rate'], (int, float), "Win rate should be numeric")
        self.assertIsInstance(stats['total_pnl'], (int, float), "Total PnL should be numeric")
        
        self.assertGreaterEqual(stats['total_trades'], 0, "Total trades should be non-negative")
        self.assertGreaterEqual(stats['profitable_trades'], 0, "Profitable trades should be non-negative")
        self.assertGreaterEqual(stats['win_rate'], 0, "Win rate should be non-negative")
        self.assertLessEqual(stats['win_rate'], 100, "Win rate should not exceed 100%")

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main()