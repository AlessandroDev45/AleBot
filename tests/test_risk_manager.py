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
from TradingSystem.Core.RiskManager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager"""

    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.data_manager = DataManager()
        cls.risk_manager = RiskManager(data_manager=cls.data_manager)
        cls.symbol = "BTCUSDT"
        
        # Get account info
        account_info = cls.data_manager.get_account_info()
        cls.balance = float(account_info.get('totalWalletBalance', 0))
        
        if cls.balance <= 0:
            raise unittest.SkipTest("Insufficient balance to run tests")
            
        current_price = float(cls.data_manager.get_market_data(
            symbol=cls.symbol,
            timeframe="1m",
            limit=1
        )['close'].iloc[-1])
        
        cls.test_amount = 0.001  # Minimum test amount
        cls.test_price = current_price
        
        logging.info(f"Account balance: {cls.balance} USDT")
        logging.info(f"Current BTC price: {current_price} USDT")
        logging.info(f"Test amount: {cls.test_amount} BTC")

    def setUp(self):
        """Setup before each test"""
        if self.balance <= 0:
            self.skipTest("Insufficient balance to run test")

    def test_validate_order(self):
        """Test order validation"""
        try:
            # Test valid order
            result = self.risk_manager.validate_order(
                order_type="LIMIT",
                quantity=self.test_amount,
                price=self.test_price
            )
            self.assertTrue(result, "Valid order should be approved")
            
            # Test order exceeding position size limit
            large_amount = self.balance / self.test_price * 2  # 200% of balance
            result = self.risk_manager.validate_order(
                order_type="LIMIT",
                quantity=large_amount,
                price=self.test_price
            )
            self.assertFalse(result, "Order exceeding position size limit should be rejected")
            
            logging.info("Order validation tests passed")
            
        except Exception as e:
            self.fail(f"Order validation test failed: {str(e)}")

    def test_daily_loss_limit(self):
        """Test daily loss limit"""
        try:
            # Record a loss trade
            loss_amount = self.balance * self.risk_manager.max_daily_loss * 0.5  # 50% of daily loss limit
            self.risk_manager.record_trade(
                quantity=self.test_amount,
                price=self.test_price,
                side="BUY"
            )
            
            # Try to validate another order
            result = self.risk_manager.validate_order(
                order_type="LIMIT",
                quantity=self.test_amount,
                price=self.test_price
            )
            self.assertTrue(result, "Order within daily loss limit should be approved")
            
            # Record another loss exceeding limit
            self.risk_manager.record_trade(
                quantity=self.test_amount,
                price=self.test_price * 0.5,  # 50% loss
                side="SELL"
            )
            
            # Try to validate order after exceeding limit
            result = self.risk_manager.validate_order(
                order_type="LIMIT",
                quantity=self.test_amount,
                price=self.test_price
            )
            self.assertFalse(result, "Order exceeding daily loss limit should be rejected")
            
            logging.info("Daily loss limit tests passed")
            
        except Exception as e:
            self.fail(f"Daily loss limit test failed: {str(e)}")

    def test_drawdown_limit(self):
        """Test drawdown limit"""
        try:
            # Get initial metrics
            initial_metrics = self.risk_manager.get_risk_metrics()
            
            # Record a drawdown
            drawdown_amount = self.balance * self.risk_manager.max_drawdown * 0.5  # 50% of max drawdown
            self.risk_manager.record_trade(
                quantity=self.test_amount,
                price=self.test_price,
                side="BUY"
            )
            self.risk_manager.record_trade(
                quantity=self.test_amount,
                price=self.test_price * 0.5,  # 50% loss
                side="SELL"
            )
            
            # Get updated metrics
            current_metrics = self.risk_manager.get_risk_metrics()
            
            # Verify drawdown calculation
            self.assertGreater(
                current_metrics['current_drawdown'],
                initial_metrics['current_drawdown'],
                "Drawdown should increase after loss"
            )
            
            logging.info("Drawdown limit tests passed")
            
        except Exception as e:
            self.fail(f"Drawdown limit test failed: {str(e)}")

    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        try:
            # Get initial metrics
            metrics = self.risk_manager.get_risk_metrics()
            
            # Verify metrics structure
            self.assertIn('daily_loss', metrics)
            self.assertIn('current_drawdown', metrics)
            self.assertIn('peak_balance', metrics)
            self.assertIn('current_balance', metrics)
            
            # Verify metrics values
            self.assertGreaterEqual(metrics['peak_balance'], metrics['current_balance'])
            self.assertGreaterEqual(metrics['current_drawdown'], 0)
            
            logging.info("Risk metrics tests passed")
            
        except Exception as e:
            self.fail(f"Risk metrics test failed: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main() 