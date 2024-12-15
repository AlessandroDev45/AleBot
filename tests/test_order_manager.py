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
from TradingSystem.Core.OrderManager import OrderManager
from TradingSystem.Core.RiskManager import RiskManager

class TestOrderManager(unittest.TestCase):
    """Test cases for OrderManager using DataManager for API access"""

    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.data_manager = DataManager()
        cls.risk_manager = RiskManager(data_manager=cls.data_manager)
        cls.order_manager = OrderManager(data_manager=cls.data_manager, risk_manager=cls.risk_manager)
        cls.symbol = "BTCUSDT"
        
        # Get account info to calculate test amount
        account_info = cls.data_manager.get_account_info()
        cls.balance = float(account_info.get('totalWalletBalance', 0))
        
        if cls.balance <= 0:
            raise unittest.SkipTest("Insufficient balance to run tests")
            
        current_price = float(cls.data_manager.get_market_data(
            symbol=cls.symbol,
            timeframe="1m",
            limit=1
        )['close'].iloc[-1])
        
        # Calculate test amount based on account balance and position size limit
        max_position_value = cls.balance * cls.risk_manager.max_position_size
        cls.test_amount = max_position_value / current_price
        
        logging.info(f"Account balance: {cls.balance} USDT")
        logging.info(f"Using test amount of {cls.test_amount} BTC")
        logging.info(f"Current BTC price: {current_price} USDT")

    def setUp(self):
        """Setup before each test"""
        if self.balance <= 0:
            self.skipTest("Insufficient balance to run test")
            
        self.data_manager.clear_cache()
        # Cancela todas as ordens pendentes antes de cada teste
        self.order_manager.cancel_all_orders(self.symbol)

    def tearDown(self):
        """Cleanup after each test"""
        self.data_manager.clear_cache()
        # Cancela todas as ordens pendentes após cada teste
        self.order_manager.cancel_all_orders(self.symbol)

    def test_market_order(self):
        """Test market order creation through DataManager"""
        try:
            # Obter preço atual
            current_price = float(self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe="1m",
                limit=1
            )['close'].iloc[-1])

            # Criar ordem de mercado de compra
            order = self.order_manager.create_order(
                symbol=self.symbol,
                side="BUY",
                order_type="MARKET",
                quantity=self.test_amount
            )

            self.assertIsNotNone(order, "Order should be created")
            self.assertEqual(order['symbol'], self.symbol)
            self.assertEqual(order['side'], "BUY")
            self.assertEqual(order.get('type'), "MARKET")
            self.assertEqual(float(order.get('origQty', 0)), self.test_amount)

            logging.info(f"Market order created successfully at {current_price}")

        except Exception as e:
            self.fail(f"Market order test failed: {str(e)}")

    def test_limit_order(self):
        """Test limit order creation through DataManager"""
        try:
            # Obter preço atual
            current_price = float(self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe="1m",
                limit=1
            )['close'].iloc[-1])

            # Criar ordem limite abaixo do preço atual
            limit_price = current_price * 0.95  # 5% abaixo do preço atual
            
            order = self.order_manager.create_order(
                symbol=self.symbol,
                side="BUY",
                order_type="LIMIT",
                quantity=self.test_amount,
                price=limit_price
            )

            self.assertIsNotNone(order, "Order should be created")
            self.assertEqual(order['symbol'], self.symbol)
            self.assertEqual(order['side'], "BUY")
            self.assertEqual(order.get('type'), "LIMIT")
            self.assertEqual(float(order.get('price', 0)), limit_price)

            # Cancelar a ordem após o teste
            if order.get('orderId'):
                cancel_result = self.order_manager.cancel_order(self.symbol, order['orderId'])
                self.assertEqual(cancel_result.get('status'), 'CANCELED')
            
            logging.info(f"Limit order created successfully at {limit_price}")

        except Exception as e:
            self.fail(f"Limit order test failed: {str(e)}")

    def test_order_cancellation(self):
        """Test order cancellation through DataManager"""
        try:
            # Obter preço atual
            current_price = float(self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe="1m",
                limit=1
            )['close'].iloc[-1])

            # Criar ordem limite para cancelar
            limit_price = current_price * 0.95
            order = self.order_manager.create_order(
                symbol=self.symbol,
                side="BUY",
                order_type="LIMIT",
                quantity=self.test_amount,
                price=limit_price
            )

            self.assertIsNotNone(order, "Order should be created")
            self.assertIsNotNone(order.get('orderId'), "Order ID should be present")

            # Cancelar a ordem
            if order.get('orderId'):
                result = self.order_manager.cancel_order(self.symbol, order['orderId'])
                self.assertEqual(result.get('status'), 'CANCELED')
                logging.info("Order cancelled successfully")
            else:
                self.fail("Order ID not present in created order")

        except Exception as e:
            self.fail(f"Order cancellation test failed: {str(e)}")

    def test_order_status(self):
        """Test order status retrieval through DataManager"""
        try:
            # Obter preço atual
            current_price = float(self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe="1m",
                limit=1
            )['close'].iloc[-1])

            # Criar ordem limite
            limit_price = current_price * 0.95
            order = self.order_manager.create_order(
                symbol=self.symbol,
                side="BUY",
                order_type="LIMIT",
                quantity=self.test_amount,
                price=limit_price
            )

            self.assertIsNotNone(order, "Order should be created")
            self.assertIsNotNone(order.get('orderId'), "Order ID should be present")

            if order.get('orderId'):
                # Verificar status
                open_orders = self.order_manager.get_open_orders(self.symbol)
                self.assertTrue(len(open_orders) > 0, "Should have at least one open order")
                
                found_order = False
                for open_order in open_orders:
                    if open_order['orderId'] == order['orderId']:
                        found_order = True
                        break
                
                self.assertTrue(found_order, "Created order should be in open orders")

                # Cancelar a ordem após o teste
                cancel_result = self.order_manager.cancel_order(self.symbol, order['orderId'])
                self.assertEqual(cancel_result.get('status'), 'CANCELED')
                logging.info(f"Order status retrieved successfully")
            else:
                self.fail("Order ID not present in created order")

        except Exception as e:
            self.fail(f"Order status test failed: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main() 