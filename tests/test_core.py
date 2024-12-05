import unittest
import asyncio
from datetime import datetime, timedelta
from config import ConfigManager
from exchange import BinanceExchange
from analysis import TechnicalAnalysis
from ml_model import MLModel
from risk_management import RiskManager

class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigManager()
        cls.exchange = BinanceExchange(cls.config)
        cls.analysis = TechnicalAnalysis(cls.config, cls.exchange)
        cls.ml_model = MLModel(cls.config, cls.analysis)
        cls.risk_manager = RiskManager(cls.config, cls.exchange)

    async def test_exchange_connection(self):
        """Test exchange connectivity"""
        try:
            await self.exchange.connect()
            status = await self.exchange.test_connection()
            self.assertTrue(status, 'Exchange connection failed')
        except Exception as e:
            self.fail(f'Exchange connection raised {e}')

    async def test_market_data(self):
        """Test market data retrieval"""
        try:
            data = await self.exchange.get_market_data('BTCUSDT', '1h')
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)
            self.assertTrue('close' in data.columns)
        except Exception as e:
            self.fail(f'Market data retrieval failed: {e}')

    async def test_technical_analysis(self):
        """Test technical analysis calculations"""
        try:
            data = await self.exchange.get_market_data('BTCUSDT', '1h')
            indicators = await self.analysis.calculate_indicators(data)
            
            self.assertIsNotNone(indicators)
            self.assertTrue('rsi' in indicators)
            self.assertTrue('macd' in indicators)
            self.assertTrue('bb_upper' in indicators)
        except Exception as e:
            self.fail(f'Technical analysis failed: {e}')

    async def test_ml_model(self):
        """Test ML model predictions"""
        try:
            prediction = await self.ml_model.predict('BTCUSDT')
            self.assertIsNotNone(prediction)
            self.assertTrue('price_prediction' in prediction)
            self.assertTrue('confidence' in prediction)
        except Exception as e:
            self.fail(f'ML model prediction failed: {e}')

    async def test_risk_management(self):
        """Test risk management calculations"""
        try:
            metrics = await self.risk_manager.calculate_risk_metrics('BTCUSDT')
            self.assertIsNotNone(metrics)
            self.assertTrue('var_95' in metrics)
            self.assertTrue('sharpe_ratio' in metrics)
        except Exception as e:
            self.fail(f'Risk management failed: {e}')

    @classmethod
    async def asyncTearDown(cls):
        await cls.exchange.close()