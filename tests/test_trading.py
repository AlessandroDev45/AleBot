import unittest
from datetime import datetime, timedelta
from trading.strategy import TradingStrategy
from trading.executor import TradeExecutor
from config import ConfigManager

class TestTrading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigManager()
        cls.strategy = TradingStrategy(cls.config)
        cls.executor = TradeExecutor(cls.config)

    async def test_signal_generation(self):
        """Test trading signal generation"""
        try:
            signals = await self.strategy.generate_signals('BTCUSDT')
            self.assertIsNotNone(signals)
            
            if signals:  # If we have signals
                signal = signals[0]
                self.assertTrue(hasattr(signal, 'symbol'))
                self.assertTrue(hasattr(signal, 'signal_type'))
                self.assertTrue(hasattr(signal, 'price'))
                self.assertTrue(hasattr(signal, 'confidence'))
        except Exception as e:
            self.fail(f'Signal generation failed: {e}')

    async def test_trade_execution(self):
        """Test trade execution"""
        try:
            # Create a test signal
            signals = await self.strategy.generate_signals('BTCUSDT')
            if signals:
                execution = await self.executor.execute_signal(signals[0])
                self.assertIsNotNone(execution)
                self.assertTrue(hasattr(execution, 'order_id'))
                self.assertTrue(hasattr(execution, 'status'))
        except Exception as e:
            self.fail(f'Trade execution failed: {e}')

    async def test_position_management(self):
        """Test position management"""
        try:
            await self.executor.update_positions()
            positions = self.executor.position_cache
            self.assertIsNotNone(positions)
        except Exception as e:
            self.fail(f'Position management failed: {e}')