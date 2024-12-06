import unittest
from datetime import datetime, timedelta
from performance.tracker import PerformanceTracker
from config import ConfigManager

class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigManager()
        cls.tracker = PerformanceTracker(cls.config)

    async def test_metrics_calculation(self):
        """Test performance metrics calculation"""
        try:
            await self.tracker.update_metrics()
            metrics = self.tracker.get_metrics()
            
            self.assertIsNotNone(metrics)
            self.assertTrue('win_rate' in metrics)
            self.assertTrue('profit_factor' in metrics)
            self.assertTrue('sharpe_ratio' in metrics)
        except Exception as e:
            self.fail(f'Metrics calculation failed: {e}')

    async def test_equity_curve(self):
        """Test equity curve calculation"""
        try:
            equity_curve = self.tracker.get_equity_curve()
            self.assertIsNotNone(equity_curve)
            self.assertTrue(len(equity_curve) > 0)
        except Exception as e:
            self.fail(f'Equity curve calculation failed: {e}')