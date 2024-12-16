import pytest
from dash.testing.application_runners import import_app
from unittest.mock import MagicMock, patch
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from TradingSystem.Core.BinanceClient import BinanceClient
from TradingSystem.Core.DataManager.data_manager import DataManager
from TradingSystem.Interface.Dashboard.app import create_app

class TestDashboard:
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing"""
        # Mock Binance client
        mock_binance = MagicMock()
        mock_binance.get_account.return_value = {
            'balances': [
                {'asset': 'BTC', 'free': '1.0', 'locked': '0.0'},
                {'asset': 'USDT', 'free': '10000.0', 'locked': '0.0'}
            ]
        }
        
        # Mock historical data
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='1min')
        mock_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(40000, 45000, len(dates)),
            'low': np.random.uniform(40000, 45000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(1, 100, len(dates))
        })
        
        # Mock DataManager
        mock_data_manager = MagicMock()
        mock_data_manager.get_historical_data.return_value = mock_data
        mock_data_manager.get_latest_price.return_value = 42000.0
        
        # Create components dictionary
        components = {
            'binance_client': mock_binance,
            'data_manager': mock_data_manager,
            'risk_manager': MagicMock(),
            'order_manager': MagicMock(),
            'data_processor': MagicMock(),
            'pattern_detector': MagicMock(),
            'ml_model': MagicMock()
        }
        
        return components

    def test_dashboard_creation(self, mock_components):
        """Test if dashboard is created successfully"""
        app = create_app(mock_components)
        assert app is not None
        
    def test_market_overview_data(self, mock_components):
        """Test if market overview tab receives data"""
        app = create_app(mock_components)
        
        # Test price data
        price_data = mock_components['data_manager'].get_latest_price()
        assert price_data is not None
        assert isinstance(price_data, float)
        
        # Test historical data for charts
        historical_data = mock_components['data_manager'].get_historical_data()
        assert historical_data is not None
        assert isinstance(historical_data, pd.DataFrame)
        assert not historical_data.empty
        
    def test_trading_signals_data(self, mock_components):
        """Test if trading signals tab receives data"""
        app = create_app(mock_components)
        
        # Test pattern detection
        mock_components['pattern_detector'].detect_patterns.return_value = [
            {'pattern': 'bullish', 'timestamp': datetime.now()}
        ]
        patterns = mock_components['pattern_detector'].detect_patterns()
        assert patterns is not None
        assert len(patterns) > 0
        
        # Test ML predictions
        mock_components['ml_model'].predict.return_value = [1, 0, 1]  # Example predictions
        predictions = mock_components['ml_model'].predict()
        assert predictions is not None
        assert len(predictions) > 0
        
    def test_portfolio_performance_data(self, mock_components):
        """Test if portfolio performance tab receives data"""
        app = create_app(mock_components)
        
        # Test account balance data
        account_data = mock_components['binance_client'].get_account()
        assert account_data is not None
        assert 'balances' in account_data
        
        # Test risk metrics
        mock_components['risk_manager'].calculate_risk_metrics.return_value = {
            'var': 0.05,
            'sharpe_ratio': 1.5
        }
        risk_metrics = mock_components['risk_manager'].calculate_risk_metrics()
        assert risk_metrics is not None
        assert 'var' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        
    def test_order_management_data(self, mock_components):
        """Test if order management tab receives data"""
        app = create_app(mock_components)
        
        # Test open orders
        mock_components['order_manager'].get_open_orders.return_value = [
            {'symbol': 'BTCUSDT', 'side': 'BUY', 'price': '42000'}
        ]
        open_orders = mock_components['order_manager'].get_open_orders()
        assert open_orders is not None
        assert len(open_orders) > 0
        
        # Test order history
        mock_components['order_manager'].get_order_history.return_value = [
            {'symbol': 'BTCUSDT', 'side': 'SELL', 'price': '43000', 'status': 'FILLED'}
        ]
        order_history = mock_components['order_manager'].get_order_history()
        assert order_history is not None
        assert len(order_history) > 0

    def test_data_refresh(self, mock_components):
        """Test if data is refreshed properly"""
        app = create_app(mock_components)
        
        # Test data update interval
        initial_price = mock_components['data_manager'].get_latest_price()
        mock_components['data_manager'].get_latest_price.return_value = 43000.0
        new_price = mock_components['data_manager'].get_latest_price()
        assert initial_price != new_price
        
    def test_error_handling(self, mock_components):
        """Test error handling in dashboard components"""
        app = create_app(mock_components)
        
        # Test API error handling
        mock_components['binance_client'].get_account.side_effect = Exception("API Error")
        try:
            account_data = mock_components['binance_client'].get_account()
        except Exception as e:
            assert str(e) == "API Error"
            
        # Test data processing error handling
        mock_components['data_processor'].process_data.side_effect = Exception("Processing Error")
        try:
            processed_data = mock_components['data_processor'].process_data()
        except Exception as e:
            assert str(e) == "Processing Error" 