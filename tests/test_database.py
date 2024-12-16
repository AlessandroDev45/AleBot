import unittest
import os
from datetime import datetime, timedelta, UTC
from TradingSystem.Core.DataManager.database import (
    Database, Trade, Position, MarketData,
    OrderSide, OrderType, OrderStatus
)

class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test database"""
        cls.db = Database('sqlite:///test_trading_system.db')
        
        # Sample data
        cls.trade_data = {
            'symbol': 'BTCUSDT',
            'order_id': '12345',
            'side': OrderSide.BUY,
            'type': OrderType.MARKET,
            'status': OrderStatus.FILLED,
            'quantity': 1.0,
            'price': 45000.0,
            'executed_qty': 1.0,
            'executed_price': 45000.0,
            'commission': 0.1,
            'commission_asset': 'BNB'
        }
        
        cls.position_data = {
            'symbol': 'BTCUSDT',
            'side': OrderSide.BUY,
            'quantity': 1.0,
            'entry_price': 45000.0,
            'current_price': 46000.0,
            'unrealized_pnl': 1000.0,
            'stop_loss': 44000.0,
            'take_profit': 48000.0
        }
        
        cls.market_data = [{
            'symbol': 'BTCUSDT',
            'timestamp': datetime.now(UTC),
            'open': 45000.0,
            'high': 45100.0,
            'low': 44900.0,
            'close': 45050.0,
            'volume': 100.0,
            'trades': 50,
            'taker_buy_volume': 60.0,
            'taker_sell_volume': 40.0
        }]
        
    @classmethod
    def tearDownClass(cls):
        """Cleanup test database"""
        cls.db.engine.dispose()
        if os.path.exists('test_trading_system.db'):
            os.remove('test_trading_system.db')
            
    def tearDown(self):
        """Clean up after each test"""
        session = self.db.Session()
        try:
            session.query(Trade).delete()
            session.query(Position).delete()
            session.query(MarketData).delete()
            session.commit()
        finally:
            session.close()

    def test_save_trade(self):
        """Test trade saving"""
        trade = self.db.save_trade(self.trade_data)
        
        self.assertIsNotNone(trade.id)
        self.assertEqual(trade.symbol, 'BTCUSDT')
        self.assertEqual(trade.order_id, '12345')
        self.assertEqual(trade.side, OrderSide.BUY)
        self.assertEqual(trade.type, OrderType.MARKET)
        self.assertEqual(trade.status, OrderStatus.FILLED)
        self.assertEqual(trade.quantity, 1.0)
        self.assertEqual(trade.price, 45000.0)

    def test_update_trade(self):
        """Test trade updating"""
        trade = self.db.save_trade(self.trade_data)
        
        update_data = {
            'status': OrderStatus.PARTIALLY_FILLED,
            'executed_qty': 0.5
        }
        
        updated_trade = self.db.update_trade(trade.order_id, update_data)
        
        self.assertEqual(updated_trade.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(updated_trade.executed_qty, 0.5)

    def test_save_position(self):
        """Test position saving"""
        position = self.db.save_position(self.position_data)
        
        self.assertIsNotNone(position.id)
        self.assertEqual(position.symbol, 'BTCUSDT')
        self.assertEqual(position.side, OrderSide.BUY)
        self.assertEqual(position.quantity, 1.0)
        self.assertEqual(position.entry_price, 45000.0)
        self.assertEqual(position.current_price, 46000.0)
        self.assertEqual(position.unrealized_pnl, 1000.0)

    def test_update_position(self):
        """Test position updating"""
        position = self.db.save_position(self.position_data)
        
        update_data = {
            'current_price': 47000.0,
            'unrealized_pnl': 2000.0
        }
        
        updated_position = self.db.update_position(position.id, update_data)
        
        self.assertEqual(updated_position.current_price, 47000.0)
        self.assertEqual(updated_position.unrealized_pnl, 2000.0)

    def test_save_market_data(self):
        """Test market data saving"""
        data_objects = self.db.save_market_data(self.market_data)
        
        self.assertEqual(len(data_objects), 1)
        data = data_objects[0]
        self.assertEqual(data.symbol, 'BTCUSDT')
        self.assertEqual(data.open, 45000.0)
        self.assertEqual(data.high, 45100.0)
        self.assertEqual(data.low, 44900.0)
        self.assertEqual(data.close, 45050.0)

    def test_get_trades(self):
        """Test trade retrieval"""
        # Save multiple trades
        trade1 = self.trade_data.copy()
        trade2 = self.trade_data.copy()
        trade2['order_id'] = '12346'
        trade2['symbol'] = 'ETHUSDT'
        
        self.db.save_trade(trade1)
        self.db.save_trade(trade2)
        
        # Test filtering
        btc_trades = self.db.get_trades(symbol='BTCUSDT')
        self.assertEqual(len(btc_trades), 1)
        self.assertEqual(btc_trades[0].symbol, 'BTCUSDT')
        
        eth_trades = self.db.get_trades(symbol='ETHUSDT')
        self.assertEqual(len(eth_trades), 1)
        self.assertEqual(eth_trades[0].symbol, 'ETHUSDT')

    def test_get_positions(self):
        """Test position retrieval"""
        # Save multiple positions
        pos1 = self.position_data.copy()
        pos2 = self.position_data.copy()
        pos2['symbol'] = 'ETHUSDT'
        
        self.db.save_position(pos1)
        self.db.save_position(pos2)
        
        # Test filtering
        btc_positions = self.db.get_positions(symbol='BTCUSDT')
        self.assertEqual(len(btc_positions), 1)
        self.assertEqual(btc_positions[0].symbol, 'BTCUSDT')
        
        eth_positions = self.db.get_positions(symbol='ETHUSDT')
        self.assertEqual(len(eth_positions), 1)
        self.assertEqual(eth_positions[0].symbol, 'ETHUSDT')

    def test_get_market_data(self):
        """Test market data retrieval"""
        # Save market data for different times
        data1 = self.market_data[0].copy()
        data2 = self.market_data[0].copy()
        data2['timestamp'] = datetime.now(UTC) + timedelta(hours=1)
        
        self.db.save_market_data([data1, data2])
        
        # Test time range filtering
        start_time = datetime.now(UTC) - timedelta(minutes=5)
        end_time = datetime.now(UTC) + timedelta(hours=2)
        
        data = self.db.get_market_data('BTCUSDT', start_time, end_time)
        self.assertEqual(len(data), 2)
        self.assertTrue(data[0].timestamp < data[1].timestamp)

    def test_backup_restore(self):
        """Test database backup and restore"""
        # Save some data
        self.db.save_trade(self.trade_data)
        self.db.save_position(self.position_data)
        self.db.save_market_data(self.market_data)
        
        # Create backup
        backup_file = self.db.create_backup('test_backups')
        self.assertTrue(os.path.exists(backup_file))
        
        # Clear database
        session = self.db.Session()
        try:
            session.query(Trade).delete()
            session.query(Position).delete()
            session.query(MarketData).delete()
            session.commit()
        finally:
            session.close()
        
        # Verify data is cleared
        self.assertEqual(len(self.db.get_trades()), 0)
        
        # Restore backup
        success = self.db.restore_backup(backup_file)
        self.assertTrue(success)
        
        # Verify data is restored
        self.assertEqual(len(self.db.get_trades()), 1)
        
        # Cleanup
        os.remove(backup_file)
        os.rmdir('test_backups')

    def test_cleanup_old_data(self):
        """Test old data cleanup"""
        # Save old and new data
        old_data = self.market_data[0].copy()
        old_data['timestamp'] = datetime.now(UTC) - timedelta(days=40)
        new_data = self.market_data[0].copy()
        new_data['timestamp'] = datetime.now(UTC)  # Ensure new data has timezone
        
        self.db.save_market_data([old_data, new_data])
        
        # Clean up data older than 30 days
        deleted = self.db.cleanup_old_data(days_to_keep=30)
        self.assertEqual(deleted, 1)
        
        # Verify only new data remains
        data = self.db.get_market_data(
            'BTCUSDT',
            datetime.now(UTC) - timedelta(days=50),
            datetime.now(UTC)
        )
        self.assertEqual(len(data), 1)
        
        # Ensure timestamp comparison uses timezone-aware datetimes
        data_age = datetime.now(UTC) - data[0].timestamp.replace(tzinfo=UTC)
        self.assertTrue(data_age.days < 30)

    def test_get_statistics(self):
        """Test statistics retrieval"""
        # Save some data
        self.db.save_trade(self.trade_data)
        self.db.save_position(self.position_data)
        self.db.save_market_data(self.market_data)
        
        stats = self.db.get_statistics()
        
        self.assertEqual(stats['total_trades'], 1)
        self.assertEqual(stats['total_positions'], 1)
        self.assertEqual(stats['market_data_points'], 1)
        self.assertEqual(stats['symbols'], ['BTCUSDT'])
        self.assertIsNotNone(stats['earliest_data'])
        self.assertIsNotNone(stats['latest_data'])

if __name__ == '__main__':
    unittest.main() 