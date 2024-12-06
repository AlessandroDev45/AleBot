from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import logging
from typing import Dict, List, Optional
from enum import Enum as PyEnum

Base = declarative_base()
logger = logging.getLogger(__name__)

class TimeFrame(PyEnum):
    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'

class OrderSide(PyEnum):
    BUY = 'BUY'
    SELL = 'SELL'

class OrderStatus(PyEnum):
    NEW = 'NEW'
    FILLED = 'FILLED'
    CANCELED = 'CANCELED'
    REJECTED = 'REJECTED'
    EXPIRED = 'EXPIRED'

class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(Enum(TimeFrame), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    status = Column(Enum(OrderStatus), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.engine = create_engine(
            config.DATABASE_URL,
            **config.get_database_config()
        )
        self.Session = sessionmaker(bind=self.engine)
        self._create_tables()

    def _create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info('Database tables created successfully')
        except Exception as e:
            logger.error(f'Error creating database tables: {e}')
            raise

    def get_market_data(self, symbol: str, timeframe: TimeFrame,
                       start_time: datetime, end_time: datetime) -> Optional[List[Dict]]:
        """Get market data from database"""
        try:
            session = self.Session()
            data = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe,
                MarketData.timestamp.between(start_time, end_time)
            ).order_by(MarketData.timestamp.asc()).all()
            
            return [{
                'timestamp': entry.timestamp,
                'open': entry.open,
                'high': entry.high,
                'low': entry.low,
                'close': entry.close,
                'volume': entry.volume
            } for entry in data]
        except Exception as e:
            logger.error(f'Error getting market data: {e}')
            return None
        finally:
            session.close()

    def insert_market_data(self, data: Dict):
        """Insert market data into database"""
        try:
            session = self.Session()
            market_data = MarketData(**data)
            session.add(market_data)
            session.commit()
        except Exception as e:
            logger.error(f'Error inserting market data: {e}')
            session.rollback()
            raise
        finally:
            session.close()

    def insert_trade(self, trade_data: Dict):
        """Insert trade record into database"""
        try:
            session = self.Session()
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            return trade.id
        except Exception as e:
            logger.error(f'Error inserting trade: {e}')
            session.rollback()
            raise
        finally:
            session.close()

    def get_trades(self, symbol: str = None, start_time: datetime = None,
                   end_time: datetime = None) -> Optional[List[Dict]]:
        """Get trade records from database"""
        try:
            session = self.Session()
            query = session.query(Trade)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if start_time:
                query = query.filter(Trade.timestamp >= start_time)
            if end_time:
                query = query.filter(Trade.timestamp <= end_time)
                
            trades = query.order_by(Trade.timestamp.asc()).all()
            
            return [{
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'price': trade.price,
                'quantity': trade.quantity,
                'status': trade.status
            } for trade in trades]
        except Exception as e:
            logger.error(f'Error getting trades: {e}')
            return None
        finally:
            session.close()