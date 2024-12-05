from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.engine = create_engine(config.DATABASE_URL)
        self.Session = sessionmaker(bind=self.engine)
        self.base = Base
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        try:
            self.base.metadata.create_all(self.engine)
            logger.info('Database tables created successfully')
        except Exception as e:
            logger.error(f'Error creating database tables: {e}')
            raise

    async def insert_market_data(self, data):
        """Insert market data into database"""
        session = self.Session()
        try:
            session.add(MarketData(**data))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f'Error inserting market data: {e}')
            raise
        finally:
            session.close()

    async def get_market_data(self, symbol, start_time, end_time):
        """Get market data from database"""
        session = self.Session()
        try:
            data = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timestamp.between(start_time, end_time)
            ).all()
            return data
        except Exception as e:
            logger.error(f'Error getting market data: {e}')
            raise
        finally:
            session.close()

# Database Models
class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    indicators = Column(JSON)

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    pnl = Column(Float)

class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)