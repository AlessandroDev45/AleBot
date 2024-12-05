import asyncio
import logging
from config import ConfigManager
from database import DatabaseManager
from exchange import BinanceExchange
from analysis import TechnicalAnalysis
from ml_model import MLModel
from risk_management import RiskManager
from dash_app import DashboardApp
from bot_commands import TelegramBot

logger = logging.getLogger(__name__)

async def main():
    try:
        # Initialize components
        config = ConfigManager()
        db = DatabaseManager(config)
        exchange = BinanceExchange(config, db)
        analysis = TechnicalAnalysis(config, db, exchange)
        ml_model = MLModel(config, db, exchange, analysis)
        risk_manager = RiskManager(config, db, exchange, analysis, ml_model)
        
        # Initialize interfaces
        dashboard = DashboardApp(config, db, exchange, analysis, ml_model, risk_manager)
        telegram_bot = TelegramBot(config, db, exchange, analysis, ml_model, risk_manager)
        
        # Start trading system
        await exchange.connect()
        await exchange.start_market_data_collection()
        
        # Run dashboard and bot
        dashboard.run()
        telegram_bot.run()
        
    except Exception as e:
        logger.error(f'Critical error: {e}')
        raise

if __name__ == '__main__':
    asyncio.run(main())