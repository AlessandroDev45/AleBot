import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Import our modules
from config import ConfigManager
from database.database_manager import DatabaseManager
from exchange.exchange_manager import BinanceExchange
from analysis.technical_analysis import TechnicalAnalysis
from ml.model import MLModel
from risk.risk_manager import RiskManager
from dashboard.app import DashboardApp
from telegram_bot.bot import TelegramBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alebot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self):
        self.initialize_components()
        self.running = False
        self.last_health_check = datetime.now()

    def initialize_components(self):
        # Create necessary directories
        Path('data').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)

        try:
            # Initialize components
            self.config = ConfigManager()
            self.db = DatabaseManager(self.config)
            self.exchange = BinanceExchange(self.config, self.db)
            self.analysis = TechnicalAnalysis(self.config, self.db, self.exchange)
            self.ml_model = MLModel(self.config, self.db, self.exchange, self.analysis)
            self.risk_manager = RiskManager(
                self.config, self.db, self.exchange, self.analysis, self.ml_model
            )
            
            # Initialize interfaces
            self.dashboard = DashboardApp(
                self.config, self.db, self.exchange,
                self.analysis, self.ml_model, self.risk_manager
            )
            self.telegram_bot = TelegramBot(
                self.config, self.db, self.exchange,
                self.analysis, self.ml_model, self.risk_manager
            )
            
            logger.info('All components initialized successfully')
            
        except Exception as e:
            logger.error(f'Error initializing components: {e}')
            raise

    async def start(self):
        try:
            logger.info('Starting trading system...')
            self.running = True

            # Initialize connections
            await self.exchange.connect()
            await self.exchange.start_market_data_collection()

            # Start monitoring tasks
            asyncio.create_task(self.monitor_system_health())
            
            # Run interfaces
            asyncio.create_task(self.dashboard.run())
            asyncio.create_task(self.telegram_bot.run())

            logger.info('Trading system started successfully')
            
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f'Critical error in trading system: {e}')
            await self.shutdown()

    async def monitor_system_health(self):
        while self.running:
            try:
                # Check component health
                await self.exchange.check_connection()
                await self.db.check_connection()
                
                # Check trading conditions
                await self.risk_manager.check_risk_levels()
                
                # Update ML model if needed
                await self.ml_model.update_if_needed()
                
                self.last_health_check = datetime.now()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f'Error in health monitoring: {e}')
                await asyncio.sleep(5)

    async def shutdown(self):
        logger.info('Shutting down trading system...')
        self.running = False
        
        try:
            # Close all positions
            await self.exchange.close_all_positions()
            
            # Cleanup components
            await self.exchange.cleanup()
            await self.risk_manager.cleanup()
            await self.analysis.cleanup()
            await self.ml_model.cleanup()
            
            logger.info('Trading system shutdown complete')
            
        except Exception as e:
            logger.error(f'Error during shutdown: {e}')
        finally:
            sys.exit(0)

async def main():
    system = TradingSystem()
    await system.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Trading system stopped by user')
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
