# main.py
import sys
import os
import logging
import traceback
from datetime import datetime
from threading import Thread
import time
from dotenv import load_dotenv
import socket
import webbrowser
from threading import Timer
from Core.BinanceClient import BinanceClient

# Add project root to Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(root_dir)
sys.path.append(project_root)

# Load environment variables from BTC.env
dotenv_path = os.path.join(root_dir, 'Config', 'BTC.env')
load_dotenv(dotenv_path)  # This line is uncommented

from Interface.Dashboard.app import create_app
from Core.RiskManager import RiskManager
from Core.OrderManager import OrderManager
from Core.DataProcessor import DataProcessor
from Core.PatternDetection import PatternDetector
from Core.TradingCore.ml_model import MLModel
from Interface.TelegramBot.bot_handler import TelegramBot
from Core.DataManager.data_manager import DataManager


def ensure_directories():
    """Ensure required directories exist"""
    directories = ['logs', 'data', 'models', 'config', 'cache']
    for directory in directories:
        os.makedirs(os.path.join(root_dir, directory), exist_ok=True)

def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join(root_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'trading_{datetime.now().strftime("%Y%m%d")}.log')
    
    # Reset any existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Ensure logs are displayed immediately
    sys.stdout.flush()
    
    return logger

def initialize_components(logger):
    """Initialize all trading system components"""
    try:
        # Initialize DataManager first
        data_manager = DataManager()
        logger.info("Data manager initialized")
        
        # Wait for DataManager to fully initialize and connect to Binance
        time.sleep(2)  # Give time for connection establishment
        
        # Initialize ML model with DataManager
        ml_model = MLModel(data_manager=data_manager)
        logger.info("ML model initialized")
        
        # Initialize other components
        risk_manager = RiskManager(data_manager=data_manager)
        order_manager = OrderManager(data_manager=data_manager, risk_manager=risk_manager)
        data_processor = DataProcessor(data_manager=data_manager)
        pattern_detector = PatternDetector()
        
        # Store all components in a dictionary
        components = {
            'data_manager': data_manager,
            'risk_manager': risk_manager,
            'order_manager': order_manager,
            'data_processor': data_processor,
            'pattern_detector': pattern_detector,
            'ml_model': ml_model
        }
        
        logger.info("All components initialized successfully")
        return components
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        # Log detailed error information
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        raise

def start_telegram_bot(components, logger):
    """Start Telegram bot in a separate thread"""
    try:
        # Get Telegram credentials from DataManager
        telegram_creds = components['data_manager'].get_telegram_credentials()
        
        if telegram_creds['bot_token'] and telegram_creds['chat_id']:
            telegram_bot = TelegramBot(components, telegram_creds['bot_token'], telegram_creds['chat_id'])
            telegram_thread = Thread(target=telegram_bot.start)
            telegram_thread.daemon = True
            telegram_thread.start()
            logger.info("Telegram bot started successfully")
        else:
            logger.warning("Telegram bot not started: missing configuration")
            
    except Exception as e:
        logger.error(f"Failed to start Telegram bot: {e}")
        logger.warning("Continuing without Telegram bot...")

def open_browser():
    """Open browser to dashboard URL"""
    webbrowser.open('http://127.0.0.1:8188')

def main():
    try:
        # Ensure required directories exist
        ensure_directories()
        
        # Setup logging
        logger = setup_logging()
        logger.info("Starting AleBot Trading System...")
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Initialize DataManager
        data_manager = DataManager()
        logger.info("DataManager initialized successfully")
        
        # Initialize MLModel
        logger.info("Initializing MLModel...")
        ml_model = MLModel(data_manager=data_manager)
        logger.info(f"MLModel initialized with state: {ml_model.__dict__}")
        
        # Initialize BinanceClient
        logger.info("Initializing BinanceClient...")
        binance_client = BinanceClient()
        logger.info("BinanceClient initialized successfully")
        
        # Create components dictionary
        components = {
            'data_manager': data_manager,
            'ml_model': ml_model,
            'binance_client': binance_client
        }
        
        # Initialize TelegramBot
        logger.info("Initializing TelegramBot...")
        telegram_creds = data_manager.get_telegram_credentials()
        if telegram_creds['bot_token'] and telegram_creds['chat_id']:
            telegram_bot = TelegramBot(components, telegram_creds['bot_token'], telegram_creds['chat_id'])
            components['telegram_bot'] = telegram_bot
            logger.info("TelegramBot initialized successfully")
        else:
            logger.warning("TelegramBot not initialized: missing credentials")
            components['telegram_bot'] = None
        
        logger.info("All components initialized successfully")
        
        # Create and start dashboard
        app = create_app(components)
        if not app:
            logger.error("Failed to create dashboard application")
            return
            
        # Start Telegram bot
        start_telegram_bot(components, logger)
        
        # Start dashboard with proper server configuration
        logger.info("Trading system started successfully")
        logger.info("Dashboard available at http://127.0.0.1:8188")
        
        # Force immediate display of logs
        sys.stdout.flush()
        
        # Configure server options for better stability
        server_config = {
            'debug': True,
            'host': '127.0.0.1',
            'port': 8188,
            'use_reloader': False,  # Disable reloader to prevent duplicate processes
            'threaded': True  # Enable threading for better performance
        }
        
        # Open browser after a short delay to ensure server is ready
        Timer(1.5, open_browser).start()
        
        app.run_server(**server_config)
        
    except Exception as e:
        if logger:
            logger.error(f"Critical error in main: {str(e)}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
        else:
            print(f"Critical error in main: {str(e)}")
            print(f"Detailed error: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup when system stops
        try:
            if logger:
                logger.info("Shutting down trading system...")
            if 'components' in locals():
                # Stop data manager threads
                if 'data_manager' in components:
                    components['data_manager'].stop()
                if logger:
                    logger.info("Components stopped successfully")
        except Exception as cleanup_error:
            if logger:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
            else:
                print(f"Error during cleanup: {str(cleanup_error)}")

if __name__ == "__main__":
    # Check if another instance is running
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Try to bind to the dashboard port
        sock.bind(('127.0.0.1', 8188))
        # Close the socket so the port is available for the dashboard
        sock.close()
        # No other instance running, start the system
        main()
    except socket.error as e:
        if e.errno == 98 or e.errno == 10048:  # Port already in use
            print("Sistema já está rodando em http://127.0.0.1:8188")
            # Open browser to existing instance
            webbrowser.open('http://127.0.0.1:8188')
            sys.exit(1)
        else:
            # If it's a different error, log it and try to start anyway
            print(f"Warning: {str(e)}")
            main()