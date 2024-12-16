import logging
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import Update
from telegram.ext import ContextTypes
import threading
import nest_asyncio
import time

logger = logging.getLogger(__name__)
nest_asyncio.apply()

class TelegramBot:
    def __init__(self, components, token, chat_id):
        self.components = components
        self.token = token
        self.chat_id = chat_id
        self.application = None
        self._is_running = False
        self._stop_event = threading.Event()
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text("Bot started. Use /help to see available commands.")
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
Available commands:
/start - Start the bot
/status - Get system status
/balance - Get account balance
/price - Get current BTC price
/help - Show this help message
"""
        await update.message.reply_text(help_text)
        
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status = self.components['data_manager'].get_system_status()
        await update.message.reply_text(f"System Status:\n{status}")
        
    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        balance = self.components['data_manager'].get_account_balance()
        await update.message.reply_text(f"Account Balance:\n{balance}")
        
    async def price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /price command"""
        price = self.components['data_manager'].get_current_price()
        await update.message.reply_text(f"Current BTC Price: ${price:,.2f}")
        
    def setup_handlers(self):
        """Setup message handlers"""
        self.application.add_handler(CommandHandler('start', self.start_command))
        self.application.add_handler(CommandHandler('help', self.help_command))
        self.application.add_handler(CommandHandler('status', self.status_command))
        self.application.add_handler(CommandHandler('balance', self.balance_command))
        self.application.add_handler(CommandHandler('price', self.price_command))
        
    async def run_async(self):
        """Run the bot asynchronously"""
        try:
            self.application = Application.builder().token(self.token).build()
            self.setup_handlers()
            self._is_running = True
            
            # Initialize and start the application
            await self.application.initialize()
            await self.application.start()
            
            # Run until stop event is set
            while not self._stop_event.is_set():
                await self.application.update_queue.get()
                
        except Exception as e:
            logger.error(f"Error running bot: {str(e)}")
            self._is_running = False
        finally:
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
                
    def run_bot(self):
        """Run the bot in a new event loop"""
        try:
            asyncio.run(self.run_async())
        except Exception as e:
            logger.error(f"Error in bot thread: {str(e)}")
            
    def start(self):
        """Start the Telegram bot"""
        try:
            # Set up the application
            self.application = Application.builder().token(self.token).build()
            
            # Add handlers
            self.add_handlers()
            
            # Start the bot with retry logic
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.application.run_polling(allowed_updates=Update.ALL_TYPES)
                    break
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error running bot (attempt {retry_count}/{max_retries}): {str(e)}")
                    if retry_count < max_retries:
                        time.sleep(5)  # Wait before retrying
                    else:
                        logger.error("Max retries reached, bot initialization failed")
                        break
            
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {str(e)}")
            
    def stop(self):
        """Stop the Telegram bot"""
        try:
            if self._is_running:
                self._stop_event.set()
                self._is_running = False
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {str(e)}")
            
    async def send_trade_alert(self, trade_data):
        """Send trade entry/exit alert"""
        try:
            message = (
                f"🔔 Trade Alert\n"
                f"Type: {trade_data['type']}\n"
                f"Symbol: {trade_data['symbol']}\n"
                f"Side: {trade_data['side']}\n"
                f"Price: {trade_data['price']}\n"
                f"Size: {trade_data['size']}\n"
                f"Stop Loss: {trade_data['stop_loss']}\n"
                f"Take Profit: {trade_data['take_profit']}"
            )
            await self.application.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logger.error(f"Error sending trade alert: {str(e)}")

    def add_handlers(self):
        """Add command handlers to the bot"""
        try:
            # Start command
            self.application.add_handler(CommandHandler("start", self.start_command))
            
            # Status command
            self.application.add_handler(CommandHandler("status", self.status_command))
            
            # Balance command
            self.application.add_handler(CommandHandler("balance", self.balance_command))
            
            # Price command
            self.application.add_handler(CommandHandler("price", self.price_command))
            
            # Help command
            self.application.add_handler(CommandHandler("help", self.help_command))
            
            logger.info("Bot handlers added successfully")
            
        except Exception as e:
            logger.error(f"Error adding bot handlers: {str(e)}")
            raise
