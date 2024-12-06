from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional

class TelegramBot:
    def __init__(self, config, exchange, analysis):
        self.config = config
        self.exchange = exchange
        self.analysis = analysis
        self.app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup command handlers"""
        self.app.add_handler(CommandHandler('start', self._start_command))
        self.app.add_handler(CommandHandler('status', self._status_command))
        self.app.add_handler(CommandHandler('price', self._price_command))
        self.app.add_handler(CommandHandler('positions', self._positions_command))
        self.app.add_handler(CommandHandler('trades', self._trades_command))
        self.app.add_handler(CallbackQueryHandler(self._button_callback))

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        keyboard = [
            [InlineKeyboardButton('Status', callback_data='status'),
             InlineKeyboardButton('Price', callback_data='price')],
            [InlineKeyboardButton('Positions', callback_data='positions'),
             InlineKeyboardButton('Trades', callback_data='trades')]
        ]

        await update.message.reply_text(
            'Welcome to AleBot! Choose an option:',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Get bot status
            positions = await self.exchange.get_positions()
            balance = await self.exchange.get_account_balance()

            status_text = (
                f'🤖 Bot Status\n\n'
                f'Balance: {balance:.2f} USDT\n'
                f'Active Positions: {len(positions)}\n'
                f'Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )

            await update.message.reply_text(status_text)

        except Exception as e:
            logging.error(f'Error in status command: {e}')
            await update.message.reply_text('Error getting status')

    async def _price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /price command"""
        try:
            prices = []
            for symbol in self.config.TRADING_CONFIG['symbols']:
                price = self.exchange.last_prices.get(symbol, 0)
                prices.append(f'{symbol}: ${price:.2f}')

            await update.message.reply_text('\n'.join(prices))

            # Generate and send price chart
            chart = await self._generate_price_chart()
            if chart:
                await update.message.reply_photo(chart)

        except Exception as e:
            logging.error(f'Error in price command: {e}')
            await update.message.reply_text('Error getting prices')

    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            positions = await self.exchange.get_positions()
            if not positions:
                await update.message.reply_text('No open positions')
                return

            position_text = '📊 Open Positions:\n\n'
            for symbol, pos in positions.items():
                pnl = (self.exchange.last_prices[symbol] - pos['entry_price']) * pos['quantity']
                position_text += (
                    f'Symbol: {symbol}\n'
                    f'Side: {pos["side"]}\n'
                    f'Size: {pos["quantity"]}\n'
                    f'Entry: ${pos["entry_price"]:.2f}\n'
                    f'Current: ${self.exchange.last_prices[symbol]:.2f}\n'
                    f'PnL: ${pnl:.2f}\n\n'
                )

            await update.message.reply_text(position_text)

        except Exception as e:
            logging.error(f'Error in positions command: {e}')
            await update.message.reply_text('Error getting positions')

    async def _trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command"""
        try:
            trades = await self.exchange.get_trades(
                start_time=datetime.now() - timedelta(days=1)
            )

            if not trades:
                await update.message.reply_text('No recent trades')
                return

            trades_text = '📈 Recent Trades:\n\n'
            for trade in trades:
                trades_text += (
                    f'Symbol: {trade["symbol"]}\n'
                    f'Side: {trade["side"]}\n'
                    f'Price: ${trade["price"]:.2f}\n'
                    f'Time: {trade["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}\n\n'
                )

            await update.message.reply_text(trades_text)

        except Exception as e:
            logging.error(f'Error in trades command: {e}')
            await update.message.reply_text('Error getting trades')

    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()

        if query.data == 'status':
            await self._status_command(update, context)
        elif query.data == 'price':
            await self._price_command(update, context)
        elif query.data == 'positions':
            await self._positions_command(update, context)
        elif query.data == 'trades':
            await self._trades_command(update, context)

    async def _generate_price_chart(self) -> Optional[bytes]:
        """Generate price chart image"""
        try:
            symbol = self.config.TRADING_CONFIG['symbols'][0]
            data = await self.exchange.get_market_data(
                symbol=symbol,
                timeframe='1h',
                limit=24
            )

            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )])

            fig.update_layout(
                title=f'{symbol} Price Chart',
                yaxis_title='Price',
                xaxis_title='Time'
            )

            img_bytes = fig.to_image(format="png")
            return io.BytesIO(img_bytes)

        except Exception as e:
            logging.error(f'Error generating chart: {e}')
            return None

    def run(self):
        """Run the Telegram bot"""
        self.app.run_polling()