import aiohttp
import hmac
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import websockets
import json
import logging
from urllib.parse import urlencode
from database import TimeFrame, OrderSide, OrderStatus

logger = logging.getLogger(__name__)

class BinanceExchange:
    BASE_URL = 'https://api.binance.com'
    WS_URL = 'wss://stream.binance.com:9443/ws'

    def __init__(self, config):
        self.config = config
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.session = None
        self.ws_connections = {}
        self.order_books = {}
        self.last_prices = {}
        self.subscribers = []

    async def connect(self):
        """Initialize connection and session"""
        try:
            self.session = aiohttp.ClientSession()
            await self.test_connection()
            await self.start_market_data_stream()
            logger.info('Successfully connected to Binance')
        except Exception as e:
            logger.error(f'Connection error: {e}')
            raise

    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            await self._make_request('GET', '/api/v3/ping')
            return True
        except Exception as e:
            logger.error(f'Connection test failed: {e}')
            return False

    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make HTTP request to Binance API"""
        if params is None:
            params = {}

        url = f'{self.BASE_URL}{endpoint}'

        if signed:
            params['timestamp'] = int(datetime.now().timestamp() * 1000)
            query_string = urlencode(sorted(params.items()))
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            headers = {'X-MBX-APIKEY': self.api_key}
        else:
            headers = {}

        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, endpoint, params, signed)

                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f'API request error: {e}')
            raise

    async def start_market_data_stream(self):
        """Start websocket streams for market data"""
        try:
            for symbol in self.config.TRADING_CONFIG['symbols']:
                await self.start_kline_socket(symbol)
            logger.info('Market data streams started')
        except Exception as e:
            logger.error(f'Error starting market data streams: {e}')
            raise

    async def start_kline_socket(self, symbol: str):
        """Start kline websocket connection"""
        stream_name = f'{symbol.lower()}@kline_{self.config.TRADING_CONFIG["base_timeframe"]}'
        
        async def handle_socket():
            while True:
                try:
                    async with websockets.connect(f'{self.WS_URL}/{stream_name}') as ws:
                        self.ws_connections[stream_name] = ws
                        async for message in ws:
                            await self._handle_kline_message(json.loads(message))
                except Exception as e:
                    logger.error(f'Websocket error: {e}')
                    await asyncio.sleep(5)

        asyncio.create_task(handle_socket())

    async def _handle_kline_message(self, msg: Dict):
        """Process kline websocket message"""
        try:
            kline = msg['k']
            
            if kline['x']:  # Candle closed
                candle_data = {
                    'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                    'symbol': msg['s'],
                    'timeframe': TimeFrame(kline['i']),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }

                self.last_prices[msg['s']] = float(kline['c'])
                
                # Notify subscribers
                for callback in self.subscribers:
                    await callback('kline', candle_data)

        except Exception as e:
            logger.error(f'Error processing kline message: {e}')

    async def place_order(self, symbol: str, side: OrderSide, quantity: float,
                         price: Optional[float] = None, order_type: str = 'MARKET') -> Dict:
        """Place a new order"""
        try:
            params = {
                'symbol': symbol,
                'side': side.value,
                'type': order_type,
                'quantity': self._format_quantity(quantity, symbol)
            }

            if order_type == 'LIMIT':
                if not price:
                    raise ValueError('Price required for LIMIT order')
                params['price'] = self._format_price(price, symbol)
                params['timeInForce'] = 'GTC'

            return await self._make_request('POST', '/api/v3/order', params, signed=True)

        except Exception as e:
            logger.error(f'Error placing order: {e}')
            raise

    def _format_quantity(self, quantity: float, symbol: str) -> str:
        """Format quantity according to symbol rules"""
        precision = self.config.TRADING_CONFIG['quantity_precision']
        return f'%.{precision}f' % quantity

    def _format_price(self, price: float, symbol: str) -> str:
        """Format price according to symbol rules"""
        precision = self.config.TRADING_CONFIG['price_precision']
        return f'%.{precision}f' % price

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol"""
        try:
            account = await self._make_request('GET', '/api/v3/account', signed=True)
            for balance in account['balances']:
                if balance['asset'] in symbol:
                    free = float(balance['free'])
                    if free > 0:
                        return {
                            'symbol': symbol,
                            'size': free,
                            'entry_price': self.last_prices.get(symbol, 0)
                        }
            return None
        except Exception as e:
            logger.error(f'Error getting position: {e}')
            return None

    async def close_position(self, symbol: str) -> bool:
        """Close position for symbol"""
        try:
            position = await self.get_position(symbol)
            if position:
                await self.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL if position['size'] > 0 else OrderSide.BUY,
                    quantity=abs(position['size'])
                )
                return True
            return False
        except Exception as e:
            logger.error(f'Error closing position: {e}')
            return False

    async def get_market_data(self, symbol: str, timeframe: TimeFrame,
                            limit: int = 1000) -> pd.DataFrame:
        """Get historical market data"""
        try:
            params = {
                'symbol': symbol,
                'interval': timeframe.value,
                'limit': limit
            }

            data = await self._make_request('GET', '/api/v3/klines', params)
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            logger.error(f'Error getting market data: {e}')
            return pd.DataFrame()

    async def cleanup(self):
        """Cleanup resources"""
        try:
            for ws in self.ws_connections.values():
                await ws.close()
            
            if self.session and not self.session.closed:
                await self.session.close()
                
            logger.info('Exchange cleanup completed')
            
        except Exception as e:
            logger.error(f'Error during cleanup: {e}')
