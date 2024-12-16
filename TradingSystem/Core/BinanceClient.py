import logging
from typing import Dict, List, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import time
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class BinanceClient:
    """Binance API client wrapper for real-time trading data"""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        """Initialize Binance client with real API connection"""
        try:
            # Get the root directory path
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, 'Config', 'BTC.env')
            
            logger.info(f"Loading config from: {config_path}")
            
            # Load environment variables from .env file
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
            load_dotenv(config_path)
            
            # Get API keys from environment variables or parameters
            self.api_key = api_key or os.getenv('BINANCE_API_KEY')
            self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
            
            if not self.api_key or not self.api_secret:
                raise ValueError("API keys not found in environment variables or parameters")
                
            logger.info("API keys loaded")
            
            # Initialize client with real API
            self.client = Client(self.api_key, self.api_secret)
            
            # Test connection and get server time
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}")
            raise

    def _test_connection(self):
        """Test API connection and permissions"""
        try:
            # Test basic connection
            server_time = self.client.get_server_time()
            logger.info(f"Connected to Binance API. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            
            # Get API key permissions
            account_status = self.client.get_account_api_trading_status()
            logger.info(f"Account API status: {account_status}")
            
            # Test spot account access
            spot_account = self.client.get_account()
            logger.info("Successfully accessed spot account")
            
            # Test permissions
            permissions = []
            try:
                # Test reading account data
                self.get_account_info()
                permissions.append('READ_INFO')
                
                # Test reading trades
                trades = self.client.get_my_trades(symbol="BTCUSDT", limit=1)
                permissions.append('READ_TRADES')
                logger.info(f"Retrieved {len(trades)} trades for BTCUSDT")
                
            except BinanceAPIException as e:
                logger.error(f"Permission test failed: {e}")
            
            logger.info(f"API permissions verified: {permissions}")
            
        except BinanceAPIException as e:
            logger.error(f"Binance API connection test failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise

    def _validate_symbol(self, symbol: str) -> str:
        """Validate and format symbol string"""
        try:
            # If symbol is a DataFrame, return default symbol
            if isinstance(symbol, pd.DataFrame):
                return "BTCUSDT"
                
            # If symbol is None or empty, return default symbol
            if not symbol:
                return "BTCUSDT"
                
            # Remove any whitespace
            symbol = str(symbol).strip()
            
            # Check if symbol matches the required pattern
            if not symbol.isalnum():
                # Remove any special characters except hyphen and underscore
                symbol = ''.join(c for c in symbol if c.isalnum() or c in '-_')
            
            # Convert to uppercase
            symbol = symbol.upper()
            
            # Validate final format
            if not symbol or len(symbol) > 20:
                return "BTCUSDT"
                
            return symbol
            
        except Exception as e:
            logger.error(f"Error validating symbol: {e}")
            return "BTCUSDT"

    def get_historical_data(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Get historical kline data"""
        try:
            # Validate symbol format
            symbol = self._validate_symbol(symbol)
            
            # Validate interval
            valid_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
            if interval not in valid_intervals:
                logger.error(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
                return pd.DataFrame()
            
            # Get kline data
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                             'taker_buy_quote', 'ignore'])
            
            # Convert timestamp to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            logger.info(f"Retrieved {len(df)} candles for {symbol} {interval}")
            logger.debug(f"Data range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()

    def get_account_info(self) -> Dict:
        """Get real account information from Binance Spot"""
        try:
            # Get spot account info
            account_info = self.client.get_account()
            logger.info("Retrieved spot account information from Binance")
            
            # Convert spot balances
            balances = [
                {
                    'asset': b['asset'],
                    'free': float(b['free']),
                    'locked': float(b['locked'])
                }
                for b in account_info.get('balances', [])
                if float(b['free']) > 0 or float(b['locked']) > 0
            ]
            
            # Calculate total balance in USDT
            total_balance = 0
            assets_info = []
            
            for balance in balances:
                asset = balance['asset']
                free_amount = balance['free']
                locked_amount = balance['locked']
                total_amount = free_amount + locked_amount
                
                if asset == 'USDT':
                    value_in_usdt = total_amount
                else:
                    try:
                        # Get current price in USDT
                        ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['price'])
                        value_in_usdt = total_amount * price
                    except:
                        value_in_usdt = 0
                        logger.warning(f"Could not get price for {asset}USDT")
                        continue
                
                total_balance += value_in_usdt
                
                assets_info.append({
                    'asset': asset,
                    'free': free_amount,
                    'locked': locked_amount,
                    'total': total_amount,
                    'value_in_usdt': value_in_usdt
                })
            
            logger.info(f"Total balance: {total_balance:.2f} USDT")
            logger.info(f"Found {len(assets_info)} assets with balance")
            
            return {
                'accountType': 'SPOT',
                'total_balance': total_balance,
                'available_balance': total_balance,  # In spot, all balance is available
                'assets_info': assets_info,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
                
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting account info: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def get_symbol_price(self, symbol: str = "BTCUSDT") -> float:
        """Get current real-time price for a symbol from Futures market"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.debug(f"Retrieved current futures price for {symbol}: {price}")
            return price
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting symbol price: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting symbol price: {e}")
            raise

    def get_positions(self) -> List[Dict]:
        """Get current real positions from account"""
        try:
            account = self.get_account_info()
            positions = []
            
            # Pega as posições do account info
            for position in account['positions']:
                amt = float(position['positionAmt'])
                if amt != 0:  # Só adiciona posições ativas
                    positions.append({
                        'symbol': position['symbol'],
                        'positionAmt': amt,
                        'entryPrice': float(position['entryPrice']),
                        'unrealizedProfit': float(position['unrealizedProfit']),
                        'leverage': int(position['leverage'])
                    })
            
            logger.info(f"Retrieved {len(positions)} active positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise

    def create_order(self, symbol: str, side: str, order_type: str, quantity: float,
                    price: Optional[float] = None, stop_price: Optional[float] = None) -> Dict:
        """Create a new order on Binance Futures"""
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if price is not None:
                params['price'] = price
                
            if stop_price is not None:
                params['stopPrice'] = stop_price
            
            order = self.client.futures_create_order(**params)
            logger.info(f"Created {order_type} {side} order for {symbol}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error creating order: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get real open orders from Binance Futures"""
        try:
            if symbol:
                orders = self.client.futures_get_open_orders(symbol=symbol)
            else:
                orders = self.client.futures_get_open_orders()
            logger.info(f"Retrieved {len(orders)} open orders")
            return orders
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting open orders: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            raise

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order on Binance Futures"""
        try:
            result = self.client.futures_cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error canceling order: {e}")
            raise
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            raise

    def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get real trade history from Binance Futures"""
        try:
            trades = self.client.futures_account_trades(symbol=symbol, limit=limit)
            logger.info(f"Retrieved {len(trades)} trades for {symbol}")
            return trades
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting trade history: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            raise

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Get exchange information from Binance"""
        try:
            if symbol:
                exchange_info = self.client.futures_exchange_info(symbol=symbol)
            else:
                exchange_info = self.client.futures_exchange_info()
            
            logger.info(f"Retrieved exchange information from Binance")
            return exchange_info
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting exchange info: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            raise

    def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 20) -> pd.DataFrame:
        """Get order book for a symbol"""
        try:
            # Validate symbol format
            symbol = self._validate_symbol(symbol)
            
            depth = self.client.futures_order_book(symbol=symbol, limit=limit)
            
            # Convert to DataFrame for easier handling
            bids_df = pd.DataFrame(depth['bids'], columns=['price', 'quantity']).astype(float)
            asks_df = pd.DataFrame(depth['asks'], columns=['price', 'quantity']).astype(float)
            
            # Add side identifier
            bids_df['side'] = 'bid'
            asks_df['side'] = 'ask'
            
            # Combine bids and asks
            df = pd.concat([bids_df, asks_df], axis=0)
            df = df.sort_values('price', ascending=False)
            
            logger.info(f"Retrieved order book for {symbol} with {limit} levels")
            return df
            
        except Exception as e:
            logger.error(f"Error getting order book: {str(e)}")
            return pd.DataFrame()

    def get_system_status(self) -> Dict:
        """Get Binance system status"""
        try:
            status = self.client.get_system_status()
            return status
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': 1, 'msg': str(e)}

    def get_symbol_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """Get current price for a symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error getting symbol ticker: {e}")
            return {'symbol': symbol, 'price': '0.0'}

    def get_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """Get 24hr ticker data from Binance"""
        try:
            ticker = self.client.futures_ticker(symbol=symbol)
            logger.info(f"Retrieved 24hr ticker data for {symbol}")
            return ticker
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting ticker: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            raise

    def get_depth(self, symbol: str = "BTCUSDT", limit: int = 100) -> Dict:
        """Get market depth (order book) data"""
        try:
            depth = self.client.futures_order_book(symbol=symbol, limit=limit)
            logger.info(f"Retrieved market depth for {symbol}, limit: {limit}")
            return depth
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting depth: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting depth: {e}")
            raise

    def get_agg_trades(self, symbol: str = "BTCUSDT", limit: int = 100) -> List[Dict]:
        """Get aggregated trades data"""
        try:
            trades = self.client.futures_agg_trades(symbol=symbol, limit=limit)
            logger.info(f"Retrieved {len(trades)} aggregated trades for {symbol}")
            return trades
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting aggregated trades: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting aggregated trades: {e}")
            raise

    def get_margin_info(self, symbol: str = "BTCUSDT") -> Dict:
        """Get margin account information"""
        try:
            margin_info = self.client.futures_account()
            logger.info(f"Retrieved margin account information")
            return margin_info
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting margin info: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting margin info: {e}")
            raise

    def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """Change leverage for symbol"""
        try:
            response = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            logger.info(f"Changed leverage for {symbol} to {leverage}x")
            return response
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error changing leverage: {e}")
            raise
        except Exception as e:
            logger.error(f"Error changing leverage: {e}")
            raise

    def create_oco_order(self, symbol: str, side: str, quantity: float,
                        price: float, stop_price: float, stop_limit_price: float) -> Dict:
        """Create OCO (One-Cancels-Other) order"""
        try:
            order = self.client.futures_create_oco_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                stopPrice=stop_price,
                stopLimitPrice=stop_limit_price
            )
            logger.info(f"Created OCO order for {symbol}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error creating OCO order: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating OCO order: {e}")
            raise

    def get_all_orders(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get all orders history"""
        try:
            orders = self.client.futures_get_all_orders(symbol=symbol, limit=limit)
            logger.info(f"Retrieved {len(orders)} orders for {symbol}")
            return orders
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting all orders: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting all orders: {e}")
            raise

    def get_account_status(self):
        """Get account trading status"""
        try:
            status = self.client.futures_account_api_trading_status()
            return status
        except Exception as e:
            logger.error(f"Error getting account status: {str(e)}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 1):
        """Get recent trades for a symbol"""
        try:
            trades = self.client.futures_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            return []

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed information about a specific trading symbol"""
        try:
            # Get exchange info first
            exchange_info = self.get_exchange_info()
            
            # Find the specific symbol info
            symbol_info = next(
                (item for item in exchange_info['symbols'] if item['symbol'] == symbol),
                None
            )
            
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found in exchange info")
            
            logger.info(f"Retrieved symbol information for {symbol}")
            return symbol_info
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting symbol info: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            raise

    def start_symbol_streams(self, symbols: List[str], callbacks: Dict = None) -> None:
        """Start real-time data streams for specified symbols"""
        try:
            # Initialize WebSocket client if not already done
            if not hasattr(self, 'ws_client'):
                self.ws_client = self.client.get_socket_manager()
            
            # Default callbacks if none provided
            if callbacks is None:
                callbacks = {
                    'trade': lambda msg: logger.debug(f"Trade: {msg}"),
                    'kline': lambda msg: logger.debug(f"Kline: {msg}"),
                    'depth': lambda msg: logger.debug(f"Depth: {msg}")
                }
            
            # Start streams for each symbol
            for symbol in symbols:
                symbol = symbol.lower()
                # Trade stream
                self.ws_client.start_trade_socket(
                    symbol=symbol,
                    callback=callbacks.get('trade')
                )
                # Kline stream
                self.ws_client.start_kline_socket(
                    symbol=symbol,
                    interval='1m',
                    callback=callbacks.get('kline')
                )
                # Depth stream
                self.ws_client.start_depth_socket(
                    symbol=symbol,
                    callback=callbacks.get('depth')
                )
            
            # Start the WebSocket client
            self.ws_client.start()
            logger.info(f"Started real-time streams for symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Error starting symbol streams: {e}")
            raise

    def get_server_time(self) -> Dict:
        """Get server time from Binance"""
        try:
            server_time = self.client.get_server_time()
            return server_time
        except Exception as e:
            logger.error(f"Error getting server time: {str(e)}")
            return {'serverTime': int(time.time() * 1000)}

    def get_account_trades(self, symbol: str = "BTCUSDT", limit: int = 500) -> List[Dict]:
        """Get account trade history"""
        try:
            trades = self.client.futures_account_trades(symbol=symbol, limit=limit)
            logger.info(f"Retrieved {len(trades)} trades for {symbol}")
            return trades
        except Exception as e:
            logger.error(f"Error getting account trades: {str(e)}")
            return []