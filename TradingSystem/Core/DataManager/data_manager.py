import logging
from typing import Dict, Optional, List, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv
import numpy as np
import threading
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Set up logging
log_dir = os.path.join(root_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'trading_system.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

from Core.BinanceClient import BinanceClient
from Core.TradingCore.pattern_detection import PatternDetector
from Core.TradingCore.technical_analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)

class DataManager:
    """Centralized data management for the trading system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, env_file: str = None):
        """Initialize DataManager with configuration"""
        if not hasattr(self, 'initialized'):
            logger.info("Initializing data manager...")
            self.env_file = env_file
            self._client = None
            self.cache = {}
            self.cache_timestamps = {}
            
            # Update intervals (in seconds)
            self.update_intervals = {
                'market_data': 60,    # 1 minute
                'order_book': 5,      # 5 seconds
                'account': 60,        # 1 minute
                'positions': 60,      # 1 minute
                'trades': 60,         # 1 minute
                'performance': 300,   # 5 minutes
                'risk': 300,         # 5 minutes
                'patterns': 60        # 1 minute
            }
            
            # Cache expiration times (in seconds)
            self.cache_expiry = {
                'market_data': 300,   # 5 minutes
                'order_book': 5,      # 5 seconds
                'account': 60,        # 1 minute
                'positions': 60,      # 1 minute
                'trades': 60,         # 1 minute
                'performance': 300,   # 5 minutes
                'risk': 300,         # 5 minutes
                'patterns': 60        # 1 minute
            }
            
            # Rate limiting
            self.request_weight = 0
            self.last_weight_reset = datetime.now()
            self.weight_limit = 6000  # Binance limit per minute
            self.weight_reset_interval = 60  # 1 minute in seconds
            
            # Cache cleanup interval
            self.cleanup_interval = 300  # 5 minutes
            self.last_cleanup = datetime.now()
            
            # Por padrão, apenas BTC
            self.active_symbols = ['BTCUSDT']
            
            # Initialize components
            self._initialize_client()
            self._pattern_detector = PatternDetector(self)
            self._technical_analysis = TechnicalAnalysis()
            
            self._start_update_threads()
            self.initialized = True
            logger.info("Data manager initialized successfully")

    def _initialize_client(self):
        """Initialize Binance client"""
        try:
            config_path = os.path.join(root_dir, 'Config', 'BTC.env')
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found at {config_path}, using environment variables")
            else:
                load_dotenv(config_path)
            
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                logger.warning("API keys not found, running in test mode")
                api_key = "test"
                api_secret = "test"
            
            self._client = BinanceClient(api_key=api_key, api_secret=api_secret)
            
            # Initialize cache
            self.cache = {}
            self.cache_timestamps = {}
            
            # Initialize pattern detector and technical analysis
            self._pattern_detector = PatternDetector(self)
            self._technical_analysis = TechnicalAnalysis()
            
            logger.info("Binance client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise

    def _test_connection(self):
        """Test API connection"""
        try:
            if not self._client:
                raise ConnectionError("Binance client not initialized")
                
            # Try to get server time first as a lightweight test
            server_time = self._client.get_server_time()
            if not server_time:
                raise ConnectionError("Could not get server time from Binance")
            
            # Then try to get account info
            account = self._client.get_account_info()
            if not account:
                logger.warning("Could not get account data, running in test mode")
                return
            
            # Log successful connection
            logger.info("Successfully connected to Binance API")
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            # Don't raise here, just log the error and continue in test mode

    def _start_update_threads(self):
        """Start background threads for data updates"""
        logger.info("Starting data update threads...")
        
        def update_market_data():
            while True:
                try:
                    for symbol in self.active_symbols:
                        self._update_market_data(symbol)
                except Exception as e:
                    logger.error(f"Error updating market data: {str(e)}")
                time.sleep(self.update_intervals['market_data'])
        
        def update_account_data():
            while True:
                try:
                    self._update_account_data()
                except Exception as e:
                    logger.error(f"Error updating account data: {str(e)}")
                time.sleep(self.update_intervals['account'])
        
        market_thread = threading.Thread(target=update_market_data, daemon=True)
        account_thread = threading.Thread(target=update_account_data, daemon=True)
        
        market_thread.start()
        account_thread.start()
        logger.info("Data update threads started successfully")

    def _update_market_data(self, symbol: str):
        """Update market data for a symbol"""
        try:
            # Update different timeframes
            for timeframe in ["1m", "5m", "1h", "4h"]:
                cache_key = f'market_data_{symbol}_{timeframe}'
                data = self._client.get_historical_data(symbol, timeframe, limit=100)
                
                if not data.empty:
                    # Add technical indicators
                    data = self._technical_analysis.add_indicators(data)
                    self._update_cache(cache_key, data)
                    
            logger.info(f"Updated market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {str(e)}")

    def _update_account_data(self):
        """Update account data"""
        try:
            account = self._client.get_account_info()
            if account:
                self._update_cache('account_data', account)
                logger.info("Updated account data")
        except Exception as e:
            logger.error(f"Error updating account data: {str(e)}")

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

    def get_market_data(self, symbol: str = "BTCUSDT", timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Get market data with indicators"""
        try:
            # Validate symbol format
            symbol = self._validate_symbol(symbol)
            cache_key = f'market_data_{symbol}_{timeframe}'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None and len(cached_data) >= limit:
                logger.info(f"Using cached data for {symbol} {timeframe}")
                return cached_data
            
            # Get fresh data
            try:
                data = self._client.get_historical_data(symbol, timeframe, limit=limit)
            except Exception as e:
                logger.error(f"Error getting market data: {e}")
                # Return empty DataFrame with correct columns in test mode
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if data is not None and not data.empty:
                # Add technical indicators
                try:
                    data = self._technical_analysis.add_indicators(data)
                except Exception as e:
                    logger.error(f"Error adding indicators: {e}")
                
                # Update cache
                self._update_cache(cache_key, data)
                return data
            else:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
        except Exception as e:
            logger.error(f"Error in get_market_data: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (std * 2)
            df['bb_lower'] = df['bb_middle'] - (std * 2)
            
            # Moving Averages
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['ma_200'] = df['close'].rolling(window=200).mean()
            
            # Volume Moving Average
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(index=prices.index), pd.Series(index=prices.index), pd.Series(index=prices.index)

    def get_account_data(self) -> Dict:
        """Get account data"""
        try:
            cached_data = self._get_from_cache('account_data')
            if cached_data is not None:
                return cached_data
            
            data = self._client.get_account_info()
            if data:
                self._update_cache('account_data', data)
                return data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting account data: {str(e)}")
            return {}

    def get_account_info(self) -> Dict:
        """Alias for get_account_data to maintain compatibility with existing code"""
        return self.get_account_data()

    def analyze_patterns(self, symbol: str = "BTCUSDT", timeframe: str = "1h") -> pd.DataFrame:
        """Analyze patterns in market data"""
        try:
            cache_key = f'patterns_{symbol}_{timeframe}'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Get market data and analyze patterns
            data = self.get_market_data(symbol, timeframe)
            if not data.empty:
                # Get raw pattern data
                patterns = self._pattern_detector.analyze_patterns(data)
                
                # Convert to DataFrame
                pattern_df = pd.DataFrame(patterns)
                if not pattern_df.empty:
                    pattern_df.index = data.index[-len(pattern_df):]
                    pattern_df['signal_strength'] = pattern_df.get('strength', 0)
                    pattern_df['signal_confidence'] = pattern_df.get('confidence', 0)
                    
                    # Cache and return
                    self._update_cache(cache_key, pattern_df)
                    return pattern_df
            
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['signal_strength', 'signal_confidence'])
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return pd.DataFrame(columns=['signal_strength', 'signal_confidence'])

    def get_technical_analysis(self, symbol: str = "BTCUSDT", timeframe: str = "1h") -> dict:
        """Get technical analysis results"""
        try:
            df = self.get_market_data(symbol=symbol, timeframe=timeframe)
            if df.empty:
                return {}
            
            # Get analysis from technical analysis module
            analysis = self._technical_analysis.analyze(df)
            
            # Add signal information
            if 'signal' in df.columns:
                latest = df.iloc[-1]
                analysis['signal'] = {
                    'direction': int(latest['signal']),
                    'strength': float(latest['signal_strength']),
                    'type': str(latest['signal_type'])
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting technical analysis: {str(e)}")
            return {}

    def _cleanup_cache(self):
        """Clean expired cache entries"""
        try:
            current_time = datetime.now()
            if (current_time - self.last_cleanup).total_seconds() < self.cleanup_interval:
                return
            
            for key in list(self.cache.keys()):
                if key in self.cache_timestamps:
                    timestamp = self.cache_timestamps[key]
                    cache_type = key.split('_')[0]
                    expiry_time = self.cache_expiry.get(cache_type, 60)
                    
                    if (current_time - timestamp).total_seconds() > expiry_time:
                        del self.cache[key]
                        del self.cache_timestamps[key]
            
            self.last_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")

    def _update_cache(self, key: str, data: any):
        """Update cache with new data"""
        try:
            self.cache[key] = data
            self.cache_timestamps[key] = datetime.now()
            self._cleanup_cache()
        except Exception as e:
            logger.error(f"Error updating cache: {str(e)}")

    def _get_from_cache(self, key: str) -> any:
        """Get data from cache if valid"""
        try:
            if key not in self.cache or key not in self.cache_timestamps:
                return None
            
            timestamp = self.cache_timestamps[key]
            cache_type = key.split('_')[0]
            expiry_time = self.cache_expiry.get(cache_type, 60)
            
            if (datetime.now() - timestamp).total_seconds() > expiry_time:
                del self.cache[key]
                del self.cache_timestamps[key]
                return None
            
            return self.cache[key]
            
        except Exception as e:
            logger.error(f"Error getting data from cache: {str(e)}")
            return None

    @property
    def binance_client(self):
        """Get the Binance client instance"""
        if not self._client:
            raise ValueError("Binance client not initialized")
        return self._client

    def get_credentials(self) -> Dict[str, str]:
        """Get API credentials from cache or config"""
        try:
            # Check cache first
            cache_key = 'credentials'
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Get credentials from config
            config_path = os.path.join(root_dir, 'Config', 'BTC.env')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            load_dotenv(config_path)
            
            credentials = {
                'api_key': os.getenv('BINANCE_API_KEY'),
                'api_secret': os.getenv('BINANCE_API_SECRET')
            }
            
            # Validate credentials
            if not credentials['api_key'] or not credentials['api_secret']:
                raise ValueError("API credentials not found in environment variables")
            
            # Store in cache
            self._update_cache(cache_key, credentials)
            
            return credentials
            
        except Exception as e:
            logger.error(f"Error getting credentials: {str(e)}")
            return {
                'api_key': None,
                'api_secret': None
            }

    def get_telegram_credentials(self) -> Dict[str, str]:
        """Get Telegram credentials from cache or config"""
        try:
            # Check cache first
            cache_key = 'telegram_credentials'
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Get credentials from config
            config_path = os.path.join(root_dir, 'Config', 'BTC.env')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            load_dotenv(config_path)
            
            credentials = {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID')
            }
            
            # Store in cache
            self._update_cache(cache_key, credentials)
            
            return credentials
            
        except Exception as e:
            logger.error(f"Error getting Telegram credentials: {str(e)}")
            return {
                'bot_token': None,
                'chat_id': None
            }

    def stop(self):
        """Stop all background threads and clean up"""
        try:
            # Clear cache
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cache cleared")
            
            # Stop threads gracefully
            if hasattr(self, '_market_thread'):
                self._market_thread.join(timeout=1)
            if hasattr(self, '_account_thread'):
                self._account_thread.join(timeout=1)
                
            logger.info("DataManager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping DataManager: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get trading metrics and statistics"""
        try:
            cache_key = 'metrics'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Calculate metrics from market data
            metrics = {}
            for symbol in self.active_symbols:
                try:
                    symbol = self._validate_symbol(symbol)
                    data = self.get_market_data(symbol)
                    if not data.empty:
                        latest = data.iloc[-1]
                        metrics[symbol] = {
                            'price': float(latest['close']),
                            'volume_24h': float(data['volume'].sum()),
                            'price_change_24h': float(data['close'].pct_change().iloc[-1] * 100),
                            'volatility': float(data['close'].pct_change().std() * 100),
                            'rsi': float(latest['rsi']),
                            'trend': latest['trend']
                        }
                except Exception as e:
                    logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
                    continue
            
            self._update_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {}

    def get_technical_indicators(self, symbol: str = "BTCUSDT", timeframe: str = "1h") -> Dict:
        """Get technical indicators for a symbol"""
        try:
            # Validate symbol format
            symbol = self._validate_symbol(symbol)
            cache_key = f'indicators_{symbol}_{timeframe}'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Get market data with indicators
            data = self.get_market_data(symbol, timeframe)
            if not data.empty:
                # Return full series for each indicator
                indicators = {
                    'rsi': data['rsi'].values.tolist(),
                    'macd': data['macd'].values.tolist(),
                    'signal': data['signal'].values.tolist(),
                    'macd_hist': data['macd_hist'].values.tolist(),
                    'ma_20': data['ma_20'].values.tolist(),
                    'ma_50': data['ma_50'].values.tolist(),
                    'bb_upper': data['bb_upper'].values.tolist(),
                    'bb_middle': data['bb_middle'].values.tolist(),
                    'bb_lower': data['bb_lower'].values.tolist(),
                    'atr': data['atr'].values.tolist(),
                    'trend': data['trend'].values.tolist(),
                    'volatility': data['volatility'].values.tolist(),
                    'volume_trend': data['volume_trend'].values.tolist()
                }
                
                self._update_cache(cache_key, indicators)
                return indicators
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting technical indicators: {str(e)}")
            return {}

    def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 20) -> Dict:
        """Get order book data"""
        try:
            # Validate symbol format
            symbol = self._validate_symbol(symbol)
            cache_key = f'order_book_{symbol}'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Get fresh order book data
            order_book = self._client.get_order_book(symbol, limit)
            if not order_book.empty:
                # Convert DataFrame to dict format
                book_data = {
                    'bids': order_book[order_book['side'] == 'bid'][['price', 'quantity']].values.tolist(),
                    'asks': order_book[order_book['side'] == 'ask'][['price', 'quantity']].values.tolist(),
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._update_cache(cache_key, book_data)
                return book_data
            
            return {'bids': [], 'asks': [], 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            
        except Exception as e:
            logger.error(f"Error getting order book: {str(e)}")
            return {'bids': [], 'asks': [], 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

    def get_volume_profile(self, timerange: str = "1d", symbol: str = "BTCUSDT") -> Dict:
        """Get volume profile analysis"""
        try:
            cache_key = f'volume_profile_{symbol}_{timerange}'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Get market data for volume profile
            limit = {"1d": 1440, "1w": 10080, "1m": 43200}.get(timerange, 1440)
            data = self.get_market_data(symbol=symbol, timeframe="1m", limit=limit)
            
            if data.empty:
                return {}
            
            # Calculate volume profile
            price_min = data['low'].min()
            price_max = data['high'].max()
            price_levels = np.linspace(price_min, price_max, 50)
            volumes = []
            
            for i in range(len(price_levels)-1):
                mask = (data['low'] >= price_levels[i]) & (data['high'] < price_levels[i+1])
                level_volume = data.loc[mask, 'volume'].sum()
                volumes.append(float(level_volume))
            
            # Find POC (Point of Control)
            poc_index = np.argmax(volumes)
            poc_price = float(price_levels[poc_index])
            
            # Calculate Value Area (70% of total volume)
            total_volume = sum(volumes)
            target_volume = total_volume * 0.7
            current_volume = volumes[poc_index]
            lower_index = upper_index = poc_index
            
            while current_volume < target_volume and (lower_index > 0 or upper_index < len(volumes)-1):
                if lower_index > 0 and (upper_index >= len(volumes)-1 or volumes[lower_index-1] > volumes[upper_index+1]):
                    lower_index -= 1
                    current_volume += volumes[lower_index]
                elif upper_index < len(volumes)-1:
                    upper_index += 1
                    current_volume += volumes[upper_index]
            
            profile = {
                'price_levels': price_levels.tolist(),
                'volumes': volumes,
                'poc_price': poc_price,
                'value_area': {
                    'low': float(price_levels[lower_index]),
                    'high': float(price_levels[upper_index])
                },
                'total_volume': float(total_volume)
            }
            
            # Update cache
            self._update_cache(cache_key, profile)
            return profile
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {}

    def get_performance_stats(self) -> Dict:
        """Get trading performance statistics"""
        try:
            cache_key = 'performance_stats'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Calculate performance metrics
            account = self.get_account_data()
            total_balance = account.get('total_balance', 0)
            available_balance = account.get('available_balance', 0)
            
            stats = {
                'balance': {
                    'total': float(total_balance),
                    'available': float(available_balance)
                },
                'pnl': {
                    'realized': float(account.get('realized_pnl', 0)),
                    'unrealized': float(account.get('unrealized_pnl', 0))
                },
                'trades': {
                    'total': int(account.get('total_trades', 0)),
                    'win_rate': float(account.get('win_rate', 0)),
                    'avg_profit': float(account.get('avg_profit', 0))
                },
                'risk': {
                    'margin_ratio': float(account.get('margin_ratio', 0)),
                    'leverage': int(account.get('leverage', 1))
                }
            }
            
            self._update_cache(cache_key, stats)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {str(e)}")
            return {}

    def get_api_status(self) -> Dict:
        """Get API connection status"""
        try:
            cache_key = 'api_status'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Test API connection
            try:
                self._test_connection()
                status = {
                    'connected': True,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'weight': self.request_weight,
                    'weight_limit': self.weight_limit
                }
            except Exception:
                status = {
                    'connected': False,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'weight': 0,
                    'weight_limit': self.weight_limit
                }
            
            self._update_cache(cache_key, status)
            return status
            
        except Exception as e:
            logger.error(f"Error getting API status: {str(e)}")
            return {
                'connected': False,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'weight': 0,
                'weight_limit': self.weight_limit
            }

    def get_training_data(self, timeframe: str = "1m", symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Get training data for ML model"""
        try:
            logger.info(f"Fetching training data for {symbol} with timeframe {timeframe}")
            
            # Verificar cache primeiro
            cache_key = f"training_data_{symbol}_{timeframe}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Map timeranges to appropriate candle settings with increased limits
            timerange_settings = {
                "1w": {"timeframe": "1h", "limit": 500},    # ~3 weeks of hourly data
                "1m": {"timeframe": "15m", "limit": 1000},  # ~10 days of 15min data
                "3m": {"timeframe": "1h", "limit": 1000},   # ~6 weeks of hourly data
                "1y": {"timeframe": "4h", "limit": 1000}    # ~6 months of 4h data
            }
            
            # Get appropriate settings or default to 1 month of 15min data
            settings = timerange_settings.get(timeframe, timerange_settings["1m"])
            
            # Get historical data directly from Binance
            data = self._get_historical_training_data(
                symbol=symbol,
                timeframe=settings["timeframe"],
                limit=settings["limit"]
            )
            
            # Validate data quality
            if data is None or data.empty:
                logger.error(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Verify data integrity
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns in historical data for {symbol}")
                return pd.DataFrame()
            
            # Verify we have enough data
            if len(data) < settings["limit"] * 0.9:  # Allow 10% missing data
                logger.warning(f"Insufficient data points for {symbol}. Expected {settings['limit']}, got {len(data)}")
                return pd.DataFrame()
            
            # Prepare features for training
            features = self.prepare_features(data)
            
            # Validate prepared features
            if features.empty or len(features.columns) < 10:  # Minimum expected features
                logger.error("Feature preparation failed or insufficient features generated")
                return pd.DataFrame()
            
            # Cache the prepared data
            self._update_cache(cache_key, features)
            
            logger.info(f"Successfully prepared {len(features)} samples with {len(features.columns)} features")
            logger.info(f"Features include: {', '.join(features.columns)}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            return pd.DataFrame()

    def cache_data(self, key: str, data: any, expiry: int = None) -> bool:
        """Cache data with optional expiry time
        Args:
            key (str): Unique identifier for the cached data
            data (any): Data to cache
            expiry (int, optional): Cache expiry time in seconds. If None, uses default expiry.
        Returns:
            bool: True if caching was successful, False otherwise
        """
        try:
            if expiry is None:
                # Get expiry based on key prefix or use default
                key_type = key.split('_')[0]
                expiry = self.cache_expiry.get(key_type, 300)  # Default 5 minutes
            
            # Store data and timestamp
            self.cache[key] = data
            self.cache_timestamps[key] = datetime.now()
            
            # Update cache expiry for this key if custom expiry provided
            if expiry is not None:
                self.cache_expiry[key] = expiry
            
            # Run cleanup to remove any expired entries
            self._cleanup_cache()
            
            logger.debug(f"Successfully cached data with key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            return False

    def clear_model_cache(self, model_id: str = None):
        """Clear model-related cache entries
        Args:
            model_id (str, optional): Specific model ID to clear cache for. If None, clears all model caches.
        """
        try:
            keys_to_remove = []
            for key in self.cache.keys():
                if model_id is None and key.startswith(('model_', 'training_', 'prediction_')):
                    keys_to_remove.append(key)
                elif model_id and key.startswith(f'model_{model_id}'):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
                if key in self.cache_expiry:
                    del self.cache_expiry[key]
            
            logger.info(f"Cleared {'all model' if model_id is None else f'model {model_id}'} cache entries")
            
        except Exception as e:
            logger.error(f"Error clearing model cache: {str(e)}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model training"""
        try:
            logger.info("Preparing features for ML model training")
            
            if data.empty:
                logger.error("Cannot prepare features from empty dataset")
                return pd.DataFrame()
            
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Verify data quality
            if df.isnull().sum().sum() > 0:
                logger.warning(f"Dataset contains {df.isnull().sum().sum()} null values")
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Technical indicators (already added by TechnicalAnalysis)
            features = pd.DataFrame(index=df.index)
            
            # Essential price features
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in df.columns:
                    features[col] = df[col]
                else:
                    logger.error(f"Missing essential column: {col}")
                    return pd.DataFrame()
            
            # Technical indicators with validation
            technical_indicators = ['rsi', 'macd', 'signal', 'macd_hist', 'ma_20', 'ma_50', 
                                  'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'volatility']
            
            for col in technical_indicators:
                if col in df.columns:
                    features[col] = df[col]
                    # Validate indicator values
                    if features[col].isnull().sum() > len(features) * 0.1:  # More than 10% missing
                        logger.warning(f"Indicator {col} has too many missing values")
            
            # Calculate additional features only if base data is valid
            if not features.empty:
                # Price changes and returns
                features['price_change'] = df['close'].diff()
                features['returns'] = df['close'].pct_change()
                features['log_returns'] = np.log(df['close']).diff()
                
                # Volatility features
                features['volatility_1h'] = df['close'].rolling(60).std()
                features['volatility_4h'] = df['close'].rolling(240).std()
                
                # Volume features
                features['volume_ma'] = df['volume'].rolling(20).mean()
                features['volume_std'] = df['volume'].rolling(20).std()
                features['volume_change'] = df['volume'].pct_change()
                
                # Price momentum
                features['momentum_1h'] = df['close'].diff(60)
                features['momentum_4h'] = df['close'].diff(240)
                
                # Target variable with validation
                features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                
                # Final validation
                features = features.dropna()
                if len(features) < len(df) * 0.8:  # Lost more than 20% of data
                    logger.warning("Significant data loss during feature preparation")
            
            logger.info(f"Successfully prepared {len(features)} samples with {len(features.columns)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _get_historical_training_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get historical data from Binance for training"""
        try:
            logger.info(f"Fetching historical data for {symbol} with timeframe {timeframe}, limit {limit}")
            
            # Get historical data directly using get_historical_data
            data = self._client.get_historical_data(
                symbol=symbol,
                interval=timeframe,  # Changed from timeframe to interval
                limit=limit
            )
            
            if data is None or data.empty:
                logger.error(f"No historical data received for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data = self._technical_analysis.add_indicators(data)
            
            logger.info(f"Successfully fetched {len(data)} historical candles")
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical training data: {str(e)}")
            return pd.DataFrame()

    def get_trading_signals(self, symbol: str = "BTCUSDT", timeframe: str = "1h") -> pd.DataFrame:
        """Get trading signals"""
        try:
            df = self.get_market_data(symbol=symbol, timeframe=timeframe)
            if df.empty:
                return pd.DataFrame()
            
            # Filter to only signal-related columns
            signal_cols = ['timestamp', 'close', 'signal', 'signal_strength', 'signal_type']
            signals_df = df[signal_cols].copy()
            
            # Filter to only rows with signals
            signals_df = signals_df[signals_df['signal'] != 0]
            
            return signals_df
            
        except Exception as e:
            logger.error(f"Error getting trading signals: {str(e)}")
            return pd.DataFrame()

    def get_backtest_results(self, symbol: str = "BTCUSDT", timeframe: str = "1h", 
                           start_date: str = None, end_date: str = None,
                           stop_loss: float = 2.0, take_profit: float = 4.0,
                           position_size: float = 100.0) -> dict:
        """Get backtest results"""
        try:
            # Get market data
            df = self.get_market_data(symbol=symbol, timeframe=timeframe)
            if df.empty:
                return {}
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Initialize results
            trades = []
            equity_curve = [position_size]
            current_position = None
            
            # Simulate trades
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Check for exit if in position
                if current_position:
                    exit_price = None
                    exit_type = None
                    
                    # Check stop loss
                    if current_position['type'] == 'long':
                        if row['low'] <= current_position['stop_loss']:
                            exit_price = current_position['stop_loss']
                            exit_type = 'stop_loss'
                        elif row['high'] >= current_position['take_profit']:
                            exit_price = current_position['take_profit']
                            exit_type = 'take_profit'
                    else:  # short position
                        if row['high'] >= current_position['stop_loss']:
                            exit_price = current_position['stop_loss']
                            exit_type = 'stop_loss'
                        elif row['low'] <= current_position['take_profit']:
                            exit_price = current_position['take_profit']
                            exit_type = 'take_profit'
                    
                    # Exit position if conditions met
                    if exit_price:
                        pnl = (exit_price - current_position['entry_price']) * position_size
                        if current_position['type'] == 'short':
                            pnl = -pnl
                            
                        trades.append({
                            'entry_time': current_position['entry_time'],
                            'exit_time': row.name,
                            'type': current_position['type'],
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'exit_type': exit_type
                        })
                        
                        equity_curve.append(equity_curve[-1] + pnl)
                        current_position = None
                
                # Check for entry if not in position
                if not current_position and row['signal'] != 0:
                    position_type = 'long' if row['signal'] > 0 else 'short'
                    entry_price = row['close']
                    
                    current_position = {
                        'type': position_type,
                        'entry_price': entry_price,
                        'entry_time': row.name,
                        'stop_loss': entry_price * (1 - stop_loss/100) if position_type == 'long' else entry_price * (1 + stop_loss/100),
                        'take_profit': entry_price * (1 + take_profit/100) if position_type == 'long' else entry_price * (1 - take_profit/100)
                    }
            
            # Calculate performance metrics
            if trades:
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t['pnl'] > 0])
                total_pnl = sum(t['pnl'] for t in trades)
                win_rate = winning_trades / total_trades * 100
                avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades else 0
                avg_loss = abs(np.mean([t['pnl'] for t in trades if t['pnl'] < 0])) if total_trades > winning_trades else 0
                profit_factor = abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) / sum(t['pnl'] for t in trades if t['pnl'] < 0)) if sum(t['pnl'] for t in trades if t['pnl'] < 0) != 0 else float('inf')
                
                return {
                    'trades': trades,
                    'equity_curve': equity_curve,
                    'metrics': {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'total_pnl': total_pnl,
                        'win_rate': win_rate,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'profit_factor': profit_factor
                    }
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting backtest results: {str(e)}")
            return {}

    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            df = self.get_market_data(symbol=symbol, timeframe="1m", limit=1)
            if df is not None and not df.empty:
                return float(df.iloc[-1]['close'])
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return None
            
    def get_previous_price(self, symbol):
        """Get previous price for a symbol"""
        try:
            df = self.get_market_data(symbol=symbol, timeframe="1m", limit=2)
            if df is not None and len(df) >= 2:
                return float(df.iloc[-2]['close'])
            return None
        except Exception as e:
            logger.error(f"Error getting previous price: {str(e)}")
            return None

    def get_cached_data(self, key: str) -> any:
        """Get data directly from cache without expiry check"""
        try:
            if key not in self.cache:
                return None
            return self.cache[key]
        except Exception as e:
            logger.error(f"Error getting cached data: {str(e)}")
            return None

    def get_trade_history(self, symbol: str = "BTCUSDT", limit: int = 100) -> list:
        """Get trade history for a symbol"""
        try:
            # Validate symbol format
            symbol = self._validate_symbol(symbol)
            cache_key = f'trade_history_{symbol}'
            
            # Check cache first
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Get fresh trade history
            trades = self._client.get_my_trades(symbol=symbol, limit=limit)
            if trades:
                self._update_cache(cache_key, trades)
                return trades
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []

    def cleanup_old_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()
            for key in list(self.cache.keys()):
                if key in self.cache_timestamps:
                    timestamp = self.cache_timestamps[key]
                    cache_type = key.split('_')[0]
                    expiry_time = self.cache_expiry.get(cache_type, 60)
                    
                    if (current_time - timestamp).total_seconds() > expiry_time:
                        del self.cache[key]
                        del self.cache_timestamps[key]
            
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up old cache: {str(e)}")

    def split_training_data(self, data: pd.DataFrame, train_split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally into train, validation and test sets."""
        try:
            if data is None or data.empty:
                raise ValueError("Input data cannot be None or empty")
            
            logger.info("Splitting data into train/val/test sets...")
            
            # Calculate split sizes
            total_size = len(data)
            validation_split = (1 - train_split) / 2  # Split remaining data equally between val and test
            
            train_size = int(total_size * train_split)
            val_size = int(total_size * validation_split)
            
            # Split data temporally
            train = data.iloc[:train_size]
            val = data.iloc[train_size:train_size + val_size]
            test = data.iloc[train_size + val_size:]
            
            # Validate splits
            if not self.validate_data_splits(train, val, test):
                raise ValueError("Data split validation failed")
            
            logger.info(f"Data split completed - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            return train, val, test
            
        except Exception as e:
            logger.error(f"Error splitting training data: {str(e)}")
            raise

    def validate_data_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> bool:
        """Validate the quality and integrity of data splits."""
        try:
            # Check for empty splits
            if train.empty or val.empty or test.empty:
                logger.error("One or more data splits are empty")
                return False
            
            # Check temporal order
            if not (train.index[-1] < val.index[0] and val.index[-1] < test.index[0]):
                logger.error("Data splits are not in correct temporal order")
                return False
            
            # Check for data leakage
            train_dates = set(train.index)
            val_dates = set(val.index)
            test_dates = set(test.index)
            
            if not (train_dates.isdisjoint(val_dates) and train_dates.isdisjoint(test_dates) and val_dates.isdisjoint(test_dates)):
                logger.error("Data leakage detected between splits")
                return False
            
            # Check for missing values
            for split_name, split_data in [("Train", train), ("Validation", val), ("Test", test)]:
                if split_data.isnull().sum().sum() > 0:
                    logger.warning(f"Missing values detected in {split_name} split")
                    return False
            
            logger.info("Data splits validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data splits: {str(e)}")
            return False

    def create_data_loader(self, data: pd.DataFrame, feature_names: List[str], feature_scaler: Any, batch_size: int = 32) -> DataLoader:
        """Create PyTorch DataLoader from DataFrame."""
        try:
            if data is None or data.empty:
                raise ValueError("Input data cannot be None or empty")
            
            # Extract features and target
            X = data[feature_names].values
            y = data['target'].values if 'target' in data.columns else np.zeros(len(data))
            
            # Scale features
            X_scaled = feature_scaler.transform(X)
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Convert to tensors on the same device
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            y_tensor = torch.FloatTensor(y).to(device)
            
            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor.unsqueeze(1))
            
            # Create data loader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=device.type == 'cpu',  # Only use pin_memory when CPU is used
                num_workers=0  # Adjust based on system
            )
            
            logger.info(f"Created DataLoader with {len(dataset)} samples on device {device}")
            return loader
            
        except Exception as e:
            logger.error(f"Error creating data loader: {str(e)}")
            raise

    def get_model_save_path(self, symbol: str, timeframe: str) -> str:
        """Get path for saving model checkpoints."""
        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Create model filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}_model.pt"
            
            # Full path
            save_path = os.path.join("models", filename)
            
            logger.info(f"Model save path: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error getting model save path: {str(e)}")
            raise

    def validate_data(self, data: pd.DataFrame, timeframe: str = None) -> Tuple[bool, str]:
        """Validate data quality and structure for training or analysis.
        
        Args:
            data (pd.DataFrame): Data to validate
            timeframe (str, optional): Timeframe of the data for time gap validation
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if data is None or data.empty:
                return False, "Data is empty or None"
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            # Check for null values
            null_counts = data[required_columns].isnull().sum()
            if null_counts.any():
                return False, f"Found null values: {null_counts[null_counts > 0].to_dict()}"
            
            # Check for invalid values
            if (data['high'] < data['low']).any():
                invalid_rows = data[data['high'] < data['low']].index.tolist()
                return False, f"Invalid high-low relationship at rows: {invalid_rows}"
            
            if (data['volume'] <= 0).any():
                invalid_rows = data[data['volume'] <= 0].index.tolist()
                return False, f"Non-positive volume values at rows: {invalid_rows}"
            
            if ((data['close'] < data['low']) | (data['close'] > data['high'])).any():
                invalid_rows = data[(data['close'] < data['low']) | (data['close'] > data['high'])].index.tolist()
                return False, f"Close price outside high-low range at rows: {invalid_rows}"
            
            if ((data['open'] < data['low']) | (data['open'] > data['high'])).any():
                invalid_rows = data[(data['open'] < data['low']) | (data['open'] > data['high'])].index.tolist()
                return False, f"Open price outside high-low range at rows: {invalid_rows}"
            
            # Check for time gaps if timeframe is provided
            if timeframe and data.index.dtype.kind == 'M' and len(data) > 1:
                time_diff = pd.Series(data.index).diff()
                expected_diff = pd.Timedelta(minutes=1)  # Default for 1m
                
                # Map timeframe to expected difference
                timeframe_map = {
                    '1m': pd.Timedelta(minutes=1),
                    '5m': pd.Timedelta(minutes=5),
                    '15m': pd.Timedelta(minutes=15),
                    '1h': pd.Timedelta(hours=1),
                    '4h': pd.Timedelta(hours=4),
                    '1d': pd.Timedelta(days=1)
                }
                
                if timeframe in timeframe_map:
                    expected_diff = timeframe_map[timeframe]
                
                large_gaps = time_diff[time_diff > expected_diff * 2]
                if not large_gaps.empty:
                    logger.warning(f"Found {len(large_gaps)} time gaps larger than {expected_diff * 2}")
            
            # Check for extreme price movements (potential data errors)
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.5]  # 50% change
            if not extreme_changes.empty:
                logger.warning(f"Found {len(extreme_changes)} extreme price changes > 50%")
            
            # Check for sufficient data points
            min_required_points = 100  # Minimum required for meaningful analysis
            if len(data) < min_required_points:
                return False, f"Insufficient data points: {len(data)} < {min_required_points}"
            
            # All checks passed
            return True, "Data validation passed"
            
        except Exception as e:
            error_msg = f"Data validation error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg