import os
import threading
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

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

    def __init__(self):
        """Initialize the DataManager."""
        if not hasattr(self, 'initialized'):
            self.cache = {}
            self.cache_timestamps = {}
            self.cache_expiry = {
                'market': 300,  # 5 minutes for market data
                'model': 3600,  # 1 hour for model data
                'analysis': 1800,  # 30 minutes for analysis data
                'default': 60  # 1 minute default
            }
            self.initialized = True
            logger.info("DataManager initialized")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        try:
            if data is None or data.empty:
                raise ValueError("Input data is None or empty")

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Required: {required_columns}")

            # Calculate technical indicators
            features = pd.DataFrame(index=data.index)
            
            # Price features
            features['price_change'] = data['close'].pct_change()
            features['price_range'] = (data['high'] - data['low']) / data['close']
            features['price_momentum'] = data['close'].pct_change(5)
            
            # Volume features
            features['volume_change'] = data['volume'].pct_change()
            features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Moving averages
            for period in [5, 10, 20, 50, 100]:
                features[f'ma_{period}'] = data['close'].rolling(period).mean()
                features[f'ma_{period}_slope'] = features[f'ma_{period}'].pct_change(5)
            
            # Volatility
            features['volatility'] = data['close'].rolling(20).std()
            features['volatility_change'] = features['volatility'].pct_change()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = data['close'].rolling(bb_period).mean()
            bb_std = data['close'].rolling(bb_period).std()
            features['bb_upper'] = bb_ma + (bb_std * bb_std)
            features['bb_lower'] = bb_ma - (bb_std * bb_std)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_ma
            
            # Target variable (next period return)
            features['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
            
            # Drop NaN values
            features = features.dropna()
            
            # Validate prepared features
            if not self.validate_data(features):
                raise ValueError("Feature validation failed")
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def get_training_data(self, timeframe: str, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get historical data for model training."""
        try:
            # Try to get from cache first
            cache_key = f"{symbol}_{timeframe}_training_data"
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

            # If not in cache, fetch from API
            data = self.binance_client.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                'buy_quote_volume', 'ignore'
            ])

            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Cache the data
            self._update_cache(cache_key, df)

            return df

        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            raise

    def validate_data_splits(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
        """Validate the data splits for model training."""
        try:
            # Check if any split is empty
            if train_data.empty or val_data.empty or test_data.empty:
                logger.error("One or more data splits are empty")
                return False
            
            # Check for minimum required samples
            min_samples = 100  # Minimum samples required for each split
            if len(train_data) < min_samples or len(val_data) < min_samples or len(test_data) < min_samples:
                logger.error(f"Insufficient samples in data splits. Minimum required: {min_samples}")
                return False
            
            # Verify feature consistency
            train_cols = set(train_data.columns)
            val_cols = set(val_data.columns)
            test_cols = set(test_data.columns)
            
            if train_cols != val_cols or train_cols != test_cols:
                logger.error("Inconsistent features across data splits")
                return False
            
            # Check for data leakage (timestamps should not overlap)
            if 'timestamp' in train_data.columns:
                train_dates = set(train_data['timestamp'])
                val_dates = set(val_data['timestamp'])
                test_dates = set(test_data['timestamp'])
                
                if len(train_dates.intersection(val_dates)) > 0 or len(train_dates.intersection(test_dates)) > 0:
                    logger.error("Data leakage detected: overlapping timestamps between splits")
                    return False
            
            # Verify data types consistency
            for col in train_cols:
                if (train_data[col].dtype != val_data[col].dtype or 
                    train_data[col].dtype != test_data[col].dtype):
                    logger.error(f"Inconsistent data types for column {col}")
                    return False
            
            # Check for NaN values
            if (train_data.isna().any().any() or 
                val_data.isna().any().any() or 
                test_data.isna().any().any()):
                logger.error("NaN values detected in data splits")
                return False
            
            # Verify target variable distribution
            if 'target' in train_cols:
                train_target_dist = train_data['target'].value_counts(normalize=True)
                val_target_dist = val_data['target'].value_counts(normalize=True)
                test_target_dist = test_data['target'].value_counts(normalize=True)
                
                # Check for extreme class imbalance
                if (abs(train_target_dist.max() - train_target_dist.min()) > 0.9 or
                    abs(val_target_dist.max() - val_target_dist.min()) > 0.9 or
                    abs(test_target_dist.max() - test_target_dist.min()) > 0.9):
                    logger.warning("Extreme class imbalance detected in data splits")
            
            logger.info("Data splits validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data splits: {str(e)}")
            return False

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality and structure."""
        try:
            if data is None or data.empty:
                logger.error("Data is None or empty")
                return False
            
            # Check for minimum required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns. Required: {required_columns}")
                return False
            
            # Check for NaN values
            if data.isna().any().any():
                logger.error("NaN values detected in data")
                return False
            
            # Check for infinite values
            if np.isinf(data.select_dtypes(include=np.number).values).any():
                logger.error("Infinite values detected in data")
                return False
            
            # Verify data types
            numeric_columns = data.select_dtypes(include=np.number).columns
            if len(numeric_columns) < len(required_columns):
                logger.error("Non-numeric values in required columns")
                return False
            
            # Check for monotonic timestamp if present
            if 'timestamp' in data.columns:
                if not data['timestamp'].is_monotonic_increasing:
                    logger.error("Timestamps are not monotonically increasing")
                    return False
            
            # Verify price consistency
            if not (data['high'] >= data['low']).all():
                logger.error("High prices are not consistently >= low prices")
                return False
            
            if not ((data['high'] >= data['open']) & (data['high'] >= data['close'])).all():
                logger.error("High prices are not consistently >= open/close prices")
                return False
            
            if not ((data['low'] <= data['open']) & (data['low'] <= data['close'])).all():
                logger.error("Low prices are not consistently <= open/close prices")
                return False
            
            # Check for reasonable price ranges
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if (data[col] <= 0).any():
                    logger.error(f"Non-positive values found in {col} prices")
                    return False
                
                # Check for extreme price changes
                pct_change = abs(data[col].pct_change())
                if (pct_change > 0.5).any():  # 50% price change threshold
                    logger.warning(f"Large price changes detected in {col}")
            
            # Verify volume consistency
            if (data['volume'] < 0).any():
                logger.error("Negative volumes detected")
                return False
            
            logger.info("Data validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False

    def cache_data(self, key: str, data: Any) -> None:
        """Cache data with the given key."""
        try:
            self.cache[key] = data
            self.cache_timestamps[key] = datetime.now()
            logger.info(f"Data cached successfully with key: {key}")
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data for the given key."""
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
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None

    def clear_model_cache(self) -> None:
        """Clear model-related cache entries."""
        try:
            model_keys = [key for key in self.cache.keys() if key.startswith('model_')]
            for key in model_keys:
                del self.cache[key]
                del self.cache_timestamps[key]
            logger.info("Model cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing model cache: {str(e)}")

    def create_data_loader(self, data: pd.DataFrame, feature_names: List[str], scaler: Any, batch_size: int = 32) -> DataLoader:
        """Create a DataLoader for model training."""
        try:
            # Extract features and target
            X = data[feature_names].values
            y = data['target'].values if 'target' in data.columns else None
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            if y is not None:
                y_tensor = torch.FloatTensor(y)
                dataset = TensorDataset(X_tensor, y_tensor)
            else:
                dataset = TensorDataset(X_tensor)
                
            # Create DataLoader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if y is not None else False,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=True  # Speed up data transfer to GPU
            )
            
            return loader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader: {str(e)}")
            raise

    def split_training_data(self, data: pd.DataFrame, train_split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training, validation and test sets."""
        try:
            # Sort data by index to ensure chronological order
            data = data.sort_index()
            
            # Calculate split points
            train_size = int(len(data) * train_split)
            val_size = int(len(data) * 0.1)  # 10% for validation
            
            # Split data
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:train_size + val_size]
            test_data = data.iloc[train_size + val_size:]
            
            # Validate splits
            if not self.validate_data_splits(train_data, val_data, test_data):
                raise ValueError("Data split validation failed")
                
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting training data: {str(e)}")
            raise

    def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear cached data. If key is provided, only clear that specific cache."""
        try:
            if key is not None:
                if key in self.cache:
                    del self.cache[key]
                    del self.cache_timestamps[key]
                    logger.info(f"Cleared cache for key: {key}")
            else:
                self.cache.clear()
                self.cache_timestamps.clear()
                logger.info("Cleared all cache")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Internal method to get data from cache with logging."""
        try:
            data = self.get_cached_data(key)
            if data is not None:
                logger.info(f"Using cached data for {key}")
            return data
        except Exception as e:
            logger.error(f"Error accessing cache: {str(e)}")
            return None

    def _update_cache(self, key: str, data: Any) -> None:
        """Internal method to update cache with logging."""
        try:
            self.cache_data(key, data)
            logger.info(f"Updated cache for {key}")
        except Exception as e:
            logger.error(f"Error updating cache: {str(e)}")

    def get_equity_curve(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """Get equity curve data for performance analysis."""
        try:
            # Get historical trades
            trades = self._get_historical_trades(symbol, timeframe, limit)
            
            # Calculate cumulative returns
            if trades is None or trades.empty:
                return pd.DataFrame(columns=['timestamp', 'equity', 'drawdown'])
            
            initial_balance = 10000  # Starting balance for simulation
            equity = [initial_balance]
            timestamps = [trades.index[0]]
            
            for i in range(len(trades)):
                trade = trades.iloc[i]
                pnl = trade.get('pnl', 0)
                equity.append(equity[-1] * (1 + pnl))
                timestamps.append(trade.name)
            
            # Create equity curve DataFrame
            equity_curve = pd.DataFrame({
                'timestamp': timestamps,
                'equity': equity[:-1]  # Remove last placeholder
            })
            
            # Calculate drawdown
            equity_curve['peak'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['equity']) / equity_curve['peak'] * 100
            
            # Add performance metrics
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            equity_curve['cumulative_returns'] = (1 + equity_curve['returns']).cumprod() - 1
            
            return equity_curve
            
        except Exception as e:
            logger.error(f"Error calculating equity curve: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'equity', 'drawdown'])

    def _get_historical_trades(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get historical trades data."""
        try:
            cache_key = f"{symbol}_{timeframe}_trades"
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

            # Get trades from database or trading history
            trades = pd.DataFrame()  # Initialize empty DataFrame
            
            # Cache the data
            self._update_cache(cache_key, trades)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting historical trades: {str(e)}")
            return pd.DataFrame()

    def validate_pattern_data(self, data: pd.DataFrame) -> bool:
        """Validate data for pattern detection."""
        try:
            if data is None or data.empty:
                logger.error("Pattern data is None or empty")
                return False
                
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns for pattern detection. Required: {required_columns}")
                return False
                
            # Check for minimum required data points
            min_points = 100  # Minimum points needed for reliable pattern detection
            if len(data) < min_points:
                logger.error(f"Insufficient data points for pattern detection. Found: {len(data)}, Required: {min_points}")
                return False
                
            # Verify data consistency
            if not (data['high'] >= data['low']).all():
                logger.error("Invalid price data: high prices not consistently >= low prices")
                return False
                
            if not ((data['high'] >= data['open']) & (data['high'] >= data['close'])).all():
                logger.error("Invalid price data: high prices not consistently >= open/close prices")
                return False
                
            if not ((data['low'] <= data['open']) & (data['low'] <= data['close'])).all():
                logger.error("Invalid price data: low prices not consistently <= open/close prices")
                return False
                
            # Check for reasonable price ranges
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if (data[col] <= 0).any():
                    logger.error(f"Invalid price data: non-positive values found in {col}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating pattern data: {str(e)}")
            return False