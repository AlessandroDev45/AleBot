import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Class responsible for technical analysis calculations"""
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        try:
            if df.empty:
                return df
            
            df = df.copy()
            
            # Make sure we have numeric data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate Moving Averages
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal']
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Add market structure analysis
            df = self._add_market_structure(df)
            
            # Generate trading signals
            df = self._generate_signals(df)
            
            # Fill NaN values
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {str(e)}")
            return df

    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure analysis"""
        try:
            if df.empty:
                return df
            
            # Trend Analysis
            df['trend'] = np.where(df['ma_20'] > df['ma_50'], 'Bullish',
                                 np.where(df['ma_20'] < df['ma_50'], 'Bearish', 'Ranging'))
            
            # Volatility Analysis
            df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
            
            # Volume Analysis
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_trend'] = np.where(df['volume'] > df['volume_sma'], 'High', 'Low')
            
            # Momentum Analysis
            df['momentum'] = df['close'].diff(periods=14)
            df['momentum_sma'] = df['momentum'].rolling(window=14).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding market structure: {str(e)}")
            return df

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on indicators"""
        try:
            if df.empty:
                return df
                
            # Initialize signal columns
            df['signal'] = 0  # 1 for buy, -1 for sell, 0 for neutral
            df['signal_strength'] = 0.0  # Signal strength from 0 to 1
            df['signal_type'] = ''  # Type of signal that triggered
            
            # RSI signals with dynamic thresholds
            rsi_lower = 30
            rsi_upper = 70
            df.loc[df['rsi'] < rsi_lower, 'signal'] = 1
            df.loc[df['rsi'] > rsi_upper, 'signal'] = -1
            
            # MACD signals with trend confirmation
            macd_cross_up = (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))
            macd_cross_down = (df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))
            
            # Only consider MACD signals with trend confirmation
            df.loc[macd_cross_up & (df['ma_20'] > df['ma_50']), 'signal'] = 1
            df.loc[macd_cross_down & (df['ma_20'] < df['ma_50']), 'signal'] = -1
            
            # Bollinger Bands signals with volume confirmation
            bb_buy = df['close'] < df['bb_lower']
            bb_sell = df['close'] > df['bb_upper']
            volume_spike = df['volume'] > df['volume'].rolling(20).mean() * 1.5
            
            df.loc[bb_buy & volume_spike, 'signal'] = 1
            df.loc[bb_sell & volume_spike, 'signal'] = -1
            
            # Moving Average signals with momentum confirmation
            ma_cross_up = (df['ma_20'] > df['ma_50']) & (df['ma_20'].shift(1) <= df['ma_50'].shift(1))
            ma_cross_down = (df['ma_20'] < df['ma_50']) & (df['ma_20'].shift(1) >= df['ma_50'].shift(1))
            momentum_up = df['momentum'] > 0
            momentum_down = df['momentum'] < 0
            
            df.loc[ma_cross_up & momentum_up, 'signal'] = 1
            df.loc[ma_cross_down & momentum_down, 'signal'] = -1
            
            # Calculate signal strength based on multiple confirmations
            confirmations = pd.DataFrame(index=df.index)
            
            # RSI confirmation
            confirmations['rsi'] = ((df['rsi'] < rsi_lower) | (df['rsi'] > rsi_upper)).astype(float)
            
            # MACD confirmation
            confirmations['macd'] = (macd_cross_up | macd_cross_down).astype(float)
            
            # Bollinger Bands confirmation
            confirmations['bb'] = ((bb_buy | bb_sell) & volume_spike).astype(float)
            
            # Moving Average confirmation
            confirmations['ma'] = ((ma_cross_up & momentum_up) | (ma_cross_down & momentum_down)).astype(float)
            
            # Weight the confirmations
            weights = {
                'rsi': 0.3,
                'macd': 0.3,
                'bb': 0.2,
                'ma': 0.2
            }
            
            # Calculate weighted signal strength
            df['signal_strength'] = sum(confirmations[indicator] * weight 
                                      for indicator, weight in weights.items())
            
            # Set signal type based on strongest indicator
            for i in df.index:
                max_confirmation = None
                max_value = 0
                
                for indicator in weights.keys():
                    if confirmations.loc[i, indicator] > max_value:
                        max_value = confirmations.loc[i, indicator]
                        max_confirmation = indicator
                
                if max_confirmation:
                    df.loc[i, 'signal_type'] = max_confirmation.upper()
            
            # Ensure signal strength is between 0 and 1
            df['signal_strength'] = df['signal_strength'].clip(0, 1)
            
            # Add trend context
            df.loc[(df['signal'] > 0) & (df['trend'] == 'Bearish'), 'signal_strength'] *= 0.8
            df.loc[(df['signal'] < 0) & (df['trend'] == 'Bullish'), 'signal_strength'] *= 0.8
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return df

    def analyze(self, df: pd.DataFrame) -> dict:
        """Perform technical analysis on data"""
        try:
            if df.empty:
                return {}
            
            # Get latest values
            latest = df.iloc[-1]
            
            return {
                'price': {
                    'current': float(latest['close']),
                    'change': float(df['close'].pct_change().iloc[-1] * 100),
                    'trend': latest['trend']
                },
                'indicators': {
                    'rsi': float(latest['rsi']),
                    'macd': {
                        'value': float(latest['macd']),
                        'signal': float(latest['signal']),
                        'histogram': float(latest['macd_hist'])
                    },
                    'bollinger': {
                        'upper': float(latest['bb_upper']),
                        'middle': float(latest['bb_middle']),
                        'lower': float(latest['bb_lower'])
                    },
                    'volume': {
                        'current': float(latest['volume']),
                        'trend': latest['volume_trend'],
                        'change': float(df['volume'].pct_change().iloc[-1] * 100)
                    },
                    'market_structure': {
                        'trend': latest['trend'],
                        'volatility': float(latest['volatility']),
                        'momentum': float(latest['momentum'])
                    }
                },
                'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return {} 