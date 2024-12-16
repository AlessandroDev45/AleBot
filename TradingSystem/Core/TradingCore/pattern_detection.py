import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class PatternDetector:
    """Class responsible for detecting chart patterns"""
    
    def __init__(self, data_manager):
        """Initialize pattern detector"""
        self.data_manager = data_manager
        
    def analyze_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze patterns in market data"""
        try:
            if df.empty:
                return []
            
            patterns = []
            
            # Detect candlestick patterns
            patterns.extend(self._detect_candlestick_patterns(df))
            
            # Detect chart patterns
            patterns.extend(self._detect_chart_patterns(df))
            
            # Calculate pattern strengths and confidence
            for pattern in patterns:
                self._calculate_pattern_metrics(pattern, df)
                
            # Convert patterns to signals
            self._add_pattern_signals(df, patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return []
            
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns"""
        try:
            patterns = []
            
            # Get last 3 candles for pattern detection
            last_candles = df.tail(3)
            
            # Doji pattern
            if self._is_doji(last_candles.iloc[-1]):
                patterns.append({
                    'type': 'candlestick',
                    'name': 'Doji',
                    'position': len(df) - 1
                })
            
            # Hammer pattern
            if self._is_hammer(last_candles.iloc[-1]):
                patterns.append({
                    'type': 'candlestick',
                    'name': 'Hammer',
                    'position': len(df) - 1
                })
            
            # Engulfing pattern
            if len(last_candles) >= 2:
                if self._is_engulfing(last_candles.iloc[-2:]):
                    patterns.append({
                        'type': 'candlestick',
                        'name': 'Engulfing',
                        'position': len(df) - 1
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return []
            
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns"""
        try:
            patterns = []
            
            # Get last 20 candles for pattern detection
            last_candles = df.tail(20)
            
            # Double top pattern
            if self._is_double_top(last_candles):
                patterns.append({
                    'type': 'chart',
                    'name': 'Double Top',
                    'position': len(df) - 1
                })
            
            # Double bottom pattern
            if self._is_double_bottom(last_candles):
                patterns.append({
                    'type': 'chart',
                    'name': 'Double Bottom',
                    'position': len(df) - 1
                })
            
            # Head and shoulders pattern
            if self._is_head_and_shoulders(last_candles):
                patterns.append({
                    'type': 'chart',
                    'name': 'Head and Shoulders',
                    'position': len(df) - 1
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {str(e)}")
            return []
            
    def _calculate_pattern_metrics(self, pattern: Dict, df: pd.DataFrame):
        """Calculate pattern strength and confidence"""
        try:
            # Get relevant data for the pattern
            position = pattern['position']
            pattern_data = df.iloc[max(0, position-20):position+1]
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(pattern_data)
            
            # Calculate volume confirmation
            volume_confirmation = self._calculate_volume_confirmation(pattern_data)
            
            # Calculate pattern reliability based on historical performance
            reliability = self._calculate_pattern_reliability(pattern['name'])
            
            # Calculate final strength and confidence
            pattern['strength'] = (trend_strength + volume_confirmation) / 2
            pattern['confidence'] = reliability * pattern['strength']
            
        except Exception as e:
            logger.error(f"Error calculating pattern metrics: {str(e)}")
            pattern['strength'] = 0
            pattern['confidence'] = 0
            
    def _is_doji(self, candle: pd.Series) -> bool:
        """Check if candle is a doji"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            total_size = candle['high'] - candle['low']
            
            return body_size <= (total_size * 0.1)
            
        except Exception as e:
            logger.error(f"Error checking doji pattern: {str(e)}")
            return False
            
    def _is_hammer(self, candle: pd.Series) -> bool:
        """Check if candle is a hammer"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            
            return (lower_wick > (body_size * 2)) and (upper_wick < (body_size * 0.5))
            
        except Exception as e:
            logger.error(f"Error checking hammer pattern: {str(e)}")
            return False
            
    def _is_engulfing(self, candles: pd.DataFrame) -> bool:
        """Check if pattern is engulfing"""
        try:
            prev_candle = candles.iloc[0]
            curr_candle = candles.iloc[1]
            
            prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
            curr_body_size = abs(curr_candle['close'] - curr_candle['open'])
            
            is_bullish = (prev_candle['close'] < prev_candle['open']) and (curr_candle['close'] > curr_candle['open'])
            is_bearish = (prev_candle['close'] > prev_candle['open']) and (curr_candle['close'] < curr_candle['open'])
            
            return (curr_body_size > prev_body_size) and (is_bullish or is_bearish)
            
        except Exception as e:
            logger.error(f"Error checking engulfing pattern: {str(e)}")
            return False
            
    def _is_double_top(self, df: pd.DataFrame) -> bool:
        """Check for double top pattern"""
        try:
            highs = df['high'].values
            peaks = self._find_peaks(highs)
            
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                peak_prices = highs[last_two_peaks]
                
                price_diff = abs(peak_prices[0] - peak_prices[1])
                avg_price = np.mean(peak_prices)
                
                return price_diff <= (avg_price * 0.01)  # 1% tolerance
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking double top pattern: {str(e)}")
            return False
            
    def _is_double_bottom(self, df: pd.DataFrame) -> bool:
        """Check for double bottom pattern"""
        try:
            lows = df['low'].values
            troughs = self._find_troughs(lows)
            
            if len(troughs) >= 2:
                last_two_troughs = troughs[-2:]
                trough_prices = lows[last_two_troughs]
                
                price_diff = abs(trough_prices[0] - trough_prices[1])
                avg_price = np.mean(trough_prices)
                
                return price_diff <= (avg_price * 0.01)  # 1% tolerance
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking double bottom pattern: {str(e)}")
            return False
            
    def _is_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """Check for head and shoulders pattern"""
        try:
            highs = df['high'].values
            peaks = self._find_peaks(highs)
            
            if len(peaks) >= 3:
                last_three_peaks = peaks[-3:]
                peak_prices = highs[last_three_peaks]
                
                middle_higher = (peak_prices[1] > peak_prices[0]) and (peak_prices[1] > peak_prices[2])
                shoulders_similar = abs(peak_prices[0] - peak_prices[2]) <= (np.mean(peak_prices) * 0.01)
                
                return middle_higher and shoulders_similar
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking head and shoulders pattern: {str(e)}")
            return False
            
    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """Find peaks in data"""
        try:
            peaks = []
            for i in range(1, len(data)-1):
                if (data[i] > data[i-1]) and (data[i] > data[i+1]):
                    peaks.append(i)
            return peaks
            
        except Exception as e:
            logger.error(f"Error finding peaks: {str(e)}")
            return []
            
    def _find_troughs(self, data: np.ndarray) -> List[int]:
        """Find troughs in data"""
        try:
            troughs = []
            for i in range(1, len(data)-1):
                if (data[i] < data[i-1]) and (data[i] < data[i+1]):
                    troughs.append(i)
            return troughs
            
        except Exception as e:
            logger.error(f"Error finding troughs: {str(e)}")
            return []
            
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            # Use ADX indicator for trend strength
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            dm_plus = df['high'].diff()
            dm_minus = -df['low'].diff()
            
            dm_plus = dm_plus.where(dm_plus > 0, 0)
            dm_minus = dm_minus.where(dm_minus > 0, 0)
            
            di_plus = 100 * (dm_plus.rolling(window=14).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(window=14).mean() / atr)
            
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=14).mean()
            
            return float(adx.iloc[-1]) / 100
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0
            
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """Calculate volume confirmation"""
        try:
            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            
            return min(1.0, current_volume / avg_volume)
            
        except Exception as e:
            logger.error(f"Error calculating volume confirmation: {str(e)}")
            return 0
            
    def _calculate_pattern_reliability(self, pattern_name: str) -> float:
        """Calculate pattern reliability based on historical performance"""
        try:
            # Historical reliability scores (could be improved with actual backtest data)
            reliability_scores = {
                'Doji': 0.6,
                'Hammer': 0.7,
                'Engulfing': 0.8,
                'Double Top': 0.75,
                'Double Bottom': 0.75,
                'Head and Shoulders': 0.8
            }
            
            return reliability_scores.get(pattern_name, 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating pattern reliability: {str(e)}")
            return 0.5
            
    def _add_pattern_signals(self, df: pd.DataFrame, patterns: List[Dict]):
        """Add pattern-based signals to DataFrame"""
        try:
            if df.empty or not patterns:
                return
                
            # Initialize pattern signal columns if they don't exist
            if 'signal' not in df.columns:
                df['signal'] = 0
            if 'signal_strength' not in df.columns:
                df['signal_strength'] = 0.0
            if 'signal_type' not in df.columns:
                df['signal_type'] = ''
                
            for pattern in patterns:
                position = pattern['position']
                
                # Skip if position is out of range
                if position >= len(df):
                    continue
                    
                # Determine signal direction based on pattern type and trend
                signal = self._get_pattern_signal(pattern)
                
                # Calculate signal strength based on multiple factors
                strength = pattern.get('confidence', 0.5) * pattern.get('strength', 0.5)
                
                # Enhance signal strength based on trend confirmation
                if signal > 0 and df.iloc[position]['trend'] == 'Bullish':
                    strength *= 1.2
                elif signal < 0 and df.iloc[position]['trend'] == 'Bearish':
                    strength *= 1.2
                
                # Volume confirmation
                volume_ratio = df.iloc[position]['volume'] / df['volume'].rolling(20).mean().iloc[position]
                if volume_ratio > 1.5:
                    strength *= 1.1
                
                # Only update if signal is stronger than existing
                if abs(signal) * strength > abs(df.loc[position, 'signal_strength']):
                    df.loc[position, 'signal'] = signal
                    df.loc[position, 'signal_strength'] = abs(signal) * strength
                    df.loc[position, 'signal_type'] = f"Pattern_{pattern['name']}"
                    
        except Exception as e:
            logger.error(f"Error adding pattern signals: {str(e)}")

    def _get_pattern_signal(self, pattern: Dict) -> int:
        """Get signal direction from pattern"""
        try:
            # Bullish patterns
            bullish_patterns = [
                'Hammer',
                'Double Bottom',
                'Inverse Head and Shoulders',
                'Bullish Engulfing',
                'Morning Star',
                'Piercing Line'
            ]
            
            # Bearish patterns
            bearish_patterns = [
                'Shooting Star',
                'Double Top',
                'Head and Shoulders',
                'Bearish Engulfing',
                'Evening Star',
                'Dark Cloud Cover'
            ]
            
            # Neutral patterns that need trend confirmation
            neutral_patterns = [
                'Doji',
                'Spinning Top',
                'Long Legged Doji'
            ]
            
            if pattern['name'] in bullish_patterns:
                return 1
            elif pattern['name'] in bearish_patterns:
                return -1
            elif pattern['name'] in neutral_patterns:
                # For neutral patterns, use trend direction
                if pattern.get('trend') == 'Bullish':
                    return 0.5
                elif pattern.get('trend') == 'Bearish':
                    return -0.5
            return 0
                
        except Exception as e:
            logger.error(f"Error getting pattern signal: {str(e)}")
            return 0