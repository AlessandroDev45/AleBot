import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

class PatternDetector:
    """Technical pattern detection"""
    
    def __init__(self):
        """Initialize pattern detector"""
        self.patterns = {
            'engulfing': self._check_engulfing,
            'doji': self._check_doji,
            'hammer': self._check_hammer,
            'shooting_star': self._check_shooting_star
        }
        
    def analyze_patterns(self, df):
        """Analyze candlestick patterns in the dataframe"""
        signals = []
        for pattern_name, pattern_func in self.patterns.items():
            pattern_signals = pattern_func(df)
            signals.extend(pattern_signals)
        return signals
    
    def _check_engulfing(self, df):
        """Check for bullish and bearish engulfing patterns"""
        signals = []
        for i in range(1, len(df)):
            # Bullish engulfing
            if (df['Close'][i] > df['Open'][i] and  # Current candle is bullish
                df['Close'][i-1] < df['Open'][i-1] and  # Previous candle is bearish
                df['Close'][i] > df['Open'][i-1] and  # Current close above previous open
                df['Open'][i] < df['Close'][i-1]):  # Current open below previous close
                signals.append({
                    'pattern': 'bullish_engulfing',
                    'position': 'long',
                    'timestamp': df.index[i],
                    'price': df['Close'][i]
                })
            
            # Bearish engulfing
            elif (df['Close'][i] < df['Open'][i] and  # Current candle is bearish
                  df['Close'][i-1] > df['Open'][i-1] and  # Previous candle is bullish
                  df['Close'][i] < df['Open'][i-1] and  # Current close below previous open
                  df['Open'][i] > df['Close'][i-1]):  # Current open above previous close
                signals.append({
                    'pattern': 'bearish_engulfing',
                    'position': 'short',
                    'timestamp': df.index[i],
                    'price': df['Close'][i]
                })
        return signals
    
    def _check_doji(self, df):
        """Check for doji patterns"""
        signals = []
        for i in range(len(df)):
            body_size = abs(df['Close'][i] - df['Open'][i])
            wick_size = df['High'][i] - df['Low'][i]
            
            if body_size <= (wick_size * 0.1):  # Body is very small compared to wicks
                signals.append({
                    'pattern': 'doji',
                    'position': 'neutral',
                    'timestamp': df.index[i],
                    'price': df['Close'][i]
                })
        return signals
    
    def _check_hammer(self, df):
        """Check for hammer patterns"""
        signals = []
        for i in range(len(df)):
            body_size = abs(df['Close'][i] - df['Open'][i])
            upper_wick = df['High'][i] - max(df['Open'][i], df['Close'][i])
            lower_wick = min(df['Open'][i], df['Close'][i]) - df['Low'][i]
            
            if (lower_wick > (body_size * 2) and  # Long lower wick
                upper_wick <= (body_size * 0.1)):  # Very small or no upper wick
                signals.append({
                    'pattern': 'hammer',
                    'position': 'long',
                    'timestamp': df.index[i],
                    'price': df['Close'][i]
                })
        return signals
    
    def _check_shooting_star(self, df):
        """Check for shooting star patterns"""
        signals = []
        for i in range(len(df)):
            body_size = abs(df['Close'][i] - df['Open'][i])
            upper_wick = df['High'][i] - max(df['Open'][i], df['Close'][i])
            lower_wick = min(df['Open'][i], df['Close'][i]) - df['Low'][i]
            
            if (upper_wick > (body_size * 2) and  # Long upper wick
                lower_wick <= (body_size * 0.1)):  # Very small or no lower wick
                signals.append({
                    'pattern': 'shooting_star',
                    'position': 'short',
                    'timestamp': df.index[i],
                    'price': df['Close'][i]
                })
        return signals