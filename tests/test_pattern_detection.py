import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from TradingSystem.Core.TradingCore.pattern_detection import PatternDetector, PatternType

class TestPatternDetector(unittest.TestCase):
    def setUp(self):
        """Setup test data"""
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        self.df = pd.DataFrame({
            'Open': np.random.uniform(45000, 46000, 100),
            'High': np.random.uniform(45500, 46500, 100),
            'Low': np.random.uniform(44500, 45500, 100),
            'Close': np.random.uniform(45000, 46000, 100),
            'Volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        # Ensure high is highest and low is lowest
        self.df['High'] = self.df[['Open', 'High', 'Close']].max(axis=1)
        self.df['Low'] = self.df[['Open', 'Low', 'Close']].min(axis=1)
        
        # Create specific patterns
        self._create_engulfing_pattern()
        self._create_doji_pattern()
        self._create_pin_bar_pattern()
        
        self.detector = PatternDetector()
        
    def _create_engulfing_pattern(self):
        """Create bullish and bearish engulfing patterns"""
        # Bullish engulfing
        self.df.loc[self.df.index[10], :] = {
            'Open': 45500,
            'High': 45600,
            'Low': 45200,
            'Close': 45300,  # Bearish candle
            'Volume': 500
        }
        self.df.loc[self.df.index[11], :] = {
            'Open': 45200,
            'High': 45700,
            'Low': 45100,
            'Close': 45600,  # Bullish engulfing
            'Volume': 800
        }
        
        # Bearish engulfing
        self.df.loc[self.df.index[20], :] = {
            'Open': 45300,
            'High': 45600,
            'Low': 45200,
            'Close': 45500,  # Bullish candle
            'Volume': 500
        }
        self.df.loc[self.df.index[21], :] = {
            'Open': 45600,
            'High': 45700,
            'Low': 45100,
            'Close': 45200,  # Bearish engulfing
            'Volume': 800
        }
        
    def _create_doji_pattern(self):
        """Create doji patterns"""
        # Regular doji
        self.df.loc[self.df.index[30], :] = {
            'Open': 45400,
            'High': 45600,
            'Low': 45200,
            'Close': 45400,
            'Volume': 500
        }
        
        # Dragonfly doji
        self.df.loc[self.df.index[31], :] = {
            'Open': 45400,
            'High': 45450,
            'Low': 45000,
            'Close': 45400,
            'Volume': 600
        }
        
        # Gravestone doji
        self.df.loc[self.df.index[32], :] = {
            'Open': 45400,
            'High': 45800,
            'Low': 45350,
            'Close': 45400,
            'Volume': 700
        }
        
    def _create_pin_bar_pattern(self):
        """Create hammer and shooting star patterns"""
        # Hammer
        self.df.loc[self.df.index[40], :] = {
            'Open': 45400,
            'High': 45450,
            'Low': 45000,
            'Close': 45350,
            'Volume': 500
        }
        
        # Shooting star
        self.df.loc[self.df.index[41], :] = {
            'Open': 45400,
            'High': 45800,
            'Low': 45350,
            'Close': 45450,
            'Volume': 600
        }
        
    def test_engulfing_patterns(self):
        """Test engulfing pattern detection"""
        patterns = self.detector._check_engulfing(self.df)
        
        bullish_engulfing = [p for p in patterns if p['pattern'] == PatternType.BULLISH_ENGULFING.value]
        bearish_engulfing = [p for p in patterns if p['pattern'] == PatternType.BEARISH_ENGULFING.value]
        
        self.assertTrue(any(p['timestamp'] == self.df.index[11] for p in bullish_engulfing))
        self.assertTrue(any(p['timestamp'] == self.df.index[21] for p in bearish_engulfing))
        
    def test_doji_patterns(self):
        """Test doji pattern detection"""
        patterns = self.detector._check_doji(self.df)
        
        regular_doji = [p for p in patterns if p['pattern'] == PatternType.DOJI.value]
        dragonfly_doji = [p for p in patterns if p['pattern'] == PatternType.DRAGONFLY_DOJI.value]
        gravestone_doji = [p for p in patterns if p['pattern'] == PatternType.GRAVESTONE_DOJI.value]
        
        self.assertTrue(any(p['timestamp'] == self.df.index[30] for p in regular_doji))
        self.assertTrue(any(p['timestamp'] == self.df.index[31] for p in dragonfly_doji))
        self.assertTrue(any(p['timestamp'] == self.df.index[32] for p in gravestone_doji))
        
    def test_pin_bar_patterns(self):
        """Test pin bar pattern detection"""
        # Print hammer data for debugging
        i = 40
        row = self.df.iloc[i]
        body_size = abs(row['Close'] - row['Open'])
        total_size = row['High'] - row['Low']
        upper_wick = row['High'] - max(row['Open'], row['Close'])
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        print(f"\nHammer data at index {i}:")
        print(f"Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}")
        print(f"Body size: {body_size}")
        print(f"Total size: {total_size}")
        print(f"Upper wick: {upper_wick}")
        print(f"Lower wick: {lower_wick}")
        print(f"Lower/Body ratio: {lower_wick/body_size if body_size > 0 else 'inf'}")
        print(f"Upper/Body ratio: {upper_wick/body_size if body_size > 0 else 'inf'}")
        print(f"Total/Body ratio: {total_size/body_size if body_size > 0 else 'inf'}")
        
        hammer_patterns = self.detector._check_hammer(self.df)
        shooting_star_patterns = self.detector._check_shooting_star(self.df)
        
        self.assertTrue(any(p['timestamp'] == self.df.index[40] for p in hammer_patterns))
        self.assertTrue(any(p['timestamp'] == self.df.index[41] for p in shooting_star_patterns))
        
    def test_analyze_all_patterns(self):
        """Test complete pattern analysis"""
        all_patterns = self.detector.analyze_patterns(self.df)
        
        pattern_types = [p['pattern'] for p in all_patterns]
        
        # Verificar se todos os padrões esperados foram encontrados
        self.assertIn(PatternType.BULLISH_ENGULFING.value, pattern_types)
        self.assertIn(PatternType.BEARISH_ENGULFING.value, pattern_types)
        self.assertIn(PatternType.DOJI.value, pattern_types)
        self.assertIn(PatternType.HAMMER.value, pattern_types)
        self.assertIn(PatternType.SHOOTING_STAR.value, pattern_types) 