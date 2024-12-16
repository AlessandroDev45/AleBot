import pandas as pd
import numpy as np
from typing import Union

def is_bullish_engulfing(data: pd.DataFrame) -> bool:
    """Check for bullish engulfing pattern."""
    try:
        if len(data) < 2:
            return False
        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]
        return (prev_candle['close'] < prev_candle['open'] and  # Previous red candle
                curr_candle['close'] > curr_candle['open'] and  # Current green candle
                curr_candle['open'] < prev_candle['close'] and  # Current opens below previous close
                curr_candle['close'] > prev_candle['open'])     # Current closes above previous open
    except:
        return False

def is_bearish_engulfing(data: pd.DataFrame) -> bool:
    """Check for bearish engulfing pattern."""
    try:
        if len(data) < 2:
            return False
        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]
        return (prev_candle['close'] > prev_candle['open'] and  # Previous green candle
                curr_candle['close'] < curr_candle['open'] and  # Current red candle
                curr_candle['open'] > prev_candle['close'] and  # Current opens above previous close
                curr_candle['close'] < prev_candle['open'])     # Current closes below previous open
    except:
        return False

def is_hammer(data: pd.DataFrame) -> bool:
    """Check for hammer pattern."""
    try:
        if len(data) < 1:
            return False
        candle = data.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return (lower_shadow > 2 * body_size and  # Long lower shadow
                upper_shadow < 0.1 * body_size)    # Very small upper shadow
    except:
        return False

def is_shooting_star(data: pd.DataFrame) -> bool:
    """Check for shooting star pattern."""
    try:
        if len(data) < 1:
            return False
        candle = data.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return (upper_shadow > 2 * body_size and  # Long upper shadow
                lower_shadow < 0.1 * body_size)    # Very small lower shadow
    except:
        return False

def is_morning_star(data: pd.DataFrame) -> bool:
    """Check for morning star pattern."""
    try:
        if len(data) < 3:
            return False
        first = data.iloc[-3]
        second = data.iloc[-2]
        third = data.iloc[-1]
        return (first['close'] < first['open'] and           # First day bearish
                abs(second['close'] - second['open']) < 0.1 * first['body_size'] and  # Second day small body
                third['close'] > third['open'] and           # Third day bullish
                third['close'] > (first['open'] + first['close']) / 2)  # Closes above first midpoint
    except:
        return False

def is_evening_star(data: pd.DataFrame) -> bool:
    """Check for evening star pattern."""
    try:
        if len(data) < 3:
            return False
        first = data.iloc[-3]
        second = data.iloc[-2]
        third = data.iloc[-1]
        return (first['close'] > first['open'] and           # First day bullish
                abs(second['close'] - second['open']) < 0.1 * first['body_size'] and  # Second day small body
                third['close'] < third['open'] and           # Third day bearish
                third['close'] < (first['open'] + first['close']) / 2)  # Closes below first midpoint
    except:
        return False

def is_doji(data: pd.DataFrame) -> bool:
    """Check for doji pattern."""
    try:
        if len(data) < 1:
            return False
        candle = data.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        total_size = candle['high'] - candle['low']
        return body_size <= 0.1 * total_size  # Body is very small compared to total range
    except:
        return False

def is_dragonfly_doji(data: pd.DataFrame) -> bool:
    """Check for dragonfly doji pattern."""
    try:
        if len(data) < 1:
            return False
        candle = data.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return (body_size <= 0.1 * lower_shadow and  # Very small body
                upper_shadow <= 0.1 * lower_shadow)   # Very small upper shadow
    except:
        return False

def is_gravestone_doji(data: pd.DataFrame) -> bool:
    """Check for gravestone doji pattern."""
    try:
        if len(data) < 1:
            return False
        candle = data.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return (body_size <= 0.1 * upper_shadow and  # Very small body
                lower_shadow <= 0.1 * upper_shadow)   # Very small lower shadow
    except:
        return False

def is_spinning_top(data: pd.DataFrame) -> bool:
    """Check for spinning top pattern."""
    try:
        if len(data) < 1:
            return False
        candle = data.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return (body_size <= 0.3 * (upper_shadow + lower_shadow) and  # Small body
                abs(upper_shadow - lower_shadow) <= 0.1 * (upper_shadow + lower_shadow))  # Similar shadow sizes
    except:
        return False 