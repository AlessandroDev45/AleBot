import pandas as pd
from Core.DataManager.data_manager import DataManager
from pattern_detection.pattern_detection_utils import is_bullish_engulfing, is_bearish_engulfing, is_hammer, is_shooting_star, is_morning_star, is_evening_star, is_doji, is_dragonfly_doji, is_gravestone_doji, is_spinning_top
from ..logger import logger

def add_pattern_signals(data: pd.DataFrame, min_pattern_bars: int = 5, max_pattern_bars: int = 50) -> pd.DataFrame:
    """Add pattern signals to the dataframe."""
    try:
        if not isinstance(data, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame")
            return data
            
        # Validate data through DataManager
        data_manager = DataManager()
        if not data_manager.validate_pattern_data(data):
            logger.error("Pattern data validation failed")
            return data
            
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Initialize pattern columns
        pattern_columns = [
            'bullish_engulfing', 'bearish_engulfing',
            'hammer', 'shooting_star',
            'morning_star', 'evening_star',
            'doji', 'dragonfly_doji',
            'gravestone_doji', 'spinning_top'
        ]
        
        for col in pattern_columns:
            df[col] = 0
            
        # Calculate patterns with proper window sizes
        window = min(len(df), max_pattern_bars)
        if window < min_pattern_bars:
            logger.warning(f"Insufficient data points for pattern detection. Found: {len(df)}, Required: {min_pattern_bars}")
            return df
            
        # Detect patterns
        for i in range(min_pattern_bars, len(df)):
            window_data = df.iloc[max(0, i-window):i+1]
            
            # Candlestick patterns
            df.loc[df.index[i], 'bullish_engulfing'] = is_bullish_engulfing(window_data)
            df.loc[df.index[i], 'bearish_engulfing'] = is_bearish_engulfing(window_data)
            df.loc[df.index[i], 'hammer'] = is_hammer(window_data)
            df.loc[df.index[i], 'shooting_star'] = is_shooting_star(window_data)
            df.loc[df.index[i], 'morning_star'] = is_morning_star(window_data)
            df.loc[df.index[i], 'evening_star'] = is_evening_star(window_data)
            df.loc[df.index[i], 'doji'] = is_doji(window_data)
            df.loc[df.index[i], 'dragonfly_doji'] = is_dragonfly_doji(window_data)
            df.loc[df.index[i], 'gravestone_doji'] = is_gravestone_doji(window_data)
            df.loc[df.index[i], 'spinning_top'] = is_spinning_top(window_data)
            
        return df
        
    except Exception as e:
        logger.error(f"Error adding pattern signals: {str(e)}")
        return data 