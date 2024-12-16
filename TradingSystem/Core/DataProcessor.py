import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import traceback
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing and technical analysis"""
    
    def __init__(self, data_manager):
        """Initialize DataProcessor with DataManager instance"""
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize settings
        self.settings = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0,
            'ma_fast': 10,
            'ma_slow': 30,
            'atr_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3
        }
        
        self.current_interval = "1h"
        self.last_update = 0
        self.data = pd.DataFrame()
        
        logger.info("Data processor initialized successfully")
    
    def check_status(self):
        """Check processor status"""
        try:
            return {
                'ok': True,
                'message': "Data processor is ready"
            }
        except Exception as e:
            return {
                'ok': False,
                'message': str(e)
            }
    
    def update_settings(self, settings):
        """Update processor settings"""
        try:
            self.settings.update(settings)
            logger.info("Data processor settings updated")
        except Exception as e:
            logger.error(f"Error updating data processor settings: {str(e)}")
            raise
    
    def get_current_rsi(self):
        """Get current RSI value"""
        try:
            if len(self.data) == 0:
                self.data = self.data_manager.get_historical_data(interval=self.current_interval)
            
            rsi = ta.rsi(self.data['close'], length=self.settings['rsi_period'])
            return float(rsi.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0
    
    def get_current_macd(self):
        """Get current MACD values"""
        try:
            if len(self.data) == 0:
                self.data = self.data_manager.get_historical_data(interval=self.current_interval)
            
            macd = ta.macd(self.data['close'],
                         fast=self.settings['macd_fast'],
                         slow=self.settings['macd_slow'],
                         signal=self.settings['macd_signal'])
            return float(macd['MACD_12_26_9'].iloc[-1]), float(macd['MACDs_12_26_9'].iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return 0.0, 0.0
    
    def update_data(self):
        """Update market data"""
        try:
            current_time = time.time()
            if current_time - self.last_update >= self.settings['update_interval']:
                self.data = self.data_manager.get_historical_data(interval=self.current_interval)
                self.last_update = current_time
                logger.debug("Market data updated successfully")
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            
    def get_historical_data(self, interval='1h'):
        """Get historical data with indicators"""
        try:
            # Update data if needed
            if interval != self.current_interval:
                self.current_interval = interval
                self.update_data()
            else:
                self.update_data()  # Regular update check
            
            # Calculate indicators
            df = self.data.copy()
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=self.settings['rsi_period'])
            
            # MACD
            macd = ta.macd(df['close'],
                          fast=self.settings['macd_fast'],
                          slow=self.settings['macd_slow'],
                          signal=self.settings['macd_signal'])
            df['macd'] = macd['MACD_12_26_9']
            df['signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bb = ta.bbands(df['close'],
                          length=self.settings['bb_period'],
                          std=self.settings['bb_std'])
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            
            # Moving Averages
            df['ma_fast'] = ta.sma(df['close'], length=self.settings['ma_fast'])
            df['ma_slow'] = ta.sma(df['close'], length=self.settings['ma_slow'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame() 

    def get_analysis_data(self, timeframe="1h"):
        """Get data for analysis"""
        try:
            # Get raw data from DataManager
            df = self.data_manager.get_market_data(timeframe=timeframe)
            if df.empty:
                return pd.DataFrame()
            
            # Process data with specific analysis logic
            df = self._process_analysis_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting analysis data: {str(e)}")
            return pd.DataFrame()

    def _process_analysis_data(self, df):
        """Process raw data for analysis"""
        try:
            df = df.copy()
            
            # Add technical indicators
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=self.settings['rsi_period'])
            
            # MACD
            macd = ta.macd(df['close'],
                          fast=self.settings['macd_fast'],
                          slow=self.settings['macd_slow'],
                          signal=self.settings['macd_signal'])
            df['macd'] = macd['MACD_12_26_9']
            df['signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bb = ta.bbands(df['close'],
                          length=self.settings['bb_period'],
                          std=self.settings['bb_std'])
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            
            # Moving Averages
            df['ma_fast'] = ta.sma(df['close'], length=self.settings['ma_fast'])
            df['ma_slow'] = ta.sma(df['close'], length=self.settings['ma_slow'])
            
            # Fill NaN values with forward fill then backward fill
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing analysis data: {str(e)}")
            return df

    def get_market_structure(self, timeframe="1h"):
        """Get market structure analysis"""
        try:
            # Get raw data from DataManager
            df = self.data_manager.get_market_data(timeframe=timeframe)
            if df.empty:
                return {}
                
            # Process market structure data
            result = self._analyze_market_structure(df)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting market structure: {str(e)}")
            return {}

    def _analyze_market_structure(self, df):
        """Analyze market structure from raw data"""
        try:
            df = df.copy()
            
            # Calculate market structure
            # Trend detection using moving averages
            df['ma_short'] = ta.sma(df['close'], length=20)
            df['ma_long'] = ta.sma(df['close'], length=50)
            
            # Determine regime
            df['trend'] = np.where(df['ma_short'] > df['ma_long'], 'Bullish',
                                 np.where(df['ma_short'] < df['ma_long'], 'Bearish', 'Ranging'))
            
            # Calculate trend strength using ADX
            adx = ta.adx(df['high'], df['low'], df['close'])
            trend_strength = float(adx['ADX_14'].iloc[-1]) / 100.0
            
            # Calculate structure quality using price volatility
            volatility = df['close'].pct_change().std()
            structure_quality = 1.0 - min(volatility * 100, 1.0)  # Normalize between 0 and 1
            
            # Support and resistance levels using pivot points
            pivots = ta.pivot(df['high'], df['low'], df['close'])
            
            return {
                'current_regime': df['trend'].iloc[-1],
                'regime_history': pd.DataFrame({'regime': df['trend']}),
                'price_levels': pd.DataFrame({
                    'support1': pivots['PP_S1'],
                    'support2': pivots['PP_S2'],
                    'pivot': pivots['PP_P'],
                    'resistance1': pivots['PP_R1'],
                    'resistance2': pivots['PP_R2']
                }),
                'trend_strength': float(trend_strength),
                'structure_quality': float(structure_quality)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {str(e)}")
            return {}

    def get_volume_profile(self, data):
        """Get volume profile data"""
        try:
            # Get raw data from DataManager
            df = self.data_manager.get_market_data(timeframe=data.get('timeframe', '1h'))
            if df.empty:
                return {}
                
            # Calculate volume profile
            price_range = np.linspace(df['low'].min(), df['high'].max(), 50)
            volumes = np.zeros(len(price_range)-1)
            
            for i in range(len(price_range)-1):
                mask = (df['close'] >= price_range[i]) & (df['close'] < price_range[i+1])
                volumes[i] = df.loc[mask, 'volume'].sum()
            
            return {
                'price': [(price_range[i] + price_range[i+1])/2 for i in range(len(price_range)-1)],
                'volume': volumes.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {str(e)}")
            return {}

    def get_correlation_data(self, timeframe="1h", assets=None):
        """Get correlation analysis data"""
        try:
            if assets is None:
                assets = ['BTCUSDT', 'ETHUSDT']
            
            # Get data for all assets from DataManager
            data = {}
            for asset in assets:
                df = self.data_manager.get_market_data(symbol=asset, timeframe=timeframe)
                if not df.empty:
                    data[asset] = df['close']
            
            if not data:
                return {}
                
            # Calculate correlations
            df = pd.DataFrame(data)
            correlation_matrix = df.corr()
            
            # Calculate rolling correlation
            window = 20
            correlations = {}
            for i in range(len(assets)):
                for j in range(i+1, len(assets)):
                    pair = f"{assets[i]}/{assets[j]}"
                    correlations[pair] = df[assets[i]].rolling(window).corr(df[assets[j]])
            
            return {
                'matrix': correlation_matrix.values,
                'assets': assets,
                'correlations': correlations
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            return {}

    def get_pattern_metrics(self):
        """Get pattern detection metrics"""
        try:
            # Get raw data from DataManager
            df = self.data_manager.get_market_data(timeframe="1h")
            if df.empty:
                return {}
                
            # Calculate metrics
            returns = df['close'].pct_change().dropna()
            
            # Sharpe ratio (assuming risk-free rate of 0.02)
            excess_returns = returns - 0.02/252  # Daily risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = cumulative/running_max - 1
            max_drawdown = drawdowns.min()
            
            # Recovery factor
            total_return = cumulative.iloc[-1] - 1
            recovery_factor = abs(total_return/max_drawdown) if max_drawdown != 0 else 0
            
            # Expectancy
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_rate = len(wins) / len(returns)
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            
            return {
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'recovery_factor': float(recovery_factor),
                'expectancy': float(expectancy)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern metrics: {str(e)}")
            return {}

    def get_backtest_results(self, params):
        """Get pattern backtest results"""
        try:
            # Get raw data from DataManager
            df = self.data_manager.get_market_data(
                symbol=params.get('symbol', 'BTCUSDT'),
                timeframe="1h"
            )
            
            if df.empty:
                return {}
                
            # Filter by date range
            if params.get('start_date') and params.get('end_date'):
                df = df[params['start_date']:params['end_date']]
            
            # Get pattern signals
            signals = self.analyze_patterns(df)
            
            # Run backtest
            trades = []
            total_pnl = 0
            wins = 0
            losses = 0
            
            for signal in signals:
                if signal['pattern'] == params.get('pattern_type'):
                    entry_price = signal['price']
                    entry_time = signal['timestamp']
                    
                    # Find exit based on stop loss or take profit
                    exit_mask = df.index > entry_time
                    if not any(exit_mask):
                        continue
                        
                    future_prices = df[exit_mask]
                    for idx, row in future_prices.iterrows():
                        price_change = (row['close'] - entry_price) / entry_price
                        
                        if signal['position'] == 'long':
                            if price_change <= -params.get('stop_loss', 2.0)/100:
                                pnl = -params.get('stop_loss', 2.0)
                                losses += 1
                                break
                            elif price_change >= params.get('take_profit', 4.0)/100:
                                pnl = params.get('take_profit', 4.0)
                                wins += 1
                                break
                        else:  # short
                            if price_change >= params.get('stop_loss', 2.0)/100:
                                pnl = -params.get('stop_loss', 2.0)
                                losses += 1
                                break
                            elif price_change <= -params.get('take_profit', 4.0)/100:
                                pnl = params.get('take_profit', 4.0)
                                wins += 1
                                break
                                
                        if idx == future_prices.index[-1]:
                            pnl = price_change * 100
                            if pnl > 0:
                                wins += 1
                            else:
                                losses += 1
                                
                    position_size = params.get('position_size', 1.0)
                    total_pnl += pnl * position_size/100
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': idx,
                        'exit_price': row['close'],
                        'pnl': pnl,
                        'position': signal['position']
                    })
            
            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            avg_return = total_pnl / total_trades if total_trades > 0 else 0
            profit_factor = abs(total_pnl / params.get('position_size', 1.0)) if params.get('position_size', 1.0) > 0 else 0
            
            return {
                'win_rate': round(win_rate, 2),
                'avg_return': round(avg_return, 2),
                'total_trades': total_trades,
                'profit_factor': round(profit_factor, 2),
                'total_pnl': round(total_pnl, 2),
                'trades': trades
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            return {}

    def get_timeframe_data(self, timeframe):
        """Get data for specific timeframe"""
        try:
            df = self.data_manager.get_market_data(timeframe=timeframe)
            if df.empty:
                return pd.DataFrame()
                
            # Add basic indicators
            df['rsi'] = ta.rsi(df['close'], length=self.settings['rsi_period'])
            df['ma_20'] = ta.sma(df['close'], length=20)
            df['ma_50'] = ta.sma(df['close'], length=50)
            
            # Fill NaN values
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting timeframe data: {str(e)}")
            return pd.DataFrame() 