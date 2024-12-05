import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

class BacktestEngine:
    def __init__(self, config, analysis, strategy, risk_manager):
        self.config = config
        self.analysis = analysis
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.results = {}
        self.trades = []
        self.equity_curve = []
        self.initial_balance = float(config.RISK_CONFIG['INITIAL_BALANCE'])
    
    async def run_backtest(self, symbol: str, start_date: datetime,
                          end_date: datetime) -> Dict:
        try:
            # Get historical data
            data = await self._get_historical_data(symbol, start_date, end_date)
            
            # Initialize state
            current_balance = self.initial_balance
            open_positions = {}
            self.trades = []
            self.equity_curve = [{'timestamp': start_date, 'equity': current_balance}]
            
            # Run simulation
            for timestamp, candle in data.iterrows():
                # Update state
                current_balance = await self._update_positions(
                    open_positions, candle, current_balance
                )
                
                # Generate signals
                signals = await self.strategy.generate_signals(
                    symbol, data.loc[:timestamp]
                )
                
                # Execute signals
                for signal in signals:
                    trade = await self._execute_trade(
                        signal, candle, current_balance
                    )
                    if trade:
                        self.trades.append(trade)
                        if trade['type'] == 'OPEN':
                            open_positions[symbol] = trade
                        else:
                            open_positions.pop(symbol, None)
                
                # Update equity curve
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_balance
                })
            
            # Calculate performance metrics
            self.results = self._calculate_metrics()
            return self.results
            
        except Exception as e:
            logger.error(f'Backtest error: {e}')
            raise
    
    async def _get_historical_data(self, symbol: str,
                                 start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
        # Get market data
        data = await self.analysis.get_market_data(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date,
            timeframe='1h'
        )
        
        # Add technical indicators
        indicators = await self.analysis.calculate_indicators(data)
        for name, values in indicators.items():
            data[name] = values
        
        return data
    
    async def _update_positions(self, positions: Dict,
                              candle: pd.Series,
                              balance: float) -> float:
        """Update open positions and return new balance"""
        new_balance = balance
        
        for symbol, position in list(positions.items()):
            # Check stop loss
            if position['side'] == 'BUY':
                if candle['low'] <= position['stop_loss']:
                    new_balance = await self._close_position(
                        position, position['stop_loss'], 'STOP_LOSS'
                    )
                    positions.pop(symbol)
                elif candle['high'] >= position['take_profit']:
                    new_balance = await self._close_position(
                        position, position['take_profit'], 'TAKE_PROFIT'
                    )
                    positions.pop(symbol)
            else:  # SELL position
                if candle['high'] >= position['stop_loss']:
                    new_balance = await self._close_position(
                        position, position['stop_loss'], 'STOP_LOSS'
                    )
                    positions.pop(symbol)
                elif candle['low'] <= position['take_profit']:
                    new_balance = await self._close_position(
                        position, position['take_profit'], 'TAKE_PROFIT'
                    )
                    positions.pop(symbol)
        
        return new_balance
    
    async def _execute_trade(self, signal: Dict, candle: pd.Series,
                           balance: float) -> Optional[Dict]:
        """Execute trade based on signal"""
        # Validate trade
        size = await self.risk_manager.calculate_position_size(
            signal['symbol'],
            signal['price'],
            balance
        )
        
        if size == 0:
            return None
        
        # Create trade record
        trade = {
            'timestamp': candle.name,
            'symbol': signal['symbol'],
            'side': signal['signal_type'],
            'price': signal['price'],
            'size': size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'type': 'OPEN'
        }
        
        return trade
    
    async def _close_position(self, position: Dict, price: float,
                            reason: str) -> float:
        """Close position and return new balance"""
        pnl = self._calculate_pnl(position, price)
        
        trade = {
            'timestamp': position['timestamp'],
            'symbol': position['symbol'],
            'side': 'SELL' if position['side'] == 'BUY' else 'BUY',
            'price': price,
            'size': position['size'],
            'pnl': pnl,
            'type': 'CLOSE',
            'reason': reason
        }
        
        self.trades.append(trade)
        return position['balance'] + pnl
    
    def _calculate_pnl(self, position: Dict, close_price: float) -> float:
        """Calculate profit/loss for a position"""
        if position['side'] == 'BUY':
            return (close_price - position['price']) * position['size']
        else:
            return (position['price'] - close_price) * position['size']
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        closed_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        
        if not closed_trades:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        returns = pd.Series([t['pnl'] for t in closed_trades])
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': sum(1 for t in closed_trades if t['pnl'] > 0),
            'losing_trades': sum(1 for t in closed_trades if t['pnl'] < 0),
            'win_rate': sum(1 for t in closed_trades if t['pnl'] > 0) / len(closed_trades),
            'average_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
            'average_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) \
                if len(returns[returns < 0]) > 0 else float('inf'),
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() \
                if len(returns) > 1 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_df['equity']),
            'total_return': (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1) * 100
        }