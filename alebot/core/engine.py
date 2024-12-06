import asyncio
from datetime import datetime
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    side: str
    price: float
    volume: float
    confidence: float
    indicators: Dict
    stop_loss: float
    take_profit: float

class TradingEngine:
    def __init__(self, config, exchange, analysis, risk_manager, ml_model):
        self.config = config
        self.exchange = exchange
        self.analysis = analysis
        self.risk_manager = risk_manager
        self.ml_model = ml_model
        self.active_positions = {}
        self.trade_history = []
        self.is_running = False

    async def start(self):
        """Start the trading engine"""
        self.is_running = True
        try:
            # Initialize connections
            await self.exchange.connect()
            
            # Start market data stream
            await self.exchange.subscribe_market_data(self.config.TRADING_CONFIG['symbols'])
            
            # Start main trading loop
            await asyncio.gather(
                self._monitor_market(),
                self._manage_positions(),
                self._update_models()
            )
            
        except Exception as e:
            logging.error(f'Error starting trading engine: {e}')
            await self.stop()

    async def stop(self):
        """Stop the trading engine and cleanup"""
        self.is_running = False
        try:
            # Close open positions if necessary
            if self.active_positions:
                await self._close_all_positions()
            
            # Cleanup connections
            await self.exchange.disconnect()
            
        except Exception as e:
            logging.error(f'Error stopping trading engine: {e}')

    async def _monitor_market(self):
        """Monitor market and generate trading signals"""
        while self.is_running:
            try:
                for symbol in self.config.TRADING_CONFIG['symbols']:
                    # Get market data
                    data = await self.exchange.get_market_data(symbol)
                    
                    # Perform technical analysis
                    analysis = await self.analysis.analyze(data)
                    
                    # Get ML predictions
                    predictions = await self.ml_model.predict(symbol)
                    
                    # Generate signals
                    signals = self._generate_signals(symbol, data, analysis, predictions)
                    
                    # Process signals
                    for signal in signals:
                        await self._process_signal(signal)
                        
                await asyncio.sleep(self.config.TRADING_CONFIG['update_interval'])
                
            except Exception as e:
                logging.error(f'Error in market monitoring: {e}')
                await asyncio.sleep(5)

    async def _manage_positions(self):
        """Manage open positions and risk"""
        while self.is_running:
            try:
                # Update position information
                positions = await self.exchange.get_positions()
                self.active_positions = positions
                
                for position in positions.values():
                    # Update trailing stops
                    await self._update_trailing_stop(position)
                    
                    # Check risk metrics
                    await self._check_position_risk(position)
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f'Error managing positions: {e}')
                await asyncio.sleep(5)

    async def _update_models(self):
        """Update ML models with new data"""
        while self.is_running:
            try:
                await self.ml_model.update()
                await asyncio.sleep(self.config.ML_CONFIG['update_interval'])
                
            except Exception as e:
                logging.error(f'Error updating models: {e}')
                await asyncio.sleep(60)

    def _generate_signals(self, symbol: str, data: Dict, analysis: Dict, predictions: Dict) -> List[TradeSignal]:
        """Generate trading signals based on analysis and predictions"""
        signals = []
        
        # Combine technical indicators and ML predictions
        technical_signal = analysis['signal']
        ml_signal = predictions['signal']
        confidence = predictions['confidence']
        
        # Generate signal only if technical and ML agree
        if technical_signal == ml_signal and confidence > self.config.ML_CONFIG['min_accuracy']:
            price = float(data['close'][-1])
            volume = float(data['volume'][-1])
            
            # Calculate stop loss and take profit
            volatility = analysis['volatility']
            if technical_signal == 'BUY':
                stop_loss = price * (1 - volatility * 2)
                take_profit = price * (1 + volatility * 5)
            else:
                stop_loss = price * (1 + volatility * 2)
                take_profit = price * (1 - volatility * 5)
            
            signals.append(TradeSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                side=technical_signal,
                price=price,
                volume=volume,
                confidence=confidence,
                indicators=analysis,
                stop_loss=stop_loss,
                take_profit=take_profit
            ))
        
        return signals

    async def _process_signal(self, signal: TradeSignal):
        """Process and execute trading signal"""
        try:
            # Check if we can take new positions
            if len(self.active_positions) >= self.config.TRADING_CONFIG['max_positions']:
                return
            
            # Validate signal with risk manager
            if not await self.risk_manager.validate_signal(signal):
                return
            
            # Calculate position size
            size = await self.risk_manager.calculate_position_size(signal)
            
            # Execute trade
            order = await self.exchange.place_order(
                symbol=signal.symbol,
                side=signal.side,
                quantity=size,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order:
                self.trade_history.append({
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'price': signal.price,
                    'size': size,
                    'order_id': order['id']
                })
            
        except Exception as e:
            logging.error(f'Error processing signal: {e}')

    async def _update_trailing_stop(self, position: Dict):
        """Update trailing stop for position"""
        try:
            new_stop = await self.risk_manager.calculate_trailing_stop(position)
            
            if new_stop != position['stop_loss']:
                await self.exchange.update_stop_loss(
                    symbol=position['symbol'],
                    order_id=position['order_id'],
                    stop_price=new_stop
                )
                
        except Exception as e:
            logging.error(f'Error updating trailing stop: {e}')

    async def _check_position_risk(self, position: Dict):
        """Check position risk metrics"""
        try:
            risk_metrics = await self.risk_manager.calculate_position_risk(position)
            
            # Close position if risk exceeds thresholds
            if risk_metrics['total_risk'] > self.config.RISK_CONFIG['max_risk_per_trade']:
                await self.exchange.close_position(position['symbol'])
                
        except Exception as e:
            logging.error(f'Error checking position risk: {e}')

    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            for position in self.active_positions.values():
                await self.exchange.close_position(position['symbol'])
                
        except Exception as e:
            logging.error(f'Error closing positions: {e}')