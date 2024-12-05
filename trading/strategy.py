from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

class SignalType(Enum):
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'

@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float
    stop_loss: float
    take_profit: float
    metadata: Dict

class TradingStrategy:
    def __init__(self, config, analysis, risk_manager, ml_model):
        self.config = config
        self.analysis = analysis
        self.risk_manager = risk_manager
        self.ml_model = ml_model
        self.min_confidence = 0.8
        self.required_confirmations = 3
    
    async def generate_signals(self, symbol: str) -> List[TradeSignal]:
        # Get market data
        data = await self.analysis.get_market_data(symbol)
        
        # Technical analysis
        tech_signals = await self.analysis.analyze(data)
        
        # ML predictions
        ml_predictions = await self.ml_model.predict(symbol)
        
        # Risk validation
        risk_metrics = await self.risk_manager.get_risk_metrics(symbol)
        
        # Combine signals
        signals = self._combine_signals(
            tech_signals,
            ml_predictions,
            risk_metrics,
            data
        )
        
        return signals
    
    def _combine_signals(self, tech_signals, ml_predictions, risk_metrics, data) -> List[TradeSignal]:
        signals = []
        
        # Check trend alignment
        trend_confirmed = self._check_trend_alignment(tech_signals)
        
        # Check ML confidence
        ml_confident = ml_predictions['confidence'] > self.min_confidence
        
        # Check risk levels
        risk_acceptable = risk_metrics['risk_level'] != 'HIGH'
        
        # Generate signal if all conditions met
        if trend_confirmed and ml_confident and risk_acceptable:
            signal = self._create_trade_signal(
                data,
                tech_signals,
                ml_predictions
            )
            signals.append(signal)
        
        return signals
    
    def _check_trend_alignment(self, tech_signals) -> bool:
        """Check if multiple timeframe trends align"""
        timeframes = ['1m', '5m', '15m', '1h']
        trends = [tech_signals[tf]['trend'] for tf in timeframes]
        
        bullish_count = sum(1 for t in trends if t == 'BULLISH')
        bearish_count = sum(1 for t in trends if t == 'BEARISH')
        
        return bullish_count >= self.required_confirmations or \
               bearish_count >= self.required_confirmations
    
    def _create_trade_signal(self, data, tech_signals, ml_predictions) -> TradeSignal:
        """Create trade signal with stop loss and take profit"""
        current_price = data['close'].iloc[-1]
        
        # Determine signal type
        signal_type = SignalType.BUY if ml_predictions['direction'] > 0 else SignalType.SELL
        
        # Calculate stop loss and take profit
        volatility = tech_signals['1h']['atr']
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (volatility * 2)
            take_profit = current_price + (volatility * 5)
        else:
            stop_loss = current_price + (volatility * 2)
            take_profit = current_price - (volatility * 5)
        
        return TradeSignal(
            timestamp=datetime.now(),
            symbol=data.name,
            signal_type=signal_type,
            price=current_price,
            confidence=ml_predictions['confidence'],
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'tech_signals': tech_signals,
                'ml_predictions': ml_predictions
            }
        )