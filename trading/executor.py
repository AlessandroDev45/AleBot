from typing import Dict, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradeExecution:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    status: str
    metadata: Dict

class TradeExecutor:
    def __init__(self, config, exchange, risk_manager):
        self.config = config
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.order_cache = {}
        self.position_cache = {}
    
    async def execute_signal(self, signal) -> Optional[TradeExecution]:
        try:
            # Validate trade
            if not await self._validate_trade(signal):
                return None
            
            # Calculate position size
            size = await self._calculate_position_size(signal)
            
            # Place order
            order = await self.exchange.place_order(
                symbol=signal.symbol,
                side=signal.signal_type.value,
                quantity=size,
                price=signal.price
            )
            
            # Create trade execution record
            execution = TradeExecution(
                order_id=order['id'],
                symbol=signal.symbol,
                side=signal.signal_type.value,
                quantity=size,
                price=signal.price,
                timestamp=datetime.now(),
                status='EXECUTED',
                metadata={
                    'signal': signal.__dict__,
                    'order': order
                }
            )
            
            # Update caches
            self.order_cache[order['id']] = execution
            
            return execution
            
        except Exception as e:
            logger.error(f'Error executing trade: {e}')
            return None
    
    async def _validate_trade(self, signal) -> bool:
        # Check if we have open position
        if signal.symbol in self.position_cache:
            return False
        
        # Check risk limits
        if not await self.risk_manager.validate_trade(
            signal.symbol,
            signal.signal_type.value,
            signal.price
        ):
            return False
        
        return True
    
    async def _calculate_position_size(self, signal) -> float:
        # Get account balance
        balance = await self.exchange.get_account_balance()
        
        # Calculate position size based on risk
        risk_per_trade = float(self.config.RISK_CONFIG['max_risk_per_trade'])
        account_risk = balance * risk_per_trade
        
        # Calculate size based on stop loss
        price_risk = abs(signal.price - signal.stop_loss)
        size = account_risk / price_risk
        
        # Apply position limits
        max_size = float(self.config.TRADING_CONFIG['max_trade_amount'])
        size = min(size, max_size)
        
        return size
    
    async def update_positions(self):
        try:
            # Get current positions
            positions = await self.exchange.get_positions()
            
            # Update cache
            self.position_cache = {
                pos['symbol']: pos for pos in positions
            }
            
            # Update trailing stops
            for symbol, position in self.position_cache.items():
                await self._update_trailing_stop(position)
        
        except Exception as e:
            logger.error(f'Error updating positions: {e}')
    
    async def _update_trailing_stop(self, position):
        try:
            # Get current price
            current_price = await self.exchange.get_current_price(
                position['symbol']
            )
            
            # Calculate new stop loss
            new_stop = await self.risk_manager.calculate_trailing_stop(
                position['symbol'],
                position['side'],
                position['entry_price'],
                current_price
            )
            
            # Update stop loss if needed
            if new_stop != position['stop_loss']:
                await self.exchange.update_stop_loss(
                    position['symbol'],
                    new_stop
                )
        
        except Exception as e:
            logger.error(f'Error updating trailing stop: {e}')