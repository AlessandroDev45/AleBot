from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.max_risk_per_trade = 0.01  # 1%
        
    async def validate_trade(self, symbol: str, size: float, price: float) -> bool:
        current_exposure = await self.calculate_exposure()
        position_value = size * price
        return position_value <= self.get_max_position_size(current_exposure)
        
    async def calculate_exposure(self) -> float:
        positions = await self.exchange.get_positions()
        return sum(pos['value'] for pos in positions)
        
    def get_max_position_size(self, current_exposure: float) -> float:
        account_value = self.exchange.get_account_value()
        return account_value * self.max_risk_per_trade