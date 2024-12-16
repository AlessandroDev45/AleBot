from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

class Trade:
    def __init__(self, symbol: str, side: OrderSide, price: float, quantity: float, timestamp: datetime):
        self.symbol = symbol
        self.side = side
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp

class Position:
    def __init__(self, symbol: str, quantity: float, entry_price: float, current_price: float):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = current_price

class MarketData:
    def __init__(self, symbol: str, timestamp: datetime, open: float, high: float, low: float, close: float, volume: float):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

class Database:
    def __init__(self):
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.market_data: Dict[str, List[MarketData]] = {} 