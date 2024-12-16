import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderManager:
    """Order management system"""
    
    def __init__(self, data_manager, risk_manager):
        """Initialize order manager"""
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        self.orders = []
        self.positions = []
        
    def create_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None, 
                    confidence_score: Optional[float] = None) -> Dict:
        """Create a new order with enhanced risk management"""
        try:
            # Get current price if not provided
            if price is None and order_type != "MARKET":
                price = float(self.data_manager.get_market_data(
                    symbol=symbol,
                    timeframe="1m",
                    limit=1
                )['close'].iloc[-1])
            
            # Get account information
            account = self.data_manager.get_account_info()
            available_balance = float(account.get('totalWalletBalance', 0))
            
            # Calculate optimal position size based on confidence
            if confidence_score is not None:
                # Adjust position size based on prediction confidence
                base_position = self.risk_manager.calculate_position_size(
                    account_balance=available_balance,
                    entry_price=price or stop_price,
                    stop_loss=stop_price,
                    win_rate=confidence_score
                )
                quantity = min(quantity, base_position)
            
            # Validate order with risk manager
            if not self.risk_manager.validate_order(order_type, quantity, price or 0):
                logger.warning("Order validation failed")
                return {
                    'status': 'ERROR',
                    'error': 'Order validation failed',
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity,
                    'price': price,
                    'orderId': None
                }
            
            # Create order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if price is not None and order_type != "MARKET":
                order_params['price'] = price
            if stop_price is not None:
                order_params['stopPrice'] = stop_price
            
            # Create order through data manager
            order = self.data_manager.create_order(**order_params)
            
            if order.get('status') == 'ERROR':
                logger.error(f"Error creating order: {order.get('error')}")
                return order
            
            # Record order if successful
            if order.get('status') in ['FILLED', 'NEW']:
                self.orders.append({
                    'id': order.get('orderId'),
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity,
                    'price': float(order.get('price', price or 0)),
                    'status': order['status'],
                    'timestamp': datetime.now(),
                    'confidence_score': confidence_score
                })
                
                # Record trade in risk manager if filled
                if order.get('status') == 'FILLED':
                    self.risk_manager.record_trade(
                        quantity=quantity,
                        price=float(order.get('price', price or 0)),
                        side=side
                    )
                
                # Update positions
                self._update_positions()
                
                # Create take profit orders if it's an entry
                if order.get('status') == 'FILLED' and confidence_score is not None:
                    self._create_exit_orders(
                        symbol=symbol,
                        entry_price=float(order.get('price', price or 0)),
                        quantity=quantity,
                        side=side,
                        confidence_score=confidence_score
                    )
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'price': price,
                'orderId': None
            }
            
    def _create_exit_orders(self, symbol: str, entry_price: float, quantity: float, 
                          side: str, confidence_score: float) -> None:
        """Create exit orders (take profit and stop loss) based on confidence"""
        try:
            # Adjust take profit levels based on confidence
            if confidence_score >= 0.8:  # High confidence
                take_profits = [
                    {'percentage': 0.4, 'target': 0.015},  # 40% at 1.5%
                    {'percentage': 0.3, 'target': 0.025},  # 30% at 2.5%
                    {'percentage': 0.3, 'target': 0.035}   # 30% at 3.5%
                ]
            elif confidence_score >= 0.6:  # Medium confidence
                take_profits = [
                    {'percentage': 0.5, 'target': 0.010},  # 50% at 1.0%
                    {'percentage': 0.3, 'target': 0.015},  # 30% at 1.5%
                    {'percentage': 0.2, 'target': 0.020}   # 20% at 2.0%
                ]
            else:  # Low confidence
                take_profits = [
                    {'percentage': 0.6, 'target': 0.005},  # 60% at 0.5%
                    {'percentage': 0.4, 'target': 0.010}   # 40% at 1.0%
                ]
            
            # Create take profit orders
            for tp in take_profits:
                tp_price = entry_price * (1 + tp['target']) if side == "BUY" else entry_price * (1 - tp['target'])
                tp_quantity = quantity * tp['percentage']
                
                self.create_order(
                    symbol=symbol,
                    side="SELL" if side == "BUY" else "BUY",
                    order_type="LIMIT",
                    quantity=tp_quantity,
                    price=tp_price
                )
            
            # Set stop loss based on confidence
            stop_distance = 0.01 * (1 - confidence_score)  # Tighter stop for higher confidence
            stop_price = entry_price * (1 - stop_distance) if side == "BUY" else entry_price * (1 + stop_distance)
            
            self.create_order(
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                order_type="STOP_LOSS_LIMIT",
                quantity=quantity,
                price=stop_price,
                stop_price=stop_price
            )
            
        except Exception as e:
            logger.error(f"Error creating exit orders: {str(e)}")
            
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, leverage: int = 1) -> Dict:
        """Place an order on Binance"""
        try:
            if not self.data_manager:
                raise ValueError("DataManager not initialized")
            
            # Set leverage first
            try:
                self.data_manager.set_leverage(symbol=symbol, leverage=leverage)
                logger.info(f"Leverage set to {leverage}x for {symbol}")
            except Exception as e:
                logger.error(f"Error setting leverage: {e}")
                raise
            
            # Prepare order parameters
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if order_type != 'MARKET' and price:
                params['price'] = price
            
            # Check with risk manager
            if not self.risk_manager.check_order(params):
                raise ValueError("Order rejected by risk manager")
            
            # Place the order through DataManager
            response = self.data_manager.place_order(params)
            
            # Log the order
            logger.info(f"Order placed successfully: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an existing order"""
        try:
            if not self.data_manager:
                raise ValueError("DataManager not initialized")
            return self.data_manager.cancel_order(symbol=symbol, order_id=order_id)
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            raise

    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Cancel all open orders for a symbol or all symbols"""
        try:
            results = self.data_manager.cancel_all_orders(symbol)
            
            # Update order status for all canceled orders
            for result in results:
                if result.get('status') == 'CANCELED':
                    for order in self.orders:
                        if order['id'] == result.get('orderId'):
                            order['status'] = 'CANCELED'
                            break
            
            return results
            
        except Exception as e:
            logger.error(f"Error canceling all orders: {str(e)}")
            return []
            
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        try:
            if not self.data_manager:
                raise ValueError("DataManager not initialized")
            return self.data_manager.get_open_orders(symbol=symbol)
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            raise

    def get_position_info(self, symbol: str = "BTCUSDT") -> Dict:
        """Get position information for a symbol"""
        try:
            if not self.data_manager:
                raise ValueError("DataManager not initialized")
            return self.data_manager.get_position_info(symbol=symbol)
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            raise

    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if not self.data_manager:
                raise ValueError("DataManager not initialized")
            return self.data_manager.get_account_info()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
            
    def get_order_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get order history"""
        try:
            return self.data_manager.get_trade_history(symbol, limit)
        except Exception as e:
            logger.error(f"Error getting order history: {str(e)}")
            return []
            
    def _update_positions(self) -> None:
        """Update current positions"""
        try:
            self.positions = self.data_manager.get_positions()
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        return self.positions
        
    def close_position(self, symbol: str) -> Dict:
        """Close a position"""
        try:
            # Find position
            position = None
            for pos in self.positions:
                if pos['asset'] == symbol:
                    position = pos
                    break
                    
            if position is None:
                raise ValueError(f"No position found for {symbol}")
                
            # Create market order to close position
            return self.create_order(
                symbol=symbol,
                side="SELL" if position['side'] == "BUY" else "BUY",
                order_type="MARKET",
                quantity=float(position['total'])
            )
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
            
    def close_all_positions(self) -> List[Dict]:
        """Close all open positions"""
        try:
            results = []
            for position in self.positions:
                result = self.close_position(position['asset'])
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return []
            
    def get_position_value(self, symbol: str) -> float:
        """Get current value of a position"""
        try:
            position = None
            for pos in self.positions:
                if pos['asset'] == symbol:
                    position = pos
                    break
                    
            if position is None:
                return 0.0
                
            current_price = self.data_manager.get_symbol_price(symbol)
            return position['total'] * current_price
            
        except Exception as e:
            logger.error(f"Error getting position value: {e}")
            return 0.0
            
    def get_total_position_value(self) -> float:
        """Get total value of all positions"""
        try:
            total = 0.0
            for position in self.positions:
                total += self.get_position_value(position['asset'])
            return total
            
        except Exception as e:
            logger.error(f"Error getting total position value: {e}")
            return 0.0
            
    def create_take_profit_orders(self, symbol: str, entry_price: float, quantity: float, side: str) -> List[Dict]:
        """Create take profit orders according to specification
        TP1: 30% position at 0.5% profit
        TP2: 30% position at 1.0% profit
        TP3: 40% position at 2.0% profit"""
        try:
            take_profits = [
                {'percentage': 0.30, 'target': 0.005},  # TP1: 30% at 0.5%
                {'percentage': 0.30, 'target': 0.010},  # TP2: 30% at 1.0%
                {'percentage': 0.40, 'target': 0.020}   # TP3: 40% at 2.0%
            ]
            
            tp_orders = []
            for tp in take_profits:
                tp_price = entry_price * (1 + tp['target']) if side == "BUY" else entry_price * (1 - tp['target'])
                tp_quantity = quantity * tp['percentage']
                
                order = self.create_order(
                    symbol=symbol,
                    side="SELL" if side == "BUY" else "BUY",
                    order_type="LIMIT",
                    quantity=tp_quantity,
                    price=tp_price
                )
                tp_orders.append(order)
                
            return tp_orders
            
        except Exception as e:
            logger.error(f"Error creating take profit orders: {e}")
            return []
            
    def create_trailing_stop(self, symbol: str, activation_price: float, callback_rate: float = 0.002) -> Dict:
        """Create a trailing stop order
        activation_price: Price at which trailing stop becomes active
        callback_rate: How far price must move back to trigger stop (0.2% default)"""
        try:
            return self.data_manager.create_trailing_stop_order(
                symbol=symbol,
                activation_price=activation_price,
                callback_rate=callback_rate
            )
            
        except Exception as e:
            logger.error(f"Error creating trailing stop: {e}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
            
    def update_trailing_stop(self, symbol: str, order_id: str, new_activation_price: float) -> Dict:
        """Update trailing stop activation price"""
        try:
            return self.data_manager.update_trailing_stop(
                symbol=symbol,
                order_id=order_id,
                activation_price=new_activation_price
            )
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return {
                'status': 'ERROR',
                'error': str(e)
            } 