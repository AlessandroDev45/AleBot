import numpy as np
import pandas as pd
import logging
import traceback
from .config_manager import ConfigManager
from typing import Dict, List

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, data_manager = None):
        """Initialize risk manager"""
        try:
            logger.info("Initializing risk manager...")
            
            # Use provided DataManager
            if data_manager is None:
                raise ValueError("DataManager must be provided")
            self.data_manager = data_manager
            
            # Load risk parameters from config
            config = ConfigManager()
            risk_config = config.get('risk', {})
            
            # Get risk parameters with defaults
            self.max_position_size = float(risk_config.get('max_position_size', 1.0))  # 100% of account for testing
            self.max_daily_loss = float(risk_config.get('max_daily_drawdown', 1.0))  # 100% daily loss for testing
            self.max_drawdown = float(risk_config.get('max_total_drawdown', 1.0))  # 100% max drawdown for testing
            
            # Initialize risk metrics
            self.daily_loss = 0.0
            self.current_drawdown = 0.0
            self.peak_value = None
            
            # Get initial account balance
            account_info = self.data_manager.get_account_info()
            if account_info:
                initial_balance = float(account_info.get('totalWalletBalance', 0))
                if initial_balance > 0:
                    self.peak_value = initial_balance
            
            logger.info("Risk manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing risk manager: {str(e)}")
            raise
    
    def check_status(self):
        """Check risk manager status"""
        try:
            return {
                'ok': True,
                'message': "Risk manager is ready"
            }
        except Exception as e:
            return {
                'ok': False,
                'message': str(e)
            }
    
    def update_settings(self, settings):
        """Update risk settings"""
        try:
            self.settings.update(settings)
            logger.info("Risk manager settings updated")
        except Exception as e:
            logger.error(f"Error updating risk settings: {str(e)}")
            raise
    
    def get_trading_stats(self):
        """Get current trading statistics"""
        try:
            return self.stats
        except Exception as e:
            logger.error(f"Error getting trading stats: {str(e)}")
            return {
                'trades_today': 0,
                'daily_pnl': 0.0,
                'amount_used': 0.0,
                'available_amount': 0.0,
                'max_daily_trades': 0
            }
    
    def is_stopped(self):
        """Check if trading is stopped"""
        return self.stopped
    
    def start_trading(self):
        """Start trading"""
        try:
            self.stopped = False
            logger.info("Trading started")
        except Exception as e:
            logger.error(f"Error starting trading: {str(e)}")
            raise
    
    def stop_trading(self):
        """Stop trading"""
        try:
            self.stopped = True
            logger.info("Trading stopped")
        except Exception as e:
            logger.error(f"Error stopping trading: {str(e)}")
            raise
    
    def check_trade(self, symbol, side, amount, price):
        """Check if trade is allowed"""
        try:
            if self.stopped:
                return False, "Trading is stopped"
            
            if self.stats['trades_today'] >= self.stats['max_daily_trades']:
                return False, "Maximum daily trades reached"
            
            if self.stats['daily_pnl'] <= -self.settings['max_daily_loss'] * self.stats['available_amount']:
                return False, "Maximum daily loss reached"
            
            position_size = amount * price / self.stats['available_amount']
            if position_size > self.settings['max_position_size']:
                return False, "Position size too large"
            
            return True, "Trade allowed"
            
        except Exception as e:
            logger.error(f"Error checking trade: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def update_trade_result(self, pnl):
        """Update trading statistics after trade"""
        try:
            self.stats['trades_today'] += 1
            self.stats['daily_pnl'] += pnl
            logger.info(f"Trade result updated: PnL = {pnl}")
        except Exception as e:
            logger.error(f"Error updating trade result: {str(e)}")
            raise
    
    def get_trade_history(self, days=30):
        """Get real trading history from DataManager"""
        try:
            trades = []
            cumulative_pnl = 0.0
            
            # Get real trade history from DataManager
            symbol = 'BTCUSDT'  # Default to BTC/USDT
            history = self.data_manager.get_trade_history(symbol)
            
            for trade in history:
                pnl = float(trade['quoteQty']) if trade['isBuyer'] else -float(trade['quoteQty'])
                cumulative_pnl += pnl
                
                trades.append({
                    'timestamp': pd.to_datetime(trade['time'], unit='ms'),
                    'symbol': trade['symbol'],
                    'side': 'BUY' if trade['isBuyer'] else 'SELL',
                    'price': float(trade['price']),
                    'amount': float(trade['qty']),
                    'pnl': pnl,
                    'cumulative_pnl': cumulative_pnl
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float, win_rate: float = None, risk_reward: float = None) -> float:
        """Calculate optimal position size using Kelly Criterion and risk management rules"""
        try:
            # Get win rate from model metrics if not provided
            if win_rate is None:
                win_rate = 0.55  # Conservative default
            
            # Calculate risk-reward ratio if not provided
            if risk_reward is None:
                if stop_loss <= 0 or entry_price <= 0:
                    risk_reward = 2.0  # Conservative default
                else:
                    risk_per_trade = abs(entry_price - stop_loss) / entry_price
                    reward_per_trade = risk_per_trade * 2  # Target 2:1 reward-to-risk
                    risk_reward = reward_per_trade / risk_per_trade
            
            # Calculate Kelly Criterion
            kelly_fraction = win_rate - ((1 - win_rate) / risk_reward)
            
            # Use fractional Kelly (25%) for safety
            conservative_kelly = kelly_fraction * 0.25
            
            # Calculate position size
            risk_amount = account_balance * max(0, conservative_kelly)
            
            # Apply risk limits
            max_position = account_balance * self.max_position_size
            min_position = account_balance * 0.01  # 1% minimum position
            
            # Calculate final position size
            position_size = min(risk_amount, max_position)
            position_size = max(position_size, min_position)
            
            # Adjust for current drawdown
            if self.current_drawdown > 0:
                drawdown_factor = 1 - (self.current_drawdown / self.max_drawdown)
                position_size *= max(0.5, drawdown_factor)  # Never reduce below 50%
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return account_balance * 0.01  # Conservative fallback
            
    def update_daily_loss(self, profit_loss: float, account_balance: float) -> bool:
        """Update and check daily loss limits"""
        try:
            if account_balance <= 0:
                return False
                
            self.daily_loss += profit_loss
            daily_loss_percent = abs(self.daily_loss) / account_balance
            
            return daily_loss_percent <= self.max_daily_loss
            
        except Exception as e:
            logger.error(f"Error updating daily loss: {str(e)}")
            return False
            
    def update_drawdown(self, current_balance: float) -> bool:
        """Update and check drawdown limits"""
        try:
            if self.peak_value is None or current_balance > self.peak_value:
                self.peak_value = current_balance
            
            if self.peak_value <= 0:
                return False
                
            self.current_drawdown = (self.peak_value - current_balance) / self.peak_value
            return self.current_drawdown <= self.max_drawdown
            
        except Exception as e:
            logger.error(f"Error updating drawdown: {str(e)}")
            return False
            
    def check_risk_limits(self, position_size, account_balance):
        """Check if position size meets risk limits"""
        try:
            # Check position size limit
            position_risk = position_size / account_balance
            if position_risk > self.max_position_size:
                return False
                
            # Check daily loss limit
            if abs(self.daily_loss) / account_balance > self.max_daily_loss:
                return False
                
            # Check drawdown limit
            if self.current_drawdown > self.max_drawdown:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
            
    def reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.daily_loss = 0
        
    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics"""
        try:
            account_info = self.data_manager.get_account_info()
            current_balance = float(account_info.get('totalWalletBalance', 0))
            
            # Calculate additional metrics
            drawdown_percentage = self.current_drawdown * 100 if self.current_drawdown is not None else 0
            daily_loss_percentage = (self.daily_loss / current_balance * 100) if current_balance > 0 else 0
            
            # Get trade history for win rate calculation
            trade_history = self.get_trade_history(days=30)
            total_trades = len(trade_history)
            winning_trades = len([t for t in trade_history if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate Sharpe Ratio (if we have enough trades)
            if total_trades >= 30:
                returns = [t['pnl'] / t['amount'] for t in trade_history]
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
            else:
                sharpe_ratio = 0
            
            return {
                'account_metrics': {
                    'current_balance': current_balance,
                    'peak_balance': self.peak_value,
                    'daily_loss': self.daily_loss,
                    'daily_loss_percentage': daily_loss_percentage,
                    'current_drawdown': drawdown_percentage
                },
                'risk_limits': {
                    'max_position_size': self.max_position_size * 100,
                    'max_daily_loss': self.max_daily_loss * 100,
                    'max_drawdown': self.max_drawdown * 100
                },
                'performance_metrics': {
                    'win_rate': win_rate * 100,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'sharpe_ratio': sharpe_ratio
                },
                'status': {
                    'can_trade': self._can_trade(current_balance),
                    'warnings': self._get_risk_warnings(current_balance)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return {
                'account_metrics': {},
                'risk_limits': {},
                'performance_metrics': {},
                'status': {'can_trade': False, 'warnings': [str(e)]}
            }
            
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio from returns"""
        if not returns:
            return 0
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 365)  # Daily risk-free rate
        if len(returns) > 1:
            return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(365)
        return 0
        
    def _can_trade(self, current_balance: float) -> bool:
        """Check if trading is allowed based on all risk metrics"""
        if current_balance <= 0:
            return False
        if self.daily_loss <= -self.max_daily_loss * current_balance:
            return False
        if self.current_drawdown >= self.max_drawdown:
            return False
        return True
        
    def _get_risk_warnings(self, current_balance: float) -> List[str]:
        """Get list of active risk warnings"""
        warnings = []
        if self.daily_loss < 0:
            daily_loss_pct = abs(self.daily_loss) / current_balance * 100
            warnings.append(f"Daily loss: {daily_loss_pct:.1f}% of {self.max_daily_loss*100}% limit")
        if self.current_drawdown > 0:
            warnings.append(f"Current drawdown: {self.current_drawdown*100:.1f}% of {self.max_drawdown*100}% limit")
        return warnings

    def record_trade(self, quantity: float, price: float, side: str):
        """Record a trade for risk tracking"""
        try:
            trade_value = quantity * price
            
            # Update daily PnL based on side
            if side == 'SELL':
                self.daily_loss -= trade_value
            else:
                self.daily_loss += trade_value
                
            # Update peak value if necessary
            account_info = self.data_manager.get_account_info()
            if account_info:
                current_balance = float(account_info.get('totalWalletBalance', 0))
                if self.peak_value is None or current_balance > self.peak_value:
                    self.peak_value = current_balance
                    
            logger.info(f"Trade recorded: {quantity} @ {price} ({side})")
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")