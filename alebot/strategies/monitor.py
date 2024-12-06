            logging.error(f'Error handling unhealthy strategy: {e}')

    async def _monitor_performance(self):
        """Monitor overall trading performance"""
        while self.is_running:
            try:
                for symbol in self.config.TRADING_CONFIG['symbols']:
                    state = self.strategy_states[symbol]
                    
                    # Calculate performance metrics
                    trades = await self._get_recent_trades(symbol)
                    metrics = await self._calculate_performance_metrics(trades)
                    state.performance_metrics = metrics

                    # Adjust strategy parameters based on performance
                    if state.active:
                        await self._optimize_strategy_parameters(symbol, metrics)
                    
                    # Check for reactivation conditions
                    if not state.active and await self._check_reactivation_conditions(symbol):
                        await self._reactivate_strategy(symbol)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logging.error(f'Error monitoring performance: {e}')
                await asyncio.sleep(5)

    async def _get_recent_trades(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get recent trading history"""
        try:
            trades = await self.exchange.get_trades(
                symbol=symbol,
                start_time=datetime.now() - timedelta(days=days)
            )
            return trades
        except Exception as e:
            logging.error(f'Error getting recent trades: {e}')
            return []

    async def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if not trades:
                return {}

            # Basic metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # PnL metrics
            total_pnl = sum(t['pnl'] for t in trades)
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = abs(np.mean([t['pnl'] for t in trades if t['pnl'] < 0])) if total_trades - winning_trades > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

            # Risk metrics
            returns = pd.Series([t['pnl'] for t in trades])
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)

            # Trade analysis
            avg_hold_time = np.mean([t['duration'].total_seconds() / 3600 for t in trades])
            avg_position_size = np.mean([t['position_size'] for t in trades])

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'avg_hold_time': avg_hold_time,
                'avg_position_size': avg_position_size
            }

        except Exception as e:
            logging.error(f'Error calculating performance metrics: {e}')
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max
        return float(abs(drawdown.min()))

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        annual_factor = np.sqrt(252)  # Annualization factor for daily returns
        excess_returns = returns - self.config.RISK_CONFIG['risk_free_rate'] / 252
        return float(annual_factor * excess_returns.mean() / returns.std())

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.config.RISK_CONFIG['risk_free_rate'] / 252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0001
        return float(np.sqrt(252) * excess_returns.mean() / downside_std)

    async def _optimize_strategy_parameters(self, symbol: str, metrics: Dict):
        """Optimize strategy parameters based on performance"""
        try:
            # Adjust position sizing based on performance
            if metrics['win_rate'] > 0.6 and metrics['profit_factor'] > 2.0:
                await self._increase_position_size(symbol)
            elif metrics['win_rate'] < 0.4 or metrics['profit_factor'] < 1.5:
                await self._decrease_position_size(symbol)

            # Adjust stop loss based on volatility
            volatility = self.strategy_states[symbol].risk_metrics['volatility']
            await self._adjust_stop_loss(symbol, volatility)

            # Adjust trade frequency based on market conditions
            await self._adjust_trade_frequency(symbol, metrics)

        except Exception as e:
            logging.error(f'Error optimizing strategy parameters: {e}')

    async def _check_reactivation_conditions(self, symbol: str) -> bool:
        """Check if strategy can be reactivated"""
        try:
            state = self.strategy_states[symbol]
            
            # Check if enough time has passed
            cool_off_period = timedelta(hours=24)
            if datetime.now() - state.last_update < cool_off_period:
                return False

            # Check market conditions
            market_analysis = await self.analysis.analyze(symbol)
            if market_analysis.trend == 'SIDEWAYS':
                return False

            # Check risk metrics
            risk_metrics = await self.risk_manager.calculate_risk_metrics(symbol)
            if risk_metrics.value_at_risk > self.config.RISK_CONFIG['max_var']:
                return False

            return True

        except Exception as e:
            logging.error(f'Error checking reactivation conditions: {e}')
            return False

    async def _reactivate_strategy(self, symbol: str):
        """Reactivate strategy with conservative parameters"""
        try:
            state = self.strategy_states[symbol]
            
            # Reset strategy parameters
            state.active = True
            state.last_signal = None
            state.performance_metrics = {}
            
            # Start with conservative position sizing
            await self._reset_position_size(symbol)
            
            logging.info(f'Strategy reactivated for {symbol}')
            
        except Exception as e:
            logging.error(f'Error reactivating strategy: {e}')
