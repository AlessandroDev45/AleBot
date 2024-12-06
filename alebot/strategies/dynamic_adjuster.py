            logging.info(f'Adjusted parameters for {symbol} based on {regime} regime')
            
        except Exception as e:
            logging.error(f'Error adjusting parameters for regime: {e}')

    async def optimize_parameters(self, symbol: str):
        """Optimize strategy parameters using historical performance"""
        try:
            # Get recent performance data
            trades = await self._get_recent_trades(symbol)
            if not trades:
                return

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(trades)
            self.performance_history[symbol].append(metrics)

            # Perform optimization if we have enough history
            if len(self.performance_history[symbol]) >= 3:
                await self._optimize_based_on_performance(symbol)

            logging.info(f'Optimized parameters for {symbol}')

        except Exception as e:
            logging.error(f'Error optimizing parameters: {e}')

    async def _get_recent_trades(self, symbol: str) -> List[Dict]:
        """Get recent trading history"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            trades = await self.exchange.get_trade_history(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            return trades
        except Exception as e:
            logging.error(f'Error getting recent trades: {e}')
            return []

    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics for optimization"""
        try:
            if not trades:
                return {}

            # Calculate base metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades

            # Calculate average returns
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = abs(np.mean([t['pnl'] for t in trades if t['pnl'] < 0])) if total_trades - winning_trades > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

            # Calculate trade efficiency
            avg_trade_duration = np.mean([t['duration'].total_seconds() for t in trades])
            execution_quality = np.mean([abs(t['entry_price'] - t['target_entry']) / t['target_entry'] for t in trades])

            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade_duration': avg_trade_duration,
                'execution_quality': execution_quality
            }

        except Exception as e:
            logging.error(f'Error calculating performance metrics: {e}')
            return {}

    async def _optimize_based_on_performance(self, symbol: str):
        """Optimize parameters based on historical performance"""
        try:
            # Get current parameters
            params = self.strategy_parameters[symbol]
            history = self.performance_history[symbol][-3:]

            # Calculate performance trends
            win_rate_trend = (history[-1]['win_rate'] - history[0]['win_rate']) / history[0]['win_rate']
            profit_factor_trend = (history[-1]['profit_factor'] - history[0]['profit_factor']) / history[0]['profit_factor']

            # Adjust entry/exit thresholds
            if win_rate_trend < -0.1:
                # Increase thresholds for more selective entries
                params.entry_threshold *= 1.1
                params.exit_threshold *= 0.9
            elif win_rate_trend > 0.1:
                # Consider slightly looser thresholds
                params.entry_threshold *= 0.95
                params.exit_threshold *= 1.05

            # Adjust position sizing
            if profit_factor_trend > 0.1 and history[-1]['profit_factor'] > 2.0:
                # Increase position size for good performance
                params.position_size_multiplier *= 1.1
            elif profit_factor_trend < -0.1 or history[-1]['profit_factor'] < 1.5:
                # Reduce position size for poor performance
                params.position_size_multiplier *= 0.9

            # Adjust trade frequency based on execution quality
            avg_execution_quality = np.mean([h['execution_quality'] for h in history])
            if avg_execution_quality > 0.02:  # More than 2% slippage
                params.trade_frequency = int(params.trade_frequency * 1.2)  # Reduce frequency
            elif avg_execution_quality < 0.01:  # Less than 1% slippage
                params.trade_frequency = int(params.trade_frequency * 0.9)  # Increase frequency

            # Apply constraints
            params = self._apply_parameter_constraints(params)

            # Update parameters
            self.strategy_parameters[symbol] = params

            logging.info(f'Optimized parameters based on performance for {symbol}')

        except Exception as e:
            logging.error(f'Error optimizing based on performance: {e}')

    def _apply_parameter_constraints(self, params: StrategyParameters) -> StrategyParameters:
        """Apply constraints to parameters to ensure they stay within acceptable ranges"""
        try:
            # Entry/exit thresholds
            params.entry_threshold = max(min(params.entry_threshold, 0.9), 0.3)
            params.exit_threshold = max(min(params.exit_threshold, 0.8), 0.2)

            # Stop loss and take profit multipliers
            params.stop_loss_multiplier = max(min(params.stop_loss_multiplier, 2.0), 0.5)
            params.take_profit_multiplier = max(min(params.take_profit_multiplier, 3.0), 1.0)

            # Position size multiplier
            params.position_size_multiplier = max(min(params.position_size_multiplier, 2.0), 0.2)

            # Trade frequency
            min_freq = self.config.TRADING_CONFIG['min_trade_interval']
            max_freq = self.config.TRADING_CONFIG['max_trade_interval']
            params.trade_frequency = max(min(params.trade_frequency, max_freq), min_freq)

            return params

        except Exception as e:
            logging.error(f'Error applying parameter constraints: {e}')
            return params

    async def get_optimized_parameters(self, symbol: str) -> Optional[StrategyParameters]:
        """Get current optimized parameters for a symbol"""
        try:
            # Update market regime
            await self.update_market_regime(symbol)
            
            # Optimize parameters
            await self.optimize_parameters(symbol)
            
            return self.strategy_parameters.get(symbol)
            
        except Exception as e:
            logging.error(f'Error getting optimized parameters: {e}')
            return None