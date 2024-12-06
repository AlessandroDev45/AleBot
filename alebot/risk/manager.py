    async def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        try:
            positions = await self.exchange.get_positions()
            total_exposure = sum(float(p['notional_value']) for p in positions.values())
            account_value = await self.exchange.get_account_value()
            return total_exposure / account_value if account_value > 0 else 0
            
        except Exception as e:
            logging.error(f'Error calculating exposure: {e}')
            return 0.0

    async def _check_correlation(self, symbol: str) -> bool:
        """Check correlation with existing positions"""
        try:
            if not self.risk_metrics.get(symbol):
                await self.calculate_risk_metrics(symbol)
            
            correlations = self.risk_metrics[symbol].correlation
            max_correlation = float(self.config.RISK_CONFIG['max_correlation'])
            
            # Check correlation with existing positions
            positions = await self.exchange.get_positions()
            for pos_symbol in positions:
                if pos_symbol in correlations:
                    if abs(correlations[pos_symbol]) > max_correlation:
                        return False
            return True
            
        except Exception as e:
            logging.error(f'Error checking correlation: {e}')
            return False

    def _check_risk_limits(self, metrics: RiskMetrics) -> bool:
        """Check if risk metrics are within acceptable limits"""
        try:
            # Check Value at Risk
            if abs(metrics.value_at_risk) > self.config.RISK_CONFIG['max_var']:
                return False
                
            # Check volatility
            if metrics.volatility > self.config.RISK_CONFIG['volatility_threshold']:
                return False
                
            # Check drawdown
            if metrics.max_drawdown > self.config.RISK_CONFIG['max_drawdown']:
                return False
                
            # Check Sharpe ratio
            if metrics.sharpe_ratio < self.config.RISK_CONFIG['min_sharpe']:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f'Error checking risk limits: {e}')
            return False

    async def _get_market_returns(self) -> np.ndarray:
        """Get market benchmark returns (using BTC)"""
        try:
            market_data = await self.exchange.get_market_data('BTCUSDT')
            return np.diff(np.log(market_data['close']))
        except Exception as e:
            logging.error(f'Error getting market returns: {e}')
            return np.array([])

    async def _calculate_correlations(self, symbol: str, returns: np.ndarray) -> Dict[str, float]:
        """Calculate correlations with other traded symbols"""
        try:
            correlations = {}
            for other_symbol in self.config.TRADING_CONFIG['symbols']:
                if other_symbol != symbol:
                    other_data = await self.exchange.get_market_data(other_symbol)
                    other_returns = np.diff(np.log(other_data['close']))
                    if len(other_returns) == len(returns):
                        corr = np.corrcoef(returns, other_returns)[0, 1]
                        correlations[other_symbol] = float(corr)
            return correlations
        except Exception as e:
            logging.error(f'Error calculating correlations: {e}')
            return {}

    async def update_risk_metrics(self):
        """Update risk metrics for all traded symbols"""
        try:
            for symbol in self.config.TRADING_CONFIG['symbols']:
                metrics = await self.calculate_risk_metrics(symbol)
                self.risk_metrics[symbol] = metrics
                
                # Log significant changes
                if metrics.value_at_risk > self.config.RISK_CONFIG['max_var']:
                    logging.warning(f'High VaR for {symbol}: {metrics.value_at_risk:.2%}')
                if metrics.volatility > self.config.RISK_CONFIG['volatility_threshold']:
                    logging.warning(f'High volatility for {symbol}: {metrics.volatility:.2%}')
                    
        except Exception as e:
            logging.error(f'Error updating risk metrics: {e}')

    async def monitor_portfolio_risk(self):
        """Continuous portfolio risk monitoring"""
        while True:
            try:
                # Update risk metrics
                await self.update_risk_metrics()
                
                # Calculate portfolio level metrics
                total_exposure = await self._calculate_total_exposure()
                if total_exposure > self.max_position_size:
                    logging.warning(f'High portfolio exposure: {total_exposure:.2%}')
                    
                # Check for risk limit breaches
                for symbol, metrics in self.risk_metrics.items():
                    if not self._check_risk_limits(metrics):
                        logging.warning(f'Risk limits breached for {symbol}')
                        # Implement risk reduction if needed
                        
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logging.error(f'Error monitoring portfolio risk: {e}')
                await asyncio.sleep(5)