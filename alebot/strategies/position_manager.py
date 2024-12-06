            base_size = self.position_states.get(symbol, {}).get('size',
                self.config.TRADING_CONFIG['initial_position_size'])

            # Kelly Criterion calculation
            win_rate = signal.get('confidence', 0.5)
            win_loss_ratio = abs(signal['take_profit'] - signal['price']) / \
                            abs(signal['price'] - signal['stop_loss'])
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Apply half-Kelly for safety
            kelly_fraction *= 0.5

            # Get account equity
            equity = await self.exchange.get_account_value()
            kelly_size = equity * kelly_fraction

            # Adjust for volatility
            volatility = await self.risk_manager.get_volatility(symbol)
            volatility_factor = self.config.RISK_CONFIG['base_volatility'] / volatility
            adjusted_size = kelly_size * volatility_factor

            # Adjust for correlation
            correlations = await self.risk_manager.get_correlations(symbol)
            correlation_factor = self._calculate_correlation_factor(correlations)
            adjusted_size *= correlation_factor

            # Apply risk limits
            max_size = min(
                self.config.TRADING_CONFIG['max_position_size'],
                equity * self.config.RISK_CONFIG['max_risk_per_trade']
            )
            min_size = self.config.TRADING_CONFIG['min_position_size']
            
            final_size = max(min(adjusted_size, max_size), min_size)

            return float(final_size)

        except Exception as e:
            logging.error(f'Error calculating optimal position size: {e}')
            return None

    def _calculate_correlation_factor(self, correlations: Dict[str, float]) -> float:
        """Calculate position size adjustment factor based on correlations"""
        try:
            if not correlations:
                return 1.0

            # Calculate average absolute correlation
            avg_correlation = sum(abs(c) for c in correlations.values()) / len(correlations)
            
            # Reduce position size as correlation increases
            correlation_factor = 1.0 - (avg_correlation * self.config.RISK_CONFIG['correlation_penalty'])
            
            # Ensure factor stays within reasonable bounds
            return max(min(correlation_factor, 1.0), 0.2)

        except Exception as e:
            logging.error(f'Error calculating correlation factor: {e}')
            return 1.0

    async def maintain_portfolio_balance(self):
        """Maintain balanced portfolio risk"""
        try:
            positions = await self.exchange.get_positions()
            if not positions:
                return

            # Calculate total exposure
            total_exposure = sum(p['notional_value'] for p in positions.values())
            account_value = await self.exchange.get_account_value()
            exposure_ratio = total_exposure / account_value

            if exposure_ratio > self.config.RISK_CONFIG['max_portfolio_exposure']:
                # Reduce all position sizes proportionally
                reduction_factor = self.config.RISK_CONFIG['max_portfolio_exposure'] / exposure_ratio
                for symbol in positions:
                    await self._decrease_position_size(symbol, factor=reduction_factor)

            logging.info(f'Portfolio balance maintained. Exposure ratio: {exposure_ratio:.2%}')

        except Exception as e:
            logging.error(f'Error maintaining portfolio balance: {e}')

    async def update_trailing_stops(self):
        """Update trailing stops for all positions"""
        try:
            positions = await self.exchange.get_positions()
            for symbol, position in positions.items():
                # Get ATR for dynamic stop distance
                atr = await self.risk_manager.get_atr(symbol)
                current_price = await self.exchange.get_current_price(symbol)

                if position['side'] == 'BUY':
                    # For long positions
                    new_stop = current_price - (atr * self.config.RISK_CONFIG['trailing_stop_atr_multiple'])
                    if new_stop > position['stop_loss']:
                        await self.exchange.modify_stop_loss(
                            symbol=symbol,
                            order_id=position['stop_loss_order'],
                            new_stop=new_stop
                        )
                else:
                    # For short positions
                    new_stop = current_price + (atr * self.config.RISK_CONFIG['trailing_stop_atr_multiple'])
                    if new_stop < position['stop_loss']:
                        await self.exchange.modify_stop_loss(
                            symbol=symbol,
                            order_id=position['stop_loss_order'],
                            new_stop=new_stop
                        )

            logging.info('Trailing stops updated successfully')

        except Exception as e:
            logging.error(f'Error updating trailing stops: {e}')

    async def adjust_take_profits(self):
        """Dynamically adjust take-profit levels based on market conditions"""
        try:
            positions = await self.exchange.get_positions()
            for symbol, position in positions.items():
                # Get volatility and trend strength
                volatility = await self.risk_manager.get_volatility(symbol)
                trend_strength = await self.risk_manager.get_trend_strength(symbol)

                # Calculate dynamic take-profit multiplier
                base_tp = self.config.RISK_CONFIG['base_take_profit_multiple']
                volatility_adjustment = volatility / self.config.RISK_CONFIG['base_volatility']
                trend_adjustment = trend_strength / 50  # Normalize trend strength

                tp_multiplier = base_tp * (1 + volatility_adjustment + trend_adjustment)

                # Calculate new take-profit level
                entry_price = position['entry_price']
                if position['side'] == 'BUY':
                    new_tp = entry_price * (1 + tp_multiplier)
                else:
                    new_tp = entry_price * (1 - tp_multiplier)

                # Update take-profit order
                await self.exchange.modify_take_profit(
                    symbol=symbol,
                    order_id=position['take_profit_order'],
                    new_tp=new_tp
                )

            logging.info('Take-profit levels adjusted successfully')

        except Exception as e:
            logging.error(f'Error adjusting take-profits: {e}')
