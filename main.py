            current_price = self.exchange.last_prices[symbol]
            stop_loss = current_price * (1 + self.config.RISK_CONFIG['stop_loss_margin'])
            take_profit = current_price * (1 - self.config.RISK_CONFIG['take_profit_margin'])

            # Place main order
            order = await self.exchange.place_order(
                symbol=symbol,
                side='SELL',
                quantity=size
            )

            if order:
                # Save position
                self.active_positions[symbol] = TradePosition(
                    symbol=symbol,
                    side='SELL',
                    entry_price=current_price,
                    quantity=size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=datetime.now()
                )

                logger.info(f'Sell order placed for {symbol}: {order}')

        except Exception as e:
            logger.error(f'Error placing sell order: {e}')

    async def _check_exit_conditions(self, symbol: str, position: TradePosition):
        """Check if position should be closed"""
        try:
            current_price = self.exchange.last_prices.get(symbol)
            if not current_price:
                return

            # Check stop loss
            if position.side == 'BUY':
                if current_price <= position.stop_loss:
                    await self._close_position(symbol, 'Stop loss hit')
                elif current_price >= position.take_profit:
                    await self._close_position(symbol, 'Take profit hit')
            else:  # SELL position
                if current_price >= position.stop_loss:
                    await self._close_position(symbol, 'Stop loss hit')
                elif current_price <= position.take_profit:
                    await self._close_position(symbol, 'Take profit hit')

        except Exception as e:
            logger.error(f'Error checking exit conditions: {e}')

    async def _close_position(self, symbol: str, reason: str):
        """Close position for symbol"""
        try:
            position = self.active_positions.get(symbol)
            if not position:
                return

            # Place closing order
            close_side = 'SELL' if position.side == 'BUY' else 'BUY'
            order = await self.exchange.place_order(
                symbol=symbol,
                side=close_side,
                quantity=position.quantity
            )

            if order:
                # Calculate P&L
                exit_price = self.exchange.last_prices[symbol]
                pnl = (exit_price - position.entry_price) * position.quantity \
                    if position.side == 'BUY' else \
                    (position.entry_price - exit_price) * position.quantity

                # Log trade
                logger.info(f'Closed {symbol} position: {reason}, PnL: {pnl:.2f}')

                # Remove from active positions
                del self.active_positions[symbol]

                # Save trade to database
                await self._save_trade_record(position, exit_price, pnl, reason)

        except Exception as e:
            logger.error(f'Error closing position: {e}')

    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            for symbol in list(self.active_positions.keys()):
                await self._close_position(symbol, 'Bot shutdown')
        except Exception as e:
            logger.error(f'Error closing all positions: {e}')

    async def _update_market_data(self, symbol: str, analysis: Dict):
        """Update market data in database"""
        try:
            # Prepare market data record
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'timeframe': TimeFrame(self.config.TRADING_CONFIG['base_timeframe']),
                'price': self.exchange.last_prices[symbol],
                'indicators': analysis['indicators'],
                'signals': analysis['signals'],
                'support_resistance': {
                    'support': analysis['support_levels'],
                    'resistance': analysis['resistance_levels']
                }
            }

            # Save to database
            await self.db.insert_market_data(market_data)

        except Exception as e:
            logger.error(f'Error updating market data: {e}')

    async def _save_trade_record(self, position: TradePosition, exit_price: float, 
                               pnl: float, reason: str):
        """Save completed trade to database"""
        try:
            trade_record = {
                'symbol': position.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
                'pnl': pnl,
                'reason': reason,
                'entry_time': position.timestamp,
                'exit_time': datetime.now()
            }

            await self.db.insert_trade(trade_record)

        except Exception as e:
            logger.error(f'Error saving trade record: {e}')

async def main():
    """Main entry point"""
    try:
        # Initialize and start bot
        bot = TradingBot()
        await bot.start()

    except KeyboardInterrupt:
        logger.info('Bot stopped by user')
        await bot.stop()

    except Exception as e:
        logger.error(f'Critical error: {e}')
        raise

if __name__ == '__main__':
    # Set up asyncio event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except Exception as e:
        logger.error(f'Fatal error: {e}')
    finally:
        loop.close()