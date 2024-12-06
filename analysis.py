                    'type': SignalType.BUY,
                    'reason': 'Price below BB lower',
                    'strength': 0.65
                })
            elif current_price > bb_upper:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': 'Price above BB upper',
                    'strength': 0.65
                })

            # Moving Average signals
            sma_20 = indicators['sma_20'].iloc[-1]
            sma_50 = indicators['sma_50'].iloc[-1]
            sma_200 = indicators['sma_200'].iloc[-1]

            if sma_20 > sma_50 and sma_50 > sma_200:
                signals.append({
                    'type': SignalType.BUY,
                    'reason': 'Moving averages aligned bullish',
                    'strength': 0.9
                })
            elif sma_20 < sma_50 and sma_50 < sma_200:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': 'Moving averages aligned bearish',
                    'strength': 0.9
                })

            return signals

        except Exception as e:
            logger.error(f'Error generating signals: {e}')
            return []

    def calculate_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            close = data['close'].values

            # Find local maxima and minima
            resistance_levels = []
            support_levels = []

            for i in range(2, len(data) - 2):
                # Resistance
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    resistance_levels.append(highs[i])

                # Support
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    support_levels.append(lows[i])

            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))

            return support_levels, resistance_levels

        except Exception as e:
            logger.error(f'Error calculating support/resistance: {e}')
            return [], []

    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        try:
            # Calculate True Range
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            pdm = [0]
            ndm = [0]
            tr = [0]

            for i in range(1, len(data)):
                tr_i = max([high[i] - low[i],
                          abs(high[i] - close[i-1]),
                          abs(low[i] - close[i-1])])
                tr.append(tr_i)

                # +DM and -DM
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]

                if up_move > down_move and up_move > 0:
                    pdm.append(up_move)
                else:
                    pdm.append(0)

                if down_move > up_move and down_move > 0:
                    ndm.append(down_move)
                else:
                    ndm.append(0)

            # Convert to arrays
            tr = np.array(tr)
            pdm = np.array(pdm)
            ndm = np.array(ndm)

            # Calculate smoothed values
            tr14 = pd.Series(tr).rolling(14).mean().values
            pdm14 = pd.Series(pdm).rolling(14).mean().values
            ndm14 = pd.Series(ndm).rolling(14).mean().values

            # Calculate +DI and -DI
            pdi = 100 * (pdm14 / tr14)
            ndi = 100 * (ndm14 / tr14)

            # Calculate ADX
            dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
            adx = pd.Series(dx).rolling(14).mean().values

            return float(adx[-1]) if not np.isnan(adx[-1]) else 0.0

        except Exception as e:
            logger.error(f'Error calculating trend strength: {e}')
            return 0.0

    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 50) -> Dict[float, float]:
        """Calculate volume profile"""
        try:
            # Create price bins
            price_bins = pd.cut(data['close'], bins=bins)
            volume_profile = data.groupby(price_bins)['volume'].sum()

            return {float(i.mid): float(v) for i, v in volume_profile.items()}

        except Exception as e:
            logger.error(f'Error calculating volume profile: {e}')
            return {}

    async def get_analysis_summary(self, symbol: str) -> Dict:
        """Get comprehensive analysis summary"""
        try:
            analysis = await self.analyze(symbol)
            if not analysis:
                return {}

            data = await self.exchange.get_market_data(
                symbol=symbol,
                timeframe=TimeFrame(self.config.TRADING_CONFIG['base_timeframe'])
            )

            indicators = analysis['indicators']
            signals = analysis['signals']

            # Calculate additional metrics
            support_levels, resistance_levels = self.calculate_support_resistance(data)
            trend_strength = self.calculate_trend_strength(data)
            volume_profile = self.calculate_volume_profile(data)

            return {
                'signals': signals,
                'trend_strength': trend_strength,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'volume_profile': volume_profile,
                'indicators': {
                    'rsi': float(indicators['rsi'].iloc[-1]),
                    'macd': float(indicators['macd'].iloc[-1]),
                    'macd_signal': float(indicators['macd_signal'].iloc[-1]),
                    'bb_upper': float(indicators['bb_upper'].iloc[-1]),
                    'bb_lower': float(indicators['bb_lower'].iloc[-1])
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f'Error getting analysis summary: {e}')
            return {}
