            return {float(i.mid): float(v) for i, v in volume_profile.items()}
            
        except Exception as e:
            logging.error(f'Error calculating volume profile: {e}')
            return {}

    def _find_peaks(self, data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Find peaks in data series using gradient analysis"""
        gradient = np.gradient(data)
        peaks = []
        
        for i in range(1, len(gradient)-1):
            if gradient[i-1] > 0 and gradient[i+1] < 0:
                if data[i] > threshold:
                    peaks.append(i)
                    
        return np.array(peaks)

    def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure for advanced pattern recognition"""
        try:
            structure = {
                'swing_highs': self._find_swing_points(data, 'high'),
                'swing_lows': self._find_swing_points(data, 'low'),
                'patterns': self._identify_patterns(data),
                'breakout_levels': self._find_breakout_levels(data),
                'trend_strength': self._calculate_trend_strength(data)
            }
            
            return structure
            
        except Exception as e:
            logging.error(f'Error analyzing market structure: {e}')
            return {}

    def _find_swing_points(self, data: pd.DataFrame, point_type: str, 
                          window: int = 5) -> List[Dict]:
        """Find swing high and low points"""
        swing_points = []
        price_col = 'high' if point_type == 'high' else 'low'
        
        for i in range(window, len(data) - window):
            window_data = data[price_col].iloc[i-window:i+window+1]
            
            if point_type == 'high':
                if window_data.iloc[window] == window_data.max():
                    swing_points.append({
                        'price': float(window_data.iloc[window]),
                        'index': i,
                        'timestamp': data.index[i]
                    })
            else:
                if window_data.iloc[window] == window_data.min():
                    swing_points.append({
                        'price': float(window_data.iloc[window]),
                        'index': i,
                        'timestamp': data.index[i]
                    })
                    
        return swing_points

    def _identify_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Identify chart patterns"""
        patterns = []
        
        # Find double tops/bottoms
        tops = self._find_swing_points(data, 'high')
        bottoms = self._find_swing_points(data, 'low')
        
        patterns.extend(self._find_double_patterns(tops, 'double_top'))
        patterns.extend(self._find_double_patterns(bottoms, 'double_bottom'))
        
        # Find head and shoulders
        patterns.extend(self._find_head_and_shoulders(tops, bottoms))
        
        # Find triangles
        patterns.extend(self._find_triangles(data, tops, bottoms))
        
        return patterns

    def _find_double_patterns(self, points: List[Dict], pattern_type: str, 
                            tolerance: float = 0.01) -> List[Dict]:
        """Find double top/bottom patterns"""
        patterns = []
        
        for i in range(len(points)-1):
            price1 = points[i]['price']
            price2 = points[i+1]['price']
            
            # Check if prices are within tolerance
            if abs(price1 - price2) / price1 <= tolerance:
                patterns.append({
                    'type': pattern_type,
                    'price_level': (price1 + price2) / 2,
                    'start': points[i]['timestamp'],
                    'end': points[i+1]['timestamp']
                })
                
        return patterns

    def _find_head_and_shoulders(self, tops: List[Dict], 
                                bottoms: List[Dict]) -> List[Dict]:
        """Find head and shoulders patterns"""
        patterns = []
        
        for i in range(len(tops)-2):
            # Check for regular head and shoulders
            if (tops[i]['price'] < tops[i+1]['price'] > tops[i+2]['price'] and
                abs(tops[i]['price'] - tops[i+2]['price']) / tops[i]['price'] <= 0.02):
                
                patterns.append({
                    'type': 'head_and_shoulders',
                    'head_price': tops[i+1]['price'],
                    'neckline': min(bottoms[i]['price'], bottoms[i+1]['price']),
                    'start': tops[i]['timestamp'],
                    'end': tops[i+2]['timestamp']
                })
                
        return patterns

    def _find_triangles(self, data: pd.DataFrame, tops: List[Dict], 
                       bottoms: List[Dict]) -> List[Dict]:
        """Find triangle patterns (ascending, descending, symmetric)"""
        patterns = []
        
        # Calculate trend lines
        for i in range(len(tops)-1):
            for j in range(len(bottoms)-1):
                top_slope = (tops[i+1]['price'] - tops[i]['price']) / \
                           (tops[i+1]['index'] - tops[i]['index'])
                bottom_slope = (bottoms[j+1]['price'] - bottoms[j]['price']) / \
                              (bottoms[j+1]['index'] - bottoms[j]['index'])
                
                if abs(top_slope) < 0.001 and bottom_slope > 0:
                    patterns.append({
                        'type': 'ascending_triangle',
                        'resistance': tops[i]['price'],
                        'start': min(tops[i]['timestamp'], bottoms[j]['timestamp']),
                        'end': max(tops[i+1]['timestamp'], bottoms[j+1]['timestamp'])
                    })
                elif abs(bottom_slope) < 0.001 and top_slope < 0:
                    patterns.append({
                        'type': 'descending_triangle',
                        'support': bottoms[j]['price'],
                        'start': min(tops[i]['timestamp'], bottoms[j]['timestamp']),
                        'end': max(tops[i+1]['timestamp'], bottoms[j+1]['timestamp'])
                    })
                elif abs(abs(top_slope) - abs(bottom_slope)) < 0.001:
                    patterns.append({
                        'type': 'symmetric_triangle',
                        'apex_price': (tops[i+1]['price'] + bottoms[j+1]['price']) / 2,
                        'start': min(tops[i]['timestamp'], bottoms[j]['timestamp']),
                        'end': max(tops[i+1]['timestamp'], bottoms[j+1]['timestamp'])
                    })
                    
        return patterns

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # ADX for trend strength
            adx = self._calculate_adx(data)
            
            # Price slope
            price_slope = (data['close'].iloc[-1] - data['close'].iloc[0]) / len(data)
            normalized_slope = price_slope / data['close'].mean()
            
            # R-squared of price regression
            x = np.arange(len(data))
            y = data['close'].values
            coeffs = np.polyfit(x, y, 1)
            r_squared = 1 - (sum((y - (coeffs[0] * x + coeffs[1])) ** 2) / \
                            sum((y - y.mean()) ** 2))
            
            # Combine metrics
            trend_strength = (0.4 * adx[-1] / 100 + 
                            0.3 * abs(normalized_slope) * 100 +
                            0.3 * r_squared)
            
            return float(trend_strength)
            
        except Exception as e:
            logging.error(f'Error calculating trend strength: {e}')
            return 0.0