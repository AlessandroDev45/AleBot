import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union

class MLModel:
    def __init__(self, config, analysis):
        self.config = config
        self.analysis = analysis
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.last_update = {}
        self.setup_models()

    def setup_models(self):
        """Initialize ML models"""
        try:
            # LSTM for sequence prediction
            self.models['lstm'] = self._create_lstm_model()
            
            # XGBoost for classification
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='gpu_hist' if self.config.ML_CONFIG['gpu_enabled'] else 'hist'
            )
            
            # LightGBM for regression
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                device='gpu' if self.config.ML_CONFIG['gpu_enabled'] else 'cpu'
            )
            
        except Exception as e:
            logging.error(f'Error setting up models: {e}')
            raise

    def _create_lstm_model(self) -> Sequential:
        """Create LSTM model architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(60, 20)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='huber',
            metrics=['mae']
        )
        
        return model

    async def prepare_features(self, symbol: str) -> pd.DataFrame:
        """Prepare features for ML models"""
        try:
            # Get market data
            data = await self.analysis.get_market_data(symbol)
            
            # Technical indicators
            indicators = await self.analysis.calculate_indicators(data)
            
            # Market microstructure features
            microstructure = await self._get_microstructure_features(symbol)
            
            # Volume profile features
            volume_profile = await self.analysis.get_volume_profile(symbol)
            
            # Combine features
            features = pd.DataFrame({
                'close': data['close'],
                'volume': data['volume'],
                'rsi': indicators['rsi'],
                'macd': indicators['macd'],
                'bb_upper': indicators['bb_upper'],
                'bb_lower': indicators['bb_lower'],
                'atr': indicators['atr'],
                'order_imbalance': microstructure['order_imbalance'],
                'spread': microstructure['spread'],
                'volume_profile': volume_profile
            })
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            self.scalers[symbol] = scaler
            
            return pd.DataFrame(scaled_features, columns=features.columns)
            
        except Exception as e:
            logging.error(f'Error preparing features: {e}')
            raise

    async def train_models(self, symbol: str):
        """Train all models"""
        try:
            # Prepare data
            features = await self.prepare_features(symbol)
            targets = await self._prepare_targets(symbol)
            
            # Split data
            train_size = int(len(features) * 0.8)
            train_features = features[:train_size]
            train_targets = targets[:train_size]
            test_features = features[train_size:]
            test_targets = targets[train_size:]
            
            # Train LSTM
            sequences = self._prepare_sequences(train_features)
            self.models['lstm'].fit(
                sequences,
                train_targets['price_direction'][60:],
                epochs=50,
                batch_size=32,
                validation_split=0.2
            )
            
            # Train XGBoost
            self.models['xgboost'].fit(
                train_features,
                train_targets['trend'],
                eval_set=[(test_features, test_targets['trend'])],
                early_stopping_rounds=10
            )
            
            # Train LightGBM
            self.models['lightgbm'].fit(
                train_features,
                train_targets['returns'],
                eval_set=[(test_features, test_targets['returns'])],
                early_stopping_rounds=10
            )
            
            # Update feature importance
            self.feature_importance[symbol] = {
                'xgboost': dict(zip(features.columns, self.models['xgboost'].feature_importances_)),
                'lightgbm': dict(zip(features.columns, self.models['lightgbm'].feature_importances_))
            }
            
            self.last_update[symbol] = datetime.now()
            
        except Exception as e:
            logging.error(f'Error training models: {e}')
            raise

    async def predict(self, symbol: str) -> Dict:
        """Generate predictions from all models"""
        try:
            # Check if models need updating
            if await self._should_update(symbol):
                await self.train_models(symbol)
            
            # Get features
            features = await self.prepare_features(symbol)
            
            # Generate predictions
            lstm_pred = self.models['lstm'].predict(self._prepare_sequences(features[-60:]))
            xgb_pred = self.models['xgboost'].predict_proba(features.iloc[-1:])
            lgb_pred = self.models['lightgbm'].predict(features.iloc[-1:])
            
            # Combine predictions
            ensemble_weights = self.config.ML_CONFIG['ensemble_weights']
            combined_score = (
                lstm_pred[-1][0] * ensemble_weights['lstm'] +
                xgb_pred[0][1] * ensemble_weights['xgboost'] +
                lgb_pred[0] * ensemble_weights['lightgbm']
            )
            
            return {
                'signal': 'BUY' if combined_score > 0.5 else 'SELL',
                'confidence': abs(combined_score - 0.5) * 2,
                'predictions': {
                    'lstm': float(lstm_pred[-1][0]),
                    'xgboost': float(xgb_pred[0][1]),
                    'lightgbm': float(lgb_pred[0])
                }
            }
            
        except Exception as e:
            logging.error(f'Error generating predictions: {e}')
            raise

    async def _should_update(self, symbol: str) -> bool:
        """Check if models need updating"""
        if symbol not in self.last_update:
            return True
            
        hours_since_update = (datetime.now() - self.last_update[symbol]).total_seconds() / 3600
        return hours_since_update >= self.config.ML_CONFIG['update_interval']