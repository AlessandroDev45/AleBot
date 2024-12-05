import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self, config, analysis):
        self.config = config
        self.analysis = analysis
        self.models = {}
        self.scalers = {}
        self.setup_models()

    def setup_models(self):
        self.models['price'] = self._create_lstm_model()
        self.models['trend'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            tree_method='gpu_hist'
        )
        self.models['volatility'] = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            device='gpu'
        )

    def _create_lstm_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(60, 20)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        return model