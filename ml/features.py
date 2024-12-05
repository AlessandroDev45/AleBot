import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class FeatureEngine:
    def __init__(self):
        self.scalers = {}

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        features = []
        feature_names = []

        # Price features
        features.extend(self._create_price_features(data))
        feature_names.extend(['price_ma', 'price_std', 'returns'])

        # Volume features
        features.extend(self._create_volume_features(data))
        feature_names.extend(['volume_ma', 'volume_std'])

        # Technical indicators
        tech_features, tech_names = self._create_technical_features(data)
        features.extend(tech_features)
        feature_names.extend(tech_names)

        return np.array(features).T, feature_names

    def _create_price_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        price = data['close'].values
        features = [
            self._moving_average(price, 20),
            self._rolling_std(price, 20),
            np.diff(np.log(price), prepend=price[0])
        ]
        return features