import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
import gc
import os
import glob
import ta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class DeepLearningModel(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()
        self.input_size = input_size
        hidden_dim = 256
        self.use_amp = True
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.transformer_out = nn.Linear(hidden_dim, 64)
        
        # CNN
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),  # First layer: input_size -> 64
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1)
        ])
        
        self.residual_layers = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=1),
            nn.Conv1d(64, 64, kernel_size=1)
        ])
        
        self.cnn_out = nn.Linear(64, 64)
        
        # GRU
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.gru_out = nn.Linear(hidden_dim * 2, 64)
        
        # Ensemble
        self.ensemble_attention = nn.MultiheadAttention(64, num_heads=4)
        self.final_ensemble = nn.Linear(64, 1)  # Removed sigmoid activation
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the model."""
        batch_size = x.size(0)
        
        # Ensure input has correct dimensions [batch, seq, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
        elif x.dim() == 4:  # [batch, channels, seq, features]
            x = x.view(batch_size, -1, self.input_size)
        
        # Transformer path
        x_transformer = self.input_projection(x)  # [batch, seq, hidden_dim]
        transformer_out = self.transformer_encoder(x_transformer)
        transformer_out = self.transformer_out(transformer_out.mean(dim=1))  # [batch, 64]
        
        # CNN path
        x_cnn = x.transpose(1, 2)  # [batch, features, seq]
        cnn_out = self.conv_layers[0](x_cnn)  # [batch, 64, seq]
        
        for conv, residual in zip(self.conv_layers[1:], self.residual_layers):
            identity = cnn_out
            cnn_out = conv(cnn_out)
            residual_out = residual(identity)
            cnn_out = cnn_out + residual_out
        
        cnn_out = self.cnn_out(cnn_out.mean(dim=2))  # [batch, 64]
        
        # GRU path
        gru_out, _ = self.gru(x)  # [batch, seq, hidden_dim*2]
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        gru_out = self.gru_out(attn_out[:, -1, :])  # [batch, 64]
        
        # Ensemble
        ensemble_input = torch.stack([
            transformer_out, cnn_out, gru_out
        ], dim=1)  # [batch, 3, 64]
        
        ensemble_out, _ = self.ensemble_attention(
            ensemble_input, ensemble_input, ensemble_input
        )
        
        # Apply final layer without sigmoid activation
        out = self.final_ensemble(ensemble_out.mean(dim=1))  # [batch, 1]
        
        return out

class MLModel:
    def __init__(self, data_manager=None):
        """Initialize ML model with default configuration"""
        # Store data_manager instance
        if data_manager is None:
            raise ValueError("DataManager must be provided")
        self.data_manager = data_manager
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.feature_names = []
        self.input_size = None
        self.feature_scaler = RobustScaler()
        
        # Load saved hyperparameters or use defaults
        saved_hyperparams_path = os.path.join("models", "hyperparameters.pt")
        if os.path.exists(saved_hyperparams_path):
            try:
                self.hyperparameters = torch.load(saved_hyperparams_path)
                logger.info("Loaded saved hyperparameters")
            except Exception as e:
                logger.warning(f"Failed to load saved hyperparameters: {e}")
                self.hyperparameters = self._get_default_hyperparameters()
        else:
            self.hyperparameters = self._get_default_hyperparameters()
        
        # Initialize model status
        self.model_status = {
            'is_training': False,
            'is_ready': False,
            'current_metrics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'trading_metrics': {
                'total_trades': 0,
                'successful_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'avg_confidence': 0.0,
                'high_confidence_win_rate': 0.0
            },
            'performance_history': [],
            'predictions': {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'strength': 0.0,
                'horizon': 'Short',
                'risk': 'Low'
            },
            'training_progress': {
                'current_epoch': 0,
                'total_epochs': 500,
                'elapsed_time': '0:00:00',
                'eta': 'N/A'
            },
            'gpu_metrics': {
                'gpu_memory_used': 0,
                'gpu_memory_cached': 0,
                'gpu_memory_total': 0,
                'gpu_memory_peak': 0,
                'gpu_utilization': 0,
                'gpu_temperature': 0
            }
        }
        
        self.training_status = {
            'epoch': 0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'metrics': {},
            'training_progress': self.model_status['training_progress']
        }
        
        logger.info("MLModel initialized with default configuration")

    def _get_default_hyperparameters(self):
        """Get default hyperparameters for the model."""
        return {
            'model_type': 'ensemble',
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'sequence_length': 30,
            'stride': 5,
            'hidden_size': 256,
            'num_layers': 3,
            'dropout_rate': 0.2,
            'weight_decay': 1e-5,
            'scheduler_mode': 'min',
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
            'scheduler_min_lr': 1e-6,
            'early_stopping_patience': 20,
            'use_amp': True,
            'confidence_threshold': 0.7,
            'position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_positions': 3
        }

    def _get_param_value(self, param_name, default_value=None):
        """Get parameter value from hyperparameters with fallback to default."""
        try:
            if isinstance(self.hyperparameters.get(param_name), dict):
                return self.hyperparameters[param_name].get('value', default_value)
            return self.hyperparameters.get(param_name, default_value)
        except Exception as e:
            logger.warning(f"Error getting parameter {param_name}: {str(e)}")
            return default_value

    def _validate_required_params(self):
        """Validate that all required parameters are present."""
        try:
            required_params = [
                'model_type',
                'learning_rate',
                'batch_size',
                'epochs',
                'sequence_length',
                'stride'
            ]
            
            for param in required_params:
                value = self._get_param_value(param)
                if value is None:
                    logger.error(f"Missing required parameter: {param}")
                    return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating parameters: {str(e)}")
            return False

    def build_model(self):
        """Build the neural network model."""
        try:
            logger.info("Building model architecture...")
            
            # Validate required parameters
            if not self._validate_required_params():
                raise ValueError("Missing required parameters")
            
            # Initialize model components
            if not self.initialize_model():
                raise ValueError("Failed to initialize model")
            
            logger.info("Model built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            self.model_status['is_ready'] = False
            return False

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        # Get price differences
        delta = prices.diff()
        
        # Create gain (positive) and loss (negative) Series
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality and structure."""
        try:
            logger.info("Starting data validation...")
            print("Validating columns:", data.columns.tolist())
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error("Missing required columns")
                print("Missing columns:", [col for col in required_columns if col not in data.columns])
                return False
            
            # Check for invalid values
            if (data['high'] < data['low']).any():
                logger.error("Invalid high-low relationship detected")
                print("Invalid high-low pairs:", data[data['high'] < data['low']])
                return False
                
            if (data['volume'] <= 0).any():
                logger.error("Non-positive volume values detected")
                print("Zero or negative volumes:", data[data['volume'] <= 0])
                return False
            
            # Check for time gaps - only for 1m timeframe
            if data.index.dtype.kind == 'M' and len(data) > 1:  # If index is datetime
                time_diff = pd.Series(data.index).diff()
                expected_diff = pd.Timedelta(minutes=1)  # For 1m timeframe
                
                # For timeframes other than 1m, adjust the expected difference
                if hasattr(data, 'timeframe'):
                    if data.timeframe == '5m':
                        expected_diff = pd.Timedelta(minutes=5)
                    elif data.timeframe == '15m':
                        expected_diff = pd.Timedelta(minutes=15)
                    elif data.timeframe == '1h':
                        expected_diff = pd.Timedelta(hours=1)
                    elif data.timeframe == '4h':
                        expected_diff = pd.Timedelta(hours=4)
                    elif data.timeframe == '1d':
                        expected_diff = pd.Timedelta(days=1)
                
                if (time_diff > expected_diff * 2).any():
                    logger.warning("Large time gaps detected in data")
                    print("Time gaps found:", time_diff[time_diff > expected_diff * 2])
                    # Don't fail validation for time gaps, just warn
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            print("Error in data validation:", str(e))
            return False

    def prepare_features(self, data: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """Prepare features for model training or prediction."""
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        # Log initial data shape
        logger.info(f"Data shape before validation: {data.shape}")
        
        # Validate data
        if not self._validate_data(data):
            raise ValueError("Data validation failed")
        
        # Create features DataFrame
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close']/data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Volume features
        features['volume_change'] = data['volume'].pct_change()
        features['volume_ma'] = data['volume'].rolling(window=20).mean()
        features['volume_std'] = data['volume'].rolling(window=20).std()
        features['volume_price_trend'] = features['volume_change'] * features['returns']
        features['volume_ma_ratio'] = data['volume'] / features['volume_ma']
        features['volume_variance'] = features['volume_std'] / features['volume_ma']
        
        # Technical indicators
        for period in [5, 10, 20]:
            features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
            features[f'momentum_{period}'] = data['close'].diff(period)
            features[f'roc_{period}'] = data['close'].pct_change(period)
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            
            # Bollinger Bands
            std = data['close'].rolling(window=period).std()
            features[f'bb_upper_{period}'] = features[f'sma_{period}'] + (2 * std)
            features[f'bb_lower_{period}'] = features[f'sma_{period}'] - (2 * std)
            features[f'bb_position_{period}'] = (data['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # Moving average crossovers
        features['ema_cross_12_26'] = (data['close'].ewm(span=12).mean() > data['close'].ewm(span=26).mean()).astype(int)
        features['sma_cross_20_50'] = (features['sma_20'] > data['close'].rolling(window=50).mean()).astype(int)
        
        # Market microstructure
        features['spread'] = data['high'] - data['low']
        features['spread_ma'] = features['spread'].rolling(window=20).mean()
        features['price_acceleration'] = features['returns'].diff()
        
        # Volatility features for all required windows
        for window in [12, 24, 48, 96]:
            high_low = np.log(data['high'] / data['low'])
            features[f'parkinson_{window}'] = np.sqrt(high_low.rolling(window=window).var() / (4 * np.log(2)))
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
        
        # Additional crypto-specific features
        features['volume_intensity'] = data['volume'] / data['volume'].rolling(window=24).mean()
        features['taker_buy_ratio'] = data['taker_buy_base'] / data['volume']
        features['price_range'] = (data['high'] - data['low']) / data['close']
        
        # Target variable
        features['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # Drop NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features

    def predict(self, data: pd.DataFrame = None, symbol: str = "BTCUSDT") -> np.ndarray:
        """Make predictions using data from DataManager"""
        try:
            # Se não recebeu dados, pegar do DataManager
            if data is None:
                data = self.data_manager.get_market_data(symbol=symbol)
            
            # Prepare features using DataManager
            features = self.data_manager.prepare_features(data)
            logger.info(f"Shape before dropna: {features.shape}")
            
            # Check if we have any data after feature preparation
            if len(features) == 0:
                logger.warning("No valid data after feature preparation")
                return np.array([])
            
            # Instead of dropping NaN values, fill them with forward fill then backward fill
            features = features.ffill().bfill()
            logger.info(f"Shape after fillna: {features.shape}")
            
            # Ensure we use the same feature columns as during training
            if self.feature_names is None:
                raise ValueError("Model must be trained before making predictions")
            
            # Select only the features used during training
            X = features[self.feature_names].values
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.cpu().numpy()
                
                # Ensure predictions are in [0,1] range
                predictions = np.clip(predictions, 0, 1)
                
                # Validate predictions
                if not np.all(np.isfinite(predictions)):
                    logger.warning("Found non-finite predictions, replacing with 0.5")
                    predictions[~np.isfinite(predictions)] = 0.5
                
                if not np.all((predictions >= 0) & (predictions <= 1)):
                    logger.warning("Found predictions outside [0,1] range, clipping")
                    predictions = np.clip(predictions, 0, 1)
                
                logger.info(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _temporal_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally into train, validation and test sets."""
        total_size = len(data)
        train_size = int(total_size * (1 - self.hyperparameters['validation_split'] - self.hyperparameters['test_split']))
        val_size = int(total_size * self.hyperparameters['validation_split'])
        
        train = data.iloc[:train_size]
        val = data.iloc[train_size:train_size + val_size]
        test = data.iloc[train_size + val_size:]
        
        return train, val, test

    def _update_training_status(self, epoch: int, train_loss: float, val_loss: float, metrics: Dict[str, float]) -> None:
        """Update training status with current metrics and confidence analysis."""
        try:
            # Calculate time-based metrics
            current_time = datetime.now()
            if hasattr(self, '_training_start_time'):
                elapsed_time = current_time - self._training_start_time
                eta = (elapsed_time / (epoch + 1)) * (self.hyperparameters['epochs'] - epoch - 1)
            else:
                self._training_start_time = current_time
                elapsed_time = timedelta(0)
                eta = timedelta(0)

            # Helper function to ensure metric is in [0,1] range
            def scale_metric(name: str, value: float) -> float:
                if name == 'loss':
                    return float(value)
                return float(np.clip(value, 0, 1))

            # Update performance history
            if 'performance_history' not in self.model_status:
                self.model_status['performance_history'] = []
                
            self.model_status['performance_history'].append({
                'timestamp': current_time.isoformat(),
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'return': float(metrics.get('return', 0.0)),
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'confidence': float(metrics.get('avg_confidence', 0.0))
            })

            # Update basic metrics
            self.training_status = {
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'metrics': {
                    'loss': float(val_loss),
                    'accuracy': scale_metric('accuracy', metrics.get('accuracy', 0.0)),
                    'precision': scale_metric('precision', metrics.get('precision', 0.0)),
                    'recall': scale_metric('recall', metrics.get('recall', 0.0)),
                    'f1': scale_metric('f1', metrics.get('f1', 0.0)),
                    'roc_auc': scale_metric('roc_auc', metrics.get('roc_auc', 0.5)),
                    'confidence_threshold': scale_metric('confidence_threshold', metrics.get('confidence_threshold', 0.5)),
                    'avg_confidence': scale_metric('avg_confidence', metrics.get('avg_confidence', 0.0)),
                    'high_confidence_rate': scale_metric('high_confidence_rate', metrics.get('high_confidence_rate', 0.0))
                },
                'training_progress': {
                    'current_epoch': epoch + 1,
                    'total_epochs': self.hyperparameters['epochs'],
                    'elapsed_time': str(elapsed_time),
                    'eta': str(eta),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
            }

            # Update predictions
            if 'predictions' not in self.model_status:
                self.model_status['predictions'] = {}
                
            self.model_status['predictions'].update({
                'confidence': float(metrics.get('avg_confidence', 0.0)),
                'signal': 'BUY' if metrics.get('prediction', 0) > 0.5 else 'SELL',
                'strength': float(metrics.get('signal_strength', 0.0)),
                'horizon': metrics.get('time_horizon', 'Short'),
                'risk': metrics.get('risk_level', 'Low')
            })

            # Update model status
            self.model_status.update({
                'current_metrics': self.training_status['metrics'],
                'validation_metrics': {
                    'loss': float(val_loss),
                    'accuracy': scale_metric('accuracy', metrics.get('accuracy', 0.0)),
                    'precision': scale_metric('precision', metrics.get('precision', 0.0)),
                    'recall': scale_metric('recall', metrics.get('recall', 0.0)),
                    'f1': scale_metric('f1', metrics.get('f1', 0.0)),
                    'roc_auc': scale_metric('roc_auc', metrics.get('roc_auc', 0.5)),
                    'confidence_threshold': scale_metric('confidence_threshold', metrics.get('confidence_threshold', 0.5)),
                    'avg_confidence': scale_metric('avg_confidence', metrics.get('avg_confidence', 0.0)),
                    'high_confidence_rate': scale_metric('high_confidence_rate', metrics.get('high_confidence_rate', 0.0))
                },
                'test_metrics': {
                    'loss': float(val_loss),
                    'accuracy': scale_metric('accuracy', metrics.get('accuracy', 0.0)),
                    'precision': scale_metric('precision', metrics.get('precision', 0.0)),
                    'recall': scale_metric('recall', metrics.get('recall', 0.0)),
                    'f1': scale_metric('f1', metrics.get('f1', 0.0)),
                    'roc_auc': scale_metric('roc_auc', metrics.get('roc_auc', 0.5)),
                    'confidence_threshold': scale_metric('confidence_threshold', metrics.get('confidence_threshold', 0.5)),
                    'avg_confidence': scale_metric('avg_confidence', metrics.get('avg_confidence', 0.0)),
                    'high_confidence_rate': scale_metric('high_confidence_rate', metrics.get('high_confidence_rate', 0.0))
                },
                'training_progress': self.training_status['training_progress'],
                'last_update': current_time.isoformat(),
                'is_ready': True
            })

            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.hyperparameters['epochs']} - "
                       f"Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                       f"Accuracy: {metrics['accuracy']:.4f} - "
                       f"ROC-AUC: {metrics.get('roc_auc', 0.5):.4f} - "
                       f"High Conf Rate: {metrics.get('high_confidence_rate', 0.0):.4f}")

        except Exception as e:
            logger.error(f"Error updating training status: {str(e)}")

    def _create_sequences(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for temporal data."""
        sequence_length = self.hyperparameters['sequence_length']
        stride = self.hyperparameters['stride']
        
        X = features[self.feature_names].values
        y = features['target'].values
        
        sequences = []
        targets = []
        
        for i in range(0, len(X) - sequence_length + 1, stride):
            sequences.append(X[i:i + sequence_length])
            targets.append(y[i + sequence_length - 1])
            
        return np.array(sequences), np.array(targets)

    def _create_data_loader(self, features: pd.DataFrame) -> DataLoader:
        """Create data loader with sequences."""
        # Create sequences
        X, y = self._create_sequences(features)
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor.unsqueeze(1))
        
        # Create data loader
        return DataLoader(
            dataset,
            batch_size=self.hyperparameters['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=0  # Adjust based on system
        )

    def _train_epoch(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Train the model for one epoch."""
        try:
            if self.model is None or self.optimizer is None:
                raise ValueError("Model or optimizer not initialized")
            
            self.model.train()
            total_loss = 0
            all_predictions = []
            all_targets = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # Ensure tensors are on correct device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Clear gradients
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                
                # Use mixed precision training
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.model.use_amp):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                
                # Scale loss and compute gradients
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights if gradients are valid
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                with torch.no_grad():
                    predictions = torch.sigmoid(outputs).detach()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            # Calculate metrics
            metrics = {
                'loss': total_loss / len(train_loader),
                'accuracy': accuracy_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions]),
                'precision': precision_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'recall': recall_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'f1': f1_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'roc_auc': roc_auc_score(all_targets, all_predictions)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training epoch: {str(e)}")
            raise

    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model on validation data."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move batch to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass with mixed precision if GPU available
                if self.gpu_available:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self._calculate_loss(output, target)
                else:
                    output = self.model(data)
                    loss = self._calculate_loss(output, target)
                
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def initialize_model(self) -> bool:
        """Initialize model architecture and optimizer."""
        try:
            logger.info("Initializing model components...")
            
            # Validate input size
            if self.input_size is None or self.input_size <= 0:
                logger.error("Invalid input size")
                return False
            
            # Validate required hyperparameters
            required_params = {
                'hidden_size': 256,
                'num_layers': 3,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            }
            
            for param, default in required_params.items():
                if param not in self.hyperparameters or self.hyperparameters[param] is None:
                    logger.warning(f"Missing {param}, using default value: {default}")
                    self.hyperparameters[param] = default
                elif isinstance(self.hyperparameters[param], dict):
                    if 'value' not in self.hyperparameters[param] or self.hyperparameters[param]['value'] is None:
                        logger.warning(f"Missing {param} value, using default: {default}")
                        self.hyperparameters[param]['value'] = default
            
            # Initialize model
            logger.info(f"Creating model with input size: {self.input_size}")
            self.model = DeepLearningModel(input_size=self.input_size)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
            
            # Get hyperparameter values, handling both dict and direct value formats
            def get_param(name, default=None):
                param = self.hyperparameters.get(name)
                if isinstance(param, dict):
                    return param.get('value', default)
                return param if param is not None else default
            
            # Initialize optimizer with correct device
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(get_param('learning_rate', 0.001)),
                weight_decay=float(get_param('weight_decay', 1e-5))
            )
            logger.info("Optimizer initialized")
            
            # Initialize learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            logger.info("Learning rate scheduler initialized")
            
            # Initialize loss function with class weights
            pos_weight = torch.tensor([2.0]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info("Loss function initialized")
            
            # Enable automatic mixed precision if GPU is available
            if self.gpu_available:
                self.model.use_amp = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("GPU optimizations enabled")
            
            # Initialize gradient scaler for mixed precision training
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.gpu_available)
            logger.info("Gradient scaler initialized")
            
            # Log model architecture and parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            
            self.model_status['is_ready'] = True
            logger.info(f"Model initialized on device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.model_status['is_ready'] = False
            return False

    def _create_data_loader(self, features: pd.DataFrame) -> DataLoader:
        X = features[self.feature_names].values
        y = features['target'].values
        
        X = self.feature_scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor.unsqueeze(1))
        return DataLoader(
            dataset,
            batch_size=self.hyperparameters['batch_size'],
            shuffle=True
        )

    def save_model(self, symbol: str = "BTCUSDT", timeframe: str = "1h") -> str:
        """Save the trained model and associated configurations."""
        if self.model is None:
            raise ValueError("No model to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{symbol}_{timeframe}_{timestamp}"
        model_path = os.path.join("models", model_filename)
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'input_size': self.input_size,
            'hyperparameters': self.hyperparameters,
            'model_status': self.model_status,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'framework_version': torch.__version__,
                'creation_date': datetime.now().isoformat()
            }
        }
        
        os.makedirs("models", exist_ok=True)
        torch.save(save_data, f"{model_path}.pt")
        logger.info(f"Model saved successfully to {model_path}.pt")
        
        return model_path
        
    def load_model(self, model_path: str) -> None:
        """Load a previously saved model and its configurations."""
        if not os.path.exists(f"{model_path}.pt"):
            raise FileNotFoundError(f"Model file not found: {model_path}.pt")
        
        checkpoint = torch.load(f"{model_path}.pt", map_location=self.device)
        
        # Validate checkpoint data
        required_keys = ['model_state_dict', 'feature_scaler', 'feature_names', 'input_size']
        if not all(key in checkpoint for key in required_keys):
            raise ValueError("Incomplete model checkpoint data")
        
        # Load model configuration
        self.input_size = checkpoint['input_size']
        self.feature_names = checkpoint['feature_names']
        self.hyperparameters = checkpoint['hyperparameters']
        
        # Initialize and load model
        self.initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load additional components
        self.feature_scaler = checkpoint['feature_scaler']
        self.model_status = checkpoint['model_status']
        
        logger.info(f"Model loaded successfully from {model_path}.pt")
        
    def cleanup(self) -> None:
        """Perform cleanup operations and free resources."""
        try:
            if self.gpu_available:
                # Move model to CPU before cleanup
                if self.model is not None:
                    self.model.cpu()
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                            param.grad = None
                    del self.model
                    self.model = None
                
                if self.optimizer is not None:
                    del self.optimizer
                    self.optimizer = None
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                    torch.cuda.reset_accumulated_memory_stats()
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    def get_model_status(self) -> Dict:
        """Retrieve the current model status and metrics."""
        self._update_gpu_metrics()
        
        # Initialize metrics if empty
        if not self.model_status['current_metrics']:
            self.model_status['current_metrics'] = {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0,
                'confidence_threshold': 0.5,
                'avg_confidence': 0.0,
                'high_confidence_rate': 0.0  # Predictions with confidence > 0.8
            }
        
        if not self.model_status['validation_metrics']:
            self.model_status['validation_metrics'] = {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0,
                'confidence_metrics': {
                    'mean': 0.0,
                    'std': 0.0,
                    'high_confidence_accuracy': 0.0
                }
            }
            
        if not self.model_status['test_metrics']:
            self.model_status['test_metrics'] = {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0,
                'confidence_metrics': {
                    'mean': 0.0,
                    'std': 0.0,
                    'high_confidence_accuracy': 0.0
                }
            }
        
        # Add trading performance metrics
        if 'trading_metrics' not in self.model_status:
            self.model_status['trading_metrics'] = {
                'total_trades': 0,
                'successful_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'avg_confidence': 0.0,
                'high_confidence_win_rate': 0.0
            }
            
        return self.model_status

    def _update_gpu_metrics(self):
        """Update GPU metrics if available"""
        try:
            if not torch.cuda.is_available():
                return
            
            # Get GPU memory usage
            gpu_memory_allocated = torch.cuda.memory_allocated(0)
            gpu_memory_cached = torch.cuda.memory_reserved(0)
            
            # Convert to MB
            gpu_memory_allocated_mb = gpu_memory_allocated / 1024 / 1024
            gpu_memory_cached_mb = gpu_memory_cached / 1024 / 1024
            
            # Update model status with GPU metrics
            self.model_status.update({
                'gpu_metrics': {
                    'memory_allocated': gpu_memory_allocated_mb,
                    'memory_cached': gpu_memory_cached_mb,
                    'device_name': torch.cuda.get_device_name(0),
                    'device_capability': torch.cuda.get_device_capability(0)
                }
            })
            
            logger.info(f"GPU Memory Allocated: {gpu_memory_allocated_mb:.2f} MB")
            logger.info(f"GPU Memory Cached: {gpu_memory_cached_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error updating GPU metrics: {str(e)}")

    def cancel_training(self):
        """Cancel current training process"""
        try:
            self.model_status['is_training'] = False
            self.model_status['training_cancelled'] = True
            logger.info("Training cancelled by user")
            
        except Exception as e:
            logger.error(f"Error cancelling training: {str(e)}")

    def train(self, train_loader: DataLoader = None, val_loader: DataLoader = None, symbol: str = None, timeframe: str = None, limit: int = 1000, train_split: float = 0.8, epochs: int = 100) -> Dict[str, Any]:
        """Train the model with either data loaders or by fetching data automatically.
        
        Args:
            train_loader (DataLoader, optional): Pre-prepared training data loader
            val_loader (DataLoader, optional): Pre-prepared validation data loader
            symbol (str, optional): Trading symbol for fetching data
            timeframe (str, optional): Timeframe for fetching data
            limit (int, optional): Number of candles to fetch
            train_split (float, optional): Train/validation split ratio
            epochs (int, optional): Number of training epochs
            
        Returns:
            Dict[str, Any]: Training results
        """
        try:
            # Reset model status at start of training
            self.model_status.update({
                'is_training': True,
                'is_ready': False,
                'training_cancelled': False,
                'current_metrics': {},
                'validation_metrics': {},
                'test_metrics': {},
                'performance_history': [],
                'predictions': {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'horizon': 'Short',
                    'risk': 'Low'
                }
            })
            
            # If data loaders are not provided, create them from market data
            if train_loader is None or val_loader is None:
                logger.info("Data loaders not provided, fetching market data...")
                
                # Get training data
                cache_key = f"{symbol}_{timeframe}_training_data"
                cached_data = self.data_manager.get_cached_data(cache_key)
                
                if cached_data is not None:
                    logger.info("Using cached training data")
                    data = cached_data
                else:
                    logger.info(f"Fetching training data for {symbol} with timeframe {timeframe}")
                    data = self.data_manager.get_training_data(timeframe=timeframe, symbol=symbol)
                    if data.empty:
                        raise ValueError("No training data available")
                    self.data_manager.cache_data(cache_key, data)
                
                # Prepare features
                logger.info("Preparing and validating features...")
                features = self.data_manager.prepare_features(data)
                
                if not self.data_manager.validate_data(features):
                    raise ValueError("Feature validation failed")
                
                # Update feature names and input size
                self.feature_names = [col for col in features.columns if col != 'target']
                self.input_size = len(self.feature_names)
                logger.info(f"Input size set to {self.input_size} features")
                
                # Initialize model components
                logger.info("Initializing model architecture...")
                if not self.initialize_model():
                    raise ValueError("Failed to initialize model")
                
                # Initialize feature scaler
                logger.info("Initializing feature scaler...")
                self.feature_scaler = RobustScaler()
                
                # Split data
                logger.info("Splitting data into train/val/test sets...")
                train_data, val_data, test_data = self.data_manager.split_training_data(
                    features, 
                    train_split=train_split
                )
                
                if not self.data_manager.validate_data_splits(train_data, val_data, test_data):
                    raise ValueError("Invalid data splits")
                
                # Fit scaler on training data only
                logger.info("Fitting feature scaler...")
                self.feature_scaler.fit(train_data[self.feature_names])
                
                # Create data loaders
                logger.info("Creating data loaders...")
                batch_size = self._get_param_value('batch_size')
                train_loader = self.data_manager.create_data_loader(
                    train_data, 
                    self.feature_names, 
                    self.feature_scaler,
                    batch_size=batch_size
                )
                val_loader = self.data_manager.create_data_loader(
                    val_data, 
                    self.feature_names, 
                    self.feature_scaler,
                    batch_size=batch_size
                )
            
            # Ensure model is initialized
            if not self.model_status['is_ready']:
                raise RuntimeError("Model not initialized. Call initialize_model() first.")
            
            logger.info(f"Starting training on device: {self.device}")
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = self._get_param_value('early_stopping_patience', 20)
            
            # Initialize mixed precision if GPU available
            scaler = torch.cuda.amp.GradScaler() if self.gpu_available else None
            
            for epoch in range(epochs):
                if self.model_status['training_cancelled']:
                    logger.info("Training cancelled by user")
                    break
                
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Move batch to device
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass with mixed precision if GPU available
                    if self.gpu_available:
                        with torch.cuda.amp.autocast():
                            output = self.model(data)
                            loss = self._calculate_loss(output, target)
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        # Standard forward and backward pass
                        output = self.model(data)
                        loss = self._calculate_loss(output, target)
                        loss.backward()
                        self.optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                val_loss = self._validate(val_loader)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1
                
                # Log progress
                avg_train_loss = train_loss / len(train_loader)
                logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
                
                # Update training status
                self.model_status.update({
                    'current_epoch': epoch + 1,
                    'total_epochs': epochs,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
                
                if patience_counter >= max_patience:
                    logger.info("Early stopping triggered")
                    break
                
                # Clear GPU cache periodically if using GPU
                if self.gpu_available and (epoch + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            return {
                'final_train_loss': avg_train_loss,
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'epochs_completed': epoch + 1
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            self.model_status.update({
                'is_training': False,
                'is_ready': False,
                'error': str(e)
            })
            raise

    def get_feature_analysis(self, categories=None) -> Dict:
        """Get feature analysis data for the dashboard."""
        try:
            if not self.feature_names:
                return {}
            
            feature_importance = {}
            feature_correlations = {}
            
            # Get feature importance if model exists
            if self.model is not None:
                # Calculate feature importance using gradient-based method
                self.model.eval()
                with torch.no_grad():
                    for name in self.feature_names:
                        feature_importance[name] = abs(self.model.input_projection[0].weight.mean(dim=0)[self.feature_names.index(name)].item())
            
            # Filter by categories if specified
            if categories:
                filtered_features = {}
                for name, value in feature_importance.items():
                    for category in categories:
                        if category in name.lower():
                            filtered_features[name] = value
                feature_importance = filtered_features
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
            
            return {
                'importance': feature_importance,
                'correlations': feature_correlations
            }
            
        except Exception as e:
            logger.error(f"Error in get_feature_analysis: {str(e)}")
            return {}

    def get_realtime_metrics(self) -> Dict:
        """Get real-time model metrics for the dashboard."""
        try:
            metrics = {
                'performance': self.model_status.get('current_metrics', {}),
                'validation': self.model_status.get('validation_metrics', {}),
                'predictions': self.model_status.get('predictions', {}),
                'training': self.model_status.get('training_progress', {})
            }
            
            # Add additional real-time metrics
            if self.model is not None and hasattr(self, 'optimizer'):
                metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                metrics['model_size'] = sum(p.numel() for p in self.model.parameters())
                metrics['trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in get_realtime_metrics: {str(e)}")
            return {}

    def get_detailed_metrics(self) -> Dict:
        """Get detailed model metrics for the dashboard."""
        try:
            return {
                'current': self.model_status.get('current_metrics', {}),
                'validation': self.model_status.get('validation_metrics', {}),
                'test': self.model_status.get('test_metrics', {}),
                'trading': self.model_status.get('trading_metrics', {}),
                'gpu': self.model_status.get('gpu_metrics', {})
            }
        except Exception as e:
            logger.error(f"Error in get_detailed_metrics: {str(e)}")
            return {}

    def get_latest_metrics(self) -> Dict:
        """Get latest metrics for the dashboard overview."""
        try:
            metrics = {}
            
            # Get performance history
            history = self.model_status.get('performance_history', [])
            if history:
                latest = history[-1]
                metrics.update({
                    'accuracy': latest.get('accuracy', 0),
                    'return': latest.get('return', 0),
                    'confidence': latest.get('confidence', 0)
                })
            
            # Get current predictions
            predictions = self.model_status.get('predictions', {})
            metrics.update({
                'signal': predictions.get('signal', 'NEUTRAL'),
                'strength': predictions.get('strength', 0),
                'risk': predictions.get('risk', 'Low')
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in get_latest_metrics: {str(e)}")
            return {}

    def get_training_status(self) -> Dict:
        """Get current training status for the dashboard."""
        try:
            return {
                'is_training': self.model_status.get('is_training', False),
                'progress': self.model_status.get('training_progress', {}),
                'hyperparameters': self.hyperparameters,
                'last_update': self.model_status.get('last_update')
            }
        except Exception as e:
            logger.error(f"Error in get_training_status: {str(e)}")
            return {}

    def update_hyperparameters(self, new_hyperparameters: Dict) -> None:
        """Update model hyperparameters and reinitialize components if needed."""
        try:
            logger.info("Atualizando parâmetros...")
            
            # Atualiza APENAS os parâmetros que foram passados, mantendo os existentes
            for param, value in new_hyperparameters.items():
                # Converte o valor para o tipo correto
                if param == 'learning_rate':
                    value = float(value)
                elif isinstance(value, (int, float)):
                    value = int(value)
                
                # Atualiza o parâmetro
                self.hyperparameters[param] = value
                logger.info(f"Parâmetro {param} atualizado para {value}")
            
            # Atualiza learning rate no otimizador se necessário
            if 'learning_rate' in new_hyperparameters and self.optimizer is not None:
                new_lr = float(new_hyperparameters['learning_rate'])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"Learning rate do otimizador atualizada para {new_lr}")
            
            # Reconstrói o modelo apenas se parâmetros arquiteturais mudaram
            if any(param in new_hyperparameters for param in ['model_type', 'hidden_size', 'num_layers']):
                self.build_model()
                logger.info("Modelo reconstruído com novos parâmetros")
            
            # Salva TODOS os parâmetros (antigos + novos)
            save_path = os.path.join("models", "hyperparameters.pt")
            os.makedirs("models", exist_ok=True)
            torch.save(self.hyperparameters, save_path)
            logger.info("Parâmetros salvos com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na atualização: {str(e)}")
            raise

    def _save_hyperparameters(self):
        """Save hyperparameters to ensure persistence between sessions."""
        try:
            save_path = os.path.join("models", "hyperparameters.pt")
            os.makedirs("models", exist_ok=True)
            torch.save(self.hyperparameters, save_path)
            logger.info("Hyperparameters saved successfully")
        except Exception as e:
            logger.error(f"Error saving hyperparameters: {str(e)}")

    def set_parameter(self, param_path: str, value: Any) -> bool:
        """
        Configura um parâmetro específico.
        Args:
            param_path: Caminho do parâmetro (ex: 'optimizer.weight_decay')
            value: Valor a ser configurado
        Returns:
            bool: True se configurado com sucesso, False caso contrário
        """
        try:
            parts = param_path.split('.')
            current = self.hyperparameters
            
            # Navega até o último nível
            for part in parts[:-1]:
                if part not in current:
                    logger.error(f"Invalid parameter path: {param_path}")
                    return False
                current = current[part]
                
            last_part = parts[-1]
            if last_part not in current:
                logger.error(f"Parameter not found: {param_path}")
                return False
                
            param_config = current[last_part]
            
            # Verifica se o parâmetro é configurável
            if not param_config.get('configurable', False):
                logger.error(f"Parameter {param_path} is not configurable")
                return False
                
            # Valida o valor
            if 'range' in param_config:
                min_val, max_val = param_config['range']
                if not (min_val <= value <= max_val):
                    logger.error(f"Value {value} out of range [{min_val}, {max_val}] for {param_path}")
                    return False
                    
            if 'options' in param_config and value not in param_config['options']:
                logger.error(f"Invalid value {value} for {param_path}. Valid options: {param_config['options']}")
                return False
                
            # Atualiza o valor
            param_config['value'] = value
            logger.info(f"Parameter {param_path} set to {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting parameter {param_path}: {str(e)}")
            return False

    def get_configurable_parameters(self) -> Dict:
        """
        Retorna todos os parâmetros configuráveis com seus valores atuais e configurações.
        Returns:
            Dict: Dicionário com os parâmetros configuráveis
        """
        configurable_params = {}
        
        def extract_configurable(params, prefix=''):
            for key, value in params.items():
                if isinstance(value, dict):
                    if 'configurable' in value and value['configurable']:
                        param_info = {
                            'value': value['value'],
                            'required': value.get('required', False),
                            'description': value.get('description', ''),
                            'range': value.get('range', None),
                            'options': value.get('options', None)
                        }
                        configurable_params[f"{prefix}{key}"] = param_info
                    elif 'configurable' not in value:
                        extract_configurable(value, f"{prefix}{key}.")
        
        extract_configurable(self.hyperparameters)
        return configurable_params

    def get_required_parameters(self) -> Dict:
        """
        Retorna apenas os parâmetros obrigatórios que ainda não foram configurados.
        Returns:
            Dict: Dicionário com os parâmetros obrigatórios não configurados
        """
        required_params = {}
        
        def extract_required(params, prefix=''):
            for key, value in params.items():
                if isinstance(value, dict):
                    if 'required' in value and value['required']:
                        if 'value' not in value or value['value'] is None:
                            param_info = {
                                'description': value.get('description', ''),
                                'range': value.get('range', None),
                                'options': value.get('options', None)
                            }
                            required_params[f"{prefix}{key}"] = param_info
                    elif 'required' not in value:
                        extract_required(value, f"{prefix}{key}.")
        
        extract_required(self.hyperparameters)
        return required_params

    def print_parameters_status(self):
        """Imprime o status atual de todos os parâmetros configuráveis."""
        configurable = self.get_configurable_parameters()
        required = self.get_required_parameters()
        
        print("\n=== Parâmetros Configuráveis ===")
        print("\nParâmetros Obrigatórios:")
        for param, info in configurable.items():
            if info['required']:
                status = "✓" if 'value' in info and info['value'] is not None else "✗"
                print(f"{status} {param}:")
                print(f"    Valor Atual: {info.get('value', 'Não configurado')}")
                print(f"    Descrição: {info['description']}")
                if info.get('range'):
                    print(f"    Range: {info['range']}")
                if info.get('options'):
                    print(f"    Opções: {info['options']}")
                print()
        
        print("\nParâmetros Opcionais:")
        for param, info in configurable.items():
            if not info['required']:
                print(f"• {param}:")
                print(f"    Valor Atual: {info.get('value', 'Não configurado')}")
                print(f"    Descrição: {info['description']}")
                if info.get('range'):
                    print(f"    Range: {info['range']}")
                if info.get('options'):
                    print(f"    Opções: {info['options']}")
                print()
        
        if required:
            print("\n⚠️ Parâmetros Obrigatórios Faltando:")
            for param, info in required.items():
                print(f"✗ {param}:")
                print(f"    Descrição: {info['description']}")
                if info.get('range'):
                    print(f"    Range: {info['range']}")
                if info.get('options'):
                    print(f"    Opções: {info['options']}")
                print()
        else:
            print("\n✓ Todos os parâmetros obrigatórios estão configurados!")

    def _calculate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with class weights on the correct device."""
        if not hasattr(self, 'criterion'):
            # Initialize loss function with class weights on the correct device
            pos_weight = torch.tensor([2.0]).to(self.device)  # Adjust based on your needs
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        return self.criterion(output, target)

    def _save_best_model(self) -> None:
        """Save the best model state with all necessary components."""
        try:
            # Create models directory if it doesn't exist
            model_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(model_dir, f'best_model_{timestamp}.pt')
            
            # Prepare checkpoint with all necessary components
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'hyperparameters': self.hyperparameters,
                'feature_names': self.feature_names,
                'input_size': self.input_size,
                'feature_scaler': self.feature_scaler,
                'model_status': self.model_status,
                'device': str(self.device)
            }
            
            # Save checkpoint
            torch.save(checkpoint, model_path)
            logger.info(f"Best model saved to {model_path}")
            
            # Update model status
            self.model_status['best_model_path'] = model_path
            
        except Exception as e:
            logger.error(f"Error saving best model: {str(e)}")
            raise