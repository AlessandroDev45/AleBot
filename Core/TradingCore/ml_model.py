import os
import gc
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DeepLearningModel(nn.Module):
    def __init__(self, input_size: int = 64):
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.transformer_out = nn.Linear(hidden_dim, 64)
        
        # CNN
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1)
        ])
        
        self.residual_layers = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=1),
            nn.Conv1d(64, 64, kernel_size=1)
        ])
        
        self.cnn_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(64)
        ])
        
        self.cnn_out = nn.Linear(64, 64)
        
        # GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.gru_out = nn.Linear(hidden_dim * 2, 64)
        
        # Ensemble
        self.ensemble_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.final_ensemble = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        cnn_out = self.conv_layers[0](x_cnn)
        cnn_out = self.cnn_norm_layers[0](cnn_out)
        cnn_out = nn.functional.gelu(cnn_out)
        
        for conv, norm, residual in zip(
            self.conv_layers[1:],
            self.cnn_norm_layers[1:],
            self.residual_layers
        ):
            identity = cnn_out
            cnn_out = conv(cnn_out)
            cnn_out = norm(cnn_out)
            cnn_out = nn.functional.gelu(cnn_out)
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
        
        # Final output
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
            'training_cancelled': False,
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
        
        logger.info(f"MLModel initialized on device: {self.device}")

    def train(self, train_loader: DataLoader = None, val_loader: DataLoader = None, symbol: str = None, timeframe: str = None, limit: int = 1000, train_split: float = 0.8, epochs: int = 100) -> Dict[str, Any]:
        """Train the model with the provided data."""
        try:
            logger.info("Starting training process...")
            
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
            
            # Get data through DataManager
            if train_loader is None or val_loader is None:
                logger.info("Getting data through DataManager...")
                
                # Use DataManager's cache if available
                cached_data = self.data_manager.get_cached_data(symbol or "BTCUSDT", timeframe or "1h")
                if cached_data is not None:
                    data = cached_data
                else:
                    data = self.data_manager.get_training_data(
                        timeframe=timeframe or "1h", 
                        symbol=symbol or "BTCUSDT",
                        limit=limit
                    )
                
                if data.empty:
                    raise ValueError("No training data available")
                
                # Use DataManager's feature preparation
                features = self.data_manager.prepare_features(data)
                if not self.data_manager.validate_data(features):
                    raise ValueError("Feature validation failed")
                
                # Update feature names and input size
                self.feature_names = self.data_manager.get_feature_names(features)
                self.input_size = len(self.feature_names)
                
                # Use DataManager's data splitting
                train_data, val_data, test_data = self.data_manager.split_training_data(features, train_split)
                
                # Use DataManager's data validation
                if not self.data_manager.validate_data_splits(train_data, val_data, test_data):
                    raise ValueError("Invalid data splits")
                
                # Use DataManager's data scaling
                train_data = self.data_manager.scale_features(train_data, self.feature_names)
                val_data = self.data_manager.scale_features(val_data, self.feature_names)
                
                # Create data loaders through DataManager
                batch_size = self._get_param_value('batch_size', default=32)
                train_loader = self.data_manager.create_data_loader(train_data, self.feature_names, batch_size=batch_size)
                val_loader = self.data_manager.create_data_loader(val_data, self.feature_names, batch_size=batch_size)
            
            # Initialize model if needed
            if not self.model_status['is_ready']:
                if not self.initialize_model():
                    raise RuntimeError("Failed to initialize model")
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = self._get_param_value('early_stopping_patience', 20)
            
            for epoch in range(epochs):
                if self.model_status['training_cancelled']:
                    break
                
                train_metrics = self._train_epoch(train_loader)
                val_metrics = self._validate(val_loader)
                
                self.scheduler.step(val_metrics['loss'])
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1
                
                self._update_training_status(epoch, train_metrics['loss'], val_metrics['loss'], train_metrics)
                
                if patience_counter >= max_patience:
                    logger.info("Early stopping triggered")
                    break
                
                if self.gpu_available and (epoch + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            return {
                'final_train_loss': train_metrics['loss'],
                'final_val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'epochs_completed': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            self.model_status.update({
                'is_training': False,
                'is_ready': False,
                'error': str(e)
            })
            raise

    def _train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train the model for one epoch."""
        try:
            if self.model is None or self.optimizer is None:
                raise ValueError("Model or optimizer not initialized")
            
            self.model.train()
            total_loss = 0
            all_predictions = []
            all_targets = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.gpu_available:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                with torch.no_grad():
                    predictions = torch.sigmoid(outputs).detach()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            metrics = {
                'loss': total_loss / len(train_loader),
                'accuracy': accuracy_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions]),
                'precision': precision_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'recall': recall_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'f1': f1_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'roc_auc': roc_auc_score(all_targets, all_predictions)
            }
            
            self.model_status['current_metrics'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training epoch: {str(e)}")
            raise

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model on validation data."""
        try:
            self.model.eval()
            total_loss = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if self.gpu_available:
                        with torch.amp.autocast(device_type='cuda'):
                            outputs = self.model(batch_X)
                            loss = self.criterion(outputs, batch_y)
                    else:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                    
                    total_loss += loss.item()
                    predictions = torch.sigmoid(outputs).detach()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            metrics = {
                'loss': total_loss / len(val_loader),
                'accuracy': accuracy_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions]),
                'precision': precision_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'recall': recall_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'f1': f1_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'roc_auc': roc_auc_score(all_targets, all_predictions),
                'avg_confidence': np.mean(all_predictions),
                'high_confidence_rate': np.mean([1 if p > 0.8 or p < 0.2 else 0 for p in all_predictions])
            }
            
            self.model_status['validation_metrics'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise

    def initialize_model(self) -> bool:
        """Initialize model architecture and components."""
        try:
            # Check if CUDA is available
            self.gpu_available = torch.cuda.is_available()
            self.device = torch.device('cuda' if self.gpu_available else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize model architecture
            if not hasattr(self, 'input_size') or self.input_size is None:
                raise ValueError("Input size not set")
            
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
            
            # Enable automatic mixed precision if GPU available
            if self.gpu_available:
                self.model.use_amp = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("GPU optimizations enabled")
            else:
                self.model.use_amp = False
            
            # Initialize gradient scaler for mixed precision training
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.gpu_available)
            logger.info("Gradient scaler initialized")
            
            # Log model architecture and parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            self.model_status.update({
                'is_ready': True,
                'device': str(self.device),
                'total_params': total_params,
                'trainable_params': trainable_params,
                'gpu_available': self.gpu_available,
                'mixed_precision': self.gpu_available
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.model_status.update({
                'is_ready': False,
                'error': str(e)
            })
            return False

    def _get_param_value(self, param_name: str, default: Any = None) -> Any:
        """Get the value of a hyperparameter."""
        param = self.hyperparameters.get(param_name)
        if isinstance(param, dict):
            return param.get('value', default)
        return param if param is not None else default

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get the default hyperparameters."""
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'early_stopping_patience': 20
        }

    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between model outputs and targets."""
        return self.criterion(outputs, targets)

    def _save_best_model(self):
        """Save the best model state."""
        try:
            model_path = os.path.join("models", "best_model.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'hyperparameters': self.hyperparameters,
                'feature_names': self.feature_names,
                'input_size': self.input_size,
                'model_status': self.model_status
            }, model_path)
            logger.info("Saved best model checkpoint")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def _update_training_status(self, epoch: int, train_loss: float, val_loss: float, train_metrics: Dict[str, float]):
        """Update the training status with current metrics."""
        self.model_status.update({
            'current_metrics': train_metrics,
            'training_progress': {
                'current_epoch': epoch + 1,
                'total_epochs': self.hyperparameters.get('epochs', 500),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss)
            }
        })
        
        if self.gpu_available:
            self.model_status['gpu_metrics'].update({
                'gpu_memory_used': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**2,
                'gpu_memory_peak': torch.cuda.max_memory_allocated() / 1024**2
            })

    def predict(self, data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Make predictions on input data."""
        try:
            self.model.eval()
            with torch.no_grad():
                data = data.to(self.device)
                if self.gpu_available:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                probabilities = torch.sigmoid(outputs)
                confidence = float(torch.mean(torch.abs(probabilities - 0.5) * 2))
                return probabilities, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def load_model(self, model_path: str) -> bool:
        """Load a saved model state."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model with saved input size
            self.input_size = checkpoint.get('input_size')
            if not self.initialize_model():
                raise ValueError("Failed to initialize model")
            
            # Load saved states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.hyperparameters = checkpoint.get('hyperparameters', self._get_default_hyperparameters())
            self.feature_names = checkpoint.get('feature_names', [])
            self.model_status.update(checkpoint.get('model_status', {}))
            self.model_status['is_ready'] = True
            
            logger.info(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data."""
        try:
            self.model.eval()
            total_loss = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.gpu_available:
                        with torch.cuda.amp.autocast():
                            output = self.model(data)
                            loss = self._calculate_loss(output, target)
                    else:
                        output = self.model(data)
                        loss = self._calculate_loss(output, target)
                    
                    total_loss += loss.item()
                    predictions = torch.sigmoid(output).cpu().numpy()
                    all_predictions.extend(predictions)
                    all_targets.extend(target.cpu().numpy())
            
            metrics = {
                'test_loss': total_loss / len(test_loader),
                'accuracy': accuracy_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions]),
                'precision': precision_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'recall': recall_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'f1': f1_score(all_targets, [1 if p > 0.5 else 0 for p in all_predictions], zero_division=0),
                'roc_auc': roc_auc_score(all_targets, all_predictions)
            }
            
            self.model_status['test_metrics'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

    def get_model_status(self) -> Dict[str, Any]:
        """Get the current model status."""
        return self.model_status.copy()

    def cancel_training(self):
        """Cancel the current training process."""
        self.model_status['training_cancelled'] = True
        logger.info("Training cancellation requested")

    def cleanup(self):
        """Clean up model resources."""
        try:
            if self.gpu_available:
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")