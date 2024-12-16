import unittest
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from TradingSystem import BinanceClient, DataManager, MLModel
from TradingSystem.Core.TradingCore.ml_model import DeepLearningModel
from sklearn.preprocessing import StandardScaler, RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMLModel(unittest.TestCase):
    def setUp(self):
        """Setup before each test"""
        print("Starting setUp...")
        self.data_manager = DataManager()
        self.ml_model = MLModel(self.data_manager)
        self.symbol = "BTCUSDT"
        self.timeframe = "1m"
        
        # Get data for testing through DataManager
        self.data = self.data_manager.get_market_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=1000
        )
        
        # Ensure we have enough data
        if len(self.data) < 500:
            self.data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=2000
            )
        
        # Prepare features through DataManager
        self.features = self.data_manager.prepare_features(self.data)
        
        print(f"MLModel initialized")
        print(f"Training data obtained: {len(self.data)} candles")
        print(f"Features prepared: {self.features.shape}")
        
        self.model_status = {}

    def test_api_connection(self):
        """Test API connection through DataManager"""
        try:
            # Verify DataManager initialization
            self.assertIsNotNone(self.data_manager)
            self.assertIsNotNone(self.data_manager._client)
            
            # Test data retrieval
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=10
            )
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)
            
            # Test feature preparation
            features = self.data_manager.prepare_features(data)
            self.assertIsNotNone(features)
            self.assertGreater(len(features), 0)
            
        except Exception as e:
            logger.error(f"Error in test_api_connection: {str(e)}")
            raise

    def test_data_quality(self):
        """Test quality of real data"""
        try:
            self.assertIsNotNone(self.data)
            self.assertFalse(self.data.empty)
            self.assertTrue(len(self.data) > 500, "Need at least 500 candles for testing")
            
            # Check for missing values
            null_counts = self.data.isnull().sum()
            self.assertTrue(null_counts.sum() == 0, f"Found null values: {null_counts[null_counts > 0]}")
            
            # Check data continuity
            time_diffs = pd.Series(self.data.index).diff()
            gaps = time_diffs[time_diffs > timedelta(hours=1)]
            self.assertTrue(len(gaps) == 0, f"Found gaps in data: {gaps}")
            
            # Check price consistency
            self.assertTrue((self.data['high'] >= self.data['low']).all(), "High should be >= Low")
            self.assertTrue((self.data['high'] >= self.data['open']).all(), "High should be >= Open")
            self.assertTrue((self.data['high'] >= self.data['close']).all(), "High should be >= Close")
            
            # Check volume validity
            self.assertTrue((self.data['volume'] > 0).all(), "Volume should be > 0")
            
            logger.info("Data quality checks passed")
            
        except Exception as e:
            logger.error(f"Error in test_data_quality: {str(e)}")
            raise
    
    def test_feature_preparation(self):
        """Test feature preparation with real data"""
        try:
            # Get features through DataManager
            features = self.data_manager.prepare_features(self.data)
            
            self.assertIsNotNone(features)
            self.assertTrue(len(features) > 0)
            
            # Check for NaN values
            null_counts = features.isnull().sum()
            self.assertTrue(null_counts.sum() == 0, f"Found NaN values: {null_counts[null_counts > 0]}")
            
            # Check required features
            required_features = [
                'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                'volume_change', 'volume_ma', 'volume_std', 'volume_price_trend', 'volume_ma_ratio', 'volume_variance',
                'sma_5', 'ema_5', 'momentum_5', 'roc_5', 'rsi_5',
                'sma_10', 'ema_10', 'momentum_10', 'roc_10', 'rsi_10',
                'sma_20', 'ema_20', 'momentum_20', 'roc_20', 'rsi_20',
                'bb_upper_5', 'bb_lower_5', 'bb_position_5',
                'bb_upper_10', 'bb_lower_10', 'bb_position_10',
                'bb_upper_20', 'bb_lower_20', 'bb_position_20',
                'ema_cross_12_26', 'sma_cross_20_50',
                'spread', 'spread_ma', 'price_acceleration',
                'parkinson_12', 'volatility_12',
                'parkinson_24', 'volatility_24',
                'target'
            ]
            
            # Check each required feature exists
            missing_features = [f for f in required_features if f not in features.columns]
            self.assertEqual(len(missing_features), 0, f"Missing features: {missing_features}")
            
            # Check feature values are valid
            self.assertTrue(features['returns'].notna().all(), "Returns contain NaN values")
            self.assertTrue(features['volume_ma'].notna().all(), "Volume MA contains NaN values")
            self.assertTrue(features['rsi_5'].between(0, 100).all(), "RSI values out of range")
            
            logger.info(f"Feature preparation successful. Shape: {features.shape}")
            
        except Exception as e:
            logger.error(f"Error in test_feature_preparation: {str(e)}")
            raise
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_optimization(self):
        """Test GPU optimization settings"""
        try:
            # Get GPU status
            status = self.ml_model.model_status
            
            # Check if GPU metrics exist
            self.assertIn('gpu_metrics', status)
            
            # If GPU available, check memory metrics
            if self.ml_model.gpu_available:
                metrics = status['gpu_metrics']
                self.assertIn('gpu_memory_used', metrics)
                self.assertIn('gpu_memory_cached', metrics)
                self.assertIn('gpu_memory_total', metrics)
                
                # gpu_utilization pode não estar disponível se pynvml não estiver instalado
                if 'gpu_utilization' in metrics:
                    self.assertIsInstance(metrics['gpu_utilization'], (int, float))
            
        except Exception as e:
            logger.error(f"Error in test_gpu_optimization: {str(e)}")
            raise
    
    def test_model_training(self):
        """Test model training with real data"""
        try:
            # Get training data
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            
            # Train model with data directly
            self.ml_model.train(data=data)
            
            # Get test data
            test_data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            # Make predictions
            predictions = self.ml_model.predict(test_data)
            
            # Basic assertions
            self.assertIsNotNone(predictions)
            self.assertTrue(isinstance(predictions, np.ndarray))
            self.assertTrue(len(predictions) > 0)
            self.assertTrue(all(0 <= p <= 1 for p in predictions.flatten()))
            
            # Check model status
            status = self.ml_model.get_model_status()
            self.assertIn('current_metrics', status)
            self.assertIn('validation_metrics', status)
            self.assertIn('test_metrics', status)
            
            logger.info("Model training test passed")
            
        except Exception as e:
            logger.error(f"Error in test_model_training: {str(e)}")
            raise
    
    def test_real_time_prediction(self):
        """Test real-time prediction with latest data"""
        try:
            # Get training data
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            
            # Train model with data directly
            self.ml_model.train(data=data)
            
            # Get latest data
            latest_data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=10
            )
            
            # Make prediction
            start_time = datetime.now()
            prediction = self.ml_model.predict(latest_data)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Basic assertions
            self.assertIsNotNone(prediction)
            self.assertTrue(isinstance(prediction, np.ndarray))
            self.assertTrue(len(prediction) > 0)
            self.assertTrue(all(0 <= p <= 1 for p in prediction))
            
            # Check prediction timing
            self.assertLess(prediction_time, 0.1, "Prediction should be fast")
            
            # Check GPU memory after prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated()
                self.assertLess(memory_allocated/torch.cuda.get_device_properties(0).total_memory, 0.5)
            
            logger.info(f"Real-time prediction: {float(prediction[0]):.4f} (took {prediction_time*1000:.2f}ms)")
            
        except Exception as e:
            logger.error(f"Error in test_real_time_prediction: {str(e)}")
            raise
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management during training"""
        try:
            initial_memory = torch.cuda.memory_allocated(0)
            
            # Get training data
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            
            # Train model with data directly
            self.ml_model.train(data=data)
            
            peak_memory = torch.cuda.max_memory_allocated(0)
            final_memory = torch.cuda.memory_allocated(0)
            
            # Check memory usage
            self.assertLess(final_memory, peak_memory, "Memory should be freed after training")
            self.assertLess(peak_memory / torch.cuda.get_device_properties(0).total_memory, 0.95,
                          "Peak memory should not exceed 95% of total GPU memory")
            
            logger.info(f"GPU memory test passed. Peak usage: {peak_memory/1024**2:.2f}MB")
            
        except Exception as e:
            logger.error(f"Error in test_gpu_memory_management: {str(e)}")
            raise
    
    def test_multi_timeframe_data(self):
        """Test getting data from multiple timeframes in parallel"""
        try:
            timeframes = ['1h', '4h', '1d']
            data = {}
            
            for tf in timeframes:
                data[tf] = self.data_manager.get_market_data(
                    symbol=self.symbol,
                    timeframe=tf,
                    limit=100
                )
                
                self.assertIsNotNone(data[tf])
                self.assertFalse(data[tf].empty)
                self.assertTrue(len(data[tf]) > 0)
                
                # Check data structure
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                self.assertTrue(all(col in data[tf].columns for col in required_columns))
                
                # Check data types
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    self.assertTrue(pd.api.types.is_numeric_dtype(data[tf][col]))
                
                # Check index is datetime
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(data[tf].index))
                
            logger.info("Multi-timeframe data tests passed")
            
        except Exception as e:
            logger.error(f"Error in test_multi_timeframe_data: {str(e)}")
            raise
            
    def test_binance_data_validation(self):
        """Test specific Binance data validation"""
        try:
            # Test valid data
            valid_data = self.data.copy()
            self.assertTrue(self.ml_model._validate_data(valid_data))
            
            # Test invalid price
            invalid_price = valid_data.copy()
            invalid_price.loc[invalid_price.index[0], 'high'] = invalid_price.loc[invalid_price.index[0], 'low'] - 1
            self.assertFalse(self.ml_model._validate_data(invalid_price))
            
            # Test invalid volume
            invalid_volume = valid_data.copy()
            invalid_volume.loc[invalid_volume.index[0], 'volume'] = 0
            self.assertFalse(self.ml_model._validate_data(invalid_volume))
            
            # Test missing columns
            missing_columns = valid_data.drop(columns=['volume']).copy()
            self.assertFalse(self.ml_model._validate_data(missing_columns))
            
            logger.info("Binance data validation tests passed")
            
        except Exception as e:
            logger.error(f"Error in test_binance_data_validation: {str(e)}")
            raise
            
    def test_ensemble_model(self):
        """Test ensemble model architecture and predictions"""
        try:
            # Get data first
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            
            # Train model with data directly
            self.ml_model.train(data=data)
            
            # Get model outputs
            test_data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            features = self.prepare_features(test_data)
            feature_cols = [col for col in features.columns if col != 'target']
            X = features[feature_cols].values
            X_scaled = self.ml_model.feature_scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.ml_model.device)
            
            # Get model outputs
            self.ml_model.model.eval()
            with torch.no_grad():
                # Ensure input has correct dimensions [batch, seq, features]
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)  # [batch, 1, features]
                
                # Get transformer output
                x_transformer = self.ml_model.model.input_projection(X_tensor)
                transformer_out = self.ml_model.model.transformer_encoder(x_transformer)
                transformer_out = self.ml_model.model.transformer_out(transformer_out.mean(dim=1))
                
                # Get CNN output
                x_cnn = X_tensor.transpose(1, 2)
                cnn_out = self.ml_model.model.conv_layers[0](x_cnn)
                for conv, residual in zip(self.ml_model.model.conv_layers[1:], self.ml_model.model.residual_layers):
                    identity = cnn_out
                    cnn_out = conv(cnn_out)
                    residual_out = residual(identity)
                    cnn_out = cnn_out + residual_out
                cnn_out = self.ml_model.model.cnn_out(cnn_out.mean(dim=2))
                
                # Get GRU output
                gru_out, _ = self.ml_model.model.gru(X_tensor)
                attn_out, _ = self.ml_model.model.attention(gru_out, gru_out, gru_out)
                gru_out = self.ml_model.model.gru_out(attn_out[:, -1, :])
                
                # Test ensemble
                ensemble_input = torch.stack([transformer_out, cnn_out, gru_out], dim=1)
                ensemble_out, _ = self.ml_model.model.ensemble_attention(
                    ensemble_input, ensemble_input, ensemble_input
                )
                final_out = self.ml_model.model.final_ensemble(ensemble_out.mean(dim=1))
                final_out = torch.sigmoid(final_out)
                
                # Validations
                self.assertEqual(transformer_out.shape[1], 64)
                self.assertEqual(cnn_out.shape[1], 64)
                self.assertEqual(gru_out.shape[1], 64)
                self.assertEqual(ensemble_input.shape[1], 3)
                self.assertEqual(final_out.shape[1], 1)
                
                # Check value ranges
                self.assertTrue(torch.all((final_out >= 0) & (final_out <= 1)))
            
            logger.info("Ensemble model tests passed")
            
        except Exception as e:
            logger.error(f"Error in test_ensemble_model: {str(e)}")
            raise

    def test_multi_timeframe(self):
        """Test model with multiple timeframes"""
        try:
            timeframes = ["1m", "5m", "15m"]
            
            for timeframe in timeframes:
                # Get data
                data = self.data_manager.get_market_data(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    limit=1000
                )
                
                self.assertIsNotNone(data)
                self.assertFalse(data.empty)
                
                # Train and predict
                self.ml_model.train(data=data)
                predictions = self.ml_model.predict(data.iloc[-100:])
                
                self.assertIsNotNone(predictions)
                self.assertTrue(len(predictions) > 0)
                
            logger.info("Multi-timeframe tests passed")
            
        except Exception as e:
            logger.error(f"Error in test_multi_timeframe: {str(e)}")
            raise

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory(self):
        """Test GPU memory usage and optimization"""
        try:
            # Clear GPU memory before test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated()
            
            # Get training data
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            
            # Train model with data directly
            self.ml_model.train(data=data)
            
            training_memory = torch.cuda.memory_allocated()
            self.assertGreater(training_memory, initial_memory)
            
            # Check memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_fraction = training_memory / total_memory
            self.assertLess(used_fraction, 0.9)  # Should use less than 90% of GPU memory
            
            # Test memory cleanup
            self.ml_model.cleanup()
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            self.assertLess(final_memory, training_memory)
            
            logger.info("GPU memory tests passed")
            
        except Exception as e:
            logger.error(f"Error in test_gpu_memory: {str(e)}")
            raise

    def test_robust_scaler(self):
        """Test RobustScaler implementation"""
        try:
            # Prepare test data
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            features = self.ml_model.prepare_features(data)
            
            # Get feature scaler
            scaler = self.ml_model.feature_scaler
            
            # Verify scaler type
            self.assertIsInstance(scaler, RobustScaler)
            
            # Test scaling
            feature_cols = [col for col in features.columns if col != 'target']
            X = features[feature_cols].values
            
            # Fit and transform
            if not hasattr(scaler, 'center_'):
                scaler.fit(X)
            X_scaled = scaler.transform(X)
            
            # Verify scaling results
            self.assertEqual(X.shape, X_scaled.shape)
            self.assertTrue(np.all(np.isfinite(X_scaled)))
            
            logger.info("RobustScaler test passed")
            
        except Exception as e:
            logger.error(f"Error in test_robust_scaler: {str(e)}")
            raise

    def test_crypto_specific_features(self):
        """Test crypto-specific features"""
        try:
            # Get features through DataManager
            features = self.data_manager.prepare_features(self.data)
            
            # 1. Price Action Features
            self.assertIn('high_low_ratio', features.columns)
            self.assertIn('close_open_ratio', features.columns)
            
            # 2. Volatility Features
            for window in [12, 24]:
                self.assertIn(f'volatility_{window}', features.columns)
                self.assertIn(f'parkinson_{window}', features.columns)
            
            # 3. Volume Features
            self.assertIn('volume_price_trend', features.columns)
            self.assertIn('volume_ma_ratio', features.columns)
            self.assertIn('volume_variance', features.columns)
            
            # 4. Market Microstructure
            self.assertIn('spread', features.columns)
            self.assertIn('spread_ma', features.columns)
            self.assertIn('price_acceleration', features.columns)
            
            # 5. Momentum Features
            for window in [5, 10, 20]:
                self.assertIn(f'rsi_{window}', features.columns)
                self.assertIn(f'momentum_{window}', features.columns)
                self.assertIn(f'roc_{window}', features.columns)
            
            # 6. Moving Average Crossovers
            self.assertIn('ema_cross_12_26', features.columns)
            self.assertIn('sma_cross_20_50', features.columns)
            
            # Validate feature values
            self.assertTrue(features['high_low_ratio'].gt(0).all(), "high_low_ratio should be positive")
            self.assertTrue(features['volume_ma_ratio'].gt(0).all(), "volume_ma_ratio should be positive")
            self.assertTrue(features['rsi_5'].between(0, 100).all(), "RSI should be between 0 and 100")
            
            logger.info("Crypto-specific features test passed")
            
        except Exception as e:
            logger.error(f"Error in test_crypto_specific_features: {str(e)}")
            raise

    def test_temporal_split(self):
        """Test temporal data split"""
        try:
            features = self.ml_model.prepare_features(self.data)
            train, val, test = self.ml_model._temporal_split(features)
            
            # Check sizes
            total_size = len(features)
            expected_train_size = int(total_size * (1 - self.ml_model.hyperparameters['validation_split'] - self.ml_model.hyperparameters['test_split']))
            expected_val_size = int(total_size * self.ml_model.hyperparameters['validation_split'])
            
            self.assertEqual(len(train), expected_train_size)
            self.assertEqual(len(val), expected_val_size)
            
            # Check temporal order
            self.assertTrue(train.index[-1] < val.index[0])
            self.assertTrue(val.index[-1] < test.index[0])
            
            logger.info("Temporal split tests passed")
            
        except Exception as e:
            logger.error(f"Error in test_temporal_split: {str(e)}")
            raise

    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        try:
            # Get training data
            data = self.data_manager.get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=1000
            )
            
            # Train model with data directly
            self.ml_model.train(data=data)
            
            # Get model status
            status = self.ml_model.get_model_status()
            
            # Check training metrics
            self.assertIn('current_metrics', status)
            self.assertIn('validation_metrics', status)
            self.assertIn('test_metrics', status)
            
            # Check metric components
            for metric_type in ['current_metrics', 'validation_metrics', 'test_metrics']:
                metrics = status[metric_type]
                self.assertIn('loss', metrics)
                self.assertIn('accuracy', metrics)
                self.assertIn('precision', metrics)
                self.assertIn('recall', metrics)
                self.assertIn('f1', metrics)
                self.assertIn('roc_auc', metrics)
                
                # Check metric values
                for metric_name, value in metrics.items():
                    self.assertIsInstance(value, float)
                    self.assertTrue(0 <= value <= 1 if metric_name != 'loss' else True)
            
            logger.info("Model evaluation tests passed")
            
        except Exception as e:
            logger.error(f"Error in test_model_evaluation: {str(e)}")
            raise

    def _update_gpu_metrics(self):
        """Update and track GPU metrics internally."""
        try:
            if not self.gpu_available:
                self.model_status['gpu_metrics'] = {
                    'status': 'GPU not available',
                    'gpu_memory_used': 0,
                    'gpu_memory_cached': 0,
                    'gpu_memory_total': 0,
                    'gpu_memory_peak': 0,
                    'gpu_utilization': 0,
                    'gpu_temperature': 0
                }
                return

            # Clear cache for accurate measurements
            torch.cuda.empty_cache()
            
            # Get basic memory metrics
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            max_memory = torch.cuda.max_memory_allocated(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Convert to GB for readability
            memory_metrics = {
                'gpu_memory_used': memory_allocated / (1024**3),
                'gpu_memory_cached': memory_reserved / (1024**3),
                'gpu_memory_total': total_memory / (1024**3),
                'gpu_memory_peak': max_memory / (1024**3),
                'device_name': torch.cuda.get_device_name(0),
                'device_capability': torch.cuda.get_device_capability(0)
            }
            
            # Try to get advanced metrics if possible
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Get utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_metrics['gpu_utilization'] = utilization.gpu
                
                # Get temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                memory_metrics['gpu_temperature'] = temperature
                
            except (ImportError, Exception) as e:
                memory_metrics['gpu_utilization'] = 0
                memory_metrics['gpu_temperature'] = 0
                logger.warning(f"Could not get advanced GPU metrics: {str(e)}")
            
            self.model_status['gpu_metrics'] = memory_metrics
            
        except Exception as e:
            logger.error(f"Error updating GPU metrics: {str(e)}")
            self.model_status['gpu_metrics'] = {
                'status': f'Error: {str(e)}',
                'gpu_memory_used': 0,
                'gpu_memory_cached': 0,
                'gpu_memory_total': 0,
                'gpu_memory_peak': 0,
                'gpu_utilization': 0,
                'gpu_temperature': 0
            }

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_rtx4090_optimizations(self):
        """Test RTX 4090 specific optimizations"""
        try:
            features = self.prepare_features(self.data)
            input_size = len([col for col in features.columns if col != 'target'])
            
            # Create model
            model = DeepLearningModel(input_size)
            if self.ml_model.gpu_available:
                model = model.to(self.ml_model.device)
            
            # Test sample input - [batch_size, sequence_length, features]
            batch_size = 32
            seq_length = 100
            sample_input = torch.randn(batch_size, seq_length, input_size)
            if self.ml_model.gpu_available:
                sample_input = sample_input.to(self.ml_model.device)
            
            # Test forward pass
            with torch.no_grad():
                output = model(sample_input)
            
            # Verify output shape
            self.assertEqual(output.shape, (batch_size, 1))
            
            # Verify GPU optimizations
            if self.ml_model.gpu_available:
                self.assertTrue(model.use_amp)
                self.assertTrue(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
            
        except Exception as e:
            logger.error(f"Error in test_rtx4090_optimizations: {str(e)}")
            raise

    def prepare_features(self, data):
        """Prepare features for testing"""
        try:
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
                features[f'rsi_{period}'] = self.ml_model._calculate_rsi(data['close'], period)
                
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

            # Volatility features
            for period in [12, 24]:
                high_low = np.log(data['high'] / data['low'])
                features[f'parkinson_{period}'] = np.sqrt(high_low.rolling(window=period).var() / (4 * np.log(2)))
                features[f'volatility_{period}'] = features['returns'].rolling(window=period).std()

            # Target variable
            features['target'] = (data['close'].shift(-1) > data['close']).astype(int)
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise 

if __name__ == '__main__':
    try:
        unittest.main(verbosity=2)
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        raise 