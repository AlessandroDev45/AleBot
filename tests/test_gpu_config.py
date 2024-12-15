import unittest
import tensorflow as tf
import torch
import logging
from TradingSystem.Config.gpu_config import configure_gpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGPUConfig(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.gpu_configured = configure_gpu()
    
    def test_tensorflow_gpu(self):
        """Test TensorFlow GPU configuration"""
        logger.info("\n=== TensorFlow GPU Configuration ===")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
        
        # Test if TensorFlow can see the GPU
        gpus = tf.config.list_physical_devices('GPU')
        self.assertTrue(len(gpus) > 0, "No GPU found for TensorFlow")
        
        # Test GPU computation
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        
        self.assertIsNotNone(c, "GPU computation failed")
        logger.info("TensorFlow GPU test passed")
    
    def test_pytorch_gpu(self):
        """Test PyTorch GPU configuration"""
        logger.info("\n=== PyTorch GPU Configuration ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test if PyTorch can see the GPU
        self.assertTrue(torch.cuda.is_available(), "CUDA not available for PyTorch")
        
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Test GPU computation
            a = torch.randn(1000, 1000, device='cuda')
            b = torch.randn(1000, 1000, device='cuda')
            c = torch.matmul(a, b)
            
            self.assertIsNotNone(c, "GPU computation failed")
            logger.info("PyTorch GPU test passed")
    
    def test_gpu_memory(self):
        """Test GPU memory allocation"""
        if torch.cuda.is_available():
            logger.info("\n=== GPU Memory Test ===")
            
            # Get initial memory usage
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"Initial GPU memory: {initial_memory / 1e9:.2f} GB")
            
            # Allocate large tensor
            tensor_size = 1000
            large_tensor = torch.randn(tensor_size, tensor_size, device='cuda')
            
            # Get memory after allocation
            allocated_memory = torch.cuda.memory_allocated()
            logger.info(f"Memory after allocation: {allocated_memory / 1e9:.2f} GB")
            
            # Clean up
            del large_tensor
            torch.cuda.empty_cache()
            
            # Get final memory
            final_memory = torch.cuda.memory_allocated()
            logger.info(f"Final GPU memory: {final_memory / 1e9:.2f} GB")
            
            self.assertTrue(allocated_memory > initial_memory, "GPU memory not allocated")
            self.assertTrue(final_memory < allocated_memory, "GPU memory not released")

if __name__ == '__main__':
    unittest.main() 