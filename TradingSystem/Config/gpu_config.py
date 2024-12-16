import tensorflow as tf
import torch
import logging

logger = logging.getLogger(__name__)

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    try:
        # TensorFlow GPU configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set TensorFlow to use only GPU 0 (RTX 4090)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Configure for mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            logger.info(f"TensorFlow GPU configured: {gpus[0].name}")
        
        # PyTorch GPU configuration
        if torch.cuda.is_available():
            # Set device to RTX 4090
            torch.cuda.set_device(0)
            
            # Enable TF32 for better performance on RTX 4090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory allocation strategy
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
            
            # Enable CUDA graphs for repeated operations
            torch.backends.cuda.enable_graphs = True
            
            # Set optimal CUDA stream priority
            torch.cuda.Stream(priority=-1)  # High priority stream
            
            logger.info(f"PyTorch GPU configured: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            return True
            
    except Exception as e:
        logger.error(f"Error configuring GPU: {str(e)}")
        return False 