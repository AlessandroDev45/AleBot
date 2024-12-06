import sys
import pkg_resources
import subprocess
import platform
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version requirements"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.error(
            f'Python {required_version[0]}.{required_version[1]} or higher is required. '
            f'You have Python {current_version[0]}.{current_version[1]}'
        )
        return False
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    requirements_path = Path(__file__).parent.parent / 'requirements.txt'
    
    try:
        with open(requirements_path) as f:
            requirements = [line.strip() for line in f if line.strip()]
        
        pkg_resources.require(requirements)
        return True
    except pkg_resources.DistributionNotFound as e:
        logger.error(f'Missing package: {e}')
        return False
    except pkg_resources.VersionConflict as e:
        logger.error(f'Version conflict: {e}')
        return False

def check_gpu_support():
    """Check if GPU support is available for TensorFlow"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f'GPU support available: {len(gpus)} GPU(s) found')
            return True
        else:
            logger.warning('No GPU support available. ML models will run on CPU')
            return False
    except Exception as e:
        logger.error(f'Error checking GPU support: {e}')
        return False

def check_api_access():
    """Check if API credentials are configured"""
    required_env_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'TELEGRAM_BOT_TOKEN'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f'Missing environment variables: {missing_vars}')
        return False
    return True

def check_disk_space():
    """Check if sufficient disk space is available"""
    required_space_gb = 5
    current_dir = Path().absolute()
    
    total, used, free = shutil.disk_usage(current_dir)
    free_gb = free // (2**30)
    
    if free_gb < required_space_gb:
        logger.error(
            f'Insufficient disk space. {required_space_gb}GB required, '
            f'{free_gb}GB available'
        )
        return False
    return True

def main():
    """Run all validation checks"""
    checks = [
        ('Python Version', check_python_version),
        ('Dependencies', check_dependencies),
        ('GPU Support', check_gpu_support),
        ('API Access', check_api_access),
        ('Disk Space', check_disk_space)
    ]
    
    all_passed = True
    
    for name, check in checks:
        logger.info(f'Checking {name}...')
        try:
            result = check()
            if result:
                logger.info(f'✅ {name} check passed')
            else:
                logger.error(f'❌ {name} check failed')
                all_passed = False
        except Exception as e:
            logger.error(f'❌ {name} check error: {e}')
            all_passed = False
    
    if all_passed:
        logger.info('\n🎉 All validation checks passed!')
        return 0
    else:
        logger.error('\n❌ Some validation checks failed')
        return 1

if __name__ == '__main__':
    sys.exit(main())