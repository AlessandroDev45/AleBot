import sys
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run_tests():
    try:
        logger.debug("Loading test module...")
        from tests.test_ml_model import TestMLModel
        
        logger.debug("Creating test suite...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMLModel)
        
        logger.debug("Running tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        logger.debug(f"Tests completed. Success: {result.wasSuccessful()}")
        return result.wasSuccessful()
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 