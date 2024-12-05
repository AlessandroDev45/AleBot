import unittest
import asyncio
import sys
from test_core import TestCore
from test_trading import TestTrading
from test_performance import TestPerformance

def run_async_test(test_case):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_case())

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCore))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTrading))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())