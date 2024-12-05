#!/bin/bash

# Exit on any error
set -e

echo "Running environment validation..."
python scripts/validate_environment.py

echo "\nRunning unit tests..."
python -m pytest tests/ -v

echo "\nRunning integration tests..."
python tests/run_tests.py

echo "\nAll tests completed successfully!"