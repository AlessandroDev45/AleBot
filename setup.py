from setuptools import setup, find_packages

setup(
    name="TradingSystem",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'python-dotenv',
        'pynvml',
    ],
) 