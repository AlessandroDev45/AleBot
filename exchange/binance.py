import aiohttp
import hmac
import hashlib
import time
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BinanceExchange:
    BASE_URL = 'https://api.binance.com'
    WS_URL = 'wss://stream.binance.com:9443/ws'

    def __init__(self, config):
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.session = None
        self.ws = None
        self.last_prices = {}

    async def connect(self):
        self.session = aiohttp.ClientSession()
        await self.test_connection()

    async def test_connection(self):
        try:
            await self.get_exchange_info()
            logger.info('Successfully connected to Binance')
        except Exception as e:
            logger.error(f'Connection test failed: {e}')
            raise

    async def get_exchange_info(self) -> Dict:
        async with self.session.get(f'{self.BASE_URL}/api/v3/exchangeInfo') as response:
            return await response.json()

    def _sign_request(self, params: Dict) -> Tuple[str, Dict]:
        params['timestamp'] = int(time.time() * 1000)
        query_string = '&'.join([f'{k}={v}' for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': self.api_key}
        return query_string, headers