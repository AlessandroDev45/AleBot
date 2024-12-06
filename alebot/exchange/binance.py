class BinanceExchange:
    BASE_URL = 'https://api.binance.com'
    WS_URL = 'wss://stream.binance.com:9443/ws'

    def __init__(self, config):
        self.config = config
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.session = None
        self.ws_connections = {}
        self.order_books = {}
        self.last_prices = {}
        self.positions = {}
        self.subscribers = []

    async def connect(self):
        """Initialize connection and session"""
        try:
            self.session = aiohttp.ClientSession()
            await self.test_connection()
            logging.info('Successfully connected to Binance')
        except Exception as e:
            logging.error(f'Connection error: {e}')
            raise

    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            await self._make_request('GET', '/api/v3/ping')
            return True
        except Exception as e:
            logging.error(f'Connection test failed: {e}')
            return False

    async def _make_request(self, method: str, endpoint: str, 
                          params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make HTTP request to Binance API"""
        if params is None:
            params = {}

        url = f'{self.BASE_URL}{endpoint}'

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f'{k}={v}' for k, v in sorted(params.items())])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            headers = {'X-MBX-APIKEY': self.api_key}
        else:
            headers = {}

        async with self.session.request(method, url, params=params, headers=headers) as response:
            if response.status == 429:  # Rate limit
                retry_after = int(response.headers.get('Retry-After', 5))
                await asyncio.sleep(retry_after)
                return await self._make_request(method, endpoint, params, signed)

            response.raise_for_status()
            return await response.json()

    async def start_market_data_stream(self):
        """Start websocket streams for market data"""
        try:
            for symbol in self.config.TRADING_CONFIG['symbols']:
                # Start kline stream
                await self._start_kline_stream(symbol)
                # Start order book stream
                await self._start_orderbook_stream(symbol)
                # Start trade stream
                await self._start_trade_stream(symbol)

            logging.info('Market data streams started successfully')
        except Exception as e:
            logging.error(f'Error starting market data streams: {e}')
            raise

    async def _start_kline_stream(self, symbol: str):
        """Start kline websocket stream"""
        stream_name = f'{symbol.lower()}@kline_1m'
        url = f'{self.WS_URL}/{stream_name}'

        async def handle_socket():
            while True:
                try:
                    async with websockets.connect(url) as ws:
                        self.ws_connections[stream_name] = ws
                        async for message in ws:
                            await self._handle_kline_message(json.loads(message))
                except Exception as e:
                    logging.error(f'Kline stream error: {e}')
                    await asyncio.sleep(5)

        asyncio.create_task(handle_socket())

    async def _handle_kline_message(self, msg: Dict):
        """Process kline websocket message"""
        try:
            kline = msg['k']
            symbol = msg['s']

            if kline['x']:  # Candle closed
                candle_data = {
                    'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }

                # Update last price
                self.last_prices[symbol] = float(kline['c'])

                # Notify subscribers
                for callback in self.subscribers:
                    await callback('kline', symbol, candle_data)

        except Exception as e:
            logging.error(f'Error processing kline message: {e}')

    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = 'MARKET', price: Optional[float] = None,
                         stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Dict:
        """Place a new order"""
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': self._format_quantity(quantity)
            }

            if order_type == 'LIMIT':
                if not price:
                    raise ValueError('Price required for LIMIT order')
                params['price'] = self._format_price(price)
                params['timeInForce'] = 'GTC'

            # Place main order
            order = await self._make_request('POST', '/api/v3/order', params, signed=True)

            # Place stop loss if specified
            if stop_loss and order['status'] == 'FILLED':
                await self._place_stop_loss(symbol, side, quantity, stop_loss)

            # Place take profit if specified
            if take_profit and order['status'] == 'FILLED':
                await self._place_take_profit(symbol, side, quantity, take_profit)

            return order

        except Exception as e:
            logging.error(f'Error placing order: {e}')
            raise

    async def get_account_balance(self) -> float:
        """Get account balance in USDT"""
        try:
            account = await self._make_request('GET', '/api/v3/account', signed=True)
            usdt_balance = next(
                (float(asset['free']) for asset in account['balances']
                 if asset['asset'] == 'USDT'),
                0.0
            )
            return usdt_balance
        except Exception as e:
            logging.error(f'Error getting account balance: {e}')
            raise

    async def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            positions = {}
            account = await self._make_request('GET', '/api/v3/account', signed=True)

            for asset in account['balances']:
                free = float(asset['free'])
                if free > 0:
                    symbol = f"{asset['asset']}USDT"
                    if symbol in self.last_prices:
                        positions[symbol] = {
                            'symbol': symbol,
                            'size': free,
                            'value': free * self.last_prices[symbol],
                            'entry_price': self._get_position_entry(symbol),
                            'current_price': self.last_prices[symbol]
                        }

            self.positions = positions
            return positions

        except Exception as e:
            logging.error(f'Error getting positions: {e}')
            raise