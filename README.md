# AleBot

Advanced cryptocurrency trading bot with ML-driven analysis and risk management.

## Features

- Real-time market data analysis
- Machine learning-based predictions
- Advanced risk management
- Multiple trading strategies
- Automated trade execution
- Web dashboard for monitoring
- Telegram bot integration

## Project Structure

```
AleBot/
├── analysis/
│   ├── technical.py
│   └── market_data.py
├── ml/
│   ├── model.py
│   └── features.py
├── risk/
│   └── management.py
├── exchange/
│   └── binance.py
├── dashboard/
│   └── app.py
├── database/
│   └── models.py
├── config/
│   └── settings.py
├── utils/
│   └── helpers.py
└── main.py
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables in `BTC.env`
4. Run the bot: `python main.py`

## Configuration

Copy `BTC.env.example` to `BTC.env` and fill in your API credentials:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_bot_token
```

## Usage

Start the bot:
```bash
python main.py
```

Access the dashboard at http://localhost:8050

## Trading Strategy

The bot implements a sophisticated trading strategy combining:
- Technical analysis
- Machine learning predictions
- Market microstructure analysis
- Risk management rules
- Multiple timeframe confirmation

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT