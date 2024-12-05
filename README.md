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

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables
4. Run the bot: `python main.py`

## Configuration

Copy `BTC.env.example` to `BTC.env` and fill in your API credentials.

## Directory Structure

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