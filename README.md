# AleBot - Cryptocurrency Trading Bot

AleBot is an advanced cryptocurrency trading bot that uses technical analysis and risk management to execute trades on Binance.

## Features

- Real-time market data analysis
- Technical analysis with multiple indicators
- Dynamic risk management
- Position management with stop-loss and take-profit
- Database storage for trades and market data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AleBot.git
cd AleBot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the bot:
```bash
cp BTC.env.example BTC.env
# Edit BTC.env with your API credentials
```

## Usage

Start the bot:
```bash
python main.py
```

## Configuration

Edit `BTC.env` with your settings:
- Add your Binance API credentials
- Adjust risk parameters
- Configure trading pairs

## Project Structure

```
AleBot/
├── main.py           # Main bot implementation
├── config.py         # Configuration management
├── database.py       # Database operations
├── exchange.py       # Exchange integration
├── analysis.py       # Technical analysis
├── requirements.txt  # Python dependencies
└── BTC.env          # Configuration file
```

## Important Notes

- Always start with small trade sizes
- Monitor the bot's performance
- Keep your API credentials secure
- Regularly backup the database

## Testing

Run the bot in test mode first:
```bash
python main.py --test
```