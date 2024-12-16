import dash_bootstrap_components as dbc
from dash import html, dcc

def create_position_controls(position_data):
    """Create position management controls"""
    return dbc.Card([
        dbc.CardHeader("Position Controls"),
        dbc.CardBody([
            # Position Information
            html.Div([
                html.H6(f"Symbol: {position_data['symbol']}"),
                html.H6(f"Side: {position_data['side']}"),
                html.H6(f"Entry: {position_data['entry_price']:.8f}"),
                html.H6(f"Current: {position_data['current_price']:.8f}"),
                html.H6(f"P&L: {position_data['pnl']:.2f}%"),
            ], className="mb-3"),
            
            # Stop Loss Control
            html.Div([
                html.Label("Modify Stop Loss"),
                dbc.Input(
                    id="modify-sl-input",
                    type="number",
                    value=position_data['stop_loss'],
                    step=0.00000001,
                    className="mb-2"
                ),
                dbc.Button(
                    "Update Stop Loss",
                    id="update-sl-button",
                    color="warning",
                    className="w-100 mb-3"
                )
            ]),
            
            # Take Profit Control
            html.Div([
                html.Label("Modify Take Profit"),
                dbc.Input(
                    id="modify-tp-input",
                    type="number",
                    value=position_data['take_profit'],
                    step=0.00000001,
                    className="mb-2"
                ),
                dbc.Button(
                    "Update Take Profit",
                    id="update-tp-button",
                    color="success",
                    className="w-100 mb-3"
                )
            ]),
            
            # Add to Position
            html.Div([
                html.Label("Add to Position"),
                dbc.Input(
                    id="add-position-amount",
                    type="number",
                    placeholder="Amount",
                    className="mb-2"
                ),
                dbc.Button(
                    "Add to Position",
                    id="add-position-button",
                    color="primary",
                    className="w-100 mb-3"
                )
            ]),
            
            # Close Position
            dbc.Button(
                "Close Position",
                id="close-position-button",
                color="danger",
                className="w-100"
            )
        ])
    ])

def create_trading_controls(trading_data):
    """Create trading controls"""
    return dbc.Card([
        dbc.CardHeader("Trading Controls"),
        dbc.CardBody([
            # Trading Pair Selection
            html.Div([
                html.Label("Trading Pair"),
                dcc.Dropdown(
                    id="trading-pair-select",
                    options=[
                        {"label": pair, "value": pair}
                        for pair in trading_data['available_pairs']
                    ],
                    value=trading_data['current_pair'],
                    className="mb-3"
                )
            ]),
            
            # Order Type Selection
            html.Div([
                html.Label("Order Type"),
                dcc.Dropdown(
                    id="order-type-select",
                    options=[
                        {"label": "Market", "value": "MARKET"},
                        {"label": "Limit", "value": "LIMIT"},
                        {"label": "Stop Market", "value": "STOP_MARKET"},
                        {"label": "Stop Limit", "value": "STOP_LIMIT"}
                    ],
                    value="MARKET",
                    className="mb-3"
                )
            ]),
            
            # Amount Controls
            html.Div([
                html.Label("Amount (USDT)"),
                dbc.Input(
                    id="order-amount-input",
                    type="number",
                    min=0,
                    step=0.1,
                    className="mb-2"
                ),
                dbc.ButtonGroup([
                    dbc.Button("25%", id="amount-25", color="secondary", size="sm"),
                    dbc.Button("50%", id="amount-50", color="secondary", size="sm"),
                    dbc.Button("75%", id="amount-75", color="secondary", size="sm"),
                    dbc.Button("100%", id="amount-100", color="secondary", size="sm")
                ], className="w-100 mb-3")
            ]),
            
            # Price Controls
            html.Div([
                html.Label("Price"),
                dbc.Input(
                    id="order-price-input",
                    type="number",
                    min=0,
                    step=0.00000001,
                    className="mb-3"
                )
            ]),
            
            # Stop Loss & Take Profit
            html.Div([
                html.Label("Stop Loss (%)"),
                dbc.Input(
                    id="sl-percent-input",
                    type="number",
                    min=0.1,
                    step=0.1,
                    value=2,
                    className="mb-2"
                ),
                html.Label("Take Profit (%)"),
                dbc.Input(
                    id="tp-percent-input",
                    type="number",
                    min=0.1,
                    step=0.1,
                    value=4,
                    className="mb-3"
                )
            ]),
            
            # Order Buttons
            dbc.ButtonGroup([
                dbc.Button("Buy", id="buy-button-3", color="success", className="w-50"),
                dbc.Button("Sell", id="sell-button-3", color="danger", className="w-50")
            ], className="w-100")
        ])
    ])

def create_chart_controls():
    """Create chart control panel"""
    return dbc.Card([
        dbc.CardHeader("Chart Controls"),
        dbc.CardBody([
            # Timeframe Selection
            html.Div([
                html.Label("Timeframe"),
                dcc.Dropdown(
                    id="timeframe-select",
                    options=[
                        {"label": "1m", "value": "1m"},
                        {"label": "5m", "value": "5m"},
                        {"label": "15m", "value": "15m"},
                        {"label": "1h", "value": "1h"},
                        {"label": "4h", "value": "4h"},
                        {"label": "1d", "value": "1d"}
                    ],
                    value="15m",
                    className="mb-3"
                )
            ]),
            
            # Indicator Selection
            html.Div([
                html.Label("Indicators"),
                dcc.Checklist(
                    id="indicator-select",
                    options=[
                        {"label": "EMA", "value": "ema"},
                        {"label": "MACD", "value": "macd"},
                        {"label": "RSI", "value": "rsi"},
                        {"label": "Bollinger Bands", "value": "bbands"},
                        {"label": "Volume Profile", "value": "volume_profile"}
                    ],
                    value=["ema", "macd"],
                    className="mb-3"
                )
            ]),
            
            # Drawing Tools
            html.Div([
                html.Label("Drawing Tools"),
                dbc.ButtonGroup([
                    dbc.Button("Trend Line", id="trend-line-tool", color="secondary", size="sm"),
                    dbc.Button("Fibonacci", id="fib-tool", color="secondary", size="sm"),
                    dbc.Button("Rectangle", id="rect-tool", color="secondary", size="sm"),
                    dbc.Button("Clear", id="clear-drawings", color="danger", size="sm")
                ], className="w-100 mb-3")
            ])
        ])
    ])
