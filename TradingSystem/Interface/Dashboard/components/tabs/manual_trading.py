import dash_bootstrap_components as dbc
from dash import html, dcc

def create_manual_trading_tab():
    """Create manual trading tab with subtabs for better organization"""
    return dbc.Container([
        # Trade Execution Modal
        dbc.Modal([
            dbc.ModalHeader("Confirm Trade"),
            dbc.ModalBody([
                html.Div(id="trade-confirmation-details"),
                dbc.Alert(
                    "Please review the trade details carefully before confirming.",
                    color="warning",
                    className="mt-2"
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Confirm", id="confirm-trade", color="success", className="me-2"),
                dbc.Button("Cancel", id="cancel-trade", color="danger")
            ])
        ], id="trade-confirmation-modal", is_open=False),

        # Main Tabs for Manual Trading
        dbc.Tabs([
            # Main Trading Tab
            dbc.Tab(label="Trading", children=[
                dbc.Row([
                    # Left Column - Order Form & Controls
                    dbc.Col([
                        # Basic Order Form
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Place Order", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="order-form-info"),
                                dbc.Tooltip(
                                    "Basic order placement form",
                                    target="order-form-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                html.Label("Trading Pair"),
                                dcc.Dropdown(
                                    id="trading-pair",
                                    options=[{"label": "BTC/USDT", "value": "BTCUSDT"}],
                                    value="BTCUSDT",
                                    className="mb-2"
                                ),
                                dbc.Tooltip(
                                    "Select the trading pair you want to trade",
                                    target="trading-pair",
                                    placement="top"
                                ),
                                html.Label("Order Type"),
                                dcc.Dropdown(
                                    id="order-type",
                                    options=[
                                        {"label": "Market", "value": "MARKET"},
                                        {"label": "Limit", "value": "LIMIT"},
                                        {"label": "Stop Loss", "value": "STOP_LOSS"},
                                        {"label": "Take Profit", "value": "TAKE_PROFIT"},
                                        {"label": "OCO", "value": "OCO"}
                                    ],
                                    value="MARKET",
                                    className="mb-2"
                                ),
                                dbc.Tooltip(
                                    "Choose between Market, Limit, Stop Loss, Take Profit, or OCO orders",
                                    target="order-type",
                                    placement="top"
                                ),
                                html.Label("Side"),
                                dcc.Dropdown(
                                    id="order-side",
                                    options=[
                                        {"label": "Buy", "value": "BUY"},
                                        {"label": "Sell", "value": "SELL"}
                                    ],
                                    value="BUY",
                                    className="mb-2"
                                ),
                                html.Label("Amount (USDT)"),
                                dbc.InputGroup([
                                    dbc.Input(id="order-amount", type="number", min=0, step=0.01),
                                    dbc.InputGroupText("USDT")
                                ], className="mb-2"),
                                dbc.Tooltip(
                                    "Enter the amount you want to trade in USDT",
                                    target="order-amount",
                                    placement="top"
                                ),
                                html.Label("Price (USDT)"),
                                dbc.InputGroup([
                                    dbc.Input(id="order-price", type="number", min=0, step=0.01),
                                    dbc.InputGroupText("USDT")
                                ], className="mb-2"),
                                dbc.Tooltip(
                                    "Set your desired entry price (for Limit orders)",
                                    target="order-price",
                                    placement="top"
                                ),
                                html.Label("Leverage"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="leverage",
                                        type="number",
                                        min=1,
                                        max=125,
                                        step=1,
                                        value=1
                                    ),
                                    dbc.InputGroupText("x")
                                ], className="mb-2"),
                                dbc.Tooltip(
                                    "Set leverage for your trade (1x to 125x)",
                                    target="leverage",
                                    placement="top"
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Buy",
                                            id="buy-button",
                                            color="success",
                                            className="w-100 mt-2"
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Button(
                                            "Sell",
                                            id="sell-button",
                                            color="danger",
                                            className="w-100 mt-2"
                                        ),
                                    ], width=6),
                                ]),
                                dbc.FormFeedback(
                                    "Please fill in all required fields",
                                    id="order-validation-feedback",
                                    type="invalid",
                                    className="d-none"
                                )
                            ])
                        ], className="mb-3"),

                        # Advanced Order Types
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Advanced Orders", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="advanced-orders-info"),
                                dbc.Tooltip(
                                    "Advanced order types and configurations",
                                    target="advanced-orders-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                # OCO Orders Interface
                                html.Label("OCO Orders"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Stop Loss"),
                                        dbc.InputGroup([
                                            dbc.Input(id="oco-stop", type="number", min=0, step=0.01),
                                            dbc.InputGroupText("USDT")
                                        ]),
                                        dbc.Tooltip(
                                            "Set stop loss price for OCO order",
                                            target="oco-stop",
                                            placement="top"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Take Profit"),
                                        dbc.InputGroup([
                                            dbc.Input(id="oco-profit", type="number", min=0, step=0.01),
                                            dbc.InputGroupText("USDT")
                                        ]),
                                        dbc.Tooltip(
                                            "Set take profit price for OCO order",
                                            target="oco-profit",
                                            placement="top"
                                        )
                                    ], width=6)
                                ], className="mb-2"),

                                # Multiple Take-Profit Levels
                                html.Label("Multiple Take-Profit Levels"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("Add Level", id="add-tp-level", color="primary", size="sm", className="mb-2"),
                                        html.Div(id="tp-levels-container"),
                                        dbc.Tooltip(
                                            "Add multiple take-profit levels for gradual position exit",
                                            target="add-tp-level",
                                            placement="top"
                                        )
                                    ])
                                ], className="mb-2"),

                                # Slippage Controls
                                html.Label("Slippage Control"),
                                dbc.InputGroup([
                                    dbc.Input(id="slippage-tolerance", type="number", min=0, max=5, step=0.1, value=0.5),
                                    dbc.InputGroupText("%")
                                ], className="mb-2"),
                                dbc.Tooltip(
                                    "Control maximum allowed slippage for market orders",
                                    target="slippage-tolerance",
                                    placement="top"
                                ),

                                # Smart Order Routing
                                html.Label("Smart Order Routing"),
                                dcc.Checklist(
                                    id="smart-routing-options",
                                    options=[
                                        {"label": "Best Price Execution", "value": "best_price"},
                                        {"label": "Minimize Impact", "value": "min_impact"},
                                        {"label": "TWAP", "value": "twap"},
                                        {"label": "VWAP", "value": "vwap"}
                                    ],
                                    value=["best_price"],
                                    className="mb-2"
                                ),
                                dbc.Tooltip(
                                    "Select smart order routing options for better execution",
                                    target="smart-routing-options",
                                    placement="top"
                                ),

                                # Time-Based Execution
                                html.Label("Time-Based Execution"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.InputGroup([
                                            dbc.Input(id="execution-duration", type="number", min=1, placeholder="Duration"),
                                            dbc.InputGroupText("min")
                                        ]),
                                        dbc.Tooltip(
                                            "Set duration for time-based order execution",
                                            target="execution-duration",
                                            placement="top"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.InputGroup([
                                            dbc.Input(id="execution-intervals", type="number", min=1, placeholder="Intervals"),
                                            dbc.InputGroupText("parts")
                                        ]),
                                        dbc.Tooltip(
                                            "Set number of intervals for order splitting",
                                            target="execution-intervals",
                                            placement="top"
                                        )
                                    ], width=6)
                                ])
                            ])
                        ], className="mb-3")
                    ], width=3),

                    # Middle Column - Charts
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                dbc.Row([
                                    dbc.Col(html.H6("Trading Chart"), width=4),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id="chart-timeframe",
                                            options=[
                                                {"label": "1m", "value": "1m"},
                                                {"label": "5m", "value": "5m"},
                                                {"label": "15m", "value": "15m"},
                                                {"label": "1h", "value": "1h"},
                                                {"label": "4h", "value": "4h"},
                                                {"label": "1d", "value": "1d"}
                                            ],
                                            value="15m",
                                            clearable=False
                                        ),
                                        dbc.Tooltip(
                                            "Select timeframe for the chart analysis",
                                            target="chart-timeframe",
                                            placement="top"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id="chart-indicators",
                                            options=[
                                                {"label": "RSI", "value": "rsi"},
                                                {"label": "MACD", "value": "macd"},
                                                {"label": "Bollinger Bands", "value": "bbands"},
                                                {"label": "EMA", "value": "ema"},
                                                {"label": "Volume Profile", "value": "volume_profile"}
                                            ],
                                            value=["rsi"],
                                            multi=True
                                        ),
                                        dbc.Tooltip(
                                            "Select technical indicators to display",
                                            target="chart-indicators",
                                            placement="top"
                                        )
                                    ], width=4)
                                ])
                            ]),
                            dbc.CardBody([
                                # Main Chart with drawing tools
                                dcc.Graph(
                                    id="trading-chart",
                                    style={"height": "400px"},
                                    config={
                                        'modeBarButtonsToAdd': [
                                            'drawline',
                                            'drawopenpath',
                                            'eraseshape',
                                            'drawrect',
                                            'drawcircle'
                                        ],
                                        'displaylogo': False
                                    }
                                ),
                                dbc.Tooltip(
                                    "Click and drag to draw support/resistance lines and patterns",
                                    target="trading-chart",
                                    placement="top"
                                ),
                                dbc.Tooltip(
                                    "Click and drag to zoom, double-click to reset",
                                    target="trading-chart",
                                    placement="top"
                                ),
                                # Volume Chart
                                dcc.Graph(
                                    id="trading-volume-chart",
                                    style={"height": "150px"}
                                ),
                                dbc.Tooltip(
                                    "Volume bars show buying and selling pressure",
                                    target="trading-volume-chart",
                                    placement="top"
                                ),
                                # Indicators Chart
                                dcc.Graph(
                                    id="trading-indicators-chart",
                                    style={"height": "150px"}
                                ),
                                dbc.Tooltip(
                                    "Technical indicators help identify potential trade opportunities",
                                    target="trading-indicators-chart",
                                    placement="top"
                                )
                            ])
                        ])
                    ], width=6),

                    # Right Column - Order Book & Positions
                    dbc.Col([
                        # Order Book
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Order Book", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="orderbook-info"),
                                dbc.Tooltip(
                                    "Real-time order book with market depth",
                                    target="orderbook-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Asks", className="text-danger text-center"),
                                        html.Div(id="orderbook-asks", style={"maxHeight": "200px", "overflow": "auto"})
                                    ], width=6),
                                    dbc.Col([
                                        html.H6("Bids", className="text-success text-center"),
                                        html.Div(id="orderbook-bids", style={"maxHeight": "200px", "overflow": "auto"})
                                    ], width=6)
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Market Depth"),
                                        dcc.Graph(id="market-depth", style={"height": "200px"})
                                    ])
                                ])
                            ])
                        ], className="mb-3"),

                        # Technical Indicators
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Technical Indicators", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="indicators-info"),
                                dbc.Tooltip(
                                    "Real-time technical indicator values and their interpretations",
                                    target="indicators-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("RSI (14)"),
                                        html.H4(id="trading-rsi", className="text-info"),
                                        dbc.Tooltip(
                                            "Relative Strength Index: Values above 70 indicate overbought, below 30 indicate oversold",
                                            target="trading-rsi",
                                            placement="top"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.H6("MACD"),
                                        html.H4(id="trading-macd", className="text-info"),
                                        dbc.Tooltip(
                                            "Moving Average Convergence Divergence: Shows trend direction and momentum",
                                            target="trading-macd",
                                            placement="top"
                                        )
                                    ], width=6)
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Signal Line"),
                                        html.H4(id="trading-macd-signal", className="text-warning"),
                                        dbc.Tooltip(
                                            "MACD Signal Line: Crossovers with MACD line indicate potential trade signals",
                                            target="trading-macd-signal",
                                            placement="top"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.H6("Histogram"),
                                        html.H4(id="trading-macd-hist", className="text-primary"),
                                        dbc.Tooltip(
                                            "MACD Histogram: Shows the difference between MACD and Signal line",
                                            target="trading-macd-hist",
                                            placement="top"
                                        )
                                    ], width=6)
                                ])
                            ])
                        ], className="mb-3"),

                        # Position Summary with Enhanced Risk Metrics
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Position Summary", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="position-info"),
                                dbc.Tooltip(
                                    "Current position information and risk metrics",
                                    target="position-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Entry Price"),
                                        html.H4(id="position-entry", className="text-info"),
                                        dbc.Tooltip(
                                            "Average entry price of current position",
                                            target="position-entry",
                                            placement="top"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.H6("Current PnL"),
                                        html.H4(id="position-pnl", className="text-success"),
                                        dbc.Tooltip(
                                            "Unrealized profit/loss of current position",
                                            target="position-pnl",
                                            placement="top"
                                        )
                                    ], width=6)
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Risk/Reward"),
                                        html.H4(id="risk-reward-ratio", className="text-warning"),
                                        dbc.Tooltip(
                                            "Current risk to reward ratio based on stop loss and take profit levels",
                                            target="risk-reward-ratio",
                                            placement="top"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.H6("Break Even"),
                                        html.H4(id="break-even-price", className="text-primary"),
                                        dbc.Tooltip(
                                            "Price needed to break even including fees",
                                            target="break-even-price",
                                            placement="top"
                                        )
                                    ], width=6)
                                ])
                            ])
                        ])
                    ], width=3)
                ])
            ]),

            # Advanced Orders Tab
            dbc.Tab(label="Advanced Orders", children=[
                dbc.Row([
                    dbc.Col([
                        # Multiple Take-Profit Interface
                        dbc.Card([
                            dbc.CardHeader("Multiple Take-Profit Setup"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Take-Profit Levels"),
                                        dbc.Button("Add Level", id="add-tp-level-advanced", color="primary", size="sm"),
                                        html.Div(id="tp-levels-container-advanced")
                                    ])
                                ])
                            ])
                        ], className="mb-3"),
                        
                        # Advanced Risk Controls
                        dbc.Card([
                            dbc.CardHeader("Risk Controls"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Position Size Control"),
                                        dbc.InputGroup([
                                            dbc.Input(id="manual-max-position-size", type="number", placeholder="Max Position Size"),
                                            dbc.InputGroupText("USDT")
                                        ], className="mb-2"),
                                        html.Label("Daily Loss Limit (%)"),
                                        dbc.InputGroup([
                                            dbc.Input(
                                                id="manual-trading-daily-loss-limit-input",
                                                type="number",
                                                min=0.1,
                                                max=100,
                                                step=0.1,
                                                value=2,
                                                placeholder="Daily Loss Limit"
                                            ),
                                            dbc.InputGroupText("%")
                                        ], className="mb-2"),
                                        dbc.Tooltip(
                                            "Maximum allowed daily loss before trading is halted",
                                            target="manual-trading-daily-loss-limit-input",
                                            placement="top"
                                        )
                                    ])
                                ])
                            ])
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        # Smart Order Routing
                        dbc.Card([
                            dbc.CardHeader("Smart Order Routing"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Execution Strategy"),
                                        dcc.Dropdown(
                                            id="execution-strategy",
                                            options=[
                                                {"label": "TWAP", "value": "twap"},
                                                {"label": "VWAP", "value": "vwap"},
                                                {"label": "Iceberg", "value": "iceberg"},
                                                {"label": "Smart Route", "value": "smart"}
                                            ],
                                            value="smart"
                                        ),
                                        html.Label("Order Split", className="mt-2"),
                                        dbc.InputGroup([
                                            dbc.Input(id="order-split", type="number", min=1, max=100, value=1),
                                            dbc.InputGroupText("parts")
                                        ]),
                                        html.Label("Time Window", className="mt-2"),
                                        dbc.InputGroup([
                                            dbc.Input(id="time-window", type="number", min=1, value=60),
                                            dbc.InputGroupText("minutes")
                                        ])
                                    ])
                                ])
                            ])
                        ])
                    ], width=6),

                    # Add Smart Order Routing Enhancement
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H6("Smart Order Routing", className="mb-0"),
                                    dbc.Tooltip(
                                        "Advanced order routing for optimal execution",
                                        target="smart-order-routing",
                                        placement="top"
                                    )
                                ]),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Execution Algorithm"),
                                            dcc.Dropdown(
                                                id="execution-algo",
                                                options=[
                                                    {"label": "TWAP", "value": "twap"},
                                                    {"label": "VWAP", "value": "vwap"},
                                                    {"label": "Iceberg", "value": "iceberg"},
                                                    {"label": "Adaptive", "value": "adaptive"}
                                                ],
                                                value="adaptive",
                                                className="mb-2"
                                            ),
                                            dbc.Tooltip(
                                                "Select the execution algorithm for your orders",
                                                target="execution-algo",
                                                placement="top"
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Urgency Level"),
                                            dcc.Slider(
                                                id="urgency-level",
                                                min=1,
                                                max=5,
                                                value=3,
                                                marks={1: "Low", 3: "Medium", 5: "High"},
                                                className="mb-2"
                                            ),
                                            dbc.Tooltip(
                                                "Set the urgency level for order execution",
                                                target="urgency-level",
                                                placement="top"
                                            )
                                        ], width=6)
                                    ])
                                ])
                            ], className="mb-3")
                        ], width=12)
                    ])
                ])
            ]),

            # Market Analysis Tab
            dbc.Tab(label="Market Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        # Market Depth Visualization
                        dbc.Card([
                            dbc.CardHeader("Market Depth"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="market-depth-chart",
                                    style={"height": "300px"}
                                ),
                                dbc.Tooltip(
                                    "Visualizes the cumulative order book depth and liquidity distribution",
                                    target="market-depth-chart",
                                    placement="top"
                                )
                            ])
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        # Position Heat Map
                        dbc.Card([
                            dbc.CardHeader("Position Heat Map"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="position-heatmap",
                                    style={"height": "300px"}
                                ),
                                dbc.Tooltip(
                                    "Darker colors indicate higher position concentration at that price level",
                                    target="position-heatmap",
                                    placement="top"
                                ),
                                html.Div([
                                    html.Label("Time Range"),
                                    dcc.Dropdown(
                                        id="heatmap-timerange",
                                        options=[
                                            {"label": "1 Hour", "value": "1h"},
                                            {"label": "4 Hours", "value": "4h"},
                                            {"label": "1 Day", "value": "1d"}
                                        ],
                                        value="4h",
                                        clearable=False
                                    ),
                                    dbc.Tooltip(
                                        "Select time range for position analysis",
                                        target="heatmap-timerange",
                                        placement="top"
                                    )
                                ], className="mt-2"),
                                html.Div([
                                    html.Label("Aggregation Level"),
                                    dcc.Slider(
                                        id="heatmap-aggregation",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=10,
                                        marks={1: "Fine", 50: "Medium", 100: "Coarse"}
                                    ),
                                    dbc.Tooltip(
                                        "Adjust the granularity of position aggregation",
                                        target="heatmap-aggregation",
                                        placement="top"
                                    )
                                ], className="mt-2")
                            ])
                        ])
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        # Enhanced Fee Analysis Panel
                        dbc.Card([
                            dbc.CardHeader("Advanced Fee Analysis"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Trading Fees"),
                                        dbc.Progress(
                                            id="fee-progress",
                                            value=0,
                                            label="0%",
                                            color="info",
                                            className="mb-2",
                                            style={"height": "20px"}
                                        ),
                                        dbc.Tooltip(
                                            "Current trading fee tier and progress to next tier",
                                            target="fee-progress",
                                            placement="top"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.H6("Fee Statistics"),
                                        html.Div([
                                            html.P("30-Day Volume:", className="mb-1"),
                                            html.H4(id="volume-30d", className="text-info mb-2"),
                                            html.P("Current Fee Rate:", className="mb-1"),
                                            html.H4(id="current-fee-rate", className="text-success")
                                        ])
                                    ], width=6)
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Fee Breakdown"),
                                        dcc.Graph(
                                            id="fee-breakdown-chart",
                                            style={"height": "200px"}
                                        ),
                                        dbc.Tooltip(
                                            "Detailed breakdown of trading fees by type",
                                            target="fee-breakdown-chart",
                                            placement="top"
                                        )
                                    ])
                                ])
                            ])
                        ])
                    ], width=12)
                ]),
                
                # Enhanced Risk Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Enhanced Risk Controls"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Position Size Limits"),
                                        dbc.InputGroup([
                                            dbc.Input(
                                                id="trading-max-position-size",
                                                type="number",
                                                placeholder="Max Position Size",
                                                className="mb-2"
                                            ),
                                            dbc.InputGroupText("USDT")
                                        ]),
                                        dbc.Tooltip(
                                            "Maximum position size allowed",
                                            target="trading-max-position-size",
                                            placement="top"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Drawdown Protection"),
                                        dbc.InputGroup([
                                            dbc.Input(
                                                id="max-drawdown",
                                                type="number",
                                                placeholder="Max Drawdown",
                                                className="mb-2"
                                            ),
                                            dbc.InputGroupText("%")
                                        ]),
                                        dbc.Tooltip(
                                            "Maximum allowed drawdown before trading is paused",
                                            target="max-drawdown",
                                            placement="top"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Daily Loss Limit"),
                                        dbc.InputGroup([
                                            dbc.Input(
                                                id="manual-trading-daily-loss-limit-2",
                                                type="number",
                                                placeholder="Daily Loss Limit",
                                                className="mb-2"
                                            ),
                                            dbc.InputGroupText("USDT")
                                        ]),
                                        dbc.Tooltip(
                                            "Maximum allowed daily loss before trading is halted",
                                            target="manual-trading-daily-loss-limit-2",
                                            placement="top"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Position Size Limit"),
                                        dbc.InputGroup([
                                            dbc.Input(
                                                id="manual-position-size-limit",
                                                type="number",
                                                placeholder="Position Size Limit",
                                                className="mb-2"
                                            ),
                                            dbc.InputGroupText("USDT")
                                        ]),
                                        dbc.Tooltip(
                                            "Maximum allowed position size",
                                            target="manual-position-size-limit",
                                            placement="top"
                                        )
                                    ], width=4)
                                ])
                            ])
                        ])
                    ], width=12)
                ], className="mb-3"),

                # Add Volume Profile Analysis
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Volume Profile Analysis", className="mb-0"),
                                dbc.Tooltip(
                                    "Analyze trading volume distribution across price levels",
                                    target="volume-profile-analysis",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="volume-profile-chart",
                                    style={"height": "300px"}
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Time Range"),
                                        dcc.Dropdown(
                                            id="vp-timerange",
                                            options=[
                                                {"label": "1 Hour", "value": "1h"},
                                                {"label": "4 Hours", "value": "4h"},
                                                {"label": "1 Day", "value": "1d"}
                                            ],
                                            value="4h"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Profile Type"),
                                        dcc.Dropdown(
                                            id="vp-type",
                                            options=[
                                                {"label": "Regular", "value": "regular"},
                                                {"label": "Visible Range", "value": "visible"},
                                                {"label": "Fixed Range", "value": "fixed"}
                                            ],
                                            value="regular"
                                        )
                                    ], width=6)
                                ], className="mt-2")
                            ])
                        ])
                    ], width=12)
                ], className="mb-3"),

                # Add Market Correlation Analysis
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Market Correlation Analysis", className="mb-0"),
                                dbc.Tooltip(
                                    "Analyze correlations between different trading pairs",
                                    target="correlation-analysis",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="manual-correlation-heatmap",
                                    style={"height": "300px"}
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Base Assets"),
                                        dcc.Dropdown(
                                            id="correlation-assets-manual",
                                            options=[
                                                {"label": "BTC", "value": "BTC"},
                                                {"label": "ETH", "value": "ETH"},
                                                {"label": "BNB", "value": "BNB"}
                                            ],
                                            value=["BTC", "ETH"],
                                            multi=True
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Timeframe"),
                                        dcc.Dropdown(
                                            id="manual-correlation-timeframe",
                                            options=[
                                                {"label": "1 Day", "value": "1d"},
                                                {"label": "1 Week", "value": "1w"},
                                                {"label": "1 Month", "value": "1M"}
                                            ],
                                            value="1d"
                                        )
                                    ], width=6)
                                ], className="mt-2")
                            ])
                        ])
                    ], width=12)
                ])
            ])
        ], id="manual-trading-tabs", active_tab="trading")
    ], fluid=True, style={"height": "100vh", "overflow-y": "auto"}) 