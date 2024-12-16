import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go

def create_auto_bot_tab():
    """Create auto bot trading tab"""
    return dbc.Container([
        # Header with Save/Load buttons and Account Info
        dbc.Row([
            # Save/Load Buttons with tooltips
            dbc.Col([
                dbc.ButtonGroup([
                    html.Div([
                        dbc.Button("Save", id="auto-bot-save", color="primary", size="sm", className="me-1"),
                        dbc.Tooltip("Save current bot configuration and settings", target="auto-bot-save")
                    ]),
                    html.Div([
                        dbc.Button("Load", id="auto-bot-load", color="primary", size="sm", className="me-1"),
                        dbc.Tooltip("Load saved bot configuration", target="auto-bot-load")
                    ]),
                    html.Div([
                        dbc.Button("Export", id="auto-bot-export", color="secondary", size="sm"),
                        dbc.Tooltip("Export trading data and performance metrics", target="auto-bot-export")
                    ])
                ])
            ], width=2),
            
            # Account Info with tooltips
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("Total Balance", className="text-muted small"),
                        html.Div(id="auto-bot-total-balance", className="h6 mb-0"),
                        dbc.Tooltip("Total account value including open positions", target="auto-bot-total-balance")
                    ]), width=3),
                    dbc.Col(html.Div([
                        html.Div("Available Balance", className="text-muted small"),
                        html.Div(id="auto-bot-available-balance", className="h6 mb-0"),
                        dbc.Tooltip("Available balance for new trades", target="auto-bot-available-balance")
                    ]), width=3),
                    dbc.Col(html.Div([
                        html.Div("Position Value", className="text-muted small"),
                        html.Div(id="auto-bot-position-value", className="h6 mb-0"),
                        dbc.Tooltip("Current value of all open positions", target="auto-bot-position-value")
                    ]), width=3),
                    dbc.Col(html.Div([
                        html.Div("Last Update", className="text-muted small"),
                        html.Div(id="auto-bot-last-update", className="h6 mb-0"),
                        dbc.Tooltip("Time of last data update", target="auto-bot-last-update")
                    ]), width=3)
                ])
            ], width=10)
        ], className="mb-3"),
        
        # Main Content with Sub-tabs
        dbc.Tabs([
            # 1. Chart & Trading Tab
            dbc.Tab([
                dbc.Row([
                    # Left side - Chart Area
                    dbc.Col([
                        # Timeframe Controls with Tooltip
                        html.Div([
                            html.I(className="fas fa-info-circle me-2", id="timeframe-info"),
                            dbc.Tooltip(
                                "Select different timeframes to analyze price action",
                                target="timeframe-info",
                                placement="top"
                            ),
                            dbc.ButtonGroup([
                                dbc.Button("1m", id="tf-1m", size="sm", color="primary", outline=True),
                                dbc.Button("5m", id="tf-5m", size="sm", color="primary", outline=True),
                                dbc.Button("15m", id="tf-15m", size="sm", color="primary", outline=True),
                                dbc.Button("1h", id="tf-1h", size="sm", color="primary", outline=True),
                                dbc.Button("4h", id="tf-4h", size="sm", color="primary", outline=True)
                            ])
                        ], className="mb-2"),
                        
                        # Main Chart with Tooltip
                        html.Div([
                            html.I(className="fas fa-info-circle me-2", id="main-chart-info"),
                            dbc.Tooltip(
                                "Main price chart with candlesticks and trading indicators. Shows price action, volume, and key technical levels.",
                                target="main-chart-info",
                                placement="top"
                            ),
                            dcc.Graph(
                                id="main-chart",
                                style={"height": "500px"}
                            )
                        ]),
                        
                        # Volume Chart with Tooltip
                        html.Div([
                            html.I(className="fas fa-info-circle me-2", id="volume-chart-info"),
                            dbc.Tooltip(
                                "Trading volume analysis showing buying and selling pressure. Helps identify strong price levels and potential reversals.",
                                target="volume-chart-info",
                                placement="top"
                            ),
                            dcc.Graph(
                                id="auto-volume-chart",
                                style={"height": "150px"}
                            )
                        ])
                    ], width=9),
                    
                    # Right side - Trading Controls with Enhanced Tooltips
                    dbc.Col([
                        # Bot Controls with Tooltip
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Bot Controls", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="bot-controls-info"),
                                dbc.Tooltip(
                                    "Main trading bot controls. Start/Stop the bot or use Emergency Stop to immediately close all positions and cancel orders. Use with caution.",
                                    target="bot-controls-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Button("Start Bot", id="start-bot", color="success", className="w-100 mb-2"),
                                dbc.Button("Stop Bot", id="stop-bot", color="danger", className="w-100 mb-2"),
                                dbc.Button("Emergency Stop", id="emergency-stop", color="warning", className="w-100")
                            ])
                        ], className="mb-3"),
                        
                        # Quick Actions with Tooltip
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Quick Actions", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="quick-actions-info"),
                                dbc.Tooltip(
                                    "Emergency actions to quickly manage positions and orders. Use with caution as these actions are immediate and cannot be undone.",
                                    target="quick-actions-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Button("Close All Positions", id="close-all", color="danger", className="w-100 mb-2"),
                                dbc.Button("Cancel All Orders", id="cancel-all", color="warning", className="w-100")
                            ])
                        ])
                    ], width=3)
                ])
            ], label="Chart & Trading", tab_id="chart-tab"),
            
            # 2. Bot Indicators and Overlays Tab
            dbc.Tab([
                dbc.Row([
                    # Left side - Indicator Controls
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Indicator Settings", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="indicator-info"),
                                dbc.Tooltip(
                                    "Configure technical indicators and overlays for the trading chart",
                                    target="indicator-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                # Volume Profile Settings with tooltip
                                dbc.Label("Volume Profile", className="mt-2"),
                                dbc.Tooltip(
                                    "Shows price levels with highest trading volume. Use this to identify key support/resistance levels.",
                                    target="volume-profile-switch"
                                ),
                                dbc.Switch(id="volume-profile-switch", label="Show", value=True),
                                
                                # Support/Resistance Settings with tooltip
                                dbc.Label("Support/Resistance", className="mt-3"),
                                dbc.Tooltip(
                                    "Automatically detected support and resistance levels based on price action and volume.",
                                    target="sr-switch"
                                ),
                                dbc.Switch(id="sr-switch", label="Show", value=True),
                                
                                # Trade Markers Settings with tooltip
                                dbc.Label("Trade Markers", className="mt-3"),
                                dbc.Tooltip(
                                    "Visual markers showing entry and exit points of trades on the chart.",
                                    target="trade-markers-switch"
                                ),
                                dbc.Switch(id="trade-markers-switch", label="Show", value=True),
                                
                                html.Hr(),
                                
                                # Signal Indicators with tooltips
                                dbc.Label("Signal Indicators"),
                                dbc.Tooltip(
                                    "Select technical indicators to display on the chart",
                                    target="signal-indicators"
                                ),
                                dbc.Checklist(
                                    id="signal-indicators",
                                    options=[
                                        {"label": "EMA Cross", "value": "ema"},
                                        {"label": "RSI", "value": "rsi"},
                                        {"label": "MACD", "value": "macd"},
                                        {"label": "Bollinger Bands", "value": "bb"}
                                    ],
                                    value=["ema", "rsi"],
                                    className="mt-2"
                                )
                            ])
                        ])
                    ], width=3),
                    
                    # Right side - Indicator Charts
                    dbc.Col([
                        # Main Chart with Overlays and tooltip
                        dbc.Card([
                            dbc.CardHeader([
                                "Price Chart with Overlays",
                                html.I(className="fas fa-info-circle ms-2", id="overlay-chart-info"),
                                dbc.Tooltip(
                                    "Main price chart with selected indicators and overlays. Use the controls on the left to customize the display.",
                                    target="overlay-chart-info"
                                )
                            ]),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="overlay-chart",
                                    style={"height": "400px"}
                                )
                            ])
                        ], className="mb-3"),
                        
                        # Signal Dashboard with tooltips
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Signal Dashboard", className="mb-0 d-flex align-items-center"),
                                html.I(className="fas fa-info-circle ms-2", id="signal-dashboard-info"),
                                dbc.Tooltip(
                                    "Real-time trading signals and indicator analysis across multiple timeframes",
                                    target="signal-dashboard-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            dcc.Graph(id="signal-chart-1", style={"height": "200px"}),
                                            dbc.Tooltip(
                                                "Signal strength indicator showing market momentum",
                                                target="signal-chart-1"
                                            )
                                        ])
                                    ], width=12),
                                    dbc.Col([
                                        html.Div([
                                            dcc.Graph(id="signal-chart-2", style={"height": "200px"}),
                                            dbc.Tooltip(
                                                "Signal confidence level and prediction accuracy",
                                                target="signal-chart-2"
                                            )
                                        ])
                                    ], width=12)
                                ])
                            ])
                        ], className="mb-3")
                    ], width=9)
                ])
            ], label="Indicators & Overlays", tab_id="indicators-tab"),
            
            # 3. Stats Cards Tab
            dbc.Tab([
                dbc.Row([
                    # Performance Card with enhanced tooltips
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Performance Overview", className="mb-0"),
                                html.I(className="fas fa-info-circle ms-2", id="performance-info"),
                                dbc.Tooltip(
                                    "Key performance metrics and returns analysis",
                                    target="performance-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Total Return"),
                                        html.H4(id="total-return", className="text-success"),
                                        dbc.Tooltip(
                                            "Total return since bot started trading",
                                            target="total-return"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.H6("Monthly Return"),
                                        html.H4(id="monthly-return", className="text-info"),
                                        dbc.Tooltip(
                                            "Return for the current month",
                                            target="monthly-return"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.H6("Daily Return"),
                                        html.H4(id="daily-return", className="text-primary"),
                                        dbc.Tooltip(
                                            "Return for the current day",
                                            target="daily-return"
                                        )
                                    ], width=4)
                                ], className="mb-3"),
                                html.Div([
                                    dcc.Graph(
                                        id="mini-equity-curve",
                                        style={"height": "150px"}
                                    ),
                                    dbc.Tooltip(
                                        "Equity curve showing account value over time",
                                        target="mini-equity-curve"
                                    )
                                ])
                            ])
                        ])
                    ], width=6),
                    
                    # Advanced Stats Card with enhanced tooltips
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Advanced Statistics", className="mb-0"),
                                html.I(className="fas fa-info-circle ms-2", id="auto-advanced-stats-info"),
                                dbc.Tooltip(
                                    "Advanced trading statistics and risk metrics",
                                    target="auto-advanced-stats-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Sortino Ratio"),
                                        html.H4(id="auto-sortino-ratio", className="text-info"),
                                        dbc.Tooltip(
                                            "Risk-adjusted return metric focusing on downside volatility",
                                            target="auto-sortino-ratio"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.H6("Calmar Ratio"),
                                        html.H4(id="auto-calmar-ratio", className="text-warning"),
                                        dbc.Tooltip(
                                            "Ratio of average annual return to maximum drawdown risk",
                                            target="auto-calmar-ratio"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.H6("Alpha"),
                                        html.H4(id="alpha-value", className="text-success"),
                                        dbc.Tooltip(
                                            "Excess return compared to market benchmark",
                                            target="alpha-value"
                                        )
                                    ], width=4)
                                ])
                            ])
                        ])
                    ], width=6)
                ], className="mb-3"),
                
                # Recent Trades Card with enhanced tooltips
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Recent Trades Overview", className="mb-0"),
                                html.I(className="fas fa-info-circle ms-2", id="recent-trades-info"),
                                dbc.Tooltip(
                                    "Latest trading activity and performance analysis",
                                    target="recent-trades-info",
                                    placement="top"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Last Trade"),
                                        html.Div(id="last-trade-info", className="text-info"),
                                        dbc.Tooltip(
                                            "Details of the most recent trade",
                                            target="last-trade-info"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.H6("Best Trade"),
                                        html.Div(id="best-trade-info", className="text-success"),
                                        dbc.Tooltip(
                                            "Best performing trade in the current session",
                                            target="best-trade-info"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.H6("Worst Trade"),
                                        html.Div(id="worst-trade-info", className="text-danger"),
                                        dbc.Tooltip(
                                            "Worst performing trade in the current session",
                                            target="worst-trade-info"
                                        )
                                    ], width=4)
                                ], className="mb-3"),
                                html.Div([
                                    html.Div(
                                        id="recent-trades-list",
                                        style={"height": "200px", "overflow": "auto"}
                                    ),
                                    dbc.Tooltip(
                                        "Scrollable list of recent trades with details",
                                        target="recent-trades-list"
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            ], label="Stats Cards", tab_id="stats-tab"),
            
            # 4. Advanced Trading Tab (New)
            dbc.Tab([
                dbc.Row([
                    # Left Column - Order Book and Position Heat Map
                    dbc.Col([
                        # Order Book Panel
                        dbc.Card([
                            dbc.CardHeader([
                                "Order Book",
                                html.I(className="fas fa-info-circle ms-2", id="order-book-info"),
                                dbc.Tooltip(
                                    "Real-time order book showing buy/sell pressure and liquidity levels",
                                    target="order-book-info"
                                )
                            ]),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="order-book-graph",
                                    style={"height": "300px"}
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("Bid Volume", className="text-muted small"),
                                        html.Div(id="total-bid-volume", className="h6")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div("Ask Volume", className="text-muted small"),
                                        html.Div(id="total-ask-volume", className="h6")
                                    ], width=6)
                                ])
                            ])
                        ], className="mb-3"),
                        
                        # Position Heat Map
                        dbc.Card([
                            dbc.CardHeader([
                                "Position Heat Map",
                                html.I(className="fas fa-info-circle ms-2", id="heat-map-info"),
                                dbc.Tooltip(
                                    "Visual representation of position concentration and risk exposure across price levels",
                                    target="heat-map-info"
                                )
                            ]),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="position-heat-map",
                                    style={"height": "250px"}
                                )
                            ])
                        ])
                    ], width=6),
                    
                    # Right Column - Take Profit and Risk Controls
                    dbc.Col([
                        # Multiple Take-Profit Interface
                        dbc.Card([
                            dbc.CardHeader([
                                "Multiple Take-Profit Strategy",
                                html.I(className="fas fa-info-circle ms-2", id="tp-strategy-info"),
                                dbc.Tooltip(
                                    "Configure multiple take-profit levels with different position sizes",
                                    target="tp-strategy-info"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("TP Level 1 (%)"),
                                        dbc.Input(id="tp-level-1", type="number", value=1.0),
                                        html.Div("Size (%)"),
                                        dbc.Input(id="tp-size-1", type="number", value=30),
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("TP Level 2 (%)"),
                                        dbc.Input(id="tp-level-2", type="number", value=2.0),
                                        html.Div("Size (%)"),
                                        dbc.Input(id="tp-size-2", type="number", value=40),
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("TP Level 3 (%)"),
                                        dbc.Input(id="tp-level-3", type="number", value=3.0),
                                        html.Div("Size (%)"),
                                        dbc.Input(id="tp-size-3", type="number", value=30),
                                    ], width=4)
                                ], className="mb-3"),
                                dbc.Button("Apply TP Strategy", id="apply-tp-strategy", color="primary", className="w-100")
                            ])
                        ], className="mb-3"),
                        
                        # Advanced Risk Controls
                        dbc.Card([
                            dbc.CardHeader([
                                "Advanced Risk Controls",
                                html.I(className="fas fa-info-circle ms-2", id="risk-controls-info"),
                                dbc.Tooltip(
                                    "Configure advanced risk management parameters and safety controls",
                                    target="risk-controls-info"
                                )
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("Max Drawdown (%)"),
                                        dbc.Input(id="autobot-max-drawdown", type="number", value=5.0),
                                        dbc.Tooltip(
                                            "Maximum allowed drawdown before stopping the bot",
                                            target="autobot-max-drawdown"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Div("Daily Loss Limit (%)"),
                                        dbc.Input(id="autobot-daily-loss-limit", type="number", value=2.0),
                                        dbc.Tooltip(
                                            "Maximum allowed daily loss before stopping trading",
                                            target="autobot-daily-loss-limit"
                                        )
                                    ], width=6)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("Position Size Limit (%)"),
                                        dbc.Input(id="autobot-position-size-limit", type="number", value=10.0),
                                        dbc.Tooltip(
                                            "Maximum position size as percentage of portfolio",
                                            target="autobot-position-size-limit"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Div("Max Open Positions"),
                                        dbc.Input(id="advanced-max-open-positions", type="number", value=3),
                                        dbc.Tooltip(
                                            "Maximum number of simultaneous open positions",
                                            target="advanced-max-open-positions"
                                        )
                                    ], width=6)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Switch(
                                            id="advanced-trailing-stop-switch",
                                            label="Enable Trailing Stop",
                                            value=True
                                        ),
                                        dbc.Tooltip(
                                            "Enable dynamic trailing stop-loss that follows price movement",
                                            target="advanced-trailing-stop-switch"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Switch(
                                            id="advanced-auto-reduce-switch",
                                            label="Auto Position Reduce",
                                            value=True
                                        ),
                                        dbc.Tooltip(
                                            "Automatically reduce position size in high volatility",
                                            target="advanced-auto-reduce-switch"
                                        )
                                    ], width=6)
                                ])
                            ])
                        ])
                    ], width=6)
                ])
            ], label="Advanced Trading", tab_id="advanced-trading-tab")
        ])
    ], fluid=True) 