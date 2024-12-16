import dash_bootstrap_components as dbc
from dash import html, dcc

def create_indicator_tooltip(indicator_id, description):
    """Create a tooltip for an indicator"""
    return [
        html.I(className="fas fa-info-circle ms-2", id=f"{indicator_id}-info"),
        dbc.Tooltip(description, target=f"{indicator_id}-info", placement="right")
    ]

def create_main_analysis_subtab():
    """Create the main analysis subtab with technical analysis chart and controls"""
    return dbc.Container([
        dbc.Row([
            # Left Column - Analysis Controls
            dbc.Col([
                # Analysis Settings
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Analysis Settings", className="mb-0 d-flex align-items-center"),
                        *create_indicator_tooltip("analysis-settings", "Configure technical analysis parameters and indicators")
                    ]),
                    dbc.CardBody([
                        # Loading state
                        dcc.Loading(
                            id="analysis-loading",
                            type="circle",
                            children=[
                                # Timeframe Selection
                                html.Label("Timeframe"),
                                dcc.Dropdown(
                                    id="analysis-timeframe",
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
                                ),
                                # Indicator Selection with Tooltips
                                html.Label("Indicators"),
                                dcc.Checklist(
                                    id="analysis-indicators",
                                    options=[
                                        {"label": " RSI", "value": "rsi"},
                                        {"label": " MACD", "value": "macd"},
                                        {"label": " Bollinger Bands", "value": "bbands"},
                                        {"label": " Moving Averages", "value": "ma"},
                                        {"label": " Support/Resistance", "value": "sr"}
                                    ],
                                    value=["rsi", "macd"],
                                    className="mb-3",
                                    labelStyle={'display': 'block', 'margin': '5px 0'}
                                )
                            ]
                        )
                    ])
                ], className="mb-3"),
                # Technical Indicators Display
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Technical Indicators", className="mb-0"),
                        *create_indicator_tooltip("technical-indicators-info", "Real-time technical analysis indicators")
                    ]),
                    dbc.CardBody(id="technical-indicators")
                ])
            ], width=3),
            
            # Right Column - Charts
            dbc.Col([
                # Price Chart
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Price Analysis", className="mb-0"),
                        *create_indicator_tooltip("price-chart-info", "Price action and technical indicators")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="price-chart",
                                style={"height": "400px"},
                                config={'displayModeBar': True}
                            ),
                            type="circle"
                        )
                    ])
                ], className="mb-3"),
                # Volume Chart
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Volume Analysis", className="mb-0"),
                        *create_indicator_tooltip("volume-chart-info", "Trading volume analysis")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="volume-chart",
                                style={"height": "200px"},
                                config={'displayModeBar': True}
                            ),
                            type="circle"
                        )
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)

def create_volume_profile_subtab():
    """Create the volume profile analysis subtab"""
    return dbc.Container([
        dbc.Row([
            # Left Column - Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Volume Profile Settings"),
                    dbc.CardBody([
                        html.Label("Profile Period"),
                        dcc.Dropdown(
                            id="volume-profile-period",
                            options=[
                                {"label": "1 Day", "value": "1d"},
                                {"label": "1 Week", "value": "1w"},
                                {"label": "1 Month", "value": "1m"}
                            ],
                            value="1d",
                            className="mb-3"
                        ),
                        dbc.Button(
                            "Update Profile",
                            id="update-volume-profile",
                            color="primary",
                            className="w-100"
                        )
                    ])
                ])
            ], width=3),
            
            # Right Column - Volume Profile Chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Volume Profile Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id="volume-profile-main-chart", style={"height": "600px"})
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)

def create_market_regime_subtab():
    """Create the market regime detection subtab"""
    return dbc.Container([
        dbc.Row([
            # Left Column - Market Regime Analysis
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Market Regime Analysis", className="mb-0 d-flex align-items-center"),
                        *create_indicator_tooltip("market-regime", "Analysis of current market conditions and trends")
                    ]),
                    dbc.CardBody([
                        # Market Regime Chart
                        dcc.Loading(
                            dcc.Graph(
                                id="market-regime-chart",
                                style={"height": "300px"},
                                config={'displayModeBar': True}
                            ),
                            type="circle"
                        ),
                        html.Div([
                            # Current Regime Display
                            html.Div([
                                html.H6("Current Regime", className="mt-3"),
                                html.Div(id="market-regime", className="h4 text-info mb-3")
                            ], className="text-center"),
                            
                            # Regime Probabilities
                            html.Div([
                                html.H6("Regime Probabilities", className="mb-2"),
                                # Dynamic progress bars updated by callbacks
                                html.Div(id="regime-probabilities", children=[
                                    html.Div([
                                        html.Label("Bullish", className="d-flex justify-content-between"),
                                        dbc.Progress(id="bullish-prob", value=0, color="success", className="mb-2")
                                    ]),
                                    html.Div([
                                        html.Label("Neutral", className="d-flex justify-content-between"),
                                        dbc.Progress(id="neutral-prob", value=0, color="warning", className="mb-2")
                                    ]),
                                    html.Div([
                                        html.Label("Bearish", className="d-flex justify-content-between"),
                                        dbc.Progress(id="bearish-prob", value=0, color="danger", className="mb-2")
                                    ])
                                ])
                            ], className="mt-3")
                        ])
                    ])
                ], className="h-100")
            ], width=6),
            
            # Right Column - Market Structure
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Market Structure", className="mb-0 d-flex align-items-center"),
                        *create_indicator_tooltip("market-structure", "Detailed analysis of market structure and patterns")
                    ]),
                    dbc.CardBody([
                        # Market Structure Chart
                        dcc.Loading(
                            dcc.Graph(
                                id="market-structure-chart",
                                style={"height": "300px"},
                                config={'displayModeBar': True}
                            ),
                            type="circle"
                        ),
                        # Structure Analysis Results
                        html.Div(id="structure-analysis-results", className="mt-3")
                    ])
                ], className="h-100")
            ], width=6)
        ], className="mb-3"),
        
        # Bottom Row - Advanced Statistics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Advanced Statistics", className="mb-0 d-flex align-items-center"),
                        *create_indicator_tooltip("advanced-stats", "Detailed market statistics and risk metrics")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            # Volatility
                            dbc.Col([
                                html.Div([
                                    html.H6("Volatility", className="mb-2"),
                                    html.Div(id="volatility-value", className="h4"),
                                    *create_indicator_tooltip("volatility", "Market price volatility measure")
                                ], className="text-center")
                            ], width=3),
                            
                            # Trend Strength
                            dbc.Col([
                                html.Div([
                                    html.H6("Trend Strength", className="mb-2"),
                                    html.Div(id="trend-strength", className="h4"),
                                    *create_indicator_tooltip("trend-strength", "Current market trend strength")
                                ], className="text-center")
                            ], width=3),
                            
                            # Market Phase
                            dbc.Col([
                                html.Div([
                                    html.H6("Market Phase", className="mb-2"),
                                    html.Div(id="market-phase", className="h4"),
                                    *create_indicator_tooltip("market-phase", "Current market cycle phase")
                                ], className="text-center")
                            ], width=3),
                            
                            # Risk Level
                            dbc.Col([
                                html.Div([
                                    html.H6("Risk Level", className="mb-2"),
                                    html.Div(id="risk-level", className="h4"),
                                    *create_indicator_tooltip("risk-level", "Current market risk assessment")
                                ], className="text-center")
                            ], width=3)
                        ])
                    ])
                ])
            ])
        ])
    ], fluid=True)

def create_pattern_backtest_subtab():
    """Create the pattern backtesting subtab"""
    return dbc.Container([
        dbc.Row([
            # Left Column - Backtest Settings
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Backtest Settings", className="mb-0"),
                        *create_indicator_tooltip("backtest-settings", "Configure pattern backtesting parameters and conditions")
                    ]),
                    dbc.CardBody([
                        html.Label("Pattern Type"),
                        dcc.Dropdown(
                            id="backtest-pattern-type",
                            options=[
                                {"label": "Candlestick Patterns", "value": "candlestick"},
                                {"label": "Chart Patterns", "value": "chart"},
                                {"label": "Indicators", "value": "indicators"},
                                {"label": "Custom Patterns", "value": "custom"}
                            ],
                            value="candlestick",
                            className="mb-2"
                        ),
                        html.Label("Time Period"),
                        dcc.DatePickerRange(
                            id="backtest-date-range",
                            className="mb-2"
                        ),
                        html.Label("Risk Parameters"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Stop Loss %"),
                            dbc.Input(id="backtest-stop-loss", type="number", value=2),
                        ], className="mb-2"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Take Profit %"),
                            dbc.Input(id="backtest-take-profit", type="number", value=4),
                        ], className="mb-2"),
                        # Advanced Settings
                        html.Label("Advanced Parameters"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Position Size %"),
                            dbc.Input(id="backtest-position-size", type="number", value=1),
                            *create_indicator_tooltip("position-size", "Percentage of account to risk per trade")
                        ], className="mb-2"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Max Trades"),
                            dbc.Input(id="backtest-max-trades", type="number", value=10),
                            *create_indicator_tooltip("max-trades", "Maximum number of concurrent trades")
                        ], className="mb-2"),
                        # Market Conditions Filter
                        html.Label("Market Conditions"),
                        dcc.Checklist(
                            id="market-conditions",
                            options=[
                                {"label": "Trending", "value": "trend"},
                                {"label": "Ranging", "value": "range"},
                                {"label": "Volatile", "value": "volatile"}
                            ],
                            value=["trend"],
                            className="mb-2"
                        ),
                        *create_indicator_tooltip("market-conditions", "Filter backtest by specific market conditions")
                    ])
                ])
            ], width=3),
            
            # Right Column - Backtest Results
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Backtest Results", className="mb-0 d-flex align-items-center"),
                        *create_indicator_tooltip("backtest-results", "Detailed pattern backtesting results and statistics")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Win Rate"),
                                html.H4(id="pattern-win-rate", className="text-success"),
                                *create_indicator_tooltip("win-rate", "Percentage of profitable trades")
                            ], width=3),
                            dbc.Col([
                                html.H6("Avg Return"),
                                html.H4(id="pattern-avg-return", className="text-info"),
                                *create_indicator_tooltip("avg-return", "Average return per trade")
                            ], width=3),
                            dbc.Col([
                                html.H6("Total Trades"),
                                html.H4(id="pattern-total-trades"),
                                *create_indicator_tooltip("total-trades", "Number of trades in backtest period")
                            ], width=3),
                            dbc.Col([
                                html.H6("Profit Factor"),
                                html.H4(id="pattern-profit-factor", className="text-warning"),
                                *create_indicator_tooltip("profit-factor", "Ratio of gross profits to gross losses")
                            ], width=3)
                        ], className="mb-3"),
                        dcc.Graph(id="backtest-results-chart", style={"height": "400px"}),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Trade Distribution"),
                                dcc.Graph(id="trade-distribution-chart", style={"height": "200px"})
                            ], width=6),
                            dbc.Col([
                                html.H6("Monthly Performance"),
                                dcc.Graph(id="monthly-performance-chart", style={"height": "200px"})
                            ], width=6)
                        ]),
                        html.Div(id="backtest-statistics", className="mt-3"),
                        # Additional Statistics
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("Advanced Statistics", className="mb-0"),
                                *create_indicator_tooltip("analysis-advanced-stats-info", "Detailed statistical analysis of backtest results")
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Sharpe Ratio"),
                                        html.H4(id="pattern-sharpe-ratio", className="text-info"),
                                        *create_indicator_tooltip("sharpe-ratio", "Risk-adjusted return metric")
                                    ], width=3),
                                    dbc.Col([
                                        html.H6("Max Drawdown"),
                                        html.H4(id="pattern-max-drawdown", className="text-danger"),
                                        *create_indicator_tooltip("max-drawdown", "Largest peak-to-trough decline")
                                    ], width=3),
                                    dbc.Col([
                                        html.H6("Recovery Factor"),
                                        html.H4(id="pattern-recovery-factor", className="text-success"),
                                        *create_indicator_tooltip("recovery-factor", "Net profit relative to max drawdown")
                                    ], width=3),
                                    dbc.Col([
                                        html.H6("Expectancy"),
                                        html.H4(id="pattern-expectancy", className="text-warning"),
                                        *create_indicator_tooltip("expectancy", "Average profit/loss per trade")
                                    ], width=3)
                                ], className="mb-3")
                            ])
                        ], className="mb-3")
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)

def create_analysis_tab():
    """Create technical analysis tab with subtabs"""
    return dbc.Container([
        # Store components for data
        dcc.Store(id="analysis-data-store"),
        dcc.Store(id="backtest-results-store"),
        
        # Subtabs
        dbc.Tabs([
            dbc.Tab(
                create_main_analysis_subtab(),
                label="Technical Analysis",
                tab_id="tech-analysis-subtab"
            ),
            dbc.Tab(
                create_volume_profile_subtab(),
                label="Volume Profile",
                tab_id="volume-profile-subtab"
            ),
            dbc.Tab(
                create_market_regime_subtab(),
                label="Market Regime",
                tab_id="market-regime-subtab"
            ),
            dbc.Tab(
                create_pattern_backtest_subtab(),
                label="Pattern Backtesting",
                tab_id="pattern-backtest-subtab"
            )
        ], id="analysis-subtabs", active_tab="tech-analysis-subtab")
    ], fluid=True) 