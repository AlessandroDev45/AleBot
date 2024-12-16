import dash_bootstrap_components as dbc
from dash import html, dcc

# Import tab components
from .tabs.auto_bot import create_auto_bot_tab
from .tabs.manual_trading import create_manual_trading_tab
from .tabs.analysis import create_analysis_tab
from .tabs.ml_model import create_ml_tab
from .tabs.settings import create_settings_tab

def create_header():
    """Create header with account info and performance bars"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("AleBot Trading Dashboard", className="text-primary d-inline"),
                    html.Div([
                        html.H3(id="btc-price", className="text-muted d-inline ms-3"),
                        html.H5(id="btc-change", className="d-inline ms-2")
                    ], className="d-inline")
                ]),
                html.P("Real-time trading and analysis platform", className="text-muted")
            ], width=8),
            dbc.Col([
                html.Div([
                    html.H6("Account Balance", className="text-muted"),
                    html.H4(id="header-balance", children="0.00 USDT")
                ]),
                html.Div([
                    html.H6("Daily PnL", className="text-muted"),
                    html.H4(id="header-pnl", children="0.00%")
                ]),
                html.Div([
                    html.H6("Trades Today", className="text-muted"),
                    html.H4(id="header-trades", children="0")
                ])
            ], width=4)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="cpu-usage", value=0, label="CPU Usage", className="mb-2"),
                dbc.Progress(id="memory-usage", value=0, label="Memory Usage", className="mb-2"),
                dbc.Progress(id="disk-usage", value=0, label="Disk Usage", className="mb-2")
            ])
        ])
    ], fluid=True, className="header-container py-3")

def create_tabs():
    """Create main navigation tabs"""
    return dbc.Tabs([
        dbc.Tab(
            create_auto_bot_tab(),
            label="Auto Bot",
            tab_id="auto-bot-tab"
        ),
        dbc.Tab(
            create_manual_trading_tab(),
            label="Manual Trading",
            tab_id="manual-trading-tab"
        ),
        dbc.Tab(
            create_analysis_tab(),
            label="Analysis",
            tab_id="analysis-tab"
        ),
        dbc.Tab(
            create_ml_tab(),
            label="ML Model",
            tab_id="ml-model-tab"
        ),
        dbc.Tab(
            create_settings_tab(),
            label="Settings",
            tab_id="settings-tab"
        )
    ], id="tabs", active_tab="auto-bot-tab")

def create_layout():
    """Create main dashboard layout"""
    return dbc.Container([
        create_header(),
        html.Hr(),
        create_tabs(),
        # Intervals for updates
        dcc.Interval(id="interval-fast", interval=5000),  # 5 seconds
        dcc.Interval(id="interval-medium", interval=5000),  # 5 seconds
        dcc.Interval(id="interval-slow", interval=60000)  # 1 minute
    ], fluid=True) 