import dash_bootstrap_components as dbc
from dash import html, dcc

def create_settings_tab():
    """Create settings tab"""
    return dbc.Container(
        dbc.Row([
            # Left Column - Trading Settings
            dbc.Col([
                # Trading Parameters
                dbc.Card([
                    dbc.CardHeader(html.H6("Trading Parameters", className="mb-0")),
                    dbc.CardBody([
                        html.Label("Default Leverage"),
                        dbc.InputGroup([
                            dbc.Input(
                                type="number",
                                id="default-leverage",
                                min=1,
                                max=100,
                                step=1,
                                value=1
                            ),
                            dbc.InputGroupText("x")
                        ], className="mb-2"),
                        html.Label("Default Margin Type"),
                        dcc.Dropdown(
                            id="default-margin-type",
                            options=[
                                {"label": "Isolated", "value": "ISOLATED"},
                                {"label": "Cross", "value": "CROSS"}
                            ],
                            value="ISOLATED",
                            className="mb-2"
                        ),
                        html.Label("Default Order Type"),
                        dcc.Dropdown(
                            id="default-order-type",
                            options=[
                                {"label": "Market", "value": "MARKET"},
                                {"label": "Limit", "value": "LIMIT"},
                                {"label": "Stop Market", "value": "STOP_MARKET"},
                                {"label": "Stop Limit", "value": "STOP_LIMIT"}
                            ],
                            value="MARKET",
                            className="mb-2"
                        ),
                        dbc.Alert(id="settings-status", className="mt-2", is_open=False)
                    ])
                ], className="mb-3"),

                # Performance Optimization
                dbc.Card([
                    dbc.CardHeader(html.H6("Performance Optimization", className="mb-0")),
                    dbc.CardBody([
                        html.Label("Update Frequency"),
                        dcc.Dropdown(
                            id="update-frequency",
                            options=[
                                {"label": "High (1s)", "value": "1s"},
                                {"label": "Medium (5s)", "value": "5s"},
                                {"label": "Low (10s)", "value": "10s"}
                            ],
                            value="5s",
                            className="mb-2"
                        ),
                        html.Label("Chart Quality"),
                        dcc.Dropdown(
                            id="chart-quality",
                            options=[
                                {"label": "High", "value": "high"},
                                {"label": "Medium", "value": "medium"},
                                {"label": "Low", "value": "low"}
                            ],
                            value="medium",
                            className="mb-2"
                        ),
                        html.Label("Data Caching"),
                        dbc.Switch(
                            id="data-caching",
                            label="Enable Data Caching",
                            value=True,
                            className="mb-2"
                        )
                    ])
                ])
            ], width=4),

            # Middle Column - Risk Settings
            dbc.Col([
                # Risk Settings
                dbc.Card([
                    dbc.CardHeader(html.H6("Risk Settings", className="mb-0")),
                    dbc.CardBody([
                        html.Label("Max Position Size (%)"),
                        dbc.InputGroup([
                            dbc.Input(
                                type="number",
                                id="max-position-size",
                                min=1,
                                max=100,
                                step=1,
                                value=20
                            ),
                            dbc.InputGroupText("%")
                        ], className="mb-2"),
                        html.Label("Max Daily Loss (%)"),
                        dbc.InputGroup([
                            dbc.Input(
                                type="number",
                                id="max-daily-loss",
                                min=0.1,
                                max=100,
                                step=0.1,
                                value=2
                            ),
                            dbc.InputGroupText("%")
                        ], className="mb-2"),
                        html.Label("Max Drawdown (%)"),
                        dbc.InputGroup([
                            dbc.Input(
                                type="number",
                                id="max-drawdown",
                                min=0.1,
                                max=100,
                                step=0.1,
                                value=5
                            ),
                            dbc.InputGroupText("%")
                        ], className="mb-2")
                    ])
                ], className="mb-3"),

                # API Status
                dbc.Card([
                    dbc.CardHeader(html.H6("API Status", className="mb-0")),
                    dbc.CardBody(
                        dbc.Alert(
                            id="api-status",
                            className="text-center mb-0",
                            is_open=True
                        )
                    )
                ])
            ], width=4),

            # Right Column - System Settings
            dbc.Col([
                # API Configuration
                dbc.Card([
                    dbc.CardHeader(html.H6("API Configuration", className="mb-0")),
                    dbc.CardBody([
                        html.Label("API Key"),
                        dbc.Input(
                            type="password",
                            id="api-key",
                            placeholder="Enter API Key",
                            className="mb-2"
                        ),
                        html.Label("API Secret"),
                        dbc.Input(
                            type="password",
                            id="api-secret",
                            placeholder="Enter API Secret",
                            className="mb-2"
                        ),
                        dbc.Checklist(
                            id="api-options",
                            options=[
                                {"label": "Live Trading", "value": "live"},
                                {"label": "Paper Trading", "value": "paper"}
                            ],
                            value=["live"],
                            switch=True,
                            className="mb-2"
                        )
                    ])
                ], className="mb-3")
            ], width=4)
        ]),
        fluid=True
    )

