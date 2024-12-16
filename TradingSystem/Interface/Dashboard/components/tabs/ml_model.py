import dash_bootstrap_components as dbc
from dash import html, dcc
import torch
import os
import dash
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

def create_ml_model_main_tab():
    """Create main ML model tab content"""
    return dbc.Card([
        dbc.CardHeader([
            html.H6("Model Performance & Predictions", className="mb-0 d-flex align-items-center"),
            dbc.Tooltip("Main model performance metrics and predictions", target="model-perf-info")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Training Progress", className="text-muted"),
                        dbc.Progress(
                            id="training-progress-main",
                            value=0,
                            label="0%",
                            className="mb-2",
                            style={"height": "25px"}
                        ),
                        dbc.Button(
                            "Cancel Training",
                            id="cancel-training",
                            color="danger",
                            className="mb-3",
                            disabled=True
                        ),
                        html.Div(id="training-status", children="Not training", className="text-muted"),
                        html.Div(id="training-error", children="", className="text-danger"),
                        html.Div(id="training-duration", children="Duration: N/A", className="text-muted"),
                        html.Div(id="training-eta", children="ETA: N/A", className="text-muted"),
                        html.Div(id="active-features", children="", className="text-muted"),
                    ], className="mb-4"),
                ], width=12)
            ]),
            dbc.Row([
                # Model Performance Metrics
                dbc.Col([
                    html.Div([
                        html.H6("Model Performance", className="text-muted"),
                        dbc.Row([
                            dbc.Col([
                                html.H4(id="accuracy-metric", children="0%", className="text-success"), 
                                html.P("Accuracy", className="text-muted"),
                                dbc.Tooltip("Percentage of correct predictions", target="accuracy-metric")
                            ], width=3),
                            dbc.Col([
                                html.H4(id="precision-metric", children="0%", className="text-info"), 
                                html.P("Precision", className="text-muted"),
                                dbc.Tooltip("Ratio of true positives to predicted positives", target="precision-metric")
                            ], width=3),
                            dbc.Col([
                                html.H4(id="recall-metric", children="0%", className="text-warning"), 
                                html.P("Recall", className="text-muted"),
                                dbc.Tooltip("Ratio of true positives to actual positives", target="recall-metric")
                            ], width=3),
                            dbc.Col([
                                html.H4(id="f1-metric", children="0%", className="text-primary"), 
                                html.P("F1 Score", className="text-muted"),
                                dbc.Tooltip("Harmonic mean of precision and recall", target="f1-metric")
                            ], width=3)
                        ])
                    ], className="mb-4"),
                    dcc.Graph(
                        id="performance-chart",
                        style={"height": "300px"},
                        config={'displayModeBar': True, 'scrollZoom': True}
                    ),
                    dbc.Tooltip(
                        "Historical performance of the model over time",
                        target="performance-chart"
                    ),
                    # Backtesting Results
                    html.Div([
                        html.H6("Backtesting Results", className="mt-3 text-muted"),
                        dbc.Row([
                            dbc.Col([
                                html.H4(id="backtest-profit", children="0%", className="text-success"),
                                html.P("Total Return", className="text-muted"),
                                dbc.Tooltip("Total return from backtesting period", target="backtest-profit")
                            ], width=4),
                            dbc.Col([
                                html.H4(id="backtest-trades", children="0", className="text-info"),
                                html.P("Total Trades", className="text-muted"),
                                dbc.Tooltip("Number of trades during backtesting", target="backtest-trades")
                            ], width=4),
                            dbc.Col([
                                html.H4(id="backtest-ratio", children="0", className="text-warning"),
                                html.P("Profit Ratio", className="text-muted"),
                                dbc.Tooltip("Ratio of winning to losing trades", target="backtest-ratio")
                            ], width=4)
                        ], className="text-center")
                    ], className="mt-3"),
                ], width=8),
                # Prediction Signals
                dbc.Col([
                    html.Div([
                        html.H6("Current Predictions", className="text-muted mb-3"),
                        html.Div([
                            html.H4(
                                id="prediction-signal",
                                children="NEUTRAL",
                                className="text-center",
                                style={"fontSize": "24px", "fontWeight": "bold"}
                            ),
                            html.P(
                                id="prediction-confidence",
                                children="Confidence: 0%",
                                className="text-center text-muted"
                            ),
                            dbc.Tooltip("Current market prediction and confidence level", target="prediction-signal")
                        ], className="text-center mb-3"),
                        dcc.Graph(
                            id="prediction-gauge",
                            style={"height": "200px"},
                            config={'displayModeBar': False}
                        ),
                        dbc.Tooltip(
                            "Visual representation of prediction confidence",
                            target="prediction-gauge"
                        ),
                        # Real-time Prediction Metrics
                        html.Div([
                            html.H6("Real-time Metrics", className="mt-3 text-muted"),
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    "Signal Strength",
                                    dbc.Badge(
                                        id="signal-strength",
                                        children="0%",
                                        className="ms-auto",
                                        color="success"
                                    )
                                ], className="d-flex justify-content-between align-items-center"),
                                dbc.ListGroupItem([
                                    "Time Horizon",
                                    dbc.Badge(
                                        id="time-horizon",
                                        children="Short",
                                        className="ms-auto",
                                        color="info"
                                    )
                                ], className="d-flex justify-content-between align-items-center"),
                                dbc.ListGroupItem([
                                    "Risk Level",
                                    dbc.Badge(
                                        id="risk-level",
                                        children="Low",
                                        className="ms-auto",
                                        color="warning"
                                    )
                                ], className="d-flex justify-content-between align-items-center")
                            ], className="mt-2")
                        ])
                    ])
                ], width=4)
            ])
        ])
    ])

def create_feature_engineering_tab():
    """Create feature engineering subtab"""
    return dbc.Card([
        dbc.CardHeader("Feature Engineering Dashboard"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # Feature Correlation Matrix
                    dbc.Card([
                        dbc.CardHeader("Feature Correlation Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="correlation-matrix", style={"height": "300px"}),
                            dbc.Tooltip(
                                "Correlation between different features used in the model",
                                target="correlation-matrix"
                            )
                        ])
                    ], className="mb-3"),
                    # Feature Importance
                    dbc.Card([
                        dbc.CardHeader("Feature Importance"),
                        dbc.CardBody([
                            dcc.Graph(id="feature-importance-plot", style={"height": "300px"}),
                            dbc.Tooltip(
                                "Relative importance of each feature in the model",
                                target="feature-importance-plot"
                            )
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    # Feature Controls
                    dbc.Card([
                        dbc.CardHeader("Feature Controls"),
                        dbc.CardBody([
                            html.Label("Feature Selection Method"),
                            dcc.Dropdown(
                                id="feature-selection-method",
                                options=[
                                    {"label": "Recursive Feature Elimination", "value": "rfe"},
                                    {"label": "LASSO", "value": "lasso"},
                                    {"label": "Random Forest", "value": "rf"}
                                ],
                                value="rf",
                                className="mb-3"
                            ),
                            html.Label("Feature Categories"),
                            dcc.Checklist(
                                id="feature-categories",
                                options=[
                                    {"label": "Technical Indicators", "value": "technical"},
                                    {"label": "Price Action", "value": "price"},
                                    {"label": "Volume", "value": "volume"},
                                    {"label": "Market Sentiment", "value": "sentiment"}
                                ],
                                value=["technical", "price", "volume"],
                                className="mb-3"
                            ),
                            dbc.Button("Update Features", id="update-features", color="primary", className="w-100"),
                            html.Div([
                                html.H6("Feature Stats", className="mt-3"),
                                html.P(id="feature-stats", className="text-muted")
                            ])
                        ])
                    ])
                ], width=4)
            ])
        ])
    ])

def create_model_metrics_tab():
    """Create real-time model metrics subtab"""
    return dbc.Card([
        dbc.CardHeader("Real-time Model Metrics"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # Performance Metrics Over Time
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="metrics-timeline", style={"height": "300px"}),
                            dbc.Tooltip(
                                "Real-time tracking of model performance metrics",
                                target="metrics-timeline"
                            )
                        ])
                    ], className="mb-3"),
                    # Confusion Matrix
                    dbc.Card([
                        dbc.CardHeader("Confusion Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="confusion-matrix", style={"height": "300px"}),
                            dbc.Tooltip(
                                "Visual representation of model prediction accuracy",
                                target="confusion-matrix"
                            )
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    # Metric Details
                    dbc.Card([
                        dbc.CardHeader("Detailed Metrics"),
                        dbc.CardBody([
                            html.Div([
                                html.H6("ROC AUC Score"),
                                html.H4(id="roc-auc-score", children="0.00"),
                                dbc.Tooltip(
                                    "Area under the ROC curve - measures model's ability to distinguish between classes",
                                    target="roc-auc-score"
                                )
                            ], className="mb-3"),
                            html.Div([
                                html.H6("Sharpe Ratio"),
                                html.H4(id="sharpe-ratio", children="0.00"),
                                dbc.Tooltip(
                                    "Risk-adjusted return metric",
                                    target="sharpe-ratio"
                                )
                            ], className="mb-3"),
                            html.Div([
                                html.H6("Max Drawdown"),
                                html.H4(id="ml-max-drawdown", children="0.00%"),
                                dbc.Tooltip(
                                    "Maximum observed loss from a peak to a trough",
                                    target="ml-max-drawdown"
                                )
                            ], className="mb-3"),
                            html.Div([
                                html.H6("Win Rate"),
                                html.H4(id="win-rate", children="0.00%"),
                                dbc.Tooltip(
                                    "Percentage of successful predictions",
                                    target="win-rate"
                                )
                            ])
                        ])
                    ]),
                    # New: Advanced Metrics Card
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("Advanced Metrics", className="mb-0"),
                            html.I(className="fas fa-info-circle ms-2", id="ml-advanced-stats-info"),
                            dbc.Tooltip(
                                "Advanced model performance metrics and risk analysis",
                                target="ml-advanced-stats-info",
                                placement="top"
                            )
                        ]),
                        dbc.CardBody([
                            html.Div([
                                html.H6("Sortino Ratio"),
                                html.H4(id="ml-sortino-ratio", children="0.00"),
                                dbc.Tooltip(
                                    "Risk-adjusted return focusing on downside volatility",
                                    target="ml-sortino-ratio"
                                )
                            ], className="mb-3"),
                            html.Div([
                                html.H6("Calmar Ratio"),
                                html.H4(id="ml-calmar-ratio", children="0.00"),
                                dbc.Tooltip(
                                    "Ratio of average annual return to maximum drawdown",
                                    target="ml-calmar-ratio"
                                )
                            ], className="mb-3"),
                            html.Div([
                                html.H6("Information Ratio"),
                                html.H4(id="ml-info-ratio", children="0.00"),
                                dbc.Tooltip(
                                    "Risk-adjusted excess returns relative to benchmark",
                                    target="ml-info-ratio"
                                )
                            ])
                        ])
                    ], className="mt-3")
                ], width=4)
            ])
        ])
    ])

def create_input_with_validation(id, label, input_type, min_val=None, max_val=None, step=None, value=None, required=False, description=None):
    """Cria um input com validação e feedback visual."""
    input_div = html.Div([
        html.Label([
            label,
            html.Span("*", className="text-danger ms-1") if required else "",
        ]),
        dbc.Input(
            id=id,
            type=input_type,
            min=min_val,
            max=max_val,
            step=step,
            value=value,
            className="mb-1"
        ),
        # Mensagem de erro
        html.Div(
            id=f"{id}-error",
            className="text-danger small",
            style={"minHeight": "20px"}
        ),
        # Tooltip com descrição
        dbc.Tooltip(
            description,
            target=id,
            placement="right"
        )
    ], className="mb-3")
    
    return input_div

def create_model_settings_tab():
    """Create model settings subtab"""
    return dbc.Card([
        dbc.CardHeader("Model Settings"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # Model Configuration
                    dbc.Card([
                        dbc.CardHeader("Model Configuration"),
                        dbc.CardBody([
                            html.Label("Model Type"),
                            dcc.Dropdown(
                                id="model-type",
                                options=[
                                    {"label": "Deep Learning", "value": "deep_learning"},
                                    {"label": "LSTM", "value": "lstm"},
                                    {"label": "Transformer", "value": "transformer"}
                                ],
                                value="deep_learning",
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="model-type"
                            ),
                            html.Label("Learning Rate"),
                            dbc.Input(
                                id="learning-rate",
                                type="number",
                                value=0.001,
                                min=0.0001,
                                max=0.1,
                                step=0.0001,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="learning-rate"
                            ),
                            html.Label("Batch Size"),
                            dbc.Input(
                                id="batch-size",
                                type="number",
                                value=32,
                                min=1,
                                max=512,
                                step=1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="batch-size"
                            ),
                            html.Label("Dropout Rate"),
                            dbc.Input(
                                id="dropout-rate",
                                type="number",
                                value=0.5,
                                min=0,
                                max=1,
                                step=0.1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="dropout-rate"
                            ),
                            html.Label("Early Stopping Patience"),
                            dbc.Input(
                                id="early-stopping-patience",
                                type="number",
                                value=20,
                                min=1,
                                max=100,
                                step=1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="early-stopping-patience"
                            ),
                            html.Label("Sequence Length"),
                            dbc.Input(
                                id="sequence-length",
                                type="number",
                                value=30,
                                min=1,
                                max=500,
                                step=1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="sequence-length"
                            ),
                            html.Label("Stride"),
                            dbc.Input(
                                id="stride",
                                type="number",
                                value=5,
                                min=1,
                                max=100,
                                step=1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="stride"
                            ),
                            html.Label("Training Time Range"),
                            dcc.Dropdown(
                                id="training-time-range",
                                options=[
                                    {"label": "1 Hour", "value": "1h"},
                                    {"label": "4 Hours", "value": "4h"},
                                    {"label": "1 Day", "value": "1d"}
                                ],
                                value="1h",
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="training-time-range"
                            ),
                            html.Label("Training/Test Split (%)"),
                            dbc.Input(
                                id="data-split",
                                type="number",
                                value=80,
                                min=50,
                                max=95,
                                step=5,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="data-split"
                            ),
                            dbc.Button(
                                "Train Model",
                                id="train-model",
                                color="primary",
                                className="w-100 mt-3",
                                disabled=True
                            )
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    # Trading Configuration
                    dbc.Card([
                        dbc.CardHeader("Trading Configuration"),
                        dbc.CardBody([
                            html.Label("Confidence Threshold"),
                            dbc.Input(
                                id="confidence-threshold",
                                type="number",
                                value=0.8,
                                min=0.5,
                                max=1,
                                step=0.05,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="confidence-threshold"
                            ),
                            html.Label("Position Sizing (%)"),
                            dbc.Input(
                                id="position-sizing",
                                type="number",
                                value=10,
                                min=1,
                                max=100,
                                step=1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="position-sizing"
                            ),
                            html.Label("Stop Loss (%)"),
                            dbc.Input(
                                id="stop-loss",
                                type="number",
                                value=2,
                                min=0.1,
                                max=10,
                                step=0.1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="stop-loss"
                            ),
                            html.Label("Take Profit (%)"),
                            dbc.Input(
                                id="take-profit",
                                type="number",
                                value=4,
                                min=0.1,
                                max=20,
                                step=0.1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="take-profit"
                            ),
                            html.Label("Max Positions"),
                            dbc.Input(
                                id="max-positions",
                                type="number",
                                value=3,
                                min=1,
                                max=10,
                                step=1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="max-positions"
                            ),
                            html.Label("Use Technical Analysis"),
                            dcc.Checklist(
                                id="use-ta",
                                options=[{"label": "Enable", "value": "true"}],
                                value=["true"],
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="use-ta"
                            ),
                            html.Label("Future Bars to Predict"),
                            dbc.Input(
                                id="future-bars",
                                type="number",
                                value=3,
                                min=1,
                                max=24,
                                step=1,
                                className="mb-3"
                            ),
                            dbc.Tooltip(
                                "Settings are saved automatically when changed",
                                target="future-bars"
                            )
                        ])
                    ])
                ], width=6)
            ])
        ])
    ])

def create_ml_tab():
    """Create machine learning tab with subtabs"""
    return dbc.Container([
        dbc.Tabs([
            dbc.Tab(
                create_ml_model_main_tab(),
                label="Overview",
                tab_id="ml-overview-tab"
            ),
            dbc.Tab(
                create_feature_engineering_tab(),
                label="Feature Engineering",
                tab_id="feature-engineering-tab"
            ),
            dbc.Tab(
                create_model_metrics_tab(),
                label="Model Metrics",
                tab_id="model-metrics-tab"
            ),
            dbc.Tab(
                create_model_settings_tab(),
                label="Settings",
                tab_id="model-settings-tab"
            )
        ], id="ml-subtabs", active_tab="ml-overview-tab")
    ], fluid=True)

@dash.callback(
    [
        Output({"type": "error-message", "index": ALL}, "children"),
        Output("validation-status", "is_open"),
        Output("validation-status", "children"),
        Output("apply-settings", "disabled"),
        Output("train-model", "disabled"),
        Output("missing-params", "children")
    ],
    [
        Input({"type": "model-input", "index": ALL}, "value"),
        Input({"type": "model-input", "index": ALL}, "id")
    ]
)
def validate_inputs(values, ids):
    """Valida todos os inputs em tempo real"""
    if not values or not ids:
        raise PreventUpdate
    
    error_messages = []
    missing_params = []
    all_valid = True
    
    validation_rules = {
        "model-type": {
            "valid_values": ["ensemble", "lstm", "transformer", "cnn"],
            "error_msg": "Tipo de modelo inválido"
        },
        "hidden-size": {
            "min": 64,
            "max": 2048,
            "error_msg": "Valor deve estar entre 64 e 2048"
        },
        "num-layers": {
            "min": 1,
            "max": 8,
            "error_msg": "Valor deve estar entre 1 e 8"
        },
        "dropout-rate": {
            "min": 0.0,
            "max": 0.9,
            "error_msg": "Valor deve estar entre 0.0 e 0.9"
        },
        "learning-rate": {
            "min": 0.000001,
            "max": 0.1,
            "error_msg": "Valor deve estar entre 0.000001 e 0.1"
        },
        "batch-size": {
            "min": 16,
            "max": 512,
            "error_msg": "Valor deve estar entre 16 e 512"
        },
        "early-stopping-patience": {
            "min": 5,
            "max": 50,
            "error_msg": "Valor deve estar entre 5 e 50"
        },
        "sequence-length": {
            "min": 10,
            "max": 100,
            "error_msg": "Valor deve estar entre 10 e 100"
        },
        "stride": {
            "min": 1,
            "max": 20,
            "error_msg": "Valor deve estar entre 1 e 20"
        }
    }
    
    for value, id_dict in zip(values, ids):
        input_id = id_dict["index"]
        error_msg = ""
        
        # Validação básica para campos obrigatórios
        if value is None or value == "":
            error_msg = "Este campo é obrigatório"
            missing_params.append(input_id)
            all_valid = False
        else:
            # Validação específica para cada campo
            if input_id in validation_rules:
                rules = validation_rules[input_id]
                
                if "valid_values" in rules:
                    if value not in rules["valid_values"]:
                        error_msg = rules["error_msg"]
                        all_valid = False
                elif "min" in rules and "max" in rules:
                    try:
                        val = float(value)
                        if not (rules["min"] <= val <= rules["max"]):
                            error_msg = rules["error_msg"]
                            all_valid = False
                    except (ValueError, TypeError):
                        error_msg = "Valor inválido"
                        all_valid = False
        
        error_messages.append(error_msg)
    
    # Status geral de validação
    validation_status = not all_valid
    validation_message = None
    if not all_valid:
        validation_message = [
            html.H6("Erros de Validação:", className="mb-2"),
            html.Ul([
                html.Li(f"Campo '{param.replace('-', ' ').title()}': {validation_rules[param]['error_msg']}")
                for param in missing_params if param in validation_rules
            ])
        ]
    
    # Lista de parâmetros faltantes
    missing_params_message = None
    if missing_params:
        missing_params_message = [
            html.Strong("Parâmetros Faltantes:"),
            html.Ul([
                html.Li(param.replace("-", " ").title()) 
                for param in missing_params
            ])
        ]
    
    return (
        error_messages,
        validation_status,
        validation_message,
        not all_valid,  # Botão Apply desabilitado se houver erros
        not all_valid,  # Botão Train desabilitado se houver erros
        missing_params_message
    )

@dash.callback(
    [
        Output("apply-settings", "disabled"),
        Output("train-model", "disabled"),
        Output("model-status", "children"),
        Output("validation-status", "is_open"),
        Output("validation-status", "children")
    ],
    [
        Input("model-type", "value"),
        Input("hidden-size", "value"),
        Input("num-layers", "value"),
        Input("dropout-rate", "value"),
        Input("learning-rate", "value"),
        Input("batch-size", "value"),
        Input("early-stopping-patience", "value"),
        Input("sequence-length", "value"),
        Input("stride", "value")
    ]
)
def validate_and_update_buttons(*values):
    """Valida os inputs e atualiza o estado dos botões"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    all_valid = True
    error_messages = []
    
    # Validação dos campos
    validation_rules = {
        "model-type": {"valid_values": ["ensemble", "lstm", "transformer", "cnn"]},
        "hidden-size": {"min": 64, "max": 2048},
        "num-layers": {"min": 1, "max": 8},
        "dropout-rate": {"min": 0.0, "max": 0.9},
        "learning-rate": {"min": 0.000001, "max": 0.1},
        "batch-size": {"min": 16, "max": 512},
        "early-stopping-patience": {"min": 5, "max": 50},
        "sequence-length": {"min": 10, "max": 100},
        "stride": {"min": 1, "max": 20}
    }
    
    input_ids = ["model-type", "hidden-size", "num-layers", "dropout-rate", 
                 "learning-rate", "batch-size", "early-stopping-patience", 
                 "sequence-length", "stride"]
    
    for value, input_id in zip(values, input_ids):
        if value is None or value == "":
            error_messages.append(f"Campo {input_id.replace('-', ' ').title()} é obrigatório")
            all_valid = False
            continue
            
        rules = validation_rules[input_id]
        if "valid_values" in rules:
            if value not in rules["valid_values"]:
                error_messages.append(f"Valor inválido para {input_id.replace('-', ' ').title()}")
                all_valid = False
        else:
            try:
                val = float(value)
                if not (rules["min"] <= val <= rules["max"]):
                    error_messages.append(
                        f"{input_id.replace('-', ' ').title()} deve estar entre {rules['min']} e {rules['max']}"
                    )
                    all_valid = False
            except (ValueError, TypeError):
                error_messages.append(f"Valor inválido para {input_id.replace('-', ' ').title()}")
                all_valid = False
    
    # Prepara mensagens de erro
    validation_message = None
    if not all_valid:
        validation_message = [
            html.H6("Erros de Validação:", className="mb-2"),
            html.Ul([html.Li(msg) for msg in error_messages])
        ]
    
    # Status do modelo
    model_status = "Configuração válida" if all_valid else "Configuração inválida"
    
    return (
        not all_valid,  # apply-settings disabled
        not all_valid,  # train-model disabled
        model_status,  # model-status
        not all_valid,  # validation-status is_open
        validation_message  # validation-status children
    )

@dash.callback(
    [
        Output("model-status", "children"),
        Output("training-status", "children"),
        Output("training-progress-main", "value"),
        Output("cancel-training", "disabled")
    ],
    [
        Input("train-model", "n_clicks"),
        Input("apply-settings", "n_clicks")
    ],
    [
        State("model-type", "value"),
        State("hidden-size", "value"),
        State("num-layers", "value"),
        State("dropout-rate", "value"),
        State("learning-rate", "value"),
        State("batch-size", "value"),
        State("early-stopping-patience", "value"),
        State("sequence-length", "value"),
        State("stride", "value")
    ],
    prevent_initial_call=True
)
def handle_model_actions(train_clicks, apply_clicks, *params):
    """Manipula as ações de treinar e aplicar configurações"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "apply-settings":
        # Salva os parâmetros
        hyperparameters = {
            "model_type": params[0],
            "hidden_size": int(params[1]),
            "num_layers": int(params[2]),
            "dropout_rate": float(params[3]),
            "learning_rate": float(params[4]),
            "batch_size": int(params[5]),
            "early_stopping_patience": int(params[6]),
            "sequence_length": int(params[7]),
            "stride": int(params[8])
        }
        
        try:
            os.makedirs("models", exist_ok=True)
            torch.save(hyperparameters, os.path.join("models", "hyperparameters.pt"))
            return "Configurações salvas com sucesso", "Pronto para treinar", 0, True
        except Exception as e:
            return f"Erro ao salvar configurações: {str(e)}", "Erro", 0, True
            
    elif button_id == "train-model":
        # Inicia o treinamento
        return "Treinando modelo...", "Treinamento iniciado", 0, False
    
    raise PreventUpdate