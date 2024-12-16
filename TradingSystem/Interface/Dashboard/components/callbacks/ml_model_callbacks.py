import logging
from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json
import torch
import os

logger = logging.getLogger(__name__)

def register(app, components):
    """Register ML model callbacks"""
    try:
        logger.info("Registering ML model callbacks...")
        
        @app.callback(
            [Output("model-status", "children", allow_duplicate=True),
             Output("model-status", "className", allow_duplicate=True),
             Output("training-status", "children", allow_duplicate=True),
             Output("training-error", "children", allow_duplicate=True),
             Output("training-progress-main", "value", allow_duplicate=True),
             Output("training-duration", "children", allow_duplicate=True),
             Output("training-eta", "children", allow_duplicate=True),
             Output("cancel-training", "disabled", allow_duplicate=True),
             Output("training-progress-main", "label", allow_duplicate=True)],
            [Input("train-model", "n_clicks"),
             Input("cancel-training", "n_clicks"),
             Input("interval-medium", "n_intervals"),
             Input("interval-fast", "n_intervals")],
            [State("model-type", "value"),
             State("learning-rate", "value"),
             State("batch-size", "value"),
             State("dropout-rate", "value"),
             State("early-stopping-patience", "value"),
             State("sequence-length", "value"),
             State("stride", "value"),
             State("training-time-range", "value"),
             State("data-split", "value"),
             State("confidence-threshold", "value"),
             State("position-sizing", "value"),
             State("stop-loss", "value"),
             State("take-profit", "value"),
             State("max-positions", "value"),
             State("use-ta", "value"),
             State("future-bars", "value")],
            prevent_initial_call=True
        )
        def handle_model_training_and_status(n_clicks, cancel_clicks, medium_intervals, fast_intervals,
                                           model_type, learning_rate, batch_size, dropout_rate,
                                           early_stopping_patience, sequence_length, stride,
                                           timeframe, data_split, confidence_threshold,
                                           position_sizing, stop_loss, take_profit,
                                           max_positions, use_ta, future_bars):
            """Combined callback to handle both model training and status updates"""
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            try:
                if 'ml_model' not in components:
                    return ("ML Model not initialized", "text-danger", 
                            "Not training", "", 0, "Duration: N/A", "ETA: N/A", True, "0%")
                
                ml_model = components['ml_model']
                
                # Handle cancel button click
                if trigger_id == "cancel-training" and cancel_clicks:
                    ml_model.cancel_training()
                    return ("Training cancelled", "text-warning",
                            "Training cancelled", "", 0,
                            "Duration: N/A", "ETA: N/A", True, "Cancelled")
                
                # Handle training initiation if train button was clicked
                if trigger_id == "train-model" and n_clicks:
                    if 'data_manager' not in components:
                        return ("Error: Components not initialized", "text-danger",
                                "Model initialization failed", "Please check system configuration.",
                                0, "Duration: N/A", "ETA: N/A", True, "0%")
                    
                    data_manager = components['data_manager']
                    
                    # Validate all required inputs
                    required_params = {
                        'Model Type': model_type,
                        'Learning Rate': learning_rate,
                        'Batch Size': batch_size,
                        'Sequence Length': sequence_length,
                        'Future Bars': future_bars,
                        'Confidence Threshold': confidence_threshold,
                        'Position Sizing': position_sizing,
                        'Stop Loss': stop_loss,
                        'Take Profit': take_profit,
                        'Max Positions': max_positions,
                        'Technical Indicators': use_ta,
                        'Data Split': data_split,
                        'Timeframe': timeframe
                    }
                    
                    missing_params = [name for name, value in required_params.items() if value is None or value == ""]
                    
                    if missing_params:
                        return (f"Error: Missing parameters: {', '.join(missing_params)}", "text-danger",
                                "Error: Missing parameters", f"Please fill in all required fields: {', '.join(missing_params)}",
                                0, "Duration: N/A", "ETA: N/A", True, "0%")
                    
                    # Configure hyperparameters
                    hyperparameters = {
                        'model_type': model_type,
                        'learning_rate': float(learning_rate),
                        'batch_size': int(batch_size),
                        'dropout_rate': float(dropout_rate),
                        'early_stopping_patience': int(early_stopping_patience),
                        'sequence_length': int(sequence_length),
                        'stride': int(stride),
                        'future_bars': int(future_bars),
                        'epochs': 500,
                        'validation_split': 0.2,
                        'shuffle': True,
                        'optimizer': 'adam',
                        'loss': 'binary_crossentropy',
                        'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                        'feature_selection': True,
                        'feature_importance_threshold': 0.05,
                        'class_weight': 'balanced',
                        'trading': {
                            'confidence_threshold': float(confidence_threshold),
                            'position_sizing': float(position_sizing),
                            'stop_loss': float(stop_loss),
                            'take_profit': float(take_profit),
                            'max_positions': int(max_positions)
                        },
                        'feature_engineering': {
                            'use_ta': len(use_ta) > 0 and use_ta[0] == 'true'
                        }
                    }

                    try:
                        logger.info(f"Starting model training with hyperparameters: {hyperparameters}")
                        
                        # Update hyperparameters
                        ml_model.hyperparameters = hyperparameters
                        
                        # Start training
                        ml_model.train(
                            symbol="BTCUSDT",
                            timeframe=timeframe,
                            limit=10000,
                            train_split=float(data_split) / 100
                        )
                        
                        logger.info("Model training started successfully")
                        return ("Training initiated", "text-warning",
                                "Training started...", "", 0,
                                "Duration: 0:00", "ETA: Calculating...", False, "0%")
                                
                    except Exception as e:
                        logger.error(f"Error in training process: {str(e)}")
                        return (f"Error: {str(e)}", "text-danger",
                                "Training failed", str(e), 0,
                                "Duration: N/A", "ETA: N/A", True, "Error")
                
                # Handle status updates for interval triggers
                status = ml_model.get_model_status()
                if not status:
                    return ("Model not initialized", "text-warning",
                            "Not training", "", 0, "Duration: N/A", "ETA: N/A", True, "0%")
                
                training_progress = status.get('training_progress', {})
                current_epoch = training_progress.get('current_epoch', 0)
                total_epochs = training_progress.get('total_epochs', 1)
                elapsed_time = training_progress.get('elapsed_time', 'N/A')
                eta = training_progress.get('eta', 'N/A')
                
                # Calculate progress percentage
                progress = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
                progress_label = f"{progress:.1f}%"
                
                if status.get('is_training', False):
                    return (
                        f"Training in progress ({progress:.1f}%)",
                        "text-warning",
                        f"Training epoch {current_epoch}/{total_epochs}",
                        "",
                        progress,
                        f"Duration: {elapsed_time}",
                        f"ETA: {eta}",
                        False,
                        progress_label
                    )
                elif status.get('is_ready', False):
                    metrics = status.get('current_metrics', {})
                    accuracy = metrics.get('accuracy', 0) * 100
                    return (
                        f"Model ready (Accuracy: {accuracy:.1f}%)",
                        "text-success",
                        "Training completed",
                        "",
                        100,
                        f"Duration: {elapsed_time}",
                        "ETA: N/A",
                        True,
                        "100%"
                    )
                else:
                    return (
                        "Model not trained",
                        "text-warning",
                        "Not training",
                        "",
                        0,
                        "Duration: N/A",
                        "ETA: N/A",
                        True,
                        "0%"
                    )
                    
            except Exception as e:
                logger.error(f"Error in model training and status handler: {str(e)}")
                return (
                    f"Error: {str(e)}",
                    "text-danger",
                    "Error occurred",
                    str(e),
                    0,
                    "Duration: N/A",
                    "ETA: N/A",
                    True,
                    "Error"
                )
        
        @app.callback(
            [Output("train-model", "disabled"),
             Output("apply-settings", "disabled")],
            [Input("model-type", "value"),
             Input("learning-rate", "value"),
             Input("batch-size", "value"),
             Input("dropout-rate", "value"),
             Input("early-stopping-patience", "value"),
             Input("sequence-length", "value"),
             Input("stride", "value"),
             Input("training-time-range", "value"),
             Input("data-split", "value"),
             Input("confidence-threshold", "value"),
             Input("position-sizing", "value"),
             Input("stop-loss", "value"),
             Input("take-profit", "value"),
             Input("max-positions", "value"),
             Input("use-ta", "value"),
             Input("future-bars", "value")],
            prevent_initial_call=True
        )
        def update_button_states(*values):
            """Update the state of train and apply buttons based on input validity"""
            try:
                # Check if any required value is missing
                if any(v is None or v == "" for v in values):
                    return True, True
                    
                # Check if ml_model is training
                if 'ml_model' in components:
                    status = components['ml_model'].get_model_status()
                    if status and status.get('is_training', False):
                        return True, True
                        
                # All values present and not training
                return False, False
                
            except Exception as e:
                logger.error(f"Error updating button states: {str(e)}")
                return True, True
        
        @app.callback(
            [Output("performance-chart", "figure"),
             Output("prediction-gauge", "figure"),
             Output("accuracy-metric", "children"),
             Output("precision-metric", "children"),
             Output("recall-metric", "children"),
             Output("f1-metric", "children"),
             Output("prediction-signal", "children"),
             Output("prediction-confidence", "children"),
             Output("signal-strength", "children"),
             Output("time-horizon", "children"),
             Output("risk-level", "children")],
            [Input("interval-fast", "n_intervals")],
            prevent_initial_call=True
        )
        def update_performance_metrics(n_intervals):
            """Update performance metrics and predictions"""
            try:
                if 'ml_model' not in components:
                    raise PreventUpdate
                    
                ml_model = components['ml_model']
                status = ml_model.get_model_status()
                
                if not status:
                    raise PreventUpdate
                    
                # Get current metrics and predictions
                metrics = status.get('current_metrics', {})
                predictions = status.get('predictions', {})
                history = status.get('performance_history', [])
                
                # Create performance chart
                fig_performance = go.Figure()
                
                if history:
                    df = pd.DataFrame(history)
                    if not df.empty and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Add accuracy line
                        if 'accuracy' in df.columns:
                            fig_performance.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=df['accuracy'] * 100,
                                name='Accuracy',
                                line=dict(color='green', width=2)
                            ))
                        
                        # Add return line
                        if 'return' in df.columns:
                            fig_performance.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=df['return'],
                                name='Return',
                                line=dict(color='blue', width=2)
                            ))
                        
                        # Add confidence line
                        if 'confidence' in df.columns:
                            fig_performance.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=df['confidence'] * 100,
                                name='Confidence',
                                line=dict(color='orange', width=2)
                            ))
                
                fig_performance.update_layout(
                    title="Model Performance",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    template="plotly_dark",
                    height=300,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Create prediction gauge
                confidence = predictions.get('confidence', 0)
                signal = predictions.get('signal', 'NEUTRAL')
                signal_color = 'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'gray'
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': f"Prediction: {signal}"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': signal_color},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    template="plotly_dark",
                    height=200,
                    margin=dict(l=30, r=30, t=30, b=30)
                )
                
                # Format metrics
                accuracy = f"{metrics.get('accuracy', 0) * 100:.1f}%"
                precision = f"{metrics.get('precision', 0) * 100:.1f}%"
                recall = f"{metrics.get('recall', 0) * 100:.1f}%"
                f1 = f"{metrics.get('f1', 0) * 100:.1f}%"
                
                # Format predictions
                signal_strength = f"{predictions.get('strength', 0) * 100:.1f}%"
                confidence_text = f"Confidence: {confidence * 100:.1f}%"
                
                return [
                    fig_performance,
                    fig_gauge,
                    accuracy,
                    precision,
                    recall,
                    f1,
                    signal,
                    confidence_text,
                    signal_strength,
                    predictions.get('horizon', 'Short'),
                    predictions.get('risk', 'Low')
                ]
                
            except Exception as e:
                logger.error(f"Error updating performance metrics: {str(e)}")
                raise PreventUpdate
                
        @app.callback(
            [Output("feature-importance-plot", "figure", allow_duplicate=True),
             Output("correlation-matrix", "figure", allow_duplicate=True),
             Output("feature-stats", "children", allow_duplicate=True)],
            [Input("interval-slow", "n_intervals"),
             Input("feature-categories", "value"),
             Input("update-features", "n_clicks")],
            prevent_initial_call=True
        )
        def update_feature_engineering(n_intervals, categories, n_clicks):
            """Update feature engineering visualizations"""
            try:
                if 'ml_model' not in components:
                    raise PreventUpdate
                    
                # Get feature analysis data
                feature_data = components['ml_model'].get_feature_analysis(categories=categories)
                if not feature_data:
                    raise PreventUpdate
                    
                # Create feature importance plot
                importance_data = feature_data.get('importance', {})
                fig_importance = go.Figure()
                
                if importance_data:
                    # Sort features by importance
                    sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
                    features, values = zip(*sorted_features)
                    
                    fig_importance.add_trace(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color='rgb(55, 83, 109)'
                    ))
                    
                    fig_importance.update_layout(
                        title="Feature Importance",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        template="plotly_dark",
                        height=600,
                        margin=dict(l=200, r=20, t=30, b=30),
                        showlegend=False
                    )
                
                # Create correlation matrix plot
                correlation_data = feature_data.get('correlations', {})
                fig_correlation = go.Figure()
                
                if correlation_data:
                    # Convert correlation data to numpy array if it's a dict
                    if isinstance(correlation_data, dict):
                        features = list(correlation_data.keys())
                        corr_matrix = np.array([[correlation_data[f1].get(f2, 0) 
                                               for f2 in features] 
                                               for f1 in features])
                    else:
                        corr_matrix = np.array(correlation_data)
                        features = [f"Feature {i+1}" for i in range(len(corr_matrix))]

                    fig_correlation.add_trace(go.Heatmap(
                        z=corr_matrix,
                        x=features,
                        y=features,
                        colorscale="RdBu",
                        zmid=0,
                        text=[[f"{val:.2f}" for val in row] for row in corr_matrix],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    fig_correlation.update_layout(
                        title="Feature Correlations",
                        template="plotly_dark",
                        height=600,
                        margin=dict(l=100, r=100, t=50, b=50),
                        xaxis=dict(tickangle=45),
                        yaxis=dict(tickangle=0)
                    )
                
                # Create feature stats text
                num_features = len(importance_data)
                top_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:5]
                stats_text = [
                    f"Total Features: {num_features}",
                    "Top 5 Most Important Features:",
                    *[f"{i+1}. {name}: {value:.4f}" for i, (name, value) in enumerate(top_features)]
                ]
                
                return [
                    fig_importance,
                    fig_correlation,
                    html.Div([html.P(text) for text in stats_text])
                ]
                
            except Exception as e:
                logger.error(f"Error updating feature engineering: {str(e)}")
                raise PreventUpdate
                
        @app.callback(
            [Output("metrics-timeline", "figure", allow_duplicate=True),
             Output("confusion-matrix", "figure", allow_duplicate=True),
             Output("roc-auc-score", "children", allow_duplicate=True),
             Output("sharpe-ratio", "children", allow_duplicate=True),
             Output("ml-max-drawdown", "children", allow_duplicate=True),
             Output("win-rate", "children", allow_duplicate=True),
             Output("sortino-ratio", "children", allow_duplicate=True),
             Output("calmar-ratio", "children", allow_duplicate=True),
             Output("info-ratio", "children", allow_duplicate=True)],
            [Input("interval-medium", "n_intervals")],
            prevent_initial_call=True
        )
        def update_model_metrics(n_intervals):
            """Update model metrics visualizations"""
            try:
                if 'ml_model' not in components:
                    raise PreventUpdate
                    
                ml_model = components['ml_model']
                status = ml_model.get_model_status()
                
                if not status:
                    raise PreventUpdate
                    
                # Get current metrics and trading metrics
                metrics = status.get('current_metrics', {})
                trading_metrics = status.get('trading_metrics', {})
                history = status.get('performance_history', [])
                
                # Create metrics timeline
                fig_timeline = go.Figure()
                
                if history:
                    df = pd.DataFrame(history)
                    if not df.empty and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        for metric in ['accuracy', 'precision', 'recall', 'f1']:
                            if metric in df.columns:
                                fig_timeline.add_trace(go.Scatter(
                                    x=df['timestamp'],
                                    y=df[metric],
                                    name=metric.capitalize(),
                                    mode='lines'
                                ))
                
                fig_timeline.update_layout(
                    title="Metrics Over Time",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    template="plotly_dark",
                    height=300,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Create confusion matrix
                fig_confusion = go.Figure()
                
                if metrics:
                    confusion_data = [
                        [metrics.get('true_negatives', 0), metrics.get('false_positives', 0)],
                        [metrics.get('false_negatives', 0), metrics.get('true_positives', 0)]
                    ]
                    
                    fig_confusion.add_trace(go.Heatmap(
                        z=confusion_data,
                        x=['Predicted Negative', 'Predicted Positive'],
                        y=['Actual Negative', 'Actual Positive'],
                        text=[[str(val) for val in row] for row in confusion_data],
                        texttemplate="%{text}",
                        textfont={"size": 16},
                        colorscale="RdBu",
                        showscale=False
                    ))
                    
                    fig_confusion.update_layout(
                        title="Confusion Matrix",
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=50, r=50, t=50, b=50),
                        xaxis=dict(side="bottom"),
                        yaxis=dict(side="left")
                    )
                
                # Format metric values
                roc_auc = f"{metrics.get('roc_auc', 0):.3f}"
                sharpe = f"{trading_metrics.get('sharpe_ratio', 0):.3f}"
                max_dd = f"{trading_metrics.get('max_drawdown', 0):.2f}%"
                win_rate = f"{trading_metrics.get('win_rate', 0):.2f}%"
                sortino = f"{trading_metrics.get('sortino_ratio', 0):.3f}"
                calmar = f"{trading_metrics.get('calmar_ratio', 0):.3f}"
                info = f"{trading_metrics.get('info_ratio', 0):.3f}"
                
                return [
                    fig_timeline,
                    fig_confusion,
                    roc_auc,
                    sharpe,
                    max_dd,
                    win_rate,
                    sortino,
                    calmar,
                    info
                ]
                
            except Exception as e:
                logger.error(f"Error updating model metrics: {str(e)}")
                raise PreventUpdate
                
        @app.callback(
            [Output("model-status", "children", allow_duplicate=True),
             Output("model-status", "className", allow_duplicate=True)],
            [Input("model-type", "value"),
             Input("learning-rate", "value"),
             Input("batch-size", "value"),
             Input("dropout-rate", "value"),
             Input("early-stopping-patience", "value"),
             Input("sequence-length", "value"),
             Input("stride", "value"),
             Input("training-time-range", "value"),
             Input("data-split", "value"),
             Input("confidence-threshold", "value"),
             Input("position-sizing", "value"),
             Input("stop-loss", "value"),
             Input("take-profit", "value"),
             Input("max-positions", "value"),
             Input("use-ta", "value"),
             Input("future-bars", "value")],
            prevent_initial_call=True
        )
        def auto_save_settings(*values):
            """Auto-save settings when any input changes"""
            try:
                if 'ml_model' not in components:
                    return "ML Model not initialized", "text-danger"
                    
                ml_model = components['ml_model']
                
                # Validate all inputs are present
                if any(v is None or v == "" for v in values):
                    return "Waiting for all parameters...", "text-warning"
                
                # Update hyperparameters
                hyperparameters = {
                    'model_type': values[0],
                    'learning_rate': float(values[1]),
                    'batch_size': int(values[2]),
                    'dropout_rate': float(values[3]),
                    'early_stopping_patience': int(values[4]),
                    'sequence_length': int(values[5]),
                    'stride': int(values[6]),
                    'timeframe': values[7],
                    'data_split': float(values[8]),
                    'trading': {
                        'confidence_threshold': float(values[9]),
                        'position_sizing': float(values[10]),
                        'stop_loss': float(values[11]),
                        'take_profit': float(values[12]),
                        'max_positions': int(values[13])
                    },
                    'feature_engineering': {
                        'use_ta': bool(values[14])
                    },
                    'future_bars': int(values[15])
                }
                
                # Save parameters
                save_path = os.path.join("models", "hyperparameters.pt")
                os.makedirs("models", exist_ok=True)
                torch.save(hyperparameters, save_path)
                
                # Update model hyperparameters
                ml_model.hyperparameters = hyperparameters
                
                return "Settings saved automatically", "text-success"
                
            except Exception as e:
                logger.error(f"Error auto-saving settings: {str(e)}")
                return f"Error: {str(e)}", "text-danger"
        
        @app.callback(
            [Output("cancel-training", "disabled", allow_duplicate=True),
             Output("training-progress-main", "label", allow_duplicate=True)],
            [Input("interval-fast", "n_intervals"),
             Input("cancel-training", "n_clicks")],
            prevent_initial_call=True
        )
        def handle_training_progress(n_intervals, cancel_clicks):
            """Handle training progress and cancellation"""
            try:
                if 'ml_model' not in components:
                    return True, "0%"
                    
                ml_model = components['ml_model']
                status = ml_model.get_model_status()
                
                # Handle cancel button click
                ctx = callback_context
                if ctx.triggered_id == "cancel-training" and cancel_clicks:
                    ml_model.cancel_training()
                    return True, "Cancelled"
                
                # Update progress
                if status.get('is_training', False):
                    progress = status.get('training_progress', {})
                    current_epoch = progress.get('current_epoch', 0)
                    total_epochs = progress.get('total_epochs', 1)
                    percent = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
                    return False, f"{percent:.1f}%"
                    
                return True, "0%"
                
            except Exception as e:
                logger.error(f"Error handling training progress: {str(e)}")
                return True, "Error"
        
        @app.callback(
            [Output("model-status", "children"),
             Output("model-status", "className")],
            [Input("apply-settings", "n_clicks")],
            [State("model-type", "value"),
             State("learning-rate", "value"),
             State("batch-size", "value"),
             State("dropout-rate", "value"),
             State("early-stopping-patience", "value"),
             State("sequence-length", "value"),
             State("stride", "value")],
            prevent_initial_call=True
        )
        def apply_model_settings(n_clicks, model_type, learning_rate, batch_size, 
                               dropout_rate, early_stopping_patience, sequence_length, stride):
            """Apply and persist model hyperparameter settings"""
            if not n_clicks:
                raise PreventUpdate
                
            try:
                if 'ml_model' not in components:
                    return "ML Model not initialized", "text-danger"
                    
                ml_model = components['ml_model']
                
                # Validate inputs
                if not all([model_type, learning_rate, batch_size, dropout_rate, 
                           early_stopping_patience, sequence_length, stride]):
                    return "Error: Missing parameters", "text-danger"
                
                # Update hyperparameters
                hyperparameters = {
                    'model_type': model_type,
                    'learning_rate': float(learning_rate),
                    'batch_size': int(batch_size),
                    'dropout_rate': float(dropout_rate),
                    'early_stopping_patience': int(early_stopping_patience),
                    'sequence_length': int(sequence_length),
                    'stride': int(stride)
                }
                
                # Save parameters before updating model
                save_path = os.path.join("models", "hyperparameters.pt")
                os.makedirs("models", exist_ok=True)
                torch.save(hyperparameters, save_path)
                
                # Update model hyperparameters
                ml_model.update_hyperparameters(hyperparameters)
                
                return "Settings applied and saved successfully", "text-success"
                
            except Exception as e:
                logger.error(f"Error applying model settings: {str(e)}")
                return f"Error: {str(e)}", "text-danger"
        
        @app.callback(
            [Output("btc-price", "children"),
             Output("btc-price", "className"),
             Output("btc-change", "children"),
             Output("btc-change", "className")],
            [Input("interval-fast", "n_intervals")],
            prevent_initial_call=True
        )
        def update_btc_price(n_intervals):
            """Update BTC price display with color indication"""
            try:
                if 'data_manager' not in components:
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                
                # Get current and previous price
                current_price = data_manager.get_current_price("BTCUSDT")
                previous_price = data_manager.get_previous_price("BTCUSDT")
                
                if current_price is None or previous_price is None:
                    return "$0.00", "text-muted d-inline ms-3", "(0.00%)", "text-muted d-inline ms-2"
                
                # Calculate price change and percentage
                price_change = current_price - previous_price
                price_change_pct = (price_change / previous_price) * 100
                
                # Determine colors based on price movement
                price_color = "text-success d-inline ms-3" if price_change >= 0 else "text-danger d-inline ms-3"
                change_color = "text-success d-inline ms-2" if price_change >= 0 else "text-danger d-inline ms-2"
                
                # Format price with commas for thousands
                formatted_price = "${:,.2f}".format(current_price)
                formatted_change = f"({price_change_pct:+.2f}%)"
                
                return formatted_price, price_color, formatted_change, change_color
                
            except Exception as e:
                logger.error(f"Error updating BTC price: {str(e)}")
                return "$0.00", "text-muted d-inline ms-3", "(0.00%)", "text-muted d-inline ms-2"
        
        logger.info("ML model callbacks registered successfully")
        
    except Exception as e:
        logger.error(f"Error registering ML model callbacks: {str(e)}")
        raise