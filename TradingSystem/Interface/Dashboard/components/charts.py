import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_price_chart(df, timeframe, indicators=None):
    """Create main price chart with candlesticks"""
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        )
    ])
    
    if indicators:
        if 'bbands' in indicators:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')))
            
        if 'ema' in indicators:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_short'], name='EMA Short', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_long'], name='EMA Long', line=dict(color='orange')))
            
    fig.update_layout(
        title=f"Price Chart ({timeframe})",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )
    
    return fig

def create_volume_chart(df):
    """Create volume chart"""
    colors = ['red' if close < open else 'green' 
              for close, open in zip(df['close'], df['open'])]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            marker_color=colors,
            name="Volume"
        )
    ])
    
    fig.update_layout(
        title="Volume",
        xaxis_title="Time",
        yaxis_title="Volume",
        template="plotly_dark"
    )
    
    return fig

def create_win_loss_distribution(trades_df):
    """Create win/loss distribution chart"""
    fig = go.Figure()
    
    # Add histogram for wins
    fig.add_trace(go.Histogram(
        x=trades_df[trades_df['pnl'] > 0]['pnl'],
        name='Wins',
        marker_color='green',
        opacity=0.75
    ))
    
    # Add histogram for losses
    fig.add_trace(go.Histogram(
        x=trades_df[trades_df['pnl'] < 0]['pnl'],
        name='Losses',
        marker_color='red',
        opacity=0.75
    ))
    
    fig.update_layout(
        title="Win/Loss Distribution",
        xaxis_title="Profit/Loss (%)",
        yaxis_title="Number of Trades",
        barmode='overlay',
        template="plotly_dark"
    )
    
    return fig

def create_drawdown_chart(equity_curve):
    """Create drawdown chart"""
    # Calculate drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve,
        name='Equity',
        line=dict(color='blue')
    ))
    
    # Add drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        name='Drawdown',
        line=dict(color='red'),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Equity Curve & Drawdown",
        xaxis_title="Time",
        yaxis_title="Equity",
        yaxis2=dict(
            title="Drawdown %",
            overlaying="y",
            side="right",
            range=[min(drawdown)*1.1, 0]  # Scale drawdown axis
        ),
        template="plotly_dark"
    )
    
    return fig

def create_volume_profile(df, price_precision=2):
    """Create volume profile chart"""
    # Calculate price levels
    price_levels = np.arange(
        df['low'].min(),
        df['high'].max(),
        (df['high'].max() - df['low'].min()) / 100
    )
    
    volumes = []
    for level in price_levels:
        mask = (df['low'] <= level) & (df['high'] >= level)
        volume = df.loc[mask, 'volume'].sum()
        volumes.append(volume)
    
    fig = go.Figure(data=[
        go.Bar(
            x=volumes,
            y=price_levels,
            orientation='h',
            name="Volume Profile",
            marker_color='rgba(158,202,225,0.6)'
        )
    ])
    
    fig.update_layout(
        title="Volume Profile",
        xaxis_title="Volume",
        yaxis_title="Price Level",
        template="plotly_dark"
    )
    
    return fig

def create_performance_metrics(performance_data):
    """Create performance metrics visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily P&L", "Win Rate", "Profit Factor", "Average Trade")
    )
    
    # Daily P&L
    fig.add_trace(
        go.Bar(x=performance_data['dates'], y=performance_data['daily_pnl']),
        row=1, col=1
    )
    
    # Win Rate
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=performance_data['win_rate'],
            gauge={'axis': {'range': [0, 100]}},
            title={'text': "Win Rate %"}
        ),
        row=1, col=2
    )
    
    # Profit Factor
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=performance_data['profit_factor'],
            delta={'reference': 1},
            title={'text': "Profit Factor"}
        ),
        row=2, col=1
    )
    
    # Average Trade
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=performance_data['avg_trade'],
            delta={'reference': 0},
            title={'text': "Avg Trade"}
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        template="plotly_dark"
    )
    
    return fig

def create_correlation_matrix(correlation_data):
    """Create correlation matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data,
        x=correlation_data.columns,
        y=correlation_data.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        template="plotly_dark"
    )
    
    return fig

def create_feature_importance(importance_df):
    """Create feature importance bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_confusion_matrix(confusion_data):
    """Create confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=confusion_data,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark"
    )
    
    return fig

def create_prediction_gauge(confidence):
    """Create prediction confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Prediction Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(template="plotly_dark")
    return fig

def create_learning_curves(history):
    """Create learning curves plot"""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("Model Loss", "Model Accuracy"))
    
    # Add loss curves
    fig.add_trace(
        go.Scatter(y=history['loss'], name="Training Loss"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_loss'], name="Validation Loss"),
        row=1, col=1
    )
    
    # Add accuracy curves
    fig.add_trace(
        go.Scatter(y=history['accuracy'], name="Training Accuracy"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_accuracy'], name="Validation Accuracy"),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Model Learning Curves",
        template="plotly_dark"
    )
    
    return fig

def create_roc_curve(fpr, tpr, auc):
    """Create ROC curve plot"""
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {auc:.2f})',
        mode='lines'
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Random'
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template="plotly_dark"
    )
    
    return fig

def create_precision_recall_curve(precision, recall, ap):
    """Create precision-recall curve plot"""
    fig = go.Figure()
    
    # Add precision-recall curve
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        name=f'PR curve (AP = {ap:.2f})',
        mode='lines'
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template="plotly_dark"
    )
    
    return fig
