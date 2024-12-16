import logging
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html
import traceback
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def register(app, components):
    """Register manual trading callbacks"""
    try:
        logger.info("Registering manual trading callbacks...")
        
        @app.callback(
            [Output("trading-chart", "figure"),
             Output("trading-volume-chart", "figure"),
             Output("trading-indicators-chart", "figure")],
            [Input("interval-fast", "n_intervals"),
             Input("chart-timeframe", "value")]
        )
        def update_trading_charts(n_intervals, timeframe):
            """Update trading charts with real data"""
            try:
                if 'data_manager' not in components:
                    logger.error("Data manager not found in components")
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                
                # Validate timeframe
                if not timeframe:
                    timeframe = "1h"
                logger.info(f"Fetching trading data for timeframe: {timeframe}")
                
                # Get market data and indicators from DataManager
                df = data_manager.get_market_data(timeframe=timeframe, limit=100)
                if df.empty:
                    logger.error("Received empty DataFrame from data_manager")
                    raise PreventUpdate
                
                indicators = data_manager.get_technical_indicators(timeframe=timeframe)
                if not indicators:
                    logger.error("Failed to retrieve indicators from DataManager")
                    raise PreventUpdate
                
                logger.info(f"Received {len(df)} candles from DataManager")
                
                # Create price chart
                price_fig = go.Figure()
                price_fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                ))
                
                price_fig.update_layout(
                    title=f"Trading Chart ({timeframe})",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    template="plotly_dark",
                    height=400,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                # Create volume chart using DataManager data
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name="Volume",
                    marker_color=['red' if close < open else 'green' 
                                for close, open in zip(df['close'], df['open'])]
                ))
                
                volume_fig.update_layout(
                    title="Volume",
                    xaxis_title="Time",
                    yaxis_title="Volume",
                    template="plotly_dark",
                    height=150,
                    showlegend=False,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=40, r=40, t=40, b=20)
                )
                
                # Create indicators chart using DataManager indicators
                indicators_fig = make_subplots(rows=2, cols=1,
                                            row_heights=[0.5, 0.5],
                                            shared_xaxes=True,
                                            vertical_spacing=0.05)
                
                # Add RSI from indicators
                if 'rsi' in indicators:
                    indicators_fig.add_trace(
                        go.Scatter(x=df.index, y=indicators['rsi'],
                                  name="RSI", line=dict(color='purple')),
                        row=1, col=1
                    )
                    indicators_fig.add_hline(y=70, line_dash="dash", line_color="red", row=1)
                    indicators_fig.add_hline(y=30, line_dash="dash", line_color="green", row=1)
                
                # Add MACD from indicators
                if all(k in indicators for k in ['macd', 'signal', 'macd_hist']):
                    indicators_fig.add_trace(
                        go.Scatter(x=df.index, y=indicators['macd'],
                                  name="MACD", line=dict(color='blue')),
                        row=2, col=1
                    )
                    indicators_fig.add_trace(
                        go.Scatter(x=df.index, y=indicators['signal'],
                                  name="Signal", line=dict(color='orange')),
                        row=2, col=1
                    )
                    
                    # Add MACD histogram
                    colors = ['red' if x < 0 else 'green' for x in indicators['macd_hist']]
                    indicators_fig.add_trace(
                        go.Bar(x=df.index, y=indicators['macd_hist'],
                              name="MACD Hist", marker_color=colors),
                        row=2, col=1
                    )
                
                indicators_fig.update_layout(
                    height=150,
                    template="plotly_dark",
                    showlegend=True,
                    margin=dict(l=40, r=40, t=20, b=20)
                )
                
                indicators_fig.update_yaxes(title_text="RSI", row=1, col=1)
                indicators_fig.update_yaxes(title_text="MACD", row=2, col=1)
                
                logger.info("Successfully updated trading charts")
                return price_fig, volume_fig, indicators_fig
                
            except Exception as e:
                logger.error(f"Error updating trading charts: {str(e)}")
                raise PreventUpdate
        
        @app.callback(
            [Output("orderbook-asks", "children"),
             Output("orderbook-bids", "children")],
            [Input("interval-fast", "n_intervals")]
        )
        def update_orderbook(n_intervals):
            """Update orderbook with real data from DataManager"""
            try:
                if 'data_manager' not in components:
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                order_book = data_manager.get_order_book(limit=10)
                
                if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                    raise PreventUpdate
                
                # Format asks (selling orders)
                asks = []
                for price, qty in order_book['asks']:
                    asks.append(
                        html.Div([
                            html.Span(f"{float(price):.2f}", style={'color': 'red'}),
                            html.Span(f" | {float(qty):.4f}")
                        ], className="orderbook-row")
                    )
                
                # Format bids (buying orders)
                bids = []
                for price, qty in order_book['bids']:
                    bids.append(
                        html.Div([
                            html.Span(f"{float(price):.2f}", style={'color': 'green'}),
                            html.Span(f" | {float(qty):.4f}")
                        ], className="orderbook-row")
                    )
                
                return asks, bids
                
            except Exception as e:
                logger.error(f"Error updating orderbook: {str(e)}")
                raise PreventUpdate
        
        @app.callback(
            Output("position-heatmap", "figure"),
            [Input("interval-medium", "n_intervals"),
             Input("heatmap-timerange", "value"),
             Input("heatmap-aggregation", "value")]
        )
        def update_position_heatmap(n_intervals, timerange, aggregation):
            """Update position heatmap using DataManager's volume profile"""
            try:
                if 'data_manager' not in components:
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                logger.info("Fetching volume profile data for heatmap")
                
                # Get volume profile data from DataManager
                profile = data_manager.get_volume_profile(
                    timerange=timerange or "1h",
                    symbol="BTCUSDT"
                )
                
                if not profile:
                    logger.warning("No volume profile data available")
                    raise PreventUpdate
                
                # Create heatmap from volume profile
                fig = go.Figure(data=go.Heatmap(
                    y=profile['price_levels'][:-1],
                    x=['Volume'],
                    z=[profile['volumes']],
                    colorscale='Viridis'
                ))
                
                fig.update_layout(
                    title="Position Heat Map",
                    yaxis_title="Price Level",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating position heatmap: {str(e)}")
                raise PreventUpdate
        
        @app.callback(
            Output("manual-correlation-heatmap", "figure"),
            [Input("interval-medium", "n_intervals"),
             Input("manual-correlation-timeframe", "value"),
             Input("correlation-assets-manual", "value")]
        )
        def update_correlation_heatmap(n_intervals, timeframe, assets):
            """Update correlation heatmap using DataManager data"""
            try:
                if not assets or len(assets) < 2:
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                
                # Get market data for each asset
                dfs = {}
                for asset in assets:
                    df = data_manager.get_market_data(
                        symbol=f"{asset}USDT",
                        timeframe=timeframe or "1d"
                    )
                    if not df.empty:
                        dfs[asset] = df['close']
                
                if len(dfs) < 2:
                    raise PreventUpdate
                
                # Calculate correlation matrix
                df_combined = pd.DataFrame(dfs)
                correlation_matrix = df_combined.corr().values
                
                # Create correlation heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix,
                    x=assets,
                    y=assets,
                    colorscale='RdBu',
                    zmid=0
                ))
                
                fig.update_layout(
                    title="Asset Correlations",
                    template="plotly_dark",
                    height=300
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating correlation heatmap: {str(e)}")
                raise PreventUpdate
        
        @app.callback(
            [Output("order-validation-feedback", "children"),
             Output("order-validation-feedback", "type"),
             Output("order-validation-feedback", "className")],
            [Input("buy-button", "n_clicks"),
             Input("sell-button", "n_clicks")],
            [State("trading-pair", "value"),
             State("order-type", "value"),
             State("order-amount", "value"),
             State("order-price", "value"),
             State("trading-max-position-size", "value"),
             State("leverage", "value")]
        )
        def validate_order(buy_clicks, sell_clicks, symbol, order_type, amount, price, max_size, leverage):
            """Validate order before execution"""
            try:
                if 'data_manager' not in components:
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                
                # Validate order parameters
                if not symbol or not order_type or not amount or not price or not max_size or not leverage:
                    return "Please fill in all fields", "danger", "order-validation-feedback"
                
                # Validate trading pair
                if not data_manager.is_valid_trading_pair(symbol):
                    return "Invalid trading pair", "danger", "order-validation-feedback"
                
                # Validate order type
                if order_type not in ['market', 'limit']:
                    return "Invalid order type", "danger", "order-validation-feedback"
                
                # Validate order amount
                if not amount or not amount.isdigit() or float(amount) <= 0:
                    return "Invalid order amount", "danger", "order-validation-feedback"
                
                # Validate order price
                if not price or not price.isdigit() or float(price) <= 0:
                    return "Invalid order price", "danger", "order-validation-feedback"
                
                # Validate max position size
                if not max_size or not max_size.isdigit() or float(max_size) <= 0:
                    return "Invalid max position size", "danger", "order-validation-feedback"
                
                # Validate leverage
                if not leverage or not leverage.isdigit() or float(leverage) <= 0:
                    return "Invalid leverage", "danger", "order-validation-feedback"
                
                return "Order validation successful", "success", "order-validation-feedback"
                
            except Exception as e:
                logger.error(f"Error validating order: {str(e)}")
                raise PreventUpdate
        
        logger.info("Successfully registered manual trading callbacks")
    except Exception as e:
        logger.error(f"Error registering manual trading callbacks: {str(e)}")
        raise