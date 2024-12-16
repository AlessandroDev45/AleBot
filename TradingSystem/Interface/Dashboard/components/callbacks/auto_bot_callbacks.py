import html
import logging
import os
from dash import Input, Output, State, callback, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from TradingSystem.Core.DataManager.data_manager import DataManager

logger = logging.getLogger(__name__)

def register(app, components):
    """Register all auto bot callbacks"""
    
    def get_data_manager():
        """Get or initialize DataManager"""
        if 'data_manager' not in components:
            logger.error("DataManager must be initialized in main.py and passed through components")
            raise PreventUpdate
        return components['data_manager']

    @app.callback(
        [Output("main-chart", "figure"),
         Output("auto-volume-chart", "figure")],
        [Input("interval-fast", "n_intervals"),
         Input("tf-1m", "n_clicks"),
         Input("tf-5m", "n_clicks"),
         Input("tf-15m", "n_clicks"),
         Input("tf-1h", "n_clicks"),
         Input("tf-4h", "n_clicks")],
        prevent_initial_call=True
    )
    def update_charts(n_intervals, clicks_1m, clicks_5m, clicks_15m, clicks_1h, clicks_4h):
        """Update main price chart and volume chart"""
        ctx = callback_context
        try:
            data_manager = get_data_manager()
            
            # Get active timeframe
            if not ctx.triggered:
                timeframe = "1h"  # default
            else:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]
                timeframe = button_id.split("-")[1] if "tf-" in button_id else "1h"
            
            logger.info(f"Fetching market data for timeframe: {timeframe}")
            
            # Get market data from DataManager's cache
            df = data_manager.get_market_data(timeframe=timeframe)
            if df is None or df.empty:
                logger.warning(f"No market data available for timeframe {timeframe}")
                raise PreventUpdate
            
            # Get technical indicators from DataManager's cache
            indicators = data_manager.get_technical_indicators(timeframe=timeframe)
            if not indicators:
                logger.warning("No technical indicators available")
            
            # Create main chart with real data
            main_fig = go.Figure(data=[
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                )
            ])
            
            # Add indicators if available
            if indicators:
                # Add Moving Averages
                if 'ma_20' in indicators and 'ma_50' in indicators:
                    main_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=indicators['ma_20'],
                        name='MA20',
                        line=dict(color='yellow', width=1)
                    ))
                    main_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=indicators['ma_50'],
                        name='MA50',
                        line=dict(color='orange', width=1)
                    ))
                
                # Add Bollinger Bands
                if all(k in indicators for k in ['bb_upper', 'bb_lower']):
                    main_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=indicators['bb_upper'],
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                    main_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=indicators['bb_lower'],
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
            
            main_fig.update_layout(
                title=f"Price Chart ({timeframe})",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # Create volume chart with real data from the same DataFrame
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['close'], df['open'])]
                     
            volume_fig = go.Figure(data=[
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    marker_color=colors,
                    name="Volume"
                )
            ])
            
            volume_fig.update_layout(
                title="Volume",
                xaxis_title="Time",
                yaxis_title="Volume",
                template="plotly_dark",
                height=150,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            logger.info(f"Successfully updated charts for timeframe {timeframe}")
            return main_fig, volume_fig
            
        except Exception as e:
            logger.error(f"Error updating charts: {str(e)}")
            raise PreventUpdate

    @app.callback(
        [Output("auto-bot-daily-pnl", "children"),
         Output("auto-bot-win-rate", "children"),
         Output("auto-bot-trades-today", "children")],
        Input("interval-medium", "n_intervals"),
        prevent_initial_call=True
    )
    def update_trading_stats(n_intervals):
        """Update trading statistics"""
        try:
            data_manager = get_data_manager()
            
            # Get trading stats from DataManager's cache
            trading_stats = data_manager.get_daily_trading_stats()
            if not trading_stats:
                logger.warning("No trading statistics available")
                raise PreventUpdate
                
            # Format PnL with color based on value
            pnl = trading_stats.get('daily_pnl', 0)
            pnl_color = 'green' if pnl > 0 else 'red' if pnl < 0 else 'white'
            pnl_text = f'${pnl:.2f}'
            
            # Format win rate with percentage
            win_rate = trading_stats.get('win_rate', 0)
            win_rate_text = f'{win_rate:.1f}%'
            
            # Get number of trades today
            trades_today = trading_stats.get('today_trades_count', 0)
            
            logger.info(f"Updated trading stats - PnL: {pnl_text}, Win Rate: {win_rate_text}, Trades: {trades_today}")
            return (
                html.Span(pnl_text, style={'color': pnl_color}),
                win_rate_text,
                str(trades_today)
            )
            
        except Exception as e:
            logger.error(f"Error updating trading stats: {str(e)}")
            raise PreventUpdate

    @app.callback(
        [Output("signal-chart-1", "figure"),
         Output("signal-chart-2", "figure")],
        [Input("interval-fast", "n_intervals")],
        prevent_initial_call=True
    )
    def update_signal_dashboard(n_intervals):
        """Update signal dashboard charts"""
        try:
            data_manager = get_data_manager()
            
            # Get market data from DataManager's cache
            df = data_manager.get_market_data(timeframe="1m", limit=30)
            if df is None or df.empty:
                logger.warning("No market data available for signal charts")
                raise PreventUpdate
            
            # Get pattern signals from DataManager
            patterns = data_manager.analyze_patterns(symbol="BTCUSDT", timeframe="1m")
            if patterns.empty:
                logger.warning("No pattern signals available")
                raise PreventUpdate
            
            # Create signal strength chart using pattern data
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=patterns.index,
                y=patterns['signal_strength'],
                mode='lines+markers',
                name='Signal Strength',
                line=dict(color='#00ff00', width=2)
            ))
            
            fig1.update_layout(
                title="Pattern Signal Strength",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_dark",
                height=200,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # Create signal confidence chart using pattern data
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=patterns.index,
                y=patterns['signal_confidence'],
                mode='lines+markers',
                name='Pattern Confidence',
                line=dict(color='#ff9900', width=2)
            ))
            
            fig2.update_layout(
                title="Pattern Confidence",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_dark",
                height=200,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            logger.info("Successfully updated signal dashboard")
            return fig1, fig2
            
        except Exception as e:
            logger.error(f"Error updating signal dashboard: {str(e)}")
            raise PreventUpdate

    @app.callback(
        Output("order-book-heatmap", "figure"),
        [Input("interval-fast", "n_intervals")],
        prevent_initial_call=True
    )
    def update_order_book_heatmap(n_intervals):
        """Update order book heatmap"""
        try:
            data_manager = get_data_manager()
            
            # Get order book data from DataManager's cache
            order_book = data_manager.get_order_book(symbol="BTCUSDT", limit=20)
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                logger.warning("No order book data available")
                raise PreventUpdate
            
            # Process order book data
            bids = order_book['bids']
            asks = order_book['asks']
            
            # Create separate traces for bids and asks
            bid_prices = [float(price) for price, _ in bids]
            bid_volumes = [float(vol) for _, vol in bids]
            ask_prices = [float(price) for price, _ in asks]
            ask_volumes = [float(vol) for _, vol in asks]
            
            fig = go.Figure()
            
            # Add bid orders (green)
            fig.add_trace(go.Bar(
                x=bid_prices,
                y=bid_volumes,
                name="Bids",
                marker_color='rgba(0, 255, 0, 0.5)'
            ))
            
            # Add ask orders (red)
            fig.add_trace(go.Bar(
                x=ask_prices,
                y=ask_volumes,
                name="Asks",
                marker_color='rgba(255, 0, 0, 0.5)'
            ))
            
            fig.update_layout(
                title="Order Book Depth",
                xaxis_title="Price",
                yaxis_title="Volume",
                template="plotly_dark",
                height=300,
                barmode='overlay',
                bargap=0,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            logger.info("Successfully updated order book heatmap")
            return fig
            
        except Exception as e:
            logger.error(f"Error updating order book heatmap: {str(e)}")
            raise PreventUpdate

    @app.callback(
        [Output("auto-bot-total-balance", "children"),
         Output("auto-bot-available-balance", "children"),
         Output("auto-bot-position-value", "children"),
         Output("auto-bot-last-update", "children")],
        Input("interval-medium", "n_intervals"),
        prevent_initial_call=True
    )
    def update_account_info(n_intervals):
        """Update account information"""
        try:
            data_manager = get_data_manager()
            
            # Get account data from DataManager's cache
            account_data = data_manager.get_account_data()
            if not account_data:
                logger.warning("No account data available")
                raise PreventUpdate
                
            # Format balances with 2 decimal places
            total_balance = float(account_data.get('total_balance', 0))
            available_balance = float(account_data.get('available_balance', 0))
            position_value = float(account_data.get('position_value', 0))
            
            logger.info(f"Updated account info - Total: ${total_balance:.2f}, Available: ${available_balance:.2f}")
            return (
                f"${total_balance:.2f}",
                f"${available_balance:.2f}",
                f"${position_value:.2f}",
                f"Last Update: {account_data.get('last_update', 'N/A')}"
            )
                
        except Exception as e:
            logger.error(f"Error updating account info: {str(e)}")
            raise PreventUpdate

    @app.callback(
        Output("start-bot", "disabled"),
        [Input("start-bot", "n_clicks"),
         Input("stop-bot", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_bot_controls(start_clicks, stop_clicks):
        """Handle bot start/stop controls"""
        ctx = callback_context
        if not ctx.triggered:
            return False
        
        try:
            data_manager = get_data_manager()
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-bot":
                # Update bot status in DataManager
                data_manager.set_bot_status(True)
                logger.info("Bot started")
                return True
            elif button_id == "stop-bot":
                # Update bot status in DataManager
                data_manager.set_bot_status(False)
                logger.info("Bot stopped")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling bot controls: {str(e)}")
            raise PreventUpdate

    @app.callback(
        [Output("total-return", "children"),
         Output("monthly-return", "children"),
         Output("daily-return", "children"),
         Output("mini-equity-curve", "figure")],
        [Input("interval-slow", "n_intervals")]
    )
    def update_performance_stats(n_intervals):
        """Update performance statistics"""
        try:
            data_manager = get_data_manager()
            
            # Get performance data from DataManager's cache
            performance_data = data_manager.get_performance_stats()
            if not performance_data:
                logger.warning("No performance data available")
                raise PreventUpdate
                
            # Format returns
            total_return = f"{performance_data.get('total_return', 0):+.2f}%"
            monthly_return = f"{performance_data.get('monthly_return', 0):+.2f}%"
            daily_return = f"{performance_data.get('daily_return', 0):+.2f}%"
            
            # Get equity curve data from DataManager
            equity_data = data_manager.get_equity_curve()
            if equity_data is None or equity_data.empty:
                logger.warning("No equity curve data available")
                raise PreventUpdate
            
            # Create equity curve figure
            fig = go.Figure(data=[
                go.Scatter(
                    x=equity_data.index,
                    y=equity_data['equity'],
                    name="Equity",
                    fill='tozeroy',
                    line=dict(color='rgb(26, 118, 255)')
                )
            ])
            
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Time",
                yaxis_title="Equity",
                template="plotly_dark",
                height=150,
                margin=dict(l=40, r=40, t=40, b=40),
                showlegend=False
            )
            
            logger.info(f"Updated performance stats - Total: {total_return}, Monthly: {monthly_return}")
            return total_return, monthly_return, daily_return, fig
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
            raise PreventUpdate

    @app.callback(
        [Output("risk-metrics", "children"),
         Output("risk-chart", "figure")],
        [Input("interval-medium", "n_intervals")]
    )
    def update_risk_metrics(n_intervals):
        """Update risk metrics"""
        try:
            data_manager = get_data_manager()
            
            # Get risk metrics from DataManager's cache
            risk_data = data_manager.get_advanced_stats()
            if not risk_data:
                logger.warning("No risk metrics available")
                raise PreventUpdate
            
            # Create metrics display
            metrics_div = html.Div([
                html.H6("Risk Metrics"),
                html.P([
                    html.Strong("Sharpe Ratio: "),
                    f"{risk_data.get('sharpe_ratio', 0):.2f}"
                ]),
                html.P([
                    html.Strong("Max Drawdown: "),
                    f"{risk_data.get('max_drawdown', 0):.2f}%"
                ]),
                html.P([
                    html.Strong("Win Rate: "),
                    f"{risk_data.get('win_rate', 0):.1f}%"
                ])
            ])
            
            # Create risk visualization
            fig = go.Figure()
            
            # Add drawdown chart
            if 'drawdown_series' in risk_data:
                fig.add_trace(go.Scatter(
                    x=risk_data['drawdown_series'].index,
                    y=risk_data['drawdown_series'].values,
                    name="Drawdown",
                    fill='tozeroy',
                    line=dict(color='red')
                ))
            
            fig.update_layout(
                title="Drawdown Chart",
                xaxis_title="Time",
                yaxis_title="Drawdown %",
                template="plotly_dark",
                height=200,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            logger.info("Successfully updated risk metrics")
            return metrics_div, fig
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {str(e)}")
            raise PreventUpdate

    @app.callback(
        [Output("auto-bot-status", "children"),
         Output("auto-bot-status", "color")],
        [Input("start-bot", "n_clicks"),
         Input("stop-bot", "n_clicks"),
         Input("interval-medium", "n_intervals")]
    )
    def update_bot_status(start_clicks, stop_clicks, n_intervals):
        """Update bot status and handle model integration"""
        try:
            if 'ml_model' not in components or 'data_manager' not in components:
                return "Components not initialized", "danger"
            
            ml_model = components['ml_model']
            data_manager = components['data_manager']
            
            # Check if model is ready
            model_status = ml_model.get_model_status()
            if not model_status.get('is_ready', False):
                return "ML Model not ready - train model first", "warning"
            
            ctx = callback_context
            if not ctx.triggered:
                button_id = "none"
            else:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-bot":
                # Start bot with model integration
                try:
                    data_manager.set_bot_status(True)
                    data_manager.set_model_enabled(True)  # Enable model for trading
                    return "Bot running with ML model enabled", "success"
                except Exception as e:
                    logger.error(f"Error starting bot: {str(e)}")
                    return f"Error starting bot: {str(e)}", "danger"
                    
            elif button_id == "stop-bot":
                try:
                    data_manager.set_bot_status(False)
                    data_manager.set_model_enabled(False)  # Disable model
                    return "Bot stopped", "warning"
                except Exception as e:
                    logger.error(f"Error stopping bot: {str(e)}")
                    return f"Error stopping bot: {str(e)}", "danger"
            
            # Return current status
            bot_status = data_manager.get_bot_status()
            model_enabled = data_manager.get_model_enabled()
            
            if bot_status and model_enabled:
                return "Bot running with ML model enabled", "success"
            elif bot_status:
                return "Bot running without ML model", "primary"
            else:
                return "Bot stopped", "warning"
                
        except Exception as e:
            logger.error(f"Error updating bot status: {str(e)}")
            return f"Error: {str(e)}", "danger"

    logger.info("Auto bot callbacks registered successfully")