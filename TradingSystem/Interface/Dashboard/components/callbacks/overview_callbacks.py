import logging
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_html_components as html

logger = logging.getLogger(__name__)

def register(app, components):
    """Register overview tab callbacks"""
    try:
        @app.callback(
            [Output("total-balance", "children"),
             Output("available-balance", "children"),
             Output("margin-balance", "children"),
             Output("balance-chart", "figure")],
            [Input("interval-medium", "n_intervals")]
        )
        def update_balance_info(n_intervals):
            """Update balance information with real data"""
            try:
                if 'data_manager' not in components:
                    logger.error("Data manager not found in components")
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                logger.info("Fetching account data for balance info")
                
                # Get account data through DataManager
                account_data = data_manager.get_account_data()
                if not account_data:
                    logger.error("Failed to get account data")
                    raise PreventUpdate
                
                # Extract balances
                total_balance = account_data.get('total_balance', 0)
                available_balance = account_data.get('available_balance', 0)
                margin_balance = account_data.get('margin_balance', 0)
                
                logger.info(f"Total Balance: {total_balance:.2f} USDT")
                logger.info(f"Available Balance: {available_balance:.2f} USDT")
                logger.info(f"Margin Balance: {margin_balance:.2f} USDT")
                
                # Create balance chart
                fig = go.Figure()
                
                # Add balance traces
                fig.add_trace(go.Bar(
                    x=['Total', 'Available', 'Margin'],
                    y=[total_balance, available_balance, margin_balance],
                    name='Balance',
                    marker_color=['green', 'blue', 'orange']
                ))
                
                fig.update_layout(
                    title="Balance Overview",
                    yaxis_title="USDT",
                    template="plotly_dark",
                    height=200,
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                logger.info("Successfully updated balance info")
                return [
                    f"${total_balance:,.2f}",
                    f"${available_balance:,.2f}",
                    f"${margin_balance:,.2f}",
                    fig
                ]
                
            except Exception as e:
                logger.error(f"Error updating balance info: {str(e)}")
                raise PreventUpdate
        
        @app.callback(
            [Output("daily-pnl", "children"),
             Output("win-rate", "children"),
             Output("trades-today", "children"),
             Output("performance-chart", "figure")],
            [Input("interval-medium", "n_intervals")]
        )
        def update_performance_stats(n_intervals):
            """Update performance statistics with real data"""
            try:
                if 'data_manager' not in components:
                    logger.error("Data manager not found in components")
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                logger.info("Fetching performance stats")
                
                # Get daily trading stats through DataManager
                stats = data_manager.get_daily_trading_stats()
                if not stats:
                    logger.error("Failed to get daily trading stats")
                    raise PreventUpdate
                
                daily_pnl = stats.get('daily_pnl', 0)
                win_rate = stats.get('win_rate', 0)
                trades_today = stats.get('today_trades_count', 0)
                
                logger.info(f"Daily PnL: {daily_pnl:.2f} USDT")
                logger.info(f"Win Rate: {win_rate:.1f}%")
                logger.info(f"Trades Today: {trades_today}")
                
                # Create performance chart
                fig = go.Figure()
                
                # Add performance metrics
                fig.add_trace(go.Bar(
                    x=['Win Rate', 'Loss Rate'],
                    y=[win_rate, 100 - win_rate],
                    marker_color=['green', 'red']
                ))
                
                fig.update_layout(
                    title="Trading Performance",
                    yaxis_title="Percentage (%)",
                    template="plotly_dark",
                    height=200,
                    showlegend=False
                )
                
                logger.info("Successfully updated performance stats")
                return [
                    f"${daily_pnl:,.2f}",
                    f"{win_rate:.1f}%",
                    str(trades_today),
                    fig
                ]
                
            except Exception as e:
                logger.error(f"Error updating performance stats: {str(e)}")
                raise PreventUpdate
        
        @app.callback(
            [Output("asset-allocation-chart", "figure"),
             Output("asset-list", "children")],
            [Input("interval-medium", "n_intervals")]
        )
        def update_asset_allocation(n_intervals):
            """Update asset allocation information with real data"""
            try:
                if 'data_manager' not in components:
                    logger.error("Data manager not found in components")
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                logger.info("Fetching account data for asset allocation")
                
                # Get account data through DataManager
                account_data = data_manager.get_account_data()
                if not account_data:
                    logger.error("Failed to get account data")
                    raise PreventUpdate
                
                # Get assets info
                assets = account_data.get('assets_info', [])
                if not assets:
                    logger.warning("No assets found in account data")
                    raise PreventUpdate
                
                logger.info(f"Found {len(assets)} assets with balance")
                
                # Prepare data for pie chart
                labels = []
                values = []
                asset_rows = []
                
                for asset in assets:
                    if asset['value_in_usdt'] > 0:
                        labels.append(asset['asset'])
                        values.append(asset['value_in_usdt'])
                        asset_rows.append(
                            html.Tr([
                                html.Td(asset['asset']),
                                html.Td(f"{asset['total']:.8f}"),
                                html.Td(f"${asset['value_in_usdt']:.2f}")
                            ])
                        )
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.3
                )])
                
                fig.update_layout(
                    title="Asset Allocation",
                    template="plotly_dark",
                    height=300,
                    showlegend=True
                )
                
                # Create asset list table
                asset_table = html.Table(
                    [html.Thead(html.Tr([
                        html.Th("Asset"),
                        html.Th("Amount"),
                        html.Th("Value (USDT)")
                    ]))] +
                    [html.Tbody(asset_rows)]
                )
                
                logger.info("Successfully updated asset allocation")
                return fig, asset_table
                
            except Exception as e:
                logger.error(f"Error updating asset allocation: {str(e)}")
                raise PreventUpdate
        
        @app.callback(
            [Output("risk-metrics-chart", "figure"),
             Output("risk-metrics-list", "children")],
            [Input("interval-medium", "n_intervals")]
        )
        def update_risk_metrics(n_intervals):
            """Update risk metrics with real data"""
            try:
                if 'data_manager' not in components:
                    logger.error("Data manager not found in components")
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                logger.info("Fetching advanced stats for risk metrics")
                
                # Get advanced trading stats through DataManager
                stats = data_manager.get_advanced_stats()
                if not stats:
                    logger.error("Failed to get advanced stats")
                    raise PreventUpdate
                
                # Extract metrics
                metrics = {
                    'Sharpe Ratio': stats.get('sharpe_ratio', 0),
                    'Sortino Ratio': stats.get('sortino_ratio', 0),
                    'Max Drawdown': stats.get('max_drawdown', 0),
                    'Risk/Reward': stats.get('risk_reward_ratio', 0)
                }
                
                logger.info(f"Risk Metrics: {metrics}")
                
                # Create metrics chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=['blue', 'green', 'red', 'orange']
                ))
                
                fig.update_layout(
                    title="Risk Metrics",
                    template="plotly_dark",
                    height=200,
                    showlegend=False
                )
                
                # Create metrics list
                metrics_rows = []
                for metric, value in metrics.items():
                    metrics_rows.append(
                        html.Tr([
                            html.Td(metric),
                            html.Td(f"{value:.2f}")
                        ])
                    )
                
                metrics_table = html.Table(
                    [html.Thead(html.Tr([
                        html.Th("Metric"),
                        html.Th("Value")
                    ]))] +
                    [html.Tbody(metrics_rows)]
                )
                
                logger.info("Successfully updated risk metrics")
                return fig, metrics_table
                
            except Exception as e:
                logger.error(f"Error updating risk metrics: {str(e)}")
                raise PreventUpdate
        
        logger.info("Successfully registered overview callbacks")
        
    except Exception as e:
        logger.error(f"Error registering overview callbacks: {str(e)}")
        raise 