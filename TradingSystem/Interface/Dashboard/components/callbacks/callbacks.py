import logging
from dash import Input, Output, State
from . import auto_bot_callbacks
from . import manual_trading_callbacks
from . import analysis_callbacks
from . import ml_model_callbacks
from . import settings_callbacks
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)

def register_callbacks(app, components):
    """Register all dashboard callbacks"""
    try:
        # Register tab-specific callbacks
        auto_bot_callbacks.register(app, components)
        manual_trading_callbacks.register(app, components)
        analysis_callbacks.register(app, components)
        ml_model_callbacks.register(app, components)
        settings_callbacks.register(app, components)
        
        # Register header callbacks
        @app.callback(
            [
                Output("header-balance", "children"),
                Output("header-pnl", "children"),
                Output("header-trades", "children")
            ],
            [Input("interval-fast", "n_intervals")]
        )
        def update_header_data(n_intervals):
            """Update header with real-time spot account data"""
            try:
                if 'data_manager' not in components:
                    logger.error("Data manager not found in components")
                    return ["0.00 USDT", "0.00%", "0"]
                
                data_manager = components['data_manager']
                
                # Get account data
                account_data = data_manager.get_account_data()
                if not account_data:
                    logger.warning("No account data received")
                    return ["0.00 USDT", "0.00%", "0"]
                
                # Get total balance and assets info
                total_balance = float(account_data.get('total_balance', 0))
                assets_info = account_data.get('assets_info', [])
                
                # Get market metrics for 24h change
                metrics = data_manager.get_metrics()
                price_change = metrics.get('price_change', 0)
                
                # Log the values for debugging
                logger.info(f"Account values - Total Balance: {total_balance:.2f} USDT")
                logger.info(f"Found {len(assets_info)} assets with balance")
                for asset in assets_info:
                    logger.info(f"{asset['asset']}: {asset['total']} ({asset['value_in_usdt']:.2f} USDT)")
                
                return [
                    f"{total_balance:,.2f} USDT",
                    f"{price_change:+.2f}%",
                    str(len(assets_info))
                ]
                
            except Exception as e:
                logger.error(f"Error updating header data: {str(e)}")
                return ["0.00 USDT", "0.00%", "0"]
        
        # Register market depth callback
        @app.callback(
            Output("market-depth", "figure"),
            [Input("interval-medium", "n_intervals")]
        )
        def update_market_depth(n_intervals):
            """Update market depth chart with real data"""
            try:
                if 'data_manager' not in components:
                    raise PreventUpdate
                
                data_manager = components['data_manager']
                order_book = data_manager.get_order_book(limit=20)
                
                if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                    raise PreventUpdate
                
                bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity']).astype(float)
                asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity']).astype(float)
                
                fig = go.Figure()
                
                # Add bids
                fig.add_trace(go.Scatter(
                    x=bids['price'],
                    y=bids['quantity'].cumsum(),
                    name='Bids',
                    fill='tozeroy',
                    line=dict(color='green')
                ))
                
                # Add asks
                fig.add_trace(go.Scatter(
                    x=asks['price'],
                    y=asks['quantity'].cumsum(),
                    name='Asks',
                    fill='tozeroy',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title="Market Depth",
                    xaxis_title="Price",
                    yaxis_title="Quantity",
                    template="plotly_dark",
                    showlegend=True
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating market depth: {str(e)}")
                raise PreventUpdate
        
        logger.info("Successfully registered all dashboard callbacks")
        
    except Exception as e:
        logger.error(f"Error registering callbacks: {str(e)}")
        raise