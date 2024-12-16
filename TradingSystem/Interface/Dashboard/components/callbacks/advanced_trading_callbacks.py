import logging
from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def register(app, components):
    """Register all advanced trading callbacks"""
    
    @app.callback(
        Output("order-book-graph", "figure"),
        [Input("interval-fast", "n_intervals")],
        prevent_initial_call=True
    )
    def update_order_book(n_intervals):
        """Update order book visualization"""
        try:
            # This would be replaced with actual order book data from your exchange
            # For now, creating sample data
            price_levels = np.linspace(30000, 31000, 20)
            bid_volumes = np.random.exponential(5, 20)
            ask_volumes = np.random.exponential(5, 20)
            
            fig = go.Figure()
            
            # Add bid bars
            fig.add_trace(go.Bar(
                x=-bid_volumes,
                y=price_levels,
                orientation='h',
                name='Bids',
                marker_color='rgba(0, 255, 0, 0.5)'
            ))
            
            # Add ask bars
            fig.add_trace(go.Bar(
                x=ask_volumes,
                y=price_levels,
                orientation='h',
                name='Asks',
                marker_color='rgba(255, 0, 0, 0.5)'
            ))
            
            fig.update_layout(
                title="Order Book Depth",
                showlegend=True,
                barmode='overlay',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Price",
                xaxis_title="Volume",
                margin=dict(l=0, r=0, t=30, b=0),
                height=300
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating order book: {str(e)}")
            raise PreventUpdate

    @app.callback(
        [Output("total-bid-volume", "children"),
         Output("total-ask-volume", "children")],
        [Input("interval-fast", "n_intervals")],
        prevent_initial_call=True
    )
    def update_order_book_totals(n_intervals):
        """Update order book volume totals"""
        try:
            # This would be replaced with actual order book data
            total_bids = f"{np.random.uniform(100, 1000):.2f} BTC"
            total_asks = f"{np.random.uniform(100, 1000):.2f} BTC"
            return total_bids, total_asks
        except Exception as e:
            logger.error(f"Error updating order book totals: {str(e)}")
            raise PreventUpdate

    @app.callback(
        Output("position-heat-map", "figure"),
        [Input("interval-medium", "n_intervals")],
        prevent_initial_call=True
    )
    def update_position_heat_map(n_intervals):
        """Update position heat map visualization"""
        try:
            # This would be replaced with actual position data
            price_levels = np.linspace(30000, 31000, 50)
            position_sizes = np.random.normal(0, 1, 50)
            
            fig = go.Figure(data=go.Scatter(
                x=price_levels,
                y=position_sizes,
                mode='markers',
                marker=dict(
                    size=10,
                    color=position_sizes,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Position Size")
                )
            ))
            
            fig.update_layout(
                title="Position Distribution",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Price Level",
                yaxis_title="Position Size",
                margin=dict(l=0, r=0, t=30, b=0),
                height=250
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating position heat map: {str(e)}")
            raise PreventUpdate

    @app.callback(
        Output("apply-tp-strategy", "disabled"),
        [Input("tp-level-1", "value"),
         Input("tp-level-2", "value"),
         Input("tp-level-3", "value"),
         Input("tp-size-1", "value"),
         Input("tp-size-2", "value"),
         Input("tp-size-3", "value")],
        prevent_initial_call=True
    )
    def validate_tp_strategy(tp1, tp2, tp3, size1, size2, size3):
        """Validate take-profit strategy inputs"""
        try:
            # Check if all inputs are provided and valid
            if None in [tp1, tp2, tp3, size1, size2, size3]:
                return True
                
            # Check if TP levels are in ascending order
            if not (tp1 < tp2 < tp3):
                return True
                
            # Check if sizes sum to 100%
            if abs(size1 + size2 + size3 - 100) > 0.01:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error validating TP strategy: {str(e)}")
            return True

    @app.callback(
        [Output("advanced-trailing-stop-switch", "disabled"),
         Output("advanced-auto-reduce-switch", "disabled")],
        [Input("advanced-max-drawdown", "value"),
         Input("advanced-risk-daily-loss-limit", "value"),
         Input("advanced-position-size-limit", "value"),
         Input("advanced-max-open-positions", "value")],
        prevent_initial_call=True
    )
    def validate_risk_controls(max_dd, daily_limit, pos_limit, max_pos):
        """Validate risk control inputs"""
        try:
            # Check if all inputs are provided and valid
            if None in [max_dd, daily_limit, pos_limit, max_pos]:
                return True, True
                
            # Check if values are within reasonable ranges
            if not (0 < max_dd <= 50 and 0 < daily_limit <= 20 and 
                    0 < pos_limit <= 100 and 0 < max_pos <= 10):
                return True, True
                
            return False, False
            
        except Exception as e:
            logger.error(f"Error validating risk controls: {str(e)}")
            return True, True

    logger.info("Advanced trading callbacks registered successfully") 