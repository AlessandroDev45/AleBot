import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

from .components.layout import create_layout
from .components.callbacks import callbacks

def create_app(components):
    """Create and configure the dashboard application"""
    try:
        # Create Dash app with Bootstrap components
        app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.DARKLY,
                dbc.icons.FONT_AWESOME,
                'https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600&display=swap'
            ],
            assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ],
            serve_locally=True
        )
        
        # Set title
        app.title = "AleBot Trading Dashboard"
        
        # Create layout
        app.layout = create_layout()
        
        # Register callbacks
        callbacks.register_callbacks(app, components)
        
        # Configure assets
        app._favicon = 'assets/favicon.ico'
        
        return app
        
    except Exception as e:
        logger.error(f"Error creating dashboard application: {str(e)}")
        raise