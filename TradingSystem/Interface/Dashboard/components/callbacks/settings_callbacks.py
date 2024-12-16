from dash import Input, Output, State, callback_context, html
import logging

logger = logging.getLogger(__name__)

def register(app, components):
    """Register settings callbacks"""
    try:
        logger.info("Registering settings callbacks...")
        
        @app.callback(
            [Output("api-status", "children"),
             Output("api-status", "color"),
             Output("api-status", "is_open")],
            [Input("interval-slow", "n_intervals")]
        )
        def update_api_status(n_intervals):
            """Update API connection status"""
            try:
                if 'data_manager' not in components:
                    return "No Data Manager Connection", "danger", True
                    
                # Test API connection through DataManager
                status = components['data_manager'].get_api_status()
                if status and status.get('status') == 0:
                    return "Connected to Binance API", "success", True
                else:
                    return "API Connection Error", "danger", True
                    
            except Exception as e:
                logger.error(f"Error updating API status: {str(e)}")
                return "API Connection Error", "danger", True
                
        @app.callback(
            [Output("settings-status", "children"),
             Output("settings-status", "color"),
             Output("settings-status", "is_open")],
            [Input("save-settings", "n_clicks")],
            [State("max-position-size", "value"),
             State("max-daily-loss", "value"),
             State("max-drawdown", "value"),
             State("default-leverage", "value"),
             State("default-margin-type", "value"),
             State("default-order-type", "value"),
             State("telegram-bot-token", "value"),
             State("telegram-chat-id", "value")]
        )
        def save_settings(n_clicks, max_position_size, max_daily_loss, max_drawdown,
                        default_leverage, default_margin_type, default_order_type,
                        telegram_token, telegram_chat_id):
            """Save trading and notification settings"""
            if not n_clicks:
                raise PreventUpdate
                
            try:
                if 'data_manager' not in components:
                    return "No Data Manager Connection", "danger", True
                    
                # Update risk settings
                risk_settings = {
                    'max_position_size': max_position_size,
                    'max_daily_loss': max_daily_loss,
                    'max_drawdown': max_drawdown
                }
                if not components['data_manager'].update_risk_settings(risk_settings):
                    return "Failed to update risk settings", "danger", True
                
                # Update trading settings
                trading_settings = {
                    'default_leverage': default_leverage,
                    'default_margin_type': default_margin_type,
                    'default_order_type': default_order_type
                }
                if not components['data_manager'].update_trading_settings(trading_settings):
                    return "Failed to update trading settings", "danger", True
                
                # Update Telegram settings
                telegram_settings = {
                    'token': telegram_token,
                    'chat_id': telegram_chat_id
                }
                if not components['data_manager'].update_telegram_settings(telegram_settings):
                    return "Failed to update Telegram settings", "danger", True
                
                return "Settings updated successfully", "success", True
                    
            except Exception as e:
                logger.error(f"Error updating settings: {str(e)}")
                return f"Error updating settings: {str(e)}", "danger", True
                
        @app.callback(
            [Output("current-settings", "children")],
            [Input("interval-slow", "n_intervals")]
        )
        def update_current_settings(n_intervals):
            """Update current settings display"""
            try:
                if 'data_manager' not in components:
                    raise PreventUpdate
                    
                # Get current settings
                risk_settings = components['data_manager'].get_risk_settings()
                trading_settings = components['data_manager'].get_trading_settings()
                telegram_settings = components['data_manager'].get_telegram_settings()
                
                settings_div = html.Div([
                    html.H6("Current Settings", className="mt-3"),
                    html.Div([
                        html.Strong("Risk Settings:"),
                        html.Ul([
                            html.Li(f"Max Position Size: {risk_settings['max_position_size']}"),
                            html.Li(f"Max Daily Loss: {risk_settings['max_daily_loss']}%"),
                            html.Li(f"Max Drawdown: {risk_settings['max_drawdown']}%")
                        ])
                    ]),
                    html.Div([
                        html.Strong("Trading Settings:"),
                        html.Ul([
                            html.Li(f"Default Leverage: {trading_settings['default_leverage']}x"),
                            html.Li(f"Margin Type: {trading_settings['default_margin_type']}"),
                            html.Li(f"Order Type: {trading_settings['default_order_type']}")
                        ])
                    ]),
                    html.Div([
                        html.Strong("Telegram Settings:"),
                        html.Ul([
                            html.Li("Bot Token: " + ("Configured" if telegram_settings['token'] else "Not Configured")),
                            html.Li("Chat ID: " + ("Configured" if telegram_settings['chat_id'] else "Not Configured"))
                        ])
                    ])
                ])
                
                return [settings_div]
                
            except Exception as e:
                logger.error(f"Error updating current settings: {str(e)}")
                raise PreventUpdate
                
        logger.info("Settings callbacks registered successfully")
        
    except Exception as e:
        logger.error(f"Error registering settings callbacks: {str(e)}")
        raise