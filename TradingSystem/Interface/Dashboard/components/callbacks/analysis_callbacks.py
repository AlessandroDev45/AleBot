import logging
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def register(app, components):
    """Register analysis tab callbacks"""
    try:
        @app.callback(
            [Output("price-chart", "figure"),
             Output("volume-chart", "figure"),
             Output("technical-indicators", "children")],
            [Input("interval-medium", "n_intervals"),
             Input("analysis-timeframe", "value"),
             Input("analysis-indicators", "value")]
        )
        def update_analysis_charts(n_intervals, timeframe, selected_indicators):
            """Update analysis charts with real data"""
            try:
                if not timeframe or not selected_indicators:
                    raise PreventUpdate
                    
                data_manager = components['data_manager']
                
                # Get market data with proper caching
                df = data_manager.get_market_data(timeframe=timeframe)
                if df.empty:
                    logger.error("Failed to get market data from API")
                    return {}, {}, html.Div("Error loading data")
                
                # Get technical analysis data
                analysis = data_manager.get_technical_analysis(timeframe=timeframe)
                
                # Create price chart with proper data validation
                price_fig = go.Figure()
                
                if not df['open'].isnull().all():  # Validate OHLC data
                    price_fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price",
                        increasing_line_color='#26a69a', 
                        decreasing_line_color='#ef5350'
                    ))
                    
                    # Add selected indicators with validation
                    if "ma" in selected_indicators:
                        for ma_period in [20, 50, 200]:
                            ma_col = f'ma_{ma_period}'
                            if ma_col in df.columns and not df[ma_col].isnull().all():
                                price_fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df[ma_col],
                                    name=f"MA{ma_period}",
                                    line=dict(
                                        color='yellow' if ma_period == 20 else 'orange' if ma_period == 50 else 'purple',
                                        width=1
                                    )
                                ))
                    
                    if "bbands" in selected_indicators and all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                        price_fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['bb_upper'],
                            name="BB Upper",
                            line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
                            showlegend=True
                        ))
                        price_fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['bb_middle'],
                            name="BB Middle",
                            line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
                            showlegend=True
                        ))
                        price_fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['bb_lower'],
                            name="BB Lower",
                            line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(173, 216, 230, 0.1)',
                            showlegend=True
                        ))
                
                price_fig.update_layout(
                    title=dict(
                        text=f"BTC/USDT - {timeframe}",
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis_title="Time",
                    yaxis_title="Price (USDT)",
                    template="plotly_dark",
                    height=400,
                    xaxis_rangeslider_visible=False,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0.5)'
                    ),
                    margin=dict(l=50, r=50, t=50, b=50),
                    yaxis=dict(side='right')
                )
                
                # Create volume chart with validation
                volume_fig = go.Figure()
                if 'volume' in df.columns and not df['volume'].isnull().all():
                    colors = ['#ef5350' if close < open else '#26a69a' 
                             for close, open in zip(df['close'], df['open'])]
                    
                    volume_fig.add_trace(go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name="Volume",
                        marker_color=colors,
                        opacity=0.8
                    ))
                
                volume_fig.update_layout(
                    title=dict(
                        text="Volume",
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis_title="Time",
                    yaxis_title="Volume (BTC)",
                    template="plotly_dark",
                    height=200,
                    showlegend=False,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=50, r=50, t=50, b=50),
                    yaxis=dict(side='right')
                )
                
                # Create indicators display with validation
                indicators_div = html.Div([
                    html.H6("Technical Indicators", className="text-center mb-3"),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Indicator", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Signal", className="text-center")
                        ], className="table-dark")),
                        html.Tbody([
                            html.Tr([
                                html.Td("RSI (14)", className="text-center"),
                                html.Td(f"{df['rsi'].iloc[-1]:.2f}", className="text-center"),
                                html.Td(get_indicator_signal("rsi", df['rsi'].iloc[-1]), className="text-center")
                            ], className="table-dark") if "rsi" in selected_indicators and 'rsi' in df.columns else None,
                            html.Tr([
                                html.Td("MACD", className="text-center"),
                                html.Td(
                                    f"{df['macd'].iloc[-1]:.2f} / {df['macd_signal'].iloc[-1]:.2f}",
                                    className="text-center"
                                ),
                                html.Td(get_indicator_signal("macd", df['macd'].iloc[-1]), className="text-center")
                            ], className="table-dark") if "macd" in selected_indicators and all(col in df.columns for col in ['macd', 'macd_signal']) else None
                        ])
                    ], className="table table-striped table-dark table-bordered")
                ], className="mt-4")
                
                logger.info("Successfully updated analysis charts")
                return price_fig, volume_fig, indicators_div
                
            except Exception as e:
                logger.error(f"Error updating analysis charts: {str(e)}")
                return {}, {}, html.Div(f"Error: {str(e)}")
        
        @app.callback(
            Output("volume-profile-main-chart", "figure"),
            [Input("interval-medium", "n_intervals"),
             Input("volume-profile-period", "value")]
        )
        def update_volume_profile(n_intervals, timerange):
            """Update volume profile chart"""
            try:
                if not timerange:
                    timerange = "1d"
                    
                data_manager = components['data_manager']
                profile_data = data_manager.get_volume_profile(timerange=timerange)
                
                if not profile_data:
                    return {}
                
                fig = go.Figure()
                
                # Add volume profile bars
                fig.add_trace(go.Bar(
                    x=profile_data['volumes'],
                    y=profile_data['price_levels'],
                    orientation='h',
                    name="Volume Profile",
                    marker=dict(
                        color='rgba(158, 202, 225, 0.6)',
                        line=dict(color='rgba(158, 202, 225, 1.0)', width=1)
                    )
                ))
                
                # Add POC line if available
                if 'poc_price' in profile_data:
                    fig.add_hline(
                        y=profile_data['poc_price'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="POC",
                        annotation_position="top right"
                    )
                
                # Add value area if available
                if 'value_area' in profile_data:
                    fig.add_hrect(
                        y0=profile_data['value_area']['low'],
                        y1=profile_data['value_area']['high'],
                        fillcolor="rgba(255, 255, 255, 0.1)",
                        line_width=0,
                        annotation_text="Value Area"
                    )
                
                fig.update_layout(
                    title=dict(
                        text=f"Volume Profile ({timerange})",
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis_title="Volume",
                    yaxis_title="Price Level",
                    template="plotly_dark",
                    height=600,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50),
                    yaxis=dict(side='right')
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating volume profile: {str(e)}")
                return {}
        
        @app.callback(
            [Output("market-regime-chart", "figure"),
             Output("market-regime", "children"),
             Output("market-structure-chart", "figure"),
             Output("structure-analysis-results", "children"),
             Output("volatility-value", "children"),
             Output("trend-strength", "children"),
             Output("market-phase", "children"),
             Output("risk-level", "children"),
             Output("bullish-prob", "value"),
             Output("neutral-prob", "value"),
             Output("bearish-prob", "value")],
            [Input("interval-medium", "n_intervals")]
        )
        def update_market_analysis(n_intervals):
            """Update market regime and structure analysis"""
            try:
                data_manager = components['data_manager']
                
                # Get market data from data manager
                df = data_manager.get_market_data(timeframe="1h", limit=500)
                if df.empty:
                    raise ValueError("No market data available")
                
                # Get technical analysis data
                analysis = data_manager.get_technical_analysis(timeframe="1h")
                
                # Calculate market regime using technical analysis data
                regime_data = {
                    'bullish': 0,
                    'neutral': 0,
                    'bearish': 0
                }
                
                # Use multiple indicators for regime calculation
                if 'rsi' in df.columns:
                    if df['rsi'].iloc[-1] > 70:
                        regime_data['bullish'] += 1
                    elif df['rsi'].iloc[-1] < 30:
                        regime_data['bearish'] += 1
                    else:
                        regime_data['neutral'] += 1
                        
                if all(col in df.columns for col in ['ma_20', 'ma_50']):
                    if df['ma_20'].iloc[-1] > df['ma_50'].iloc[-1]:
                        regime_data['bullish'] += 1
                    elif df['ma_20'].iloc[-1] < df['ma_50'].iloc[-1]:
                        regime_data['bearish'] += 1
                    else:
                        regime_data['neutral'] += 1
                        
                if 'macd' in df.columns and 'macd_signal' in df.columns:
                    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                        regime_data['bullish'] += 1
                    elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
                        regime_data['bearish'] += 1
                    else:
                        regime_data['neutral'] += 1
                
                # Calculate probabilities
                total = sum(regime_data.values())
                if total > 0:
                    bullish_prob = (regime_data['bullish'] / total) * 100
                    neutral_prob = (regime_data['neutral'] / total) * 100
                    bearish_prob = (regime_data['bearish'] / total) * 100
                else:
                    bullish_prob = neutral_prob = bearish_prob = 33.33
                
                # Determine current regime
                current_regime = max(regime_data, key=regime_data.get)
                regime_color = {
                    'bullish': '#26a69a',
                    'neutral': '#ffa726',
                    'bearish': '#ef5350'
                }[current_regime]
                
                # Create regime chart
                regime_fig = go.Figure()
                
                # Add price and regime indicator
                regime_fig.add_trace(go.Scatter(
                    x=df.index[-100:],
                    y=df['close'][-100:],
                    name="Price",
                    line=dict(color='cyan', width=1)
                ))
                
                # Add regime zones
                if 'ma_20' in df.columns and 'ma_50' in df.columns:
                    regime_fig.add_trace(go.Scatter(
                        x=df.index[-100:],
                        y=df['ma_20'][-100:],
                        name="MA20",
                        line=dict(color='yellow', width=1)
                    ))
                    regime_fig.add_trace(go.Scatter(
                        x=df.index[-100:],
                        y=df['ma_50'][-100:],
                        name="MA50",
                        line=dict(color='orange', width=1)
                    ))
                
                regime_fig.update_layout(
                    title=dict(
                        text="Market Regime Analysis",
                        x=0.5,
                        xanchor='center'
                    ),
                    template="plotly_dark",
                    height=300,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0.5)'
                    ),
                    margin=dict(l=50, r=50, t=50, b=50),
                    yaxis=dict(side='right')
                )
                
                # Create structure chart
                structure_fig = go.Figure()
                structure_fig.add_trace(go.Candlestick(
                    x=df.index[-50:],
                    open=df['open'][-50:],
                    high=df['high'][-50:],
                    low=df['low'][-50:],
                    close=df['close'][-50:],
                    name="Price",
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ))
                
                # Add support/resistance levels if available
                if 'support_levels' in analysis and 'resistance_levels' in analysis:
                    for level in analysis['support_levels']:
                        structure_fig.add_hline(
                            y=level,
                            line_dash="dash",
                            line_color="green",
                            opacity=0.5
                        )
                    for level in analysis['resistance_levels']:
                        structure_fig.add_hline(
                            y=level,
                            line_dash="dash",
                            line_color="red",
                            opacity=0.5
                        )
                
                structure_fig.update_layout(
                    title=dict(
                        text="Market Structure",
                        x=0.5,
                        xanchor='center'
                    ),
                    template="plotly_dark",
                    height=300,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50),
                    yaxis=dict(side='right')
                )
                
                # Calculate market metrics
                volatility = f"{df['close'].pct_change().std() * 100:.2f}%"
                
                # Calculate trend strength based on multiple indicators
                trend_signals = []
                if 'adx' in df.columns:
                    trend_signals.append(1 if df['adx'].iloc[-1] > 25 else 0)
                if 'ma_20' in df.columns and 'ma_50' in df.columns:
                    trend_signals.append(1 if abs(df['ma_20'].iloc[-1] - df['ma_50'].iloc[-1]) / df['ma_50'].iloc[-1] > 0.02 else 0)
                
                trend_strength = "Strong" if sum(trend_signals) / len(trend_signals) > 0.7 else "Moderate"
                
                # Determine market phase
                if 'ma_20' in df.columns and 'ma_50' in df.columns and 'volume' in df.columns:
                    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
                    price_trend = df['close'].iloc[-1] > df['ma_50'].iloc[-1]
                    vol_trend = df['volume'].iloc[-1] > vol_avg
                    
                    if price_trend and vol_trend:
                        market_phase = "Accumulation"
                    elif not price_trend and vol_trend:
                        market_phase = "Distribution"
                    elif price_trend:
                        market_phase = "Uptrend"
                    else:
                        market_phase = "Downtrend"
                else:
                    market_phase = "Unknown"
                
                # Calculate risk level
                vol_risk = df['close'].pct_change().std() * 100
                if vol_risk > 3:
                    risk_level = html.Span("High", style={'color': '#ef5350'})
                elif vol_risk > 1.5:
                    risk_level = html.Span("Moderate", style={'color': '#ffa726'})
                else:
                    risk_level = html.Span("Low", style={'color': '#26a69a'})
                
                # Create structure analysis results
                structure_analysis = html.Div([
                    html.P(f"Market Phase: {market_phase}", className="mb-1"),
                    html.P(f"Trend Direction: {current_regime.capitalize()}", className="mb-1"),
                    html.P(f"Trend Strength: {trend_strength}", className="mb-1")
                ])
                
                return (
                    regime_fig,
                    html.Span(current_regime.capitalize(), style={'color': regime_color}),
                    structure_fig,
                    structure_analysis,
                    volatility,
                    trend_strength,
                    market_phase,
                    risk_level,
                    bullish_prob,
                    neutral_prob,
                    bearish_prob
                )
                
            except Exception as e:
                logger.error(f"Error updating market analysis: {str(e)}")
                return {}, "N/A", {}, html.Div("Error"), "N/A", "N/A", "N/A", "N/A", 0, 0, 0
        
        @app.callback(
            [Output("backtest-results-chart", "figure"),
             Output("trade-distribution-chart", "figure"),
             Output("monthly-performance-chart", "figure"),
             Output("pattern-win-rate", "children"),
             Output("pattern-avg-return", "children"),
             Output("pattern-total-trades", "children"),
             Output("pattern-profit-factor", "children"),
             Output("pattern-sharpe-ratio", "children"),
             Output("pattern-max-drawdown", "children"),
             Output("pattern-recovery-factor", "children"),
             Output("pattern-expectancy", "children"),
             Output("backtest-statistics", "children")],
            [Input("backtest-pattern-type", "value"),
             Input("backtest-date-range", "start_date"),
             Input("backtest-date-range", "end_date"),
             Input("backtest-stop-loss", "value"),
             Input("backtest-take-profit", "value"),
             Input("backtest-position-size", "value"),
             Input("backtest-max-trades", "value"),
             Input("market-conditions", "value")]
        )
        def update_backtest_results(pattern_type, start_date, end_date, stop_loss, take_profit, position_size, max_trades, market_conditions):
            """Update pattern backtesting results"""
            try:
                if not all([pattern_type, stop_loss, take_profit, position_size, max_trades]):
                    raise PreventUpdate
                
                data_manager = components['data_manager']
                
                # Get historical data for backtesting
                df = data_manager.get_market_data(timeframe="1h", limit=1000)
                if df.empty:
                    raise ValueError("No market data available for backtesting")
                
                # Initialize results dictionary
                results = {
                    'trades': [],
                    'returns': [],
                    'equity_curve': [100],  # Start with 100 as initial capital
                    'monthly_returns': {},
                    'statistics': {}
                }
                
                # Pattern detection based on type
                patterns = []
                if pattern_type == "candlestick":
                    patterns = data_manager.analyze_patterns(timeframe="1h")
                elif pattern_type == "chart":
                    # Detect chart patterns
                    if 'ma_20' in df.columns and 'ma_50' in df.columns:
                        for i in range(len(df)-1):
                            if df['ma_20'].iloc[i] < df['ma_50'].iloc[i] and df['ma_20'].iloc[i+1] > df['ma_50'].iloc[i+1]:
                                patterns.append({
                                    'signal': 'buy',
                                    'index': i+1,
                                    'type': 'Golden Cross'
                                })
                
                # Apply market conditions filter
                if market_conditions:
                    filtered_patterns = []
                    for pattern in patterns:
                        idx = pattern['index']
                        if idx >= len(df):
                            continue
                            
                        # Check trend condition
                        if 'trend' in market_conditions:
                            if 'ma_50' in df.columns:
                                is_trend = df['close'].iloc[idx] > df['ma_50'].iloc[idx]
                                if not is_trend:
                                    continue
                
                        # Check volatility condition
                        if 'volatile' in market_conditions:
                            volatility = df['close'].pct_change().rolling(20).std().iloc[idx] * 100
                            if volatility < 2:  # Less than 2% daily volatility
                                continue
                
                        filtered_patterns.append(pattern)
                    patterns = filtered_patterns
                
                # Simulate trades
                current_trades = 0
                equity = 100  # Initial capital
                
                for pattern in patterns:
                    if current_trades >= max_trades:
                        break
                        
                    idx = pattern['index']
                    if idx >= len(df):
                        continue
                        
                    entry_price = df['close'].iloc[idx]
                    sl_price = entry_price * (1 - stop_loss/100)
                    tp_price = entry_price * (1 + take_profit/100)
                    
                    # Simulate trade
                    trade = simulate_trade(df, idx, entry_price, sl_price, tp_price, position_size)
                    if trade:
                        results['trades'].append(trade)
                        current_trades += 1
                        
                        # Update equity curve
                        trade_return = trade['return'] * (position_size/100)
                        equity *= (1 + trade_return/100)
                        results['equity_curve'].append(equity)
                        results['returns'].append(trade_return)
                        
                        # Update monthly returns
                        month = trade['exit_time'].strftime('%Y-%m')
                        if month not in results['monthly_returns']:
                            results['monthly_returns'][month] = 0
                        results['monthly_returns'][month] += trade_return
                
                # Calculate performance metrics
                if results['trades']:
                    returns = results['returns']
                    results['statistics'] = {
                        'total_trades': len(results['trades']),
                        'win_rate': sum(1 for r in returns if r > 0) / len(returns) * 100,
                        'avg_return': sum(returns) / len(returns),
                        'profit_factor': abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if sum(r for r in returns if r < 0) != 0 else float('inf'),
                        'max_drawdown': calculate_max_drawdown(results['equity_curve']),
                        'sharpe_ratio': calculate_sharpe_ratio(returns),
                        'recovery_factor': calculate_recovery_factor(results['equity_curve']),
                        'expectancy': calculate_expectancy(returns)
                    }
                    
                    # Create results charts
                    results_fig = go.Figure()
                    
                    # Add equity curve
                    results_fig.add_trace(go.Scatter(
                        x=df.index[:len(results['equity_curve'])],
                        y=results['equity_curve'],
                        name="Equity Curve",
                        line=dict(color='cyan', width=2)
                    ))
                    
                    results_fig.update_layout(
                        title=dict(
                            text="Backtest Results",
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title="Date",
                        yaxis_title="Equity",
                        template="plotly_dark",
                        height=400,
                        showlegend=True,
                        margin=dict(l=50, r=50, t=50, b=50),
                        yaxis=dict(side='right')
                    )
                    
                    # Create trade distribution chart
                    dist_fig = go.Figure()
                    dist_fig.add_trace(go.Histogram(
                        x=returns,
                        nbinsx=20,
                        name="Trade Returns",
                        marker_color='cyan'
                    ))
                    
                    dist_fig.update_layout(
                        title=dict(
                            text="Trade Distribution",
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=200,
                        showlegend=False,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    
                    # Create monthly performance chart
                    monthly_fig = go.Figure()
                    months = list(results['monthly_returns'].keys())
                    returns_values = list(results['monthly_returns'].values())
                    
                    monthly_fig.add_trace(go.Bar(
                        x=months,
                        y=returns_values,
                        name="Monthly Returns",
                        marker_color=['#26a69a' if x > 0 else '#ef5350' for x in returns_values]
                    ))
                    
                    monthly_fig.update_layout(
                        title=dict(
                            text="Monthly Performance",
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title="Month",
                        yaxis_title="Return (%)",
                        template="plotly_dark",
                        height=200,
                        showlegend=False,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    
                    # Create statistics summary
                    stats = results['statistics']
                    statistics_div = html.Div([
                        html.H6("Detailed Statistics", className="mb-3"),
                        html.Table([
                            html.Tbody([
                                html.Tr([
                                    html.Td("Total Trades:", className="pe-3"),
                                    html.Td(f"{stats['total_trades']}", className="text-end")
                                ]),
                                html.Tr([
                                    html.Td("Win Rate:", className="pe-3"),
                                    html.Td(f"{stats['win_rate']:.2f}%", className="text-end")
                                ]),
                                html.Tr([
                                    html.Td("Average Return:", className="pe-3"),
                                    html.Td(f"{stats['avg_return']:.2f}%", className="text-end")
                                ]),
                                html.Tr([
                                    html.Td("Profit Factor:", className="pe-3"),
                                    html.Td(f"{stats['profit_factor']:.2f}", className="text-end")
                                ])
                            ])
                        ], className="table table-sm table-dark")
                    ])
                    
                    return (
                        results_fig,
                        dist_fig,
                        monthly_fig,
                        f"{stats['win_rate']:.1f}%",
                        f"{stats['avg_return']:.2f}%",
                        str(stats['total_trades']),
                        f"{stats['profit_factor']:.2f}",
                        f"{stats['sharpe_ratio']:.2f}",
                        f"{stats['max_drawdown']:.2f}%",
                        f"{stats['recovery_factor']:.2f}",
                        f"{stats['expectancy']:.2f}",
                        statistics_div
                    )
                else:
                    raise ValueError("No trades generated during backtest")
                
            except Exception as e:
                logger.error(f"Error updating backtest results: {str(e)}")
                return {}, {}, {}, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", html.Div("Error calculating backtest results")
        
        logger.info("Successfully registered analysis callbacks")
        
    except Exception as e:
        logger.error(f"Error registering analysis callbacks: {str(e)}")
        raise

def get_indicator_signal(indicator, value):
    """Get indicator signal with styling"""
    if indicator == "rsi":
        if value > 70:
            return html.Span("Overbought", style={'color': '#ef5350'})  # Red
        elif value < 30:
            return html.Span("Oversold", style={'color': '#26a69a'})    # Green
        return html.Span("Neutral", style={'color': '#ffffff'})         # White
    elif indicator == "macd":
        if value > 0:
            return html.Span("Bullish", style={'color': '#26a69a'})     # Green
        elif value < 0:
            return html.Span("Bearish", style={'color': '#ef5350'})     # Red
        return html.Span("Neutral", style={'color': '#ffffff'})         # White
    return "N/A"

def simulate_trade(df, entry_index, entry_price, sl_price, tp_price, position_size):
    """Simulate a single trade"""
    try:
        # Get future prices
        future_prices = df['close'].iloc[entry_index+1:]
        if future_prices.empty:
            return None
        
        # Initialize trade
        trade = {
            'entry_price': entry_price,
            'entry_time': df.index[entry_index],
            'position_size': position_size
        }
        
        # Simulate price movement
        for i, price in enumerate(future_prices):
            if price <= sl_price:
                trade['exit_price'] = price
                trade['exit_time'] = future_prices.index[i]
                trade['return'] = ((price - entry_price) / entry_price) * 100
                return trade
            elif price >= tp_price:
                trade['exit_price'] = price
                trade['exit_time'] = future_prices.index[i]
                trade['return'] = ((price - entry_price) / entry_price) * 100
                return trade
        
        # If no SL or TP hit, close at last price
        trade['exit_price'] = future_prices.iloc[-1]
        trade['exit_time'] = future_prices.index[-1]
        trade['return'] = ((future_prices.iloc[-1] - entry_price) / entry_price) * 100
        return trade
        
    except Exception as e:
        logger.error(f"Error simulating trade: {str(e)}")
        return None

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    try:
        if not equity_curve:
            return 0
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {str(e)}")
        return 0

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    try:
        if not returns:
            return 0
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0

def calculate_recovery_factor(equity_curve):
    """Calculate recovery factor"""
    try:
        if not equity_curve:
            return 0
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        max_dd = calculate_max_drawdown(equity_curve)
        if max_dd == 0:
            return float('inf')
        return total_return / max_dd
    except Exception as e:
        logger.error(f"Error calculating recovery factor: {str(e)}")
        return 0

def calculate_expectancy(returns):
    """Calculate system expectancy"""
    try:
        if not returns:
            return 0
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not wins or not losses:
            return 0
            
        win_rate = len(wins) / len(returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    except Exception as e:
        logger.error(f"Error calculating expectancy: {str(e)}")
        return 0