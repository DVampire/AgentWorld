"""Tracer visualization module for displaying agent execution records."""

import json
import threading
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from flask import Flask, render_template, jsonify
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


class TracerVisualizer:
    """Visualizer for tracer JSON files."""
    
    def __init__(self, tracer_json_path: str, port: int = 5000):
        """Initialize the visualizer.
        
        Args:
            tracer_json_path: Path to the tracer.json file
            port: Port number for the web server (default: 5000)
        """
        self.tracer_json_path = Path(tracer_json_path)
        self.port = port
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / "templates"),
                        static_folder=str(Path(__file__).parent / "static"))
        self.records: List[Dict[str, Any]] = []
        self._data_lock = threading.Lock()
        self._stop_reload_thread = threading.Event()
        self._setup_routes()
        self._load_data()
        
        # Start auto-reload thread (reload every 60 seconds)
        self._reload_thread = threading.Thread(target=self._auto_reload_data, daemon=True)
        self._reload_thread.start()
    
    def _load_data(self):
        """Load data from tracer.json file."""
        if not self.tracer_json_path.exists():
            print(f"Warning: Tracer JSON file not found: {self.tracer_json_path}")
            return
        
        try:
            with open(self.tracer_json_path, 'r', encoding='utf-8') as f:
                new_records = json.load(f)
            
            # Update records with thread lock
            with self._data_lock:
                self.records = new_records
            
            print(f"Data loaded: {len(self.records)} records from {self.tracer_json_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _auto_reload_data(self):
        """Auto-reload data every 60 seconds in background thread."""
        while not self._stop_reload_thread.is_set():
            time.sleep(60)  # Wait 60 seconds
            if not self._stop_reload_thread.is_set():
                print("Auto-reloading data...")
                self._load_data()
    
    def _extract_account_value(self, record: Dict[str, Any]) -> Optional[float]:
        """Extract account value from a record."""
        try:
            observation = record.get("observation", {})
            hyperliquid = observation.get("hyperliquid", {})
            account = hyperliquid.get("account", {})
            if not isinstance(account, dict):
                return None
            
            # New tracer format exposes account_value directly
            direct_value = account.get("account_value")
            if direct_value is not None:
                return float(direct_value)
            
            # Fallback to nested margin summary
            account_inner = account.get("account", {})
            if isinstance(account_inner, dict):
                margin_summary = account_inner.get("margin_summary", {})
                account_value = margin_summary.get("accountValue")
                if account_value is not None:
                    return float(account_value)
        except (KeyError, ValueError, TypeError):
            pass
        return None
    
    def _extract_actions(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actions from a record."""
        actions = []
        
        # Try different possible structures
        action_data = record.get("action", {})
        if isinstance(action_data, dict):
            action_list = action_data.get("action", [])
            thinking = action_data.get("thinking")
        else:
            action_list = []
            thinking = None
        
        # If action_list is empty, try direct access
        if not action_list and isinstance(action_data, list):
            action_list = action_data
        
        for action_item in action_list:
            if isinstance(action_item, dict):
                action_name = action_item.get("name", "")
                if action_name == "step":
                    args = action_item.get("args", {})
                    if isinstance(args, dict):
                        action_type = args.get("action", "").upper()
                        if action_type in ["LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT"]:
                            actions.append({
                                "type": action_type,
                                "symbol": args.get("symbol", ""),
                                "qty": args.get("qty", 0),
                                "leverage": args.get("leverage", 1),
                                "stop_loss_price": args.get("stop_loss_price"),
                                "take_profit_price": args.get("take_profit_price"),
                                "thinking": thinking,
                                "all_actions": action_list
                            })
        
        return actions
    
    def _calculate_returns(self, account_values: List[float]) -> List[float]:
        """Calculate returns from account values."""
        if not account_values or len(account_values) < 2:
            return [0.0] * len(account_values)
        
        initial_value = account_values[0]
        returns = []
        
        for value in account_values:
            if initial_value > 0:
                return_pct = ((value - initial_value) / initial_value) * 100
            else:
                return_pct = 0.0
            returns.append(return_pct)
        
        return returns
    
    def _extract_crypto_prices(self, record: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Extract cryptocurrency prices from a record (latest candle close per symbol)."""
        prices: Dict[str, Optional[float]] = {}
        try:
            observation = record.get("observation", {})
            hyperliquid = observation.get("hyperliquid", {})
            input_data = hyperliquid.get("input", {})
            data = input_data.get("data", {})
            
            if not isinstance(data, dict):
                return prices
            
            for symbol, symbol_data in data.items():
                if not isinstance(symbol_data, dict):
                    continue
                
                candles = symbol_data.get("candles")
                if candles is None:
                    candles = symbol_data.get("candle")
                if not isinstance(candles, list) or not candles:
                    continue
                
                # tracer stores candles in chronological order; take the last (latest) closed candle
                latest_candle = candles[-1] if isinstance(candles[-1], dict) else None
                if not isinstance(latest_candle, dict):
                    continue
                
                close_price = latest_candle.get("close")
                if close_price is None:
                    continue
                
                try:
                    prices[symbol] = float(close_price)
                except (ValueError, TypeError):
                    prices[symbol] = None
        except (KeyError, ValueError, TypeError):
            pass
        
        return prices
    
    def _prepare_chart_data(self) -> Dict[str, Any]:
        """Prepare data for charts."""
        timestamps = []
        account_values = []
        action_points = []  # List of (index, action_info)
        returns = []
        crypto_prices = {}  # Dict[symbol] -> List[price]
        crypto_symbols = set()  # Track all symbols
        
        # Create a copy of records with thread lock
        with self._data_lock:
            records_copy = self.records.copy()
        
        for idx, record in enumerate(records_copy):
            timestamp = record.get("timestamp", "")
            account_value = self._extract_account_value(record)
            
            if account_value is not None:
                timestamps.append(timestamp)
                account_values.append(account_value)
                
                # Extract cryptocurrency prices
                prices = self._extract_crypto_prices(record)
                for symbol, price in prices.items():
                    if symbol not in crypto_prices:
                        crypto_prices[symbol] = []
                    crypto_symbols.add(symbol)
                    crypto_prices[symbol].append(price)
                
                # Ensure all symbols have the same length
                for symbol in crypto_symbols:
                    if symbol not in prices:
                        # If this record doesn't have this symbol, append None
                        if symbol in crypto_prices:
                            crypto_prices[symbol].append(None)
                
                # Extract actions
                actions = self._extract_actions(record)
                if actions:
                    # Check if any action is LONG, SHORT, CLOSE_LONG, or CLOSE_SHORT (not all HOLD)
                    has_trading_action = any(a["type"] in ["LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT"] for a in actions)
                    if has_trading_action:
                        # Get thinking from action_data
                        action_data = record.get("action", {})
                        thinking = action_data.get("thinking") if isinstance(action_data, dict) else None
                        
                        action_points.append({
                            "index": len(account_values) - 1,
                            "timestamp": timestamp,
                            "account_value": account_value,
                            "actions": actions,
                            "thinking": thinking  # Add thinking to action point
                        })
        
        # Calculate returns
        if account_values:
            returns = self._calculate_returns(account_values)
        
        # Debug output
        print(f"Prepared chart data: {len(timestamps)} timestamps, {len(account_values)} account values, {len(action_points)} action points")
        print(f"Crypto symbols found: {sorted(crypto_symbols)}")
        
        return {
            "timestamps": timestamps,
            "account_values": account_values,
            "action_points": action_points,
            "returns": returns,
            "crypto_prices": crypto_prices,
            "crypto_symbols": sorted(crypto_symbols)
        }
    
    def _create_account_value_chart(self, data: Dict[str, Any]) -> str:
        """Create account value chart with action markers."""
        fig = go.Figure()
        
        # Check if we have data
        if not data["timestamps"] or not data["account_values"]:
            # Return empty chart with error message
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(title='Account Value Over Time - No Data')
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        # Main account value line (left Y-axis)
        fig.add_trace(go.Scatter(
            x=data["timestamps"],
            y=data["account_values"],
            mode='lines',
            name='Account Value',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Time:</b> %{x}<br><b>Account Value:</b> $%{y:,.2f}<extra></extra>',
            yaxis='y'
        ))
        
        # Add cryptocurrency price lines (right Y-axis) - normalized as percentage change
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        if data.get("crypto_prices") and data.get("crypto_symbols"):
            for idx, symbol in enumerate(data["crypto_symbols"]):
                prices = data["crypto_prices"].get(symbol, [])
                if prices and any(p is not None for p in prices):
                    # Find first valid price as baseline
                    baseline_price = None
                    for price in prices:
                        if price is not None:
                            baseline_price = price
                            break
                    
                    if baseline_price and baseline_price > 0:
                        # Convert to percentage change from baseline
                        price_changes = []
                        original_prices = []
                        for price in prices:
                            if price is not None:
                                pct_change = ((price - baseline_price) / baseline_price) * 100
                                price_changes.append(pct_change)
                                original_prices.append(price)
                            else:
                                price_changes.append(None)
                                original_prices.append(None)
                        
                        color = colors[idx % len(colors)]
                        # Create custom hover data with both percentage and original price
                        hover_texts = []
                        for i, (pct, orig, ts) in enumerate(zip(price_changes, original_prices, data["timestamps"])):
                            if pct is not None and orig is not None:
                                hover_texts.append(
                                    f"<b>Time:</b> {ts}<br>"
                                    f"<b>{symbol} Price:</b> ${orig:,.2f}<br>"
                                    f"<b>Change:</b> {pct:+.2f}%"
                                )
                            else:
                                hover_texts.append(None)
                        
                        fig.add_trace(go.Scatter(
                            x=data["timestamps"],
                            y=price_changes,
                            mode='lines',
                            name=f'{symbol} Price',
                            line=dict(color=color, width=1.5, dash='dot'),
                            customdata=original_prices,  # Store original prices for hover
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=hover_texts,
                            yaxis='y2',
                            visible=True,  # Visible by default
                            connectgaps=False  # Don't connect gaps where price is None
                        ))
        
        # Add action markers
        long_x = []
        long_y = []
        long_data = []
        short_x = []
        short_y = []
        short_data = []
        close_long_x = []
        close_long_y = []
        close_long_data = []
        close_short_x = []
        close_short_y = []
        close_short_data = []
        
        for point in data["action_points"]:
            timestamp = point["timestamp"]
            account_value = point["account_value"]
            actions = point["actions"]
            
            # Create hover text
            hover_text = f"<b>Time:</b> {timestamp}<br>"
            hover_text += f"<b>Account Value:</b> ${account_value:,.2f}<br>"
            hover_text += "<b>Actions:</b><br>"
            for action in actions:
                action_type = action['type']
                symbol = action['symbol']
                if action_type in ['LONG', 'SHORT']:
                    hover_text += f"  • {action_type} {symbol} (qty: {action.get('qty', 0)}, leverage: {action.get('leverage', 1)}x)<br>"
                else:
                    hover_text += f"  • {action_type} {symbol}<br>"
            
            # Separate LONG, SHORT, CLOSE_LONG, and CLOSE_SHORT actions
            has_long = any(a["type"] == "LONG" for a in actions)
            has_short = any(a["type"] == "SHORT" for a in actions)
            has_close_long = any(a["type"] == "CLOSE_LONG" for a in actions)
            has_close_short = any(a["type"] == "CLOSE_SHORT" for a in actions)
            
            if has_long:
                long_x.append(timestamp)
                long_y.append(account_value)
                long_data.append(point)
            
            if has_short:
                short_x.append(timestamp)
                short_y.append(account_value)
                short_data.append(point)
            
            if has_close_long:
                close_long_x.append(timestamp)
                close_long_y.append(account_value)
                close_long_data.append(point)
            
            if has_close_short:
                close_short_x.append(timestamp)
                close_short_y.append(account_value)
                close_short_data.append(point)
        
        # Add LONG markers
        if long_x:
            # Create hover text for each point
            long_hover = []
            for point in long_data:
                hover_text = f"<b>Time:</b> {point['timestamp']}<br>"
                hover_text += f"<b>Account Value:</b> ${point['account_value']:,.2f}<br>"
                hover_text += "<b>Actions:</b><br>"
                for action in point['actions']:
                    if action['type'] == 'LONG':
                        hover_text += f"  • {action['type']} {action['symbol']} (qty: {action['qty']}, leverage: {action['leverage']}x)<br>"
                hover_text += "<b>Click for details</b>"
                long_hover.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=long_x,
                y=long_y,
                mode='markers',
                name='LONG Action',
                marker=dict(
                    size=12,
                    color='green',
                    symbol='triangle-up',
                    line=dict(width=2, color='white')
                ),
                customdata=long_data,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=long_hover,
                showlegend=True
            ))
        
        # Add SHORT markers
        if short_x:
            # Create hover text for each point
            short_hover = []
            for point in short_data:
                hover_text = f"<b>Time:</b> {point['timestamp']}<br>"
                hover_text += f"<b>Account Value:</b> ${point['account_value']:,.2f}<br>"
                hover_text += "<b>Actions:</b><br>"
                for action in point['actions']:
                    if action['type'] == 'SHORT':
                        hover_text += f"  • {action['type']} {action['symbol']} (qty: {action['qty']}, leverage: {action['leverage']}x)<br>"
                hover_text += "<b>Click for details</b>"
                short_hover.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=short_x,
                y=short_y,
                mode='markers',
                name='SHORT Action',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='triangle-down',
                    line=dict(width=2, color='white')
                ),
                customdata=short_data,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=short_hover,
                showlegend=True
            ))
        
        # Add CLOSE_LONG markers
        if close_long_x:
            # Create hover text for each point
            close_long_hover = []
            for point in close_long_data:
                hover_text = f"<b>Time:</b> {point['timestamp']}<br>"
                hover_text += f"<b>Account Value:</b> ${point['account_value']:,.2f}<br>"
                hover_text += "<b>Actions:</b><br>"
                for action in point['actions']:
                    if action['type'] == 'CLOSE_LONG':
                        hover_text += f"  • {action['type']} {action['symbol']}<br>"
                hover_text += "<b>Click for details</b>"
                close_long_hover.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=close_long_x,
                y=close_long_y,
                mode='markers',
                name='CLOSE_LONG Action',
                marker=dict(
                    size=12,
                    color='lightgreen',
                    symbol='square',
                    line=dict(width=2, color='white')
                ),
                customdata=close_long_data,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=close_long_hover,
                showlegend=True
            ))
        
        # Add CLOSE_SHORT markers
        if close_short_x:
            # Create hover text for each point
            close_short_hover = []
            for point in close_short_data:
                hover_text = f"<b>Time:</b> {point['timestamp']}<br>"
                hover_text += f"<b>Account Value:</b> ${point['account_value']:,.2f}<br>"
                hover_text += "<b>Actions:</b><br>"
                for action in point['actions']:
                    if action['type'] == 'CLOSE_SHORT':
                        hover_text += f"  • {action['type']} {action['symbol']}<br>"
                hover_text += "<b>Click for details</b>"
                close_short_hover.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=close_short_x,
                y=close_short_y,
                mode='markers',
                name='CLOSE_SHORT Action',
                marker=dict(
                    size=12,
                    color='lightcoral',
                    symbol='square',
                    line=dict(width=2, color='white')
                ),
                customdata=close_short_data,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=close_short_hover,
                showlegend=True
            ))
        
        # Create layout with dual Y-axes
        layout = dict(
            title='Account Value & Cryptocurrency Prices Over Time',
            xaxis_title='Time',
            yaxis=dict(
                title=dict(text='Account Value ($)', font=dict(color='blue')),
                tickfont=dict(color='blue'),
                side='left'
            ),
            yaxis2=dict(
                title=dict(text='Price Change (%)', font=dict(color='orange')),
                tickfont=dict(color='orange'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            hovermode='closest',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(b=100)  # Add bottom margin for horizontal legend
        )
        
        fig.update_layout(**layout)
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_returns_chart(self, data: Dict[str, Any]) -> str:
        """Create returns chart."""
        fig = go.Figure()
        
        # Check if we have data
        if not data["timestamps"] or not data["returns"]:
            # Return empty chart with error message
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(title='Returns Over Time - No Data')
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        fig.add_trace(go.Scatter(
            x=data["timestamps"],
            y=data["returns"],
            mode='lines',
            name='Returns (%)',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
            hovertemplate='<b>Time:</b> %{x}<br><b>Returns:</b> %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='Returns Over Time',
            xaxis_title='Time',
            yaxis_title='Returns (%)',
            hovermode='closest',
            height=500,
            template='plotly_white'
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_crypto_prices_chart(self, data: Dict[str, Any]) -> str:
        """Create cryptocurrency prices chart."""
        fig = go.Figure()
        
        # Check if we have data
        if not data["timestamps"] or not data["crypto_symbols"]:
            # Return empty chart with error message
            fig.add_annotation(
                text="No cryptocurrency price data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(title='Cryptocurrency Prices - No Data')
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        # Add a trace for each cryptocurrency
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for idx, symbol in enumerate(data["crypto_symbols"]):
            prices = data["crypto_prices"].get(symbol, [])
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=data["timestamps"],
                y=prices,
                mode='lines',
                name=symbol,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>Time:</b> %{{x}}<br><b>{symbol} Price:</b> $%{{y:,.2f}}<extra></extra>',
                connectgaps=False  # Don't connect gaps where price is None
            ))
        
        fig.update_layout(
            title='Cryptocurrency Prices Over Time',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            hovermode='closest',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html')
        
        @self.app.route('/api/data')
        def get_data():
            """Get chart data."""
            try:
                # Reload data on each request (when page refreshes)
                self._load_data()
                
                data = self._prepare_chart_data()
                
                # Check if we have data
                if not data["timestamps"]:
                    return jsonify({
                        "error": "No data available",
                        "message": "No account values found in records"
                    }), 400
                
                account_value_chart = self._create_account_value_chart(data)
                returns_chart = self._create_returns_chart(data)
                crypto_prices_chart = self._create_crypto_prices_chart(data)
                
                return jsonify({
                    "account_value_chart": account_value_chart,
                    "returns_chart": returns_chart,
                    "crypto_prices_chart": crypto_prices_chart,
                    "action_points": data["action_points"]
                })
            except Exception as e:
                import traceback
                error_msg = f"Error preparing chart data: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return jsonify({
                    "error": "Failed to prepare chart data",
                    "message": str(e)
                }), 500
        
        @self.app.route('/api/action/<int:point_index>')
        def get_action_details(point_index: int):
            """Get action details for a specific point."""
            data = self._prepare_chart_data()
            if 0 <= point_index < len(data["action_points"]):
                point = data["action_points"][point_index]
                return jsonify({
                    "timestamp": point["timestamp"],
                    "account_value": point["account_value"],
                    "actions": point["actions"],
                    "thinking": point["actions"][0]["thinking"] if point["actions"] else None
                })
            return jsonify({"error": "Invalid point index"}), 404
    
    def stop(self):
        """Stop the auto-reload thread."""
        self._stop_reload_thread.set()
        if self._reload_thread.is_alive():
            self._reload_thread.join(timeout=2.0)
    
    def run(self, debug: bool = False):
        """Run the web server."""
        print(f"Starting visualization server on http://localhost:{self.port}")
        print(f"Open your browser and navigate to http://localhost:{self.port}")
        print(f"Auto-reload enabled: data will refresh every 60 seconds")
        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=debug)
        finally:
            self.stop()

