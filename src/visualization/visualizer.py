"""Tracer visualization module for displaying agent execution records."""

import json
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
        self._setup_routes()
        self._load_data()
    
    def _load_data(self):
        """Load data from tracer.json file."""
        if not self.tracer_json_path.exists():
            raise FileNotFoundError(f"Tracer JSON file not found: {self.tracer_json_path}")
        
        with open(self.tracer_json_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
    
    def _extract_account_value(self, record: Dict[str, Any]) -> Optional[float]:
        """Extract account value from a record."""
        try:
            observation = record.get("observation", {})
            hyperliquid = observation.get("hyperliquid", {})
            account = hyperliquid.get("account", {})
            # Account structure: account.account.margin_summary.accountValue
            if isinstance(account, dict):
                account_inner = account.get("account", {})
                if isinstance(account_inner, dict):
                    margin_summary = account_inner.get("margin_summary", {})
                    account_value = margin_summary.get("accountValue")
                    if account_value:
                        return float(account_value)
        except (KeyError, ValueError, TypeError) as e:
            # Debug: print error for troubleshooting
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
                        if action_type in ["LONG", "SHORT"]:
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
        """Extract cryptocurrency prices from a record.
        
        Returns:
            Dictionary mapping symbol to close price (latest candle)
        """
        prices = {}
        try:
            observation = record.get("observation", {})
            hyperliquid = observation.get("hyperliquid", {})
            input_data = hyperliquid.get("input", {})
            data = input_data.get("data", {})
            
            if isinstance(data, dict):
                for symbol, symbol_data in data.items():
                    if isinstance(symbol_data, dict):
                        candles = symbol_data.get("candle", [])
                        if candles and isinstance(candles, list) and len(candles) > 0:
                            # Get the latest candle (first one in the list)
                            latest_candle = candles[0]
                            if isinstance(latest_candle, dict):
                                close_price = latest_candle.get("close")
                                if close_price is not None:
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
        
        for idx, record in enumerate(self.records):
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
                    # Check if any action is LONG or SHORT (not all HOLD)
                    has_trading_action = any(a["type"] in ["LONG", "SHORT"] for a in actions)
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
        
        # Main account value line
        fig.add_trace(go.Scatter(
            x=data["timestamps"],
            y=data["account_values"],
            mode='lines',
            name='Account Value',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Time:</b> %{x}<br><b>Account Value:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Add action markers
        long_x = []
        long_y = []
        long_data = []
        short_x = []
        short_y = []
        short_data = []
        
        for point in data["action_points"]:
            timestamp = point["timestamp"]
            account_value = point["account_value"]
            actions = point["actions"]
            
            # Create hover text
            hover_text = f"<b>Time:</b> {timestamp}<br>"
            hover_text += f"<b>Account Value:</b> ${account_value:,.2f}<br>"
            hover_text += "<b>Actions:</b><br>"
            for action in actions:
                hover_text += f"  • {action['type']} {action['symbol']} (qty: {action['qty']}, leverage: {action['leverage']}x)<br>"
            
            # Separate LONG and SHORT actions
            has_long = any(a["type"] == "LONG" for a in actions)
            has_short = any(a["type"] == "SHORT" for a in actions)
            
            if has_long:
                long_x.append(timestamp)
                long_y.append(account_value)
                long_data.append(point)
            
            if has_short:
                short_x.append(timestamp)
                short_y.append(account_value)
                short_data.append(point)
        
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
        
        fig.update_layout(
            title='Account Value Over Time',
            xaxis_title='Time',
            yaxis_title='Account Value ($)',
            hovermode='closest',
            height=500,
            template='plotly_white'
        )
        
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
    
    def run(self, debug: bool = False):
        """Run the web server."""
        print(f"Starting visualization server on http://localhost:{self.port}")
        print(f"Open your browser and navigate to http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

