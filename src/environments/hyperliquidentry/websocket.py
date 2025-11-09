"""Hyperliquid WebSocket implementation for real-time data streaming."""
import json
import threading
import websocket
import time
from datetime import datetime, timezone
from typing import Dict, Callable, Optional, List, Literal
from src.logger import logger


class HyperliquidWebSocket:
    """Hyperliquid WebSocket client for minute-level candle, trades, and l2Book streaming.
    
    This WebSocket is designed for minute-level (1m) candle data streaming.
    All candle subscriptions default to 1-minute intervals.
    """
    
    def __init__(
        self,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_open: Optional[Callable] = None,
        testnet: bool = False
    ):
        """Initialize Hyperliquid WebSocket client for minute-level data streaming.
        
        Args:
            on_message: Callback function for messages (ws, channel, data)
            on_error: Callback function for errors (ws, error)
            on_close: Callback function for close events (ws)
            on_open: Callback function for open events (ws)
            testnet: Whether to use testnet
        """
        self.testnet = testnet
        # Hyperliquid WebSocket URLs
        if testnet:
            self.base_url = "wss://api.hyperliquid-testnet.xyz/ws"
        else:
            self.base_url = "wss://api.hyperliquid.xyz/ws"
        
        self.on_message_callback = on_message
        self.on_error_callback = on_error
        self.on_close_callback = on_close
        self.on_open_callback = on_open
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._should_reconnect = True  # Flag to control reconnection
        self._reconnect_delay = 5  # Delay before reconnecting (seconds)
        self._max_reconnect_attempts = 10  # Maximum reconnection attempts
        self._reconnect_attempts = 0  # Current reconnection attempts
        # Default interval for minute-level streaming
        self._default_interval = "1m"
        # Store subscriptions: {channel_type: [symbols]}
        self._subscribed_candles: List[str] = []  # candle subscriptions (1m only)
        self._subscribed_trades: List[str] = []  # trades subscriptions
        self._subscribed_l2books: List[str] = []  # l2Book subscriptions
    
    def _on_message(self, ws, message):
        """Internal message handler - processes candle, trades, and l2Book data."""
        try:
            # Parse message
            if isinstance(message, str):
                msg = json.loads(message)
            else:
                msg = message
            
            # Log all received messages for debugging (show full message)
            logger.info(f"| 📨 Raw WebSocket message received: {json.dumps(msg, indent=2) if isinstance(msg, dict) else str(msg)}")
            
            if not isinstance(msg, dict):
                logger.warning(f"| 📊 Received non-dict message: {type(msg)}")
                return
            
            channel = msg.get("channel")
            data = msg.get("data")
            
            # Handle subscription confirmation
            if channel == "subscriptionResponse":
                subscription = data.get("subscription", {}) if data else {}
                coin = subscription.get("coin", "unknown")
                sub_type = subscription.get("type", "unknown")
                interval = subscription.get("interval", "unknown")
                logger.info(f"| ✅ Subscription confirmed: coin={coin}, type={sub_type}, interval={interval}")
                return
            
            # Handle error messages
            if "error" in msg:
                logger.error(f"| ❌ WebSocket error: {msg.get('error')}")
                return
            
            # Route to appropriate handler based on channel
            # Only process minute-level candle data
            if channel == "candle":
                self._handle_candle(data)
            elif channel == "trades":
                # Ignore trades - only minute-level candle is supported
                logger.debug(f"| 📊 Ignoring trades data (only minute-level candle is supported)")
            elif channel == "l2Book":
                # Ignore l2Book - only minute-level candle is supported
                logger.debug(f"| 📊 Ignoring l2Book data (only minute-level candle is supported)")
            else:
                logger.debug(f"| 📊 Received unknown channel: {channel}")
                
        except Exception as e:
            logger.error(f"| ❌ Error in on_message handler: {e}", exc_info=True)
    
    def _handle_candle(self, data):
        """Handle candle (OHLCV) data.
        
        Args:
            data: Candle[] array or single Candle dict from Hyperliquid
        """
        # Log raw candle data received from stream
        logger.info(f"| 📊 Raw candle data from stream: {json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)}")
        
        # Handle both single candle object (dict) and candle array (list)
        if isinstance(data, dict):
            # Single candle object - convert to list for processing
            candles = [data]
        elif isinstance(data, list):
            # Candle array
            candles = data
        else:
            logger.warning(f"| ⚠️  [CANDLE] Unexpected format: {data}")
            return
        
        # Log received candle data for debugging
        logger.info(f"| 📊 Parsed {len(candles)} candle(s) from Hyperliquid (may contain multiple symbols)")
        
        # Process each candle in the array (may contain candles for different symbols)
        for c in candles:
            try:
                # Hyperliquid candle format:
                # t/T: 开/收时间 (ms), s: symbol, i: interval,
                # o/c/h/l: 价格, v: volume, n: trades 数量
                symbol = c.get("s", "").upper()
                if not symbol:
                    logger.warning(f"| ⚠️  [CANDLE] Missing symbol in candle data: {c}")
                    continue
                
                # Get timestamps (in milliseconds)
                open_time_ms = int(c.get("t", 0))
                close_time_ms = int(c.get("T", 0))
                
                # For 1-minute candle, timestamp should be the minute start time
                # Hyperliquid's 't' is already the minute start time, so timestamp should match open_time
                timestamp_ms = open_time_ms  # Use open_time as timestamp (minute start time)
                
                open_time = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                # Check if candle is closed (for informational purposes only)
                current_time_ms = int(time.time() * 1000)
                is_closed = close_time_ms < current_time_ms
                
                processed_data = {
                    "symbol": symbol,
                    "interval": c.get("i", "1m"),
                    "timestamp": timestamp,  # Use timestamp (minute start time, e.g., 14:55:00)
                    "open_time": open_time,  # Use minute start time (open_time)
                    "close_time": close_time,  # Use minute end time (close_time)
                    "open": float(c.get("o", 0)),
                    "high": float(c.get("h", 0)),
                    "low": float(c.get("l", 0)),
                    "close": float(c.get("c", 0)),
                    "volume": float(c.get("v", 0)),
                    "quote_volume": 0.0,  # Hyperliquid doesn't provide quote_volume
                    "trade_count": int(c.get("n", 0)),
                    "taker_buy_base_volume": 0.0,  # Not provided
                    "taker_buy_quote_volume": 0.0,  # Not provided
                    "is_closed": is_closed,  # Whether candle is closed
                }
                
                logger.info(f"| 📊 Processing 1m candle for {symbol} (timestamp: {timestamp}, close_time: {close_time}, is_closed: {is_closed})")
                
                # Call callback with processed data
                if self.on_message_callback:
                    logger.debug(f"| 📤 Calling callback for {symbol} candle data")
                    self.on_message_callback(ws=self.ws, channel="candle", data=processed_data)
                else:
                    logger.warning(f"| ⚠️  No callback registered for {symbol} candle data")
                    
            except Exception as e:
                logger.error(f"| ❌ Error processing candle data: {e}", exc_info=True)
    
    def _handle_trades(self, data):
        """Handle trades data.
        
        Args:
            data: WsTrade[] array from Hyperliquid
        """
        if not isinstance(data, list):
            logger.warning(f"| ⚠️  [TRADES] Unexpected format: {data}")
            return
        
        for t in data:
            try:
                # Hyperliquid trade format:
                # coin: str, time: int (ms), side: str, px: str (price), sz: str (size)
                coin = t.get("coin", "").upper()
                if not coin:
                    continue
                
                timestamp_ms = int(t.get("time", 0))
                dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                
                processed_data = {
                    "symbol": coin,
                    "timestamp": timestamp,
                    "price": float(t.get("px", 0)),
                    "size": float(t.get("sz", 0)),
                    "side": t.get("side", ""),  # "A" for ask, "B" for bid
                    "trade_id": str(t.get("tid", "")),  # trade ID if available
                }
                
                logger.debug(f"| 📊 [TRADE] {coin} {timestamp} {processed_data['side']} px:{processed_data['price']} sz:{processed_data['size']}")
                
                # Call callback with processed data
                if self.on_message_callback:
                    self.on_message_callback(ws=self.ws, channel="trades", data=processed_data)
                    
            except Exception as e:
                logger.error(f"| ❌ Error processing trade data: {e}", exc_info=True)
    
    def _handle_l2book(self, data):
        """Handle L2 order book data.
        
        Args:
            data: WsBook dict from Hyperliquid
        """
        if not isinstance(data, dict):
            logger.warning(f"| ⚠️  [L2BOOK] Unexpected format: {data}")
            return
        
        try:
            # Hyperliquid l2Book format:
            # {
            #   "coin": str,
            #   "levels": [ [WsLevel...], [WsLevel...] ], # [bids, asks]
            #   "time": int (ms)
            # }
            # WsLevel: { px: string, sz: string, n: int }
            
            coin = data.get("coin", "").upper()
            if not coin:
                logger.warning(f"| ⚠️  [L2BOOK] Missing coin: {data}")
                return
            
            levels = data.get("levels", [])
            if not levels or len(levels) != 2:
                logger.warning(f"| ⚠️  [L2BOOK] Missing or invalid levels: {data}")
                return
            
            bids, asks = levels
            
            timestamp_ms = int(data.get("time", 0))
            dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            
            processed_data = {
                "symbol": coin,
                "timestamp": timestamp,
                "bids": bids,  # List of [px, sz, n]
                "asks": asks,  # List of [px, sz, n]
                "best_bid_price": float(bids[0]["px"]) if bids and len(bids) > 0 else None,
                "best_bid_size": float(bids[0]["sz"]) if bids and len(bids) > 0 else None,
                "best_ask_price": float(asks[0]["px"]) if asks and len(asks) > 0 else None,
                "best_ask_size": float(asks[0]["sz"]) if asks and len(asks) > 0 else None,
            }
            
            if processed_data["best_bid_price"] and processed_data["best_ask_price"]:
                logger.debug(f"| 📊 [L2BOOK] {coin} Bid:{processed_data['best_bid_price']} ({processed_data['best_bid_size']}) Ask:{processed_data['best_ask_price']} ({processed_data['best_ask_size']})")
            
            # Call callback with processed data
            if self.on_message_callback:
                self.on_message_callback(ws=self.ws, channel="l2Book", data=processed_data)
                
        except Exception as e:
            logger.error(f"| ❌ Error processing l2Book data: {e}", exc_info=True)
    
    def _on_error(self, ws, error):
        """Internal error handler."""
        try:
            logger.error(f"| ❌ WebSocket error: {error}")
            if self.on_error_callback:
                self.on_error_callback(ws, error)
        except Exception as e:
            logger.error(f"| ❌ Error in on_error callback: {e}", exc_info=True)
    
    def _on_close(self, ws, close_status_code=None, close_msg=None):
        """Internal close handler."""
        try:
            logger.warning(f"| 🛑 WebSocket closed: status={close_status_code}, msg={close_msg}")
            self._running = False
            
            # Check if we should reconnect
            # Status 1000 with "Inactive" message means server closed due to inactivity
            # We should reconnect in this case
            if self._should_reconnect and close_status_code == 1000 and close_msg == "Inactive":
                logger.info(f"| 🔄 Server closed connection due to inactivity. Will attempt to reconnect...")
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    logger.info(f"| 🔄 Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {self._reconnect_delay} seconds...")
                    time.sleep(self._reconnect_delay)
                    # Restart WebSocket connection
                    self._restart_connection()
                else:
                    logger.error(f"| ❌ Maximum reconnection attempts ({self._max_reconnect_attempts}) reached. Stopping reconnection.")
            elif self._should_reconnect and close_status_code != 1000:
                # Unexpected close - try to reconnect
                logger.warning(f"| ⚠️  Unexpected WebSocket close. Will attempt to reconnect...")
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    logger.info(f"| 🔄 Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {self._reconnect_delay} seconds...")
                    time.sleep(self._reconnect_delay)
                    self._restart_connection()
                else:
                    logger.error(f"| ❌ Maximum reconnection attempts ({self._max_reconnect_attempts}) reached. Stopping reconnection.")
            
            if self.on_close_callback:
                self.on_close_callback(ws)
        except Exception as e:
            logger.error(f"| ❌ Error in on_close callback: {e}", exc_info=True)
    
    def _on_open(self, ws):
        """Internal open handler - sends subscription messages."""
        try:
            logger.info("| ✅ WebSocket opened")
            
            # Subscribe to candles (minute-level only)
            logger.info(f"| 📡 Subscribing to {len(self._subscribed_candles)} symbols: {self._subscribed_candles}")
            for symbol in self._subscribed_candles:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "candle",
                        "coin": symbol,
                        "interval": self._default_interval  # Always 1m for minute-level streaming
                    }
                }
                logger.info(f"| 📤 Sending subscription message for {symbol}: {json.dumps(subscribe_msg)}")
                ws.send(json.dumps(subscribe_msg))
                logger.info(f"| 📡 Subscribed to minute-level candle: {symbol} (interval: {self._default_interval})")
                time.sleep(0.1)  # Small delay between subscriptions
            
            # Subscribe to trades
            for symbol in self._subscribed_trades:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "trades",
                        "coin": symbol
                    }
                }
                ws.send(json.dumps(subscribe_msg))
                logger.info(f"| 📡 Subscribed to trades: {symbol}")
                time.sleep(0.1)
            
            # Subscribe to l2Book
            for symbol in self._subscribed_l2books:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "l2Book",
                        "coin": symbol
                    }
                }
                ws.send(json.dumps(subscribe_msg))
                logger.info(f"| 📡 Subscribed to l2Book: {symbol}")
                time.sleep(0.1)
            
            if self.on_open_callback:
                self.on_open_callback(ws)
        except Exception as e:
            logger.error(f"| ❌ Error in on_open callback: {e}", exc_info=True)
    
    def subscribe_candle(self, symbol: str, interval: str = "1m"):
        """Subscribe to minute-level candle stream for a symbol.
        
        Note: This WebSocket is designed for minute-level (1m) data streaming.
        Only 1-minute intervals are supported.
        
        Args:
            symbol: Symbol to subscribe (e.g., 'BTC')
            interval: Candle interval - must be '1m' for minute-level data (default: "1m")
        """
        if interval != "1m":
            logger.warning(f"| ⚠️  Only 1m interval is supported for minute-level streaming. Got: {interval}, forcing to 1m")
            interval = "1m"
        
        symbol_upper = symbol.upper()
        if symbol_upper not in self._subscribed_candles:
            self._subscribed_candles.append(symbol_upper)
            logger.info(f"| 📡 Added minute-level candle subscription: {symbol_upper} (interval: {interval})")
    
    def subscribe_trades(self, symbol: str):
        """Subscribe to trades stream for a symbol.
        
        Args:
            symbol: Symbol to subscribe (e.g., 'BTC')
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self._subscribed_trades:
            self._subscribed_trades.append(symbol_upper)
            logger.info(f"| 📡 Added trades subscription: {symbol_upper}")
    
    def subscribe_l2book(self, symbol: str):
        """Subscribe to l2Book stream for a symbol.
        
        Args:
            symbol: Symbol to subscribe (e.g., 'BTC')
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self._subscribed_l2books:
            self._subscribed_l2books.append(symbol_upper)
            logger.info(f"| 📡 Added l2Book subscription: {symbol_upper}")
    
    def unsubscribe_candle(self, symbol: str):
        """Unsubscribe from candle stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._subscribed_candles:
            self._subscribed_candles.remove(symbol_upper)
            logger.info(f"| 📡 Removed candle subscription: {symbol_upper}")
    
    def unsubscribe_trades(self, symbol: str):
        """Unsubscribe from trades stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._subscribed_trades:
            self._subscribed_trades.remove(symbol_upper)
            logger.info(f"| 📡 Removed trades subscription: {symbol_upper}")
    
    def unsubscribe_l2book(self, symbol: str):
        """Unsubscribe from l2Book stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._subscribed_l2books:
            self._subscribed_l2books.remove(symbol_upper)
            logger.info(f"| 📡 Removed l2Book subscription: {symbol_upper}")
    
    def _restart_connection(self):
        """Restart WebSocket connection after close."""
        try:
            logger.info(f"| 🔄 Restarting WebSocket connection...")
            # Reset running flag
            self._running = False
            # Wait a bit before restarting
            time.sleep(1)
            # Start new connection
            self.start()
        except Exception as e:
            logger.error(f"| ❌ Error restarting WebSocket connection: {e}", exc_info=True)
    
    def start(self):
        """Start WebSocket connection."""
        if self._running:
            logger.warning("| ⚠️  WebSocket already running")
            return
        
        if not self._subscribed_candles:
            raise ValueError("No candle streams subscribed. Call subscribe_candle() first. Only minute-level candle subscriptions are supported.")
        
        # Reset reconnect attempts on successful start
        self._reconnect_attempts = 0
        
        def run_websocket():
            try:
                logger.info(f"| 🚀 Starting Hyperliquid WebSocket: {self.base_url}")
                
                self.ws = websocket.WebSocketApp(
                    self.base_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.on_open = self._on_open
                
                self._running = True
                # Auto ping to keep connection alive (reduce interval to prevent inactivity)
                # Hyperliquid may close inactive connections, so we ping more frequently
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"| ❌ Error in WebSocket thread: {e}", exc_info=True)
                self._running = False
                # Try to reconnect if error occurs
                if self._should_reconnect and self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    logger.info(f"| 🔄 Error occurred. Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {self._reconnect_delay} seconds...")
                    time.sleep(self._reconnect_delay)
                    self._restart_connection()
        
        self._thread = threading.Thread(target=run_websocket, daemon=True)
        self._thread.start()
        logger.info("| ✅ Hyperliquid WebSocket thread started")
    
    def stop(self):
        """Stop WebSocket connection."""
        self._should_reconnect = False  # Disable reconnection when explicitly stopping
        if not self._running:
            logger.warning("| ⚠️  WebSocket not running")
            return
        
        logger.info("| 🛑 Stopping Hyperliquid WebSocket...")
        self._running = False
        
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.warning(f"| ⚠️  Error closing WebSocket: {e}")
        
        # Only join if not called from within the thread itself
        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            try:
                self._thread.join(timeout=5.0)
            except RuntimeError:
                # Ignore if trying to join current thread
                pass
        
        logger.info("| ✅ Hyperliquid WebSocket stopped")
    
    def is_running(self) -> bool:
        """Check if WebSocket is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running
