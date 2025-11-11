"""Hyperliquid WebSocket implementation for real-time data streaming (Async version)."""
import json
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Callable, Optional, List, Literal
from src.logger import logger

try:
    import websockets
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    _WEBSOCKETS_AVAILABLE = False
    logger.error("websockets library not available. Install it: pip install websockets")

try:
    # Optional: used to discover all tradable coins for auto-subscription
    from hyperliquid.info import Info  # type: ignore
    _HYPERLIQUID_SDK_AVAILABLE = True
except Exception:
    _HYPERLIQUID_SDK_AVAILABLE = False


class HyperliquidWebSocket:
    """Async Hyperliquid WebSocket client for minute-level candle, trades, and l2Book streaming.
    
    This WebSocket is designed for minute-level (1m) candle data streaming.
    All candle subscriptions default to 1-minute intervals.
    """
    
    def __init__(
        self,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_open: Optional[Callable] = None,
        testnet: bool = False,
        auto_subscribe_all_candles: bool = False
    ):
        """Initialize Hyperliquid WebSocket client for minute-level data streaming.
        
        Args:
            on_message: Async callback function for messages (ws, channel, data)
            on_error: Async callback function for errors (ws, error)
            on_close: Async callback function for close events (ws)
            on_open: Async callback function for open events (ws)
            testnet: Whether to use testnet
            auto_subscribe_all_candles: On (re)connect, subscribe candle channel for ALL tradable coins
        """
        if not _WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library is required for async WebSocket. Install it: pip install websockets")
        
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
        self.auto_subscribe_all_candles = auto_subscribe_all_candles
        
        self.ws = None  # websockets.WebSocketClientProtocol
        self._running = False
        self._task: Optional[asyncio.Task] = None
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
    
    async def _on_message(self, message: str):
        """Internal message handler - processes candle, trades, and l2Book data."""
        try:
            # Parse message
            msg = json.loads(message)
            
            # Log all received messages for debugging (show full message)
            logger.debug(f"| 📨 Raw WebSocket message received: {json.dumps(msg, indent=2) if isinstance(msg, dict) else str(msg)}")
            
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
                logger.debug(f"| 📊 Routing to candle handler...")
                await self._handle_candle(data)
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
    
    async def _handle_candle(self, data):
        """Handle candle (OHLCV) data.
        
        Args:
            data: Candle[] array or single Candle dict from Hyperliquid
        """
        # Log raw candle data received from stream
        logger.debug(f"| 📊 Raw candle data from stream: {json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)}")
        
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
        logger.debug(f"| 📊 Parsed {len(candles)} candle(s) from Hyperliquid (may contain multiple symbols)")
        
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
                # Hyperliquid's 'T' is already the minute end time, so timestamp should match close_time + 1 second
                timestamp_ms = close_time_ms + 1000  # Use close_time + 1 second as timestamp (minute start time)
                
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
                
                logger.debug(f"| 📊 Received 1m candle: {symbol} @ {timestamp} (close: {close_time}, is_closed: {is_closed})")
                
                # Call callback with processed data (async)
                if self.on_message_callback:
                    logger.debug(f"| 📤 Calling async callback for {symbol} candle data")
                    if asyncio.iscoroutinefunction(self.on_message_callback):
                        await self.on_message_callback(ws=self.ws, channel="candle", data=processed_data)
                    else:
                        # Fallback to sync callback
                        self.on_message_callback(ws=self.ws, channel="candle", data=processed_data)
                else:
                    # No callback registered, just log the data summary
                    logger.info(f"| ✅ [CANDLE DATA] {symbol} @ {timestamp} | O:{processed_data['open']} H:{processed_data['high']} L:{processed_data['low']} C:{processed_data['close']} V:{processed_data['volume']}")
                    
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
                logger.debug(f"| 🔄 Server closed connection due to inactivity. Will attempt to reconnect...")
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    logger.debug(f"| 🔄 Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {self._reconnect_delay} seconds...")
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
                    logger.debug(f"| 🔄 Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {self._reconnect_delay} seconds...")
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
            logger.debug("| ✅ WebSocket opened")
            
            # Ensure candle subscription list contains ALL coins when enabled
            if self.auto_subscribe_all_candles:
                self._populate_all_candle_symbols()
            
            # Subscribe to candles (minute-level only)
            logger.debug(f"| 📡 Subscribing to {len(self._subscribed_candles)} symbols: {self._subscribed_candles}")
            for symbol in self._subscribed_candles:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "candle",
                        "coin": symbol,
                        "interval": self._default_interval  # Always 1m for minute-level streaming
                    }
                }
                logger.debug(f"| 📤 Sending subscription message for {symbol}: {json.dumps(subscribe_msg)}")
                ws.send(json.dumps(subscribe_msg))
                logger.debug(f"| 📡 Subscribed to minute-level candle: {symbol} (interval: {self._default_interval})")
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
                logger.debug(f"| 📡 Subscribed to trades: {symbol}")
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
                logger.debug(f"| 📡 Subscribed to l2Book: {symbol}")
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
            logger.debug(f"| 📡 Added minute-level candle subscription: {symbol_upper} (interval: {interval})")
    
    def subscribe_trades(self, symbol: str):
        """Subscribe to trades stream for a symbol.
        
        Args:
            symbol: Symbol to subscribe (e.g., 'BTC')
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self._subscribed_trades:
            self._subscribed_trades.append(symbol_upper)
            logger.debug(f"| 📡 Added trades subscription: {symbol_upper}")
    
    def subscribe_l2book(self, symbol: str):
        """Subscribe to l2Book stream for a symbol.
        
        Args:
            symbol: Symbol to subscribe (e.g., 'BTC')
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self._subscribed_l2books:
            self._subscribed_l2books.append(symbol_upper)
            logger.debug(f"| 📡 Added l2Book subscription: {symbol_upper}")
    
    def unsubscribe_candle(self, symbol: str):
        """Unsubscribe from candle stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._subscribed_candles:
            self._subscribed_candles.remove(symbol_upper)
            logger.debug(f"| 📡 Removed candle subscription: {symbol_upper}")
    
    def unsubscribe_trades(self, symbol: str):
        """Unsubscribe from trades stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._subscribed_trades:
            self._subscribed_trades.remove(symbol_upper)
            logger.debug(f"| 📡 Removed trades subscription: {symbol_upper}")
    
    def unsubscribe_l2book(self, symbol: str):
        """Unsubscribe from l2Book stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._subscribed_l2books:
            self._subscribed_l2books.remove(symbol_upper)
            logger.debug(f"| 📡 Removed l2Book subscription: {symbol_upper}")
    
    async def _run_forever(self):
        """Main async loop to receive WebSocket messages."""
        try:
            while self._running and self._should_reconnect:
                try:
                    logger.info(f"| 🚀 Connecting to Hyperliquid WebSocket: {self.base_url}")
                    
                    async with websockets.connect(
                        self.base_url,
                        ping_interval=10,
                        ping_timeout=5
                    ) as websocket:
                        self.ws = websocket
                        logger.info("| ✅ WebSocket connected")
                        
                        # Reset reconnect attempts on successful connection
                        self._reconnect_attempts = 0
                        
                        # Send subscriptions
                        await self._send_subscriptions()
                        
                        # Call on_open callback
                        if self.on_open_callback:
                            if asyncio.iscoroutinefunction(self.on_open_callback):
                                await self.on_open_callback(websocket)
                            else:
                                self.on_open_callback(websocket)
                        
                        # Receive messages
                        async for message in websocket:
                            if not self._running:
                                break
                            await self._on_message(message)
                            
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"| 🛑 WebSocket closed: {e}")
                    if self.on_close_callback:
                        if asyncio.iscoroutinefunction(self.on_close_callback):
                            await self.on_close_callback(self.ws)
                        else:
                            self.on_close_callback(self.ws)
                    
                    # Reconnect logic
                    if self._should_reconnect and self._reconnect_attempts < self._max_reconnect_attempts:
                        self._reconnect_attempts += 1
                        logger.info(f"| 🔄 Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {self._reconnect_delay} seconds...")
                        await asyncio.sleep(self._reconnect_delay)
                    else:
                        logger.error(f"| ❌ Maximum reconnection attempts reached or reconnection disabled")
                        break
                        
                except Exception as e:
                    logger.error(f"| ❌ WebSocket error: {e}", exc_info=True)
                    if self.on_error_callback:
                        if asyncio.iscoroutinefunction(self.on_error_callback):
                            await self.on_error_callback(self.ws, e)
                        else:
                            self.on_error_callback(self.ws, e)
                    
                    if self._should_reconnect and self._reconnect_attempts < self._max_reconnect_attempts:
                        self._reconnect_attempts += 1
                        logger.info(f"| 🔄 Error occurred. Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {self._reconnect_delay} seconds...")
                        await asyncio.sleep(self._reconnect_delay)
                    else:
                        break
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            logger.info("| ℹ️  WebSocket task cancelled")
        except Exception as e:
            logger.error(f"| ❌ Unexpected error in _run_forever: {e}", exc_info=True)
    
    async def _send_subscriptions(self):
        """Send subscription messages for all registered symbols."""
        # Subscribe to candles (minute-level only)
        logger.info(f"| 📡 Subscribing to {len(self._subscribed_candles)} candle symbols: {self._subscribed_candles}")
        for symbol in self._subscribed_candles:
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {
                    "type": "candle",
                    "coin": symbol,
                    "interval": self._default_interval
                }
            }
            logger.debug(f"| 📤 Sending subscription message for {symbol}")
            await self.ws.send(json.dumps(subscribe_msg))
            await asyncio.sleep(0.1)  # Small delay between subscriptions
        
        # Subscribe to trades
        for symbol in self._subscribed_trades:
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {
                    "type": "trades",
                    "coin": symbol
                }
            }
            await self.ws.send(json.dumps(subscribe_msg))
            logger.debug(f"| 📡 Subscribed to trades: {symbol}")
            await asyncio.sleep(0.1)
        
        # Subscribe to l2Book
        for symbol in self._subscribed_l2books:
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {
                    "type": "l2Book",
                    "coin": symbol
                }
            }
            await self.ws.send(json.dumps(subscribe_msg))
            logger.debug(f"| 📡 Subscribed to l2Book: {symbol}")
            await asyncio.sleep(0.1)
    
    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start WebSocket connection (creates async task).
        
        Args:
            loop: Event loop to run in. If None, uses current running loop.
        """
        if self._running:
            logger.warning("| ⚠️  WebSocket already running")
            return
        
        # Auto-populate on first start if enabled and list empty
        if not self._subscribed_candles and self.auto_subscribe_all_candles:
            self._populate_all_candle_symbols()
        if not self._subscribed_candles:
            raise ValueError("No candle streams subscribed. Call subscribe_candle() first.")
        
        self._running = True
        self._should_reconnect = True
        
        # Get or create event loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error("| ❌ No running event loop found. WebSocket must be started from within an async context.")
                raise
        
        # Create task
        self._task = loop.create_task(self._run_forever())
        logger.info("| ✅ Hyperliquid WebSocket task started")
    
    async def stop(self):
        """Stop WebSocket connection (async)."""
        self._should_reconnect = False
        self._running = False
        
        logger.info("| 🛑 Stopping Hyperliquid WebSocket...")
        
        # First cancel the task to interrupt any blocking operations
        if self._task and not self._task.done():
            self._task.cancel()
        
        # Then close the WebSocket connection
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"| ⚠️  Error closing WebSocket: {e}")
        
        # Wait for task to complete (should be quick now)
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("| ⚠️  Task cancellation timed out")
            except Exception as e:
                logger.warning(f"| ⚠️  Error waiting for task: {e}")
        
        logger.info("| ✅ Hyperliquid WebSocket stopped")
    
    def is_running(self) -> bool:
        """Check if WebSocket is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running
    
    # ------------------------- Helpers: auto populate all coins -------------------------
    def _populate_all_candle_symbols(self) -> None:
        """Populate self._subscribed_candles with ALL tradable coins from Hyperliquid Info."""
        if not _HYPERLIQUID_SDK_AVAILABLE:
            logger.warning("| ⚠️  hyperliquid SDK unavailable; cannot auto-subscribe all coins. Add subscriptions manually.")
            return
        try:
            base_url = "https://api.hyperliquid-testnet.xyz" if self.testnet else "https://api.hyperliquid.xyz"
            info = Info(base_url=base_url)  # type: ignore
            meta = info.meta()
            universe = meta.get("universe", [])
            added = 0
            for coin_info in universe:
                if isinstance(coin_info, dict):
                    name = coin_info.get("name", "")
                else:
                    name = str(coin_info)
                sym = (name or "").upper().strip()
                if sym and sym not in self._subscribed_candles:
                    self._subscribed_candles.append(sym)
                    added += 1
            logger.info(f"| 📡 Auto-populated ALL candle subscriptions: +{added} symbols (total {len(self._subscribed_candles)})")
        except Exception as e:
            logger.warning(f"| ⚠️  Failed to auto-populate all candle symbols: {e}")
