"""Hyperliquid WebSocket implementation for real-time data streaming (Async version)."""
import json
import asyncio
from typing import Dict, Callable, Optional, List

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
    """Async Hyperliquid WebSocket client for minute-level candle streaming.
    
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
        # Store candle subscriptions
        self._subscribed_candles: List[str] = []  # candle subscriptions (1m only)
    
    async def _on_message(self, message: str):
        """Internal message handler - processes candle data and pushes to callback."""
        try:
            # Parse message
            msg = json.loads(message)
            
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
            
            # Push raw candle data to callback
            if channel == "candle" and self.on_message_callback:
                if asyncio.iscoroutinefunction(self.on_message_callback):
                    await self.on_message_callback(ws=self.ws, channel="candle", data=data)
                else:
                    self.on_message_callback(ws=self.ws, channel="candle", data=data)
                
        except Exception as e:
            logger.error(f"| ❌ Error in on_message handler: {e}", exc_info=True)
    
    async def _on_error(self, ws, error):
        """Internal error handler."""
        try:
            logger.error(f"| ❌ WebSocket error: {error}")
            if self.on_error_callback:
                if asyncio.iscoroutinefunction(self.on_error_callback):
                    await self.on_error_callback(ws, error)
                else:
                    self.on_error_callback(ws, error)
        except Exception as e:
            logger.error(f"| ❌ Error in on_error callback: {e}", exc_info=True)
    
    async def _on_close(self, ws, close_status_code=None, close_msg=None):
        """Internal close handler."""
        try:
            logger.warning(f"| 🛑 WebSocket closed: status={close_status_code}, msg={close_msg}")
            self._running = False
            
            if self.on_close_callback:
                if asyncio.iscoroutinefunction(self.on_close_callback):
                    await self.on_close_callback(ws)
                else:
                    self.on_close_callback(ws)
        except Exception as e:
            logger.error(f"| ❌ Error in on_close callback: {e}", exc_info=True)
    
    async def _on_open(self, ws):
        """Internal open handler - sends subscription messages."""
        try:
            logger.debug("| ✅ WebSocket opened")
            
            # Ensure candle subscription list contains ALL coins when enabled
            if self.auto_subscribe_all_candles:
                await self._populate_all_candle_symbols()
            
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
                await ws.send(json.dumps(subscribe_msg))
                await asyncio.sleep(0.1)  # Small delay between subscriptions
            
            if self.on_open_callback:
                if asyncio.iscoroutinefunction(self.on_open_callback):
                    await self.on_open_callback(ws)
                else:
                    self.on_open_callback(ws)
        except Exception as e:
            logger.error(f"| ❌ Error in on_open callback: {e}", exc_info=True)
    
    async def subscribe_candle(self, symbol: str, interval: str = "1m"):
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
    
    async def unsubscribe_candle(self, symbol: str):
        """Unsubscribe from candle stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._subscribed_candles:
            self._subscribed_candles.remove(symbol_upper)
            logger.info(f"| 📡 Removed candle subscription: {symbol_upper}")
    
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
                        
                        # Call _on_open to handle subscriptions and callbacks
                        await self._on_open(websocket)
                        
                        # Receive messages
                        async for message in websocket:
                            if not self._running:
                                break
                            await self._on_message(message)
                            
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"| 🛑 WebSocket closed: {e}")
                    await self._on_close(self.ws)
                    
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
                    await self._on_error(self.ws, e)
                    
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
    
    async def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start WebSocket connection (creates async task).
        
        Args:
            loop: Event loop to run in. If None, uses current running loop.
        """
        if self._running:
            logger.warning("| ⚠️  WebSocket already running")
            return
        
        # Auto-populate on first start if enabled and list empty
        if not self._subscribed_candles and self.auto_subscribe_all_candles:
            await self._populate_all_candle_symbols()
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
    
    async def is_running(self) -> bool:
        """Check if WebSocket is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running
    
    # ------------------------- Helpers: auto populate all coins -------------------------
    async def _populate_all_candle_symbols(self) -> None:
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
