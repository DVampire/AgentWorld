"""Binance Futures WebSocket implementation."""
import json
import threading
import websocket
from datetime import datetime, timezone
from typing import Dict, Callable, Optional, List
from src.logger import logger


class BinanceFuturesWebSocket:
    """Binance Futures WebSocket client for klines streaming."""
    
    def __init__(
        self,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_open: Optional[Callable] = None,
        testnet: bool = False
    ):
        """Initialize Binance Futures WebSocket client.
        
        Args:
            on_message: Callback function for messages (ws, message)
            on_error: Callback function for errors (ws, error)
            on_close: Callback function for close events (ws)
            on_open: Callback function for open events (ws)
            testnet: Whether to use testnet
        """
        self.testnet = testnet
        # Binance Futures WebSocket URLs
        # Live: wss://fstream.binance.com
        # Testnet: wss://stream.binancefuture.com
        self.base_url = "wss://stream.binancefuture.com" if testnet else "wss://fstream.binance.com"
        
        self.on_message_callback = on_message
        self.on_error_callback = on_error
        self.on_close_callback = on_close
        self.on_open_callback = on_open
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._subscribed_streams: List[str] = []
    
    def _on_message(self, ws, message):
        """Internal message handler - filters and processes minute-level klines only."""
        try:
            # Parse message
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
            
            # Log all received messages for debugging
            logger.debug(f"| 📨 Raw WebSocket message received: {str(data)[:200]}")
            
            # Handle combined stream format
            if isinstance(data, dict) and "stream" in data:
                stream_name = data.get("stream", "")
                kline_data = data.get("data", {})
                logger.debug(f"| 📡 Combined stream format: stream={stream_name}")
            else:
                kline_data = data
                stream_name = None
            
            # Only process kline events
            if isinstance(kline_data, dict) and "e" in kline_data and kline_data["e"] == "kline":
                logger.debug(f"| 📊 Kline event received: symbol={kline_data.get('s', 'unknown')}, event={kline_data.get('e')}")
                kline = kline_data.get("k", {})
                interval = kline.get("i", "")
                is_closed = kline.get("x", False)
                symbol = kline.get("s", "")
                
                logger.debug(f"| 📊 Kline details: symbol={symbol}, interval={interval}, is_closed={is_closed}")
                
                # Only process closed 1-minute klines
                if interval == "1m" and is_closed:
                    # Extract and format data
                    symbol = kline.get("s", "").upper()
                    
                    open_time_ms = int(kline.get("t", 0))
                    close_time_ms = int(kline.get("T", 0))
                    # For 1-minute kline, timestamp should be the minute start time
                    # Add 1 second to close_time to get the next minute start (since close_time is :59)
                    timestamp_ms = close_time_ms + 1000  # Add 1 second to get minute start
                    
                    open_time = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    
                    processed_data = {
                        "symbol": symbol,
                        "interval": interval,
                        "timestamp": timestamp,  # Use timestamp (minute start time, e.g., 14:55:00)
                        "open_time": open_time,  # Use minute start time (open_time)
                        "close_time": close_time,  # Use minute end time (close_time)
                        "open": float(kline.get("o", 0)),
                        "high": float(kline.get("h", 0)),
                        "low": float(kline.get("l", 0)),
                        "close": float(kline.get("c", 0)),
                        "volume": float(kline.get("v", 0)),
                        "quote_volume": float(kline.get("q", 0)),
                        "trade_count": int(kline.get("n", 0)),
                        "taker_buy_base_volume": float(kline.get("V", 0)),
                        "taker_buy_quote_volume": float(kline.get("Q", 0)),
                        "is_closed": True,
                    }
                    
                    logger.info(f"| 📊 Processing closed 1m kline for {symbol} (timestamp: {timestamp}, close_time: {close_time})")
                    
                    # Call callback with processed data
                    if self.on_message_callback:
                        self.on_message_callback(ws, processed_data)
                elif interval == "1m" and not is_closed:
                    # Log real-time updates (for debugging)
                    logger.debug(f"| 📊 Received real-time update (not closed) for {symbol} interval: {interval}")
                elif interval != "1m":
                    logger.debug(f"| 📊 Skipping non-1m interval: {interval} for {symbol}")
                # Skip real-time updates (x=false) and non-1m intervals
            elif isinstance(data, dict) and "result" in data:
                # Subscription confirmation - log but don't process
                logger.info(f"| ✅ Subscription confirmed: {data}")
            else:
                # Unknown message format - log but don't process
                logger.warning(f"| 📊 Received unknown message format (keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}): {str(data)[:200]}")
                
        except Exception as e:
            logger.error(f"| ❌ Error in on_message handler: {e}", exc_info=True)
    
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
            logger.info(f"| 🛑 WebSocket closed: status={close_status_code}, msg={close_msg}")
            self._running = False
            if self.on_close_callback:
                self.on_close_callback(ws)
        except Exception as e:
            logger.error(f"| ❌ Error in on_close callback: {e}", exc_info=True)
    
    def _on_open(self, ws):
        """Internal open handler."""
        try:
            logger.info("| ✅ WebSocket opened")
            if self.on_open_callback:
                self.on_open_callback(ws)
        except Exception as e:
            logger.error(f"| ❌ Error in on_open callback: {e}", exc_info=True)
    
    def subscribe_kline(self, symbol: str, interval: str = "1m"):
        """Subscribe to kline stream for a symbol.
        
        Note: Only 1-minute (1m) closed klines are processed and passed to callback.
        
        Args:
            symbol: Symbol to subscribe (e.g., 'BTCUSDT')
            interval: Kline interval - must be '1m' for minute-level data
        """
        if interval != "1m":
            logger.warning(f"| ⚠️  Only 1m interval is supported for minute-level streaming. Got: {interval}")
            interval = "1m"
        
        stream_name = f"{symbol.lower()}@kline_{interval}"
        if stream_name not in self._subscribed_streams:
            self._subscribed_streams.append(stream_name)
            logger.info(f"| 📡 Added subscription: {stream_name} (only closed 1m klines will be processed)")
    
    def unsubscribe_kline(self, symbol: str, interval: str = "1m"):
        """Unsubscribe from kline stream for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe
            interval: Kline interval
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        if stream_name in self._subscribed_streams:
            self._subscribed_streams.remove(stream_name)
            logger.info(f"| 📡 Removed subscription: {stream_name}")
    
    def _build_stream_url(self) -> str:
        """Build WebSocket stream URL.
        
        Returns:
            WebSocket URL with all streams
        """
        if not self._subscribed_streams:
            raise ValueError("No streams subscribed")
        
        # For single stream: wss://fstream.binance.com/ws/btcusdt@kline_1m
        if len(self._subscribed_streams) == 1:
            return f"{self.base_url}/ws/{self._subscribed_streams[0]}"
        else:
            # For multiple streams: wss://fstream.binance.com/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m
            stream_names = "/".join(self._subscribed_streams)
            return f"{self.base_url}/stream?streams={stream_names}"
    
    def start(self):
        """Start WebSocket connection."""
        if self._running:
            logger.warning("| ⚠️  WebSocket already running")
            return
        
        if not self._subscribed_streams:
            raise ValueError("No streams subscribed. Call subscribe_kline() first.")
        
        def run_websocket():
            try:
                stream_url = self._build_stream_url()
                logger.info(f"| 🚀 Starting Futures WebSocket: {stream_url}")
                
                self.ws = websocket.WebSocketApp(
                    stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.on_open = self._on_open
                
                self._running = True
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"| ❌ Error in WebSocket thread: {e}", exc_info=True)
                self._running = False
        
        self._thread = threading.Thread(target=run_websocket, daemon=True)
        self._thread.start()
        logger.info("| ✅ Futures WebSocket thread started")
    
    def stop(self):
        """Stop WebSocket connection."""
        if not self._running:
            logger.warning("| ⚠️  WebSocket not running")
            return
        
        logger.info("| 🛑 Stopping Futures WebSocket...")
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
        
        logger.info("| ✅ Futures WebSocket stopped")
    
    def is_running(self) -> bool:
        """Check if WebSocket is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running

