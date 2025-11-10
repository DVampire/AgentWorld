"""Data producer: receives data from Hyperliquid WebSocket streams and writes to database."""
from __future__ import annotations
import threading
import asyncio
from typing import Optional, List, Dict, TYPE_CHECKING

from src.logger import logger
from src.environments.hyperliquidentry.candle import CandleHandler
from src.environments.hyperliquidentry.trades import TradesHandler
from src.environments.hyperliquidentry.l2book import L2BookHandler
from src.environments.hyperliquidentry.types import DataStreamType
from src.environments.hyperliquidentry.websocket import HyperliquidWebSocket

if TYPE_CHECKING:
    from src.environments.hyperliquidentry.types import AccountInfo


class DataProducer:
    """Producer: receives data from Hyperliquid WebSocket streams and writes to database."""
    
    def __init__(
        self,
        account: 'AccountInfo',
        candle_handler: CandleHandler,
        trades_handler: TradesHandler,
        l2book_handler: L2BookHandler,
        symbols: Dict[str, Dict],
        max_concurrent_writes: int = 10,
        testnet: bool = False,
    ):
        """Initialize data producer.
        
        Args:
            account: Hyperliquid account information
            candle_handler: Candle data handler
            trades_handler: Trades data handler
            l2book_handler: L2Book data handler
            symbols: Symbol dictionary
            max_concurrent_writes: Maximum concurrent database writes
            testnet: Whether to use testnet
        """
        self.account = account
        self.symbols = symbols
        self.testnet = testnet
        
        self._candle_handler = candle_handler
        self._trades_handler = trades_handler
        self._l2book_handler = l2book_handler
        
        self._data_queue: Optional[asyncio.Queue] = None
        self._data_semaphore: Optional[asyncio.Semaphore] = None
        self._data_stream_running: bool = False
        self._data_stream_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._max_concurrent_writes = max_concurrent_writes
        self._ws_client: Optional[HyperliquidWebSocket] = None
        
        # Candle minute buffer:
        # Key: symbol, Value: {"close_time": str, "data": Dict}
        # Buffer the latest data for current minute, only insert when minute changes
        self._candle_minute_buffer: Dict[str, Dict] = {}
    
    async def _handle_data(self, data: Dict, symbol: str, data_type: DataStreamType) -> None:
        """Handle incoming data and write to database.
        
        Args:
            data: Processed data from Hyperliquid WebSocket stream
            symbol: Symbol name
            data_type: Data stream type
        """
        async with self._data_semaphore:
            try:
                if data_type == DataStreamType.CANDLE:
                    # Process candle data with minute-level aggregation
                    await self._process_candle_data(data, symbol)
                elif data_type == DataStreamType.TRADES:
                    result = await self._trades_handler.stream_insert(data, symbol)
                    if result:
                        logger.debug(f"| ✅ Trades data inserted for {symbol}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert trades data for {symbol}")
                elif data_type == DataStreamType.L2BOOK:
                    result = await self._l2book_handler.stream_insert(data, symbol)
                    if result:
                        logger.debug(f"| ✅ L2Book data inserted for {symbol}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert l2Book data for {symbol}")
            except Exception as e:
                logger.error(f"| ❌ Error handling {data_type.value} data for {symbol}: {e}", exc_info=True)
    
    async def _process_candle_data(self, data: Dict, symbol: str) -> None:
        """Process candle data - buffer current minute, insert when minute changes.
        
        Strategy:
        1. WebSocket pushes updates every few seconds for the SAME minute (e.g., 07:14:00-07:14:59)
        2. We buffer the LATEST update for current minute
        3. When close_time changes (e.g., from 07:14:59 to 07:15:59), we know previous minute is complete
        4. Insert the buffered data (which contains the final OHLCV for that minute)
        
        This ensures:
        - Only 1 database write per minute per symbol
        - We capture the final/complete state of each 1-minute candle
        
        Args:
            data: Candle data from WebSocket (contains close_time)
            symbol: Symbol name (uppercase)
        """
        try:
            close_time = data.get("close_time")
            if not close_time:
                logger.warning(f"| ⚠️  Candle data missing close_time for {symbol}")
                return
            
            # Check if we have buffered data for this symbol
            if symbol in self._candle_minute_buffer:
                buffered_close_time = self._candle_minute_buffer[symbol]["close_time"]
                
                if close_time != buffered_close_time:
                    # Minute changed! Insert the buffered (complete) candle from previous minute
                    buffered_data = self._candle_minute_buffer[symbol]["data"]
                    logger.info(f"| 🕐 Minute changed for {symbol}: {buffered_close_time} → {close_time}")
                    logger.info(f"| 📤 Inserting complete candle (close_time: {buffered_close_time})")
                    
                    result = await self._candle_handler.stream_insert(buffered_data, symbol)
                    success = result.success if hasattr(result, 'success') else result.get("success", False)
                    message = result.message if hasattr(result, 'message') else result.get("message", "")
                    
                    if success:
                        logger.info(f"| ✅ Complete candle inserted for {symbol} (close_time: {buffered_close_time})")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert candle for {symbol}: {message}")
                else:
                    # Same minute, just update buffer silently
                    logger.debug(f"| 📦 Updating buffer for {symbol} (same minute: {close_time})")
            else:
                # First candle for this symbol
                logger.info(f"| 🆕 First candle data for {symbol} (close_time: {close_time})")
            
            # Update buffer with current data (latest state of current minute)
            self._candle_minute_buffer[symbol] = {
                "close_time": close_time,
                "data": data
            }
            
        except Exception as e:
            logger.error(f"| ❌ Error processing candle data for {symbol}: {e}", exc_info=True)
    
    async def _data_processor(self) -> None:
        """Process data from queue."""
        logger.info(f"| 🔄 Data processor started, waiting for data...")
        while self._data_stream_running:
            try:
                # Get data from queue with timeout
                data, symbol, data_type = await asyncio.wait_for(
                    self._data_queue.get(), timeout=1.0
                )
                logger.debug(f"| 📦 Processing {data_type.value} data for {symbol} from queue")
                await self._handle_data(data, symbol, data_type)
                self._data_queue.task_done()
            except asyncio.TimeoutError:
                # Normal timeout, continue waiting
                continue
            except Exception as e:
                logger.error(f"| ❌ Error in data processor: {e}", exc_info=True)
    
    def _create_message_handler(self):
        """Create unified async message handler for WebSocket client."""
        async def on_message(ws, channel: str, data: Dict):
            """Handle incoming processed data from WebSocket (async).
            
            Args:
                ws: WebSocket connection
                channel: Channel type ("candle", "trades", "l2Book")
                data: Processed data dictionary
            """
            try:
                if not self._data_stream_running:
                    return
                
                if not isinstance(data, dict) or "symbol" not in data:
                    logger.warning(f"| 📊 Received unexpected data format: {data}")
                    return
                
                symbol = data["symbol"].upper()  # Keep consistent with subscription (uppercase)
                
                # Only process minute-level candle data
                if channel == "candle":
                    data_type = DataStreamType.CANDLE
                    logger.debug(f"| 📡 Received candle data for {symbol} (timestamp: {data.get('timestamp')})")
                else:
                    # Ignore trades and l2Book - only minute-level candle is supported
                    logger.debug(f"| 📊 Ignoring {channel} data (only minute-level candle is supported)")
                    return
                
                # Process data directly (we're already in async context)
                await self._data_handler_wrapper(data, symbol, data_type)
                    
            except Exception as e:
                logger.error(f"| ❌ Error in message handler: {e}", exc_info=True)
        
        return on_message
    
    async def _data_handler_wrapper(self, processed_data: Dict, symbol: str, data_type: DataStreamType):
        """Wrapper for data handler.
        
        Args:
            processed_data: Processed data from WebSocket
            symbol: Symbol name (uppercase)
            data_type: Data stream type
        """
        try:
            logger.debug(f"| 📥 Data handler wrapper called for {symbol} ({data_type.value})")
            if self._data_queue:
                try:
                    # Add with timeout to avoid blocking forever if queue is full
                    await asyncio.wait_for(
                        self._data_queue.put((processed_data, symbol, data_type)),
                        timeout=5.0
                    )
                    logger.debug(f"| ✅ Added {symbol} ({data_type.value}) data to queue")
                except asyncio.TimeoutError:
                    logger.warning(f"| ⚠️  Queue full, dropping {symbol} ({data_type.value}) data")
            else:
                logger.warning(f"| ⚠️  Data queue not initialized when {data_type.value} data received for {symbol}")
        except Exception as e:
            logger.error(f"| ❌ Error in data handler wrapper for {symbol}: {e}", exc_info=True)
    
    def _data_stream_worker(
        self,
        symbols: List[str],
        data_types: Optional[List[DataStreamType]] = None
    ):
        """Worker thread for running data streams.
        
        Args:
            symbols: List of symbols to subscribe to
            data_types: Optional list of data types to subscribe to (default: all types)
        """
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            
            async def setup_and_run():
                nonlocal data_types  # Allow modification of outer scope variable
                
                self._data_stream_running = True
                self._data_queue = asyncio.Queue(maxsize=1000)
                self._data_semaphore = asyncio.Semaphore(self._max_concurrent_writes)
                
                logger.info(f"| ✅ Data queue initialized")
                
                processor_task = asyncio.create_task(self._data_processor())
                logger.info(f"| ✅ Data processor started")
                
                # Default to candle only (minute-level) if not specified
                if data_types is None:
                    data_types = [DataStreamType.CANDLE]
                
                # Only support minute-level candle subscriptions
                if DataStreamType.TRADES in data_types or DataStreamType.L2BOOK in data_types:
                    logger.warning(f"| ⚠️  Only CANDLE (minute-level) subscriptions are supported. Ignoring TRADES and L2BOOK.")
                    data_types = [DataStreamType.CANDLE]
                
                # Ensure tables exist for all symbols (only candle)
                for symbol in symbols:
                    if DataStreamType.CANDLE in data_types:
                        await self._candle_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Candle table created/verified for {symbol}")
                
                # Create unified async message handler
                on_message = self._create_message_handler()
                
                # Initialize WebSocket client
                logger.info(f"| 🚀 Initializing Hyperliquid WebSocket for {len(symbols)} symbols...")
                self._ws_client = HyperliquidWebSocket(
                    on_message=on_message,
                    on_error=lambda ws, err: logger.error(f"| ❌ WS error: {err}"),
                    on_close=lambda ws: logger.info("| 🛑 WebSocket closed"),
                    on_open=lambda ws: logger.info("| ✅ WebSocket opened"),
                    testnet=self.testnet
                )
                
                # Subscribe to data streams (only minute-level candle)
                for symbol in symbols:
                    if DataStreamType.CANDLE in data_types:
                        self._ws_client.subscribe_candle(symbol, interval="1m")
                        logger.info(f"| 📡 Subscribed to minute-level candle: {symbol} (interval: 1m)")
                    # TRADES and L2BOOK subscriptions are disabled - only minute-level candle is supported
                
                # Start WebSocket (pass current event loop)
                self._ws_client.start(loop=loop)
                
                logger.info(f"| ✅ All subscriptions completed for {len(symbols)} symbols")
                
                # Keep the stream running
                while self._data_stream_running:
                    await asyncio.sleep(1)
                
            loop.run_until_complete(setup_and_run())
            
        except Exception as e:
            logger.error(f"| ❌ Error in data stream worker: {e}", exc_info=True)
        finally:
            if loop:
                loop.close()
    
    def start(
        self,
        symbols: List[str],
        data_types: Optional[List[DataStreamType]] = None
    ) -> None:
        """Start data stream for given symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            data_types: Optional list of data types to subscribe to (default: all types)
        """
        if self._data_stream_running:
            logger.warning("| ⚠️  Data stream already running")
            return
        
        # Normalize symbols to uppercase for consistency
        symbols = [s.upper() for s in symbols]
        
        logger.info(f"| 📡 Starting data stream for {len(symbols)} symbols: {symbols}")
        
        self._data_stream_thread = threading.Thread(
            target=self._data_stream_worker,
            args=(symbols, data_types),
            daemon=True
        )
        self._data_stream_thread.start()
        
        logger.info("| ✅ Data stream thread started")
    
    def stop(self) -> None:
        """Stop the data stream."""
        if not self._data_stream_running:
            logger.warning("| ⚠️  Data stream not running")
            return
        
        logger.info("| 🛑 Stopping data stream...")
        self._data_stream_running = False
        
        # Flush buffered candles before stopping
        try:
            if self._event_loop and self._candle_handler:
                logger.info("| 🔄 Flushing buffered candle data...")
                future = asyncio.run_coroutine_threadsafe(
                    self._flush_minute_buffer(),
                    self._event_loop
                )
                future.result(timeout=5.0)
        except Exception as e:
            logger.warning(f"| ⚠️  Error flushing minute buffer: {e}")
        
        # Stop WebSocket client (async)
        try:
            if self._ws_client and self._event_loop:
                logger.info("| 🛑 Stopping WebSocket client...")
                future = asyncio.run_coroutine_threadsafe(
                    self._ws_client.stop(),
                    self._event_loop
                )
                future.result(timeout=5.0)
        except Exception as e:
            logger.warning(f"| ⚠️  Error stopping WebSocket client: {e}")
        
        # Wait for thread to finish
        if self._data_stream_thread and self._data_stream_thread.is_alive():
            self._data_stream_thread.join(timeout=5.0)
        
        logger.info("| ✅ Data stream stopped")
    
    async def _flush_minute_buffer(self) -> None:
        """Flush all buffered minute data before shutdown.
        
        This ensures the last minute's data is not lost when stopping.
        """
        if not self._candle_minute_buffer:
            logger.info("| ℹ️  No buffered candles to flush")
            return
        
        logger.info(f"| 🔄 Flushing {len(self._candle_minute_buffer)} buffered candles...")
        
        for symbol, buffer_info in self._candle_minute_buffer.items():
            try:
                close_time = buffer_info["close_time"]
                data = buffer_info["data"]
                
                logger.info(f"| 📤 Flushing final candle for {symbol} (close_time: {close_time})")
                
                result = await self._candle_handler.stream_insert(data, symbol)
                success = result.success if hasattr(result, 'success') else result.get("success", False)
                message = result.message if hasattr(result, 'message') else result.get("message", "")
                
                if success:
                    logger.info(f"| ✅ Flushed candle for {symbol}")
                else:
                    logger.warning(f"| ⚠️  Failed to flush candle for {symbol}: {message}")
            except Exception as e:
                logger.error(f"| ❌ Error flushing candle for {symbol}: {e}", exc_info=True)
        
        self._candle_minute_buffer.clear()
        logger.info("| ✅ Minute buffer flushed and cleared")

