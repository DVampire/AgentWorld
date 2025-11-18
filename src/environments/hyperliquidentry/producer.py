"""Data producer: receives data from Hyperliquid WebSocket streams and writes to database."""
from __future__ import annotations
import asyncio
from typing import Optional, List, Dict, TYPE_CHECKING

from src.logger import logger
from src.environments.hyperliquidentry.candle import CandleHandler
from src.environments.hyperliquidentry.types import DataStreamType
from src.environments.hyperliquidentry.websocket import HyperliquidWebSocket

if TYPE_CHECKING:
    from src.environments.hyperliquidentry.types import AccountInfo


class DataProducer:
    """Producer: receives data from Hyperliquid WebSocket streams and writes to database.
    
    Runs as a background async task, not blocking the main event loop.
    """
    
    def __init__(
        self,
        account: 'AccountInfo',
        candle_handler: CandleHandler,
        symbols: Dict[str, Dict],
        max_concurrent_writes: int = 10,
        testnet: bool = False,
    ):
        """Initialize data producer.
        
        Args:
            account: Hyperliquid account information
            candle_handler: Candle data handler
            symbols: Symbol dictionary
            max_concurrent_writes: Maximum concurrent database writes
            testnet: Whether to use testnet
        """
        self.account = account
        self.symbols = symbols
        self.testnet = testnet
        
        self._candle_handler = candle_handler
        
        self._data_queue: Optional[asyncio.Queue] = None
        self._data_semaphore: Optional[asyncio.Semaphore] = None
        self._data_stream_running: bool = False
        self._stream_task: Optional[asyncio.Task] = None
        self._max_concurrent_writes = max_concurrent_writes
        self._ws_client: Optional[HyperliquidWebSocket] = None
        
        # Candle data buffer: store latest candle data for current minute
        # Key: symbol, Value: latest candle data dict
        # Hyperliquid pushes multiple updates for same minute, we keep only the latest
        self._candle_buffer: Dict[str, Dict] = {}
        
        # Track current minute being buffered for each symbol
        # Key: symbol, Value: current_minute_timestamp (e.g., "10:00:00")
        self._current_minute: Dict[str, str] = {}
        
        # Track if we're waiting for first complete minute
        # Key: symbol, Value: bool (True if waiting)
        self._waiting_for_new_minute: Dict[str, bool] = {}
    
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
            except Exception as e:
                logger.error(f"| ❌ Error handling {data_type.value} data for {symbol}: {e}", exc_info=True)
    
    async def _process_candle_data(self, data: Dict, symbol: str) -> None:
        """Process candle data - keep latest update for current minute, insert when minute changes.
        
        Strategy:
        Hyperliquid pushes multiple updates for the SAME minute (e.g., 10:01:00)
        - Each update contains the latest OHLCV state for that minute
        - We keep only the LATEST update (most complete data)
        - When new minute starts, insert the latest data from previous minute
        
        Example timeline:
        - Program starts at 10:00:35 → Wait for new minute
        - 10:00:45 update arrives → Skip (incomplete minute)
        - 10:01:02 update #1 arrives → Buffer (replace any previous 10:01 data)
        - 10:01:05 update #2 arrives → Buffer (replace update #1)
        - 10:01:30 update #3 arrives → Buffer (replace update #2)
        - 10:01:55 update #4 arrives → Buffer (replace update #3) ← Latest state
        - 10:02:00 update arrives → Insert buffered 10:01 data (update #4) → Start buffering 10:02
        
        This ensures:
        - Only 1 database write per minute
        - We get the most complete/accurate data (last update of the minute)
        - No need to aggregate - Hyperliquid already provides complete OHLCV
        
        Args:
            data: Complete candle data from WebSocket (OHLCV for current minute)
            symbol: Symbol name (uppercase)
        """
        try:
            timestamp = data.get("t", 0)  # Minute timestamp (e.g., "10:00:00")
            
            # Initialize tracking for this symbol
            if symbol not in self._waiting_for_new_minute:
                self._waiting_for_new_minute[symbol] = True
                logger.info(f"| ⏳ Waiting for first complete minute for {symbol}")
            
            current_minute = self._current_minute.get(symbol)
            
            logger.debug(f"| 📦 Current minute for {symbol}: {current_minute}, timestamp: {timestamp}")
            
            # Check if new minute started
            if current_minute is None:
                # First data point
                if self._waiting_for_new_minute[symbol]:
                    # Skip first incomplete minute
                    logger.info(f"| 🚫 Skipping incomplete minute for {symbol} @ {timestamp}")
                    self._waiting_for_new_minute[symbol] = False
                    self._current_minute[symbol] = timestamp
                    self._candle_buffer[symbol] = data
                    return
                else:
                    # Should not happen, but handle gracefully
                    self._current_minute[symbol] = timestamp
                    self._candle_buffer[symbol] = data
            
            elif timestamp != current_minute:
                # New minute started! Insert previous minute's latest data
                logger.info(f"| 🕐 New minute for {symbol}: {current_minute} → {timestamp}")
                
                if symbol in self._candle_buffer and self._candle_buffer[symbol]:
                    # Insert the latest (most complete) data from previous minute
                    latest_data = self._candle_buffer[symbol]
                    logger.info(f"| 📤 Inserting latest data for {symbol} @ {current_minute}")
                    
                    # Insert candle data into database
                    try:
                        result = await self._candle_handler.stream_insert(latest_data, symbol)
                        success = result.success if hasattr(result, 'success') else result.get("success", False)
                        message = result.message if hasattr(result, 'message') else result.get("message", "")
                        
                        if success:
                            logger.info(f"| ✅ Candle inserted for {symbol} @ {current_minute}")
                        else:
                            logger.warning(f"| ⚠️  Failed to insert candle for {symbol} @ {current_minute}: {message}")
                    except Exception as e:
                        logger.error(f"| ❌ Error inserting candle for {symbol} @ {current_minute}: {e}", exc_info=True)
                else:
                    logger.warning(f"| ⚠️  No buffered data for {symbol} @ {current_minute}")
                
                # Start buffering new minute
                self._current_minute[symbol] = timestamp
                self._candle_buffer[symbol] = data
            else:
                # Same minute - update buffer with latest data (replace previous)
                self._candle_buffer[symbol] = data
                logger.debug(f"| 📦 Updated buffer for {symbol} @ {timestamp} (keeping latest)")
            
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
    
    async def _create_message_handler(self):
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
                
                symbol = data.get("s", "").upper()  # Keep consistent with subscription (uppercase)
                
                # Only process minute-level candle data
                if channel == "candle":
                    data_type = DataStreamType.CANDLE
                
                # Process data directly (we're already in async context)
                await self._data_handler_wrapper(data, symbol, data_type)
                    
            except Exception as e:
                logger.error(f"| ❌ Error in message handler: {e}", exc_info=True)
        
        return on_message
    
    async def _data_handler_wrapper(self, data: Dict, symbol: str, data_type: DataStreamType):
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
                        self._data_queue.put((data, symbol, data_type)),
                        timeout=5.0
                    )
                    logger.debug(f"| ✅ Added {symbol} ({data_type.value}) data to queue")
                except asyncio.TimeoutError:
                    logger.warning(f"| ⚠️  Queue full, dropping {symbol} ({data_type.value}) data")
            else:
                logger.warning(f"| ⚠️  Data queue not initialized when {data_type.value} data received for {symbol}")
        except Exception as e:
            logger.error(f"| ❌ Error in data handler wrapper for {symbol}: {e}", exc_info=True)
    
    async def _data_stream_worker(
        self,
        symbols: List[str],
        data_types: Optional[List[DataStreamType]] = None
    ):
        """Worker for running data streams.
        
        Args:
            symbols: List of symbols to subscribe to
            data_types: Optional list of data types to subscribe to (default: all types)
        """
        try:
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
            on_message = await self._create_message_handler()
            
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
                    await self._ws_client.subscribe_candle(symbol, interval="1m")
                    logger.info(f"| 📡 Subscribed to minute-level candle: {symbol} (interval: 1m)")
            
            # Start WebSocket (pass current event loop)
            loop = asyncio.get_running_loop()
            await self._ws_client.start(loop=loop)
            
            logger.info(f"| ✅ All subscriptions completed for {len(symbols)} symbols")
            
            # Keep the stream running
            while self._data_stream_running:
                await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            logger.info("| ℹ️  Data stream worker cancelled")
        except Exception as e:
            logger.error(f"| ❌ Error in data stream worker: {e}", exc_info=True)
    
    async def start(
        self,
        symbols: List[str],
        data_types: Optional[List[DataStreamType]] = None
    ) -> None:
        """Start data stream for given symbols as a background task.
        
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
        
        # Start as a background task (non-blocking)
        self._stream_task = asyncio.create_task(self._data_stream_worker(symbols, data_types))
        
        logger.info("| ✅ Data stream task started (running in background)")
    
    async def stop(self) -> None:
        """Stop the data stream."""
        if not self._data_stream_running:
            logger.warning("| ⚠️  Data stream not running")
            return
        
        logger.info("| 🛑 Stopping data stream...")
        self._data_stream_running = False
        
        # Flush buffered candles before stopping
        try:
            logger.info("| 🔄 Flushing buffered candle data...")
            await self._flush_candle_buffers()
        except Exception as e:
            logger.warning(f"| ⚠️  Error flushing candle buffers: {e}")
        
        # Stop WebSocket client
        try:
            if self._ws_client:
                logger.info("| 🛑 Stopping WebSocket client...")
                await self._ws_client.stop()
        except Exception as e:
            logger.warning(f"| ⚠️  Error stopping WebSocket client: {e}")
        
        # Cancel the stream task
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
        
        logger.info("| ✅ Data stream stopped")
    
    async def _flush_candle_buffers(self) -> None:
        """Flush all buffered candle data before shutdown.
        
        Inserts the latest buffered data for all symbols.
        """
        if not self._candle_buffer:
            logger.info("| ℹ️  No buffered candles to flush")
            return
        
        logger.info(f"| 🔄 Flushing buffers for {len(self._candle_buffer)} symbols...")
        
        for symbol, latest_data in self._candle_buffer.items():
            if not latest_data:
                continue
                
            try:
                current_minute = self._current_minute.get(symbol)
                if current_minute:
                    logger.info(f"| 📤 Flushing latest data for {symbol} @ {current_minute}")
                    # Insert the latest buffered data
                    result = await self._candle_handler.stream_insert(latest_data, symbol)
                    success = result.success if hasattr(result, 'success') else result.get("success", False)
                    message = result.message if hasattr(result, 'message') else result.get("message", "")
                    
                    if success:
                        logger.info(f"| ✅ Candle inserted for {symbol} @ {current_minute}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert candle for {symbol} @ {current_minute}: {message}")
            except Exception as e:
                logger.error(f"| ❌ Error flushing buffer for {symbol}: {e}", exc_info=True)
        
        # Clear all buffers
        self._candle_buffer.clear()
        self._current_minute.clear()
        self._waiting_for_new_minute.clear()
        logger.info("| ✅ Candle buffers flushed and cleared")

