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
                    result = await self._candle_handler.stream_insert(data, symbol)
                    if result.get("success"):
                        logger.info(f"| ✅ Candle data inserted for {symbol} (timestamp: {data.get('timestamp')})")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert candle data for {symbol}: {result.get('message')}")
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
    
    async def _data_processor(self) -> None:
        """Process data from queue."""
        while self._data_stream_running:
            try:
                # Get data from queue with timeout
                data, symbol, data_type = await asyncio.wait_for(
                    self._data_queue.get(), timeout=1.0
                )
                await self._handle_data(data, symbol, data_type)
                self._data_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"| ❌ Error in data processor: {e}", exc_info=True)
    
    def _create_message_handler(self):
        """Create unified message handler for WebSocket client."""
        def on_message(ws, channel: str, data: Dict):
            """Handle incoming processed data from WebSocket.
            
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
                
                symbol = data["symbol"].lower()
                
                # Map channel to DataStreamType
                if channel == "candle":
                    data_type = DataStreamType.CANDLE
                    logger.info(f"| 📡 Received candle data for {symbol} (timestamp: {data.get('timestamp')})")
                elif channel == "trades":
                    data_type = DataStreamType.TRADES
                    logger.debug(f"| 📡 Received trades data for {symbol}")
                elif channel == "l2Book":
                    data_type = DataStreamType.L2BOOK
                    logger.debug(f"| 📡 Received l2Book data for {symbol}")
                else:
                    logger.warning(f"| 📊 Unknown channel: {channel}")
                    return
                
                # Run the handler in the event loop
                if self._event_loop and self._event_loop.is_running() and not self._event_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._data_handler_wrapper(data, symbol, data_type),
                        self._event_loop
                    )
                else:
                    logger.warning(f"| ⚠️  Event loop not available for processing {channel} data")
                    
            except RuntimeError as e:
                if "interpreter shutdown" in str(e) or "cannot schedule" in str(e).lower():
                    logger.debug(f"| Cannot schedule coroutine (interpreter shutdown): {e}")
                else:
                    logger.warning(f"| ⚠️  Error scheduling coroutine: {e}")
            except Exception as e:
                logger.error(f"| ❌ Error in message handler: {e}", exc_info=True)
        
        return on_message
    
    async def _data_handler_wrapper(self, processed_data: Dict, symbol: str, data_type: DataStreamType):
        """Wrapper for data handler.
        
        Args:
            processed_data: Processed data from WebSocket
            symbol: Symbol name (lowercase)
            data_type: Data stream type
        """
        try:
            if self._data_queue:
                await self._data_queue.put((processed_data, symbol, data_type))
            else:
                logger.warning(f"| ⚠️  Data queue not initialized when {data_type.value} data received")
        except Exception as e:
            logger.error(f"| ❌ Error in data handler wrapper: {e}", exc_info=True)
    
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
                self._data_stream_running = True
                self._data_queue = asyncio.Queue(maxsize=1000)
                self._data_semaphore = asyncio.Semaphore(self._max_concurrent_writes)
                
                logger.info(f"| ✅ Data queue initialized")
                
                processor_task = asyncio.create_task(self._data_processor())
                logger.info(f"| ✅ Data processor started")
                
                # Default to all data types if not specified
                if data_types is None:
                    data_types = [DataStreamType.CANDLE, DataStreamType.TRADES, DataStreamType.L2BOOK]
                
                # Ensure tables exist for all symbols
                for symbol in symbols:
                    if DataStreamType.CANDLE in data_types:
                        await self._candle_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Candle table created/verified for {symbol}")
                    if DataStreamType.TRADES in data_types:
                        await self._trades_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Trades table created/verified for {symbol}")
                    if DataStreamType.L2BOOK in data_types:
                        await self._l2book_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ L2Book table created/verified for {symbol}")
                
                # Create unified message handler
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
                
                # Subscribe to data streams
                for symbol in symbols:
                    if DataStreamType.CANDLE in data_types:
                        self._ws_client.subscribe_candle(symbol, interval="1m")
                        logger.info(f"| 📡 Subscribed to candle: {symbol}")
                    if DataStreamType.TRADES in data_types:
                        self._ws_client.subscribe_trades(symbol)
                        logger.info(f"| 📡 Subscribed to trades: {symbol}")
                    if DataStreamType.L2BOOK in data_types:
                        self._ws_client.subscribe_l2book(symbol)
                        logger.info(f"| 📡 Subscribed to l2Book: {symbol}")
                
                # Start WebSocket
                self._ws_client.start()
                
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
        
        # Stop WebSocket client
        try:
            if self._ws_client:
                self._ws_client.stop()
        except Exception as e:
            logger.warning(f"| ⚠️  Error stopping WebSocket client: {e}")
        
        # Wait for thread to finish
        if self._data_stream_thread and self._data_stream_thread.is_alive():
            self._data_stream_thread.join(timeout=5.0)
        
        logger.info("| ✅ Data stream stopped")

