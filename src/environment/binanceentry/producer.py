"""Data producer: receives data from Binance WebSocket streams and writes to database."""
from __future__ import annotations
import threading
import asyncio
from typing import Optional, List, Dict, TYPE_CHECKING

from src.logger import logger
from src.environment.binanceentry.klines import KlinesHandler
from src.environment.binanceentry.types import DataStreamType, TradeType
from src.environment.binanceentry.spot_websocket import BinanceSpotWebSocket
from src.environment.binanceentry.futures_websocket import BinanceFuturesWebSocket

if TYPE_CHECKING:
    from src.environment.binanceentry.types import AccountInfo


class DataProducer:
    """Producer: receives data from Binance WebSocket streams and writes to database."""
    
    def __init__(
        self,
        account: 'AccountInfo',
        klines_handler: KlinesHandler,
        symbols: Dict[str, Dict],
        max_concurrent_writes: int = 10,
        testnet: bool = False,
        default_trade_type: TradeType = TradeType.PERPETUAL,
    ):
        """Initialize data producer.
        
        Args:
            account: Binance account information
            klines_handler: Klines data handler
            symbols: Symbol dictionary
            max_concurrent_writes: Maximum concurrent database writes
            testnet: Whether to use testnet
            default_trade_type: Default trade type to use (SPOT or PERPETUAL). Default: PERPETUAL
        """
        self.account = account
        self.symbols = symbols
        self.default_trade_type = default_trade_type
        
        # Initialize WebSocket clients
        self._spot_ws_client: Optional[BinanceSpotWebSocket] = None
        self._futures_ws_client: Optional[BinanceFuturesWebSocket] = None
        self.testnet = testnet
        
        self._klines_handler = klines_handler
        
        self._data_queue: Optional[asyncio.Queue] = None
        self._data_semaphore: Optional[asyncio.Semaphore] = None
        self._data_stream_running: bool = False
        self._data_stream_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._max_concurrent_writes = max_concurrent_writes
        self._active_subscriptions: Dict[str, str] = {}  # symbol -> stream_id
    
    async def _handle_data(self, data: Dict, symbol: str, data_type: DataStreamType) -> None:
        """Handle incoming data and write to database.
        
        Args:
            data: Raw data from Binance WebSocket stream
            symbol: Symbol name
            data_type: Data stream type
        """
        async with self._data_semaphore:
            try:
                if data_type == DataStreamType.KLINES:
                    result = await self._klines_handler.stream_insert(data, symbol)
                    if result.success:
                        logger.info(f"| ✅ Klines data inserted for {symbol} (timestamp: {data.get('timestamp')})")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert klines data for {symbol}: {result.message}")
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
    
    async def _klines_handler_wrapper(self, processed_data: Dict, symbol: str):
        """Wrapper for klines data handler.
        
        Args:
            processed_data: Processed kline data from WebSocket (already filtered for closed 1m klines)
            symbol: Symbol name (lowercase)
        """
        try:
            # Data is already processed by WebSocket class (only closed 1m klines)
            # Format: {
            #   "symbol": "BTCUSDT",
            #   "interval": "1m",
            #   "timestamp": "YYYY-MM-DD HH:MM:SS" (minute start time, UTC),
            #   "open_time": "YYYY-MM-DD HH:MM:SS" (minute start time, UTC),
            #   "close_time": "YYYY-MM-DD HH:MM:SS" (minute end time, UTC),
            #   "open": float, "high": float, "low": float, "close": float,
            #   "volume": float, "trade_count": int, ...
            # }
            if isinstance(processed_data, dict) and "symbol" in processed_data:
                # Use original symbol from processed_data (UPPERCASE) for consistency
                original_symbol = processed_data["symbol"].upper()
                # Update processed_data to use lowercase symbol for database operations
                processed_data["symbol"] = original_symbol
                if self._data_queue:
                    await self._data_queue.put((processed_data, symbol, DataStreamType.KLINES))
                    logger.info(f"| 📊 Klines data queued for {symbol} (original: {original_symbol}, timestamp: {processed_data.get('timestamp')})")
                else:
                    logger.warning(f"| ⚠️  Data queue not initialized when klines data received")
            else:
                logger.warning(f"| 📊 Received unexpected data format for {symbol}: {processed_data}")
        except Exception as e:
            logger.error(f"| ❌ Error in klines handler wrapper: {e}", exc_info=True)
    
    def _create_message_handler(self):
        """Create unified message handler for WebSocket client.
        
        Note: WebSocket classes now pass processed data (only closed 1m klines)
        with format: {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "timestamp": "YYYY-MM-DD HH:MM:SS" (minute start time, UTC),
            "open_time": "YYYY-MM-DD HH:MM:SS" (minute start time, UTC),
            "close_time": "YYYY-MM-DD HH:MM:SS" (minute end time, UTC),
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            ...
        }
        """
        def on_message(ws, processed_data):
            """Handle incoming processed kline data from WebSocket.
            
            Args:
                ws: WebSocket connection
                processed_data: Processed kline data (dict) - already filtered for closed 1m klines
            """
            try:
                if not self._data_stream_running:
                    return
                
                # Data is already processed by WebSocket class (only closed 1m klines)
                if isinstance(processed_data, dict) and "symbol" in processed_data:
                    symbol = processed_data["symbol"].lower()
                    logger.info(f"| 📡 Received klines data for {symbol} (timestamp: {processed_data.get('timestamp')})")
                    # Run the handler in the event loop
                    if self._event_loop and self._event_loop.is_running() and not self._event_loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._klines_handler_wrapper(processed_data, symbol),
                            self._event_loop
                        )
                    else:
                        logger.warning(f"| ⚠️  Event loop not available for processing klines data")
                else:
                    logger.warning(f"| 📊 Received unexpected data format: {processed_data}")
                    
            except RuntimeError as e:
                if "interpreter shutdown" in str(e) or "cannot schedule" in str(e).lower():
                    logger.debug(f"| Cannot schedule coroutine (interpreter shutdown): {e}")
                else:
                    logger.warning(f"| ⚠️  Error scheduling coroutine: {e}")
            except Exception as e:
                logger.error(f"| ❌ Error in message handler: {e}", exc_info=True)
        
        return on_message
    
    def _data_stream_worker(self, symbols: List[str], intervals: Optional[Dict[str, str]] = None, trade_types: Optional[Dict[str, TradeType]] = None):
        """Worker thread for running data streams.
        
        Args:
            symbols: List of symbols to subscribe to
            intervals: Optional dictionary mapping symbol to interval (e.g., "1m", "5m", "1h")
            trade_types: Optional dictionary mapping symbol to trade type (SPOT or PERPETUAL)
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
                
                # Ensure tables exist for all symbols
                for symbol in symbols:
                    await self._klines_handler.ensure_table_exists(symbol)
                    logger.info(f"| ✅ Klines table created/verified for {symbol}")
                
                # Create symbol to interval mapping
                symbol_to_interval = {
                    symbol: (intervals or {}).get(symbol, "1m") 
                    for symbol in symbols
                }
                
                # Create symbol to trade type mapping (use default_trade_type if not specified)
                symbol_to_trade_type = {
                    symbol: (trade_types or {}).get(symbol, self.default_trade_type)
                    for symbol in symbols
                }
                
                # Create unified message handler
                on_message = self._create_message_handler()
                
                # Separate symbols by trade type
                spot_symbols = [s for s in symbols if symbol_to_trade_type[s] == TradeType.SPOT]
                futures_symbols = [s for s in symbols if symbol_to_trade_type[s] == TradeType.PERPETUAL]
                
                # Initialize Spot WebSocket if needed
                if spot_symbols:
                    logger.info(f"| 🚀 Initializing Spot WebSocket for {len(spot_symbols)} symbols...")
                    self._spot_ws_client = BinanceSpotWebSocket(
                        on_message=on_message,
                        on_error=lambda ws, err: logger.error(f"| ❌ Spot WS error: {err}"),
                        on_close=lambda ws: logger.info("| 🛑 Spot WebSocket closed"),
                        on_open=lambda ws: logger.info("| ✅ Spot WebSocket opened"),
                        testnet=self.testnet
                    )
                    
                    # Subscribe to spot klines
                    for symbol in spot_symbols:
                        interval = symbol_to_interval[symbol]
                        self._spot_ws_client.subscribe_kline(symbol, interval)
                        stream_name = f"{symbol.lower()}@kline_{interval}"
                        self._active_subscriptions[symbol] = stream_name
                        logger.info(f"| 📡 Subscribed to spot klines: {symbol} (interval: {interval})")
                    
                    self._spot_ws_client.start()
                
                # Initialize Futures WebSocket if needed
                if futures_symbols:
                    logger.info(f"| 🚀 Initializing Futures WebSocket for {len(futures_symbols)} symbols...")
                    self._futures_ws_client = BinanceFuturesWebSocket(
                        on_message=on_message,
                        on_error=lambda ws, err: logger.error(f"| ❌ Futures WS error: {err}"),
                        on_close=lambda ws: logger.info("| 🛑 Futures WebSocket closed"),
                        on_open=lambda ws: logger.info("| ✅ Futures WebSocket opened"),
                        testnet=self.testnet
                    )
                    
                    # Subscribe to futures klines
                    for symbol in futures_symbols:
                        interval = symbol_to_interval[symbol]
                        self._futures_ws_client.subscribe_kline(symbol, interval)
                        stream_name = f"{symbol.lower()}@kline_{interval}"
                        self._active_subscriptions[symbol] = stream_name
                        logger.info(f"| 📡 Subscribed to futures klines: {symbol} (interval: {interval})")
                    
                    self._futures_ws_client.start()
                
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
    
    def start(self, symbols: List[str], intervals: Optional[Dict[str, str]] = None, trade_types: Optional[Dict[str, TradeType]] = None) -> None:
        """Start data stream for given symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            intervals: Optional dictionary mapping symbol to interval (e.g., "1m", "5m", "1h")
            trade_types: Optional dictionary mapping symbol to trade type (SPOT or PERPETUAL)
        """
        if self._data_stream_running:
            logger.warning("| ⚠️  Data stream already running")
            return
        
        # Normalize symbols to uppercase for consistency
        symbols = [s.upper() for s in symbols]
        
        logger.info(f"| 📡 Starting data stream for {len(symbols)} symbols: {symbols}")
        
        self._data_stream_thread = threading.Thread(
            target=self._data_stream_worker,
            args=(symbols, intervals, trade_types),
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
        
        # Stop WebSocket clients
        try:
            if self._spot_ws_client:
                self._spot_ws_client.stop()
            if self._futures_ws_client:
                self._futures_ws_client.stop()
        except Exception as e:
            logger.warning(f"| ⚠️  Error stopping WebSocket clients: {e}")
        
        # Wait for thread to finish
        if self._data_stream_thread and self._data_stream_thread.is_alive():
            self._data_stream_thread.join(timeout=5.0)
        
        logger.info("| ✅ Data stream stopped")

