"""Data producer: receives data from Alpaca streams and writes to database."""
from __future__ import annotations
import threading
import asyncio
from typing import Optional, List, Dict, TYPE_CHECKING
import concurrent.futures

from alpaca.data.live import CryptoDataStream, StockDataStream, NewsDataStream
from alpaca.trading.enums import AssetClass

from src.logger import logger
from src.environments.alpacaentry.bars import BarsHandler
from src.environments.alpacaentry.quotes import QuotesHandler
from src.environments.alpacaentry.trades import TradesHandler
from src.environments.alpacaentry.orderbooks import OrderbooksHandler
from src.environments.alpacaentry.news import NewsHandler
from src.environments.alpacaentry.types import DataStreamType
from src.environments.alpacaentry.exceptions import AlpacaError

if TYPE_CHECKING:
    from src.environments.alpacaentry.types import AccountInfo


class DataProducer:
    """Producer: receives data from Alpaca streams and writes to database."""
    
    def __init__(
        self,
        account: 'AccountInfo',
        bars_handler: BarsHandler,
        quotes_handler: QuotesHandler,
        trades_handler: TradesHandler,
        orderbooks_handler: OrderbooksHandler,
        news_handler: NewsHandler,
        symbols: Dict[str, Dict],
        max_concurrent_writes: int = 10
    ):
        """Initialize data producer.
        
        Args:
            account: Alpaca account information
            bars_handler: Bars data handler
            quotes_handler: Quotes data handler
            trades_handler: Trades data handler
            orderbooks_handler: Orderbooks data handler
            news_handler: News data handler
            symbols: Symbol dictionary, format: {symbol: {asset_class: ...}}
            max_concurrent_writes: Maximum concurrent database writes
        """
        self.symbols = symbols
        
        # Initialize data streams
        self._crypto_data_stream = CryptoDataStream(
            api_key=account.api_key,
            secret_key=account.secret_key,
            raw_data=False
        )
        
        self._stock_data_stream = StockDataStream(
            api_key=account.api_key,
            secret_key=account.secret_key,
            raw_data=False
        )
        
        self._news_data_stream = NewsDataStream(
            api_key=account.api_key,
            secret_key=account.secret_key,
            raw_data=False
        )
        
        self._bars_handler = bars_handler
        self._quotes_handler = quotes_handler
        self._trades_handler = trades_handler
        self._orderbooks_handler = orderbooks_handler
        self._news_handler = news_handler
        
        self._data_queue: Optional[asyncio.Queue] = None
        self._data_semaphore: Optional[asyncio.Semaphore] = None
        self._data_stream_running: bool = False
        self._data_stream_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._max_concurrent_writes = max_concurrent_writes
        self._worker_asset_types: Optional[Dict[str, AssetClass]] = None
    
    async def _handle_data(self, data: Dict, 
                           symbol: str, 
                           asset_type: AssetClass, 
                           data_type: DataStreamType) -> None:
        """Handle incoming data and write to database.
        
        Args:
            data: Raw data from Alpaca stream
            symbol: Symbol name
            asset_type: Asset class
            data_type: Data stream type
        """
        async with self._data_semaphore:
            try:
                if data_type == DataStreamType.BARS:
                    result = await self._bars_handler.stream_insert(data, symbol, asset_type)
                    if result:
                        logger.info(f"| ✅ Bars data inserted for {symbol}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert bars data for {symbol}")
                elif data_type == DataStreamType.QUOTES:
                    result = await self._quotes_handler.stream_insert(data, symbol, asset_type)
                    if result:
                        logger.debug(f"| ✅ Quotes data inserted for {symbol}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert quotes data for {symbol}")
                elif data_type == DataStreamType.TRADES:
                    result = await self._trades_handler.stream_insert(data, symbol, asset_type)
                    if result:
                        logger.debug(f"| ✅ Trades data inserted for {symbol}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert trades data for {symbol}")
                elif data_type == DataStreamType.ORDERBOOKS:
                    result = await self._orderbooks_handler.stream_insert(data, symbol)
                    if result:
                        logger.debug(f"| ✅ Orderbooks data inserted for {symbol}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert orderbooks data for {symbol}")
                elif data_type == DataStreamType.NEWS:
                    result = await self._news_handler.stream_insert(data, symbol)
                    if result:
                        logger.debug(f"| ✅ News data inserted for {symbol}")
                    else:
                        logger.warning(f"| ⚠️  Failed to insert news data for {symbol}")
            except Exception as e:
                logger.error(f"| ❌ Error in data handler for {symbol} ({data_type}): {e}", exc_info=True)
    
    async def _data_processor(self) -> None:
        """Background task to process data from queue."""
        logger.info("| 🔄 Data processor started, waiting for data...")
        while self._data_stream_running:
            try:
                item = await asyncio.wait_for(self._data_queue.get(), timeout=1.0)
                if item is None:  # Poison pill
                    logger.info("| 🛑 Data processor received poison pill, stopping...")
                    break
                
                if len(item) == 4:
                    data, symbol, asset_type, data_type_str = item
                    data_type = DataStreamType(data_type_str)
                else:
                    logger.error(f"| ❌ Unexpected queue item format: {item}")
                    continue
                
                await self._handle_data(data, symbol, asset_type, data_type)
                self._data_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"| ❌ Error in data processor: {e}", exc_info=True)
    
    async def _quotes_handler_wrapper(self, data, asset_type: AssetClass, symbol: str):
        """Wrapper for quotes data handler."""
        try:
            data = data.model_dump()
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            if data_symbol and self._data_queue:
                await self._data_queue.put((data, data_symbol, asset_type, "quotes"))
                logger.debug(f"| 📊 Quotes data queued for {data_symbol}")
            else:
                if not data_symbol:
                    logger.warning(f"| ⚠️  Quotes data missing symbol: {data}")
                if not self._data_queue:
                    logger.warning(f"| ⚠️  Data queue not initialized when quotes data received")
        except Exception as e:
            logger.error(f"| ❌ Error in quotes handler wrapper: {e}", exc_info=True)
    
    async def _trades_handler_wrapper(self, data, asset_type: AssetClass, symbol: str):
        """Wrapper for trades data handler."""
        try:
            data = data.model_dump()
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            if data_symbol and self._data_queue:
                await self._data_queue.put((data, data_symbol, asset_type, "trades"))
                logger.debug(f"| 📊 Trades data queued for {data_symbol}")
            else:
                if not data_symbol:
                    logger.warning(f"| ⚠️  Trades data missing symbol: {data}")
                if not self._data_queue:
                    logger.warning(f"| ⚠️  Data queue not initialized when trades data received")
        except Exception as e:
            logger.error(f"| ❌ Error in trades handler wrapper: {e}", exc_info=True)
    
    async def _bars_handler_wrapper(self, data, asset_type: AssetClass, symbol: str):
        """Wrapper for bars data handler."""
        try:
            data = data.model_dump()
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            if data_symbol and self._data_queue:
                await self._data_queue.put((data, data_symbol, asset_type, "bars"))
                logger.info(f"| 📊 Bars data queued for {data_symbol}")
            else:
                if not data_symbol:
                    logger.warning(f"| ⚠️  Bars data missing symbol: {data}")
                if not self._data_queue:
                    logger.warning(f"| ⚠️  Data queue not initialized when bars data received")
        except Exception as e:
            logger.error(f"| ❌ Error in bars handler wrapper: {e}", exc_info=True)
    
    async def _orderbooks_handler_wrapper(self, data, asset_type: AssetClass, symbol: str):
        """Wrapper for orderbooks data handler."""
        try:
            data = data.model_dump()
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            if data_symbol and self._data_queue:
                await self._data_queue.put((data, data_symbol, asset_type, "orderbooks"))
                logger.debug(f"| 📊 Orderbooks data queued for {data_symbol}")
            else:
                if not data_symbol:
                    logger.warning(f"| ⚠️  Orderbooks data missing symbol: {data}")
                if not self._data_queue:
                    logger.warning(f"| ⚠️  Data queue not initialized when orderbooks data received")
        except Exception as e:
            logger.error(f"| ❌ Error in orderbooks handler wrapper: {e}", exc_info=True)
    
    async def _news_handler_wrapper(self, data, asset_type: AssetClass, symbol: str):
        """Wrapper for news data handler."""
        try:
            data = data.model_dump()
            data_symbol = data.get("symbols", [])
            if data_symbol:
                data_symbol = data_symbol[0] if isinstance(data_symbol, list) else data_symbol
            else:
                data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol if symbol else ""
            if self._data_queue:
                await self._data_queue.put((data, data_symbol, asset_type, "news"))
                logger.debug(f"| 📊 News data queued for {data_symbol if data_symbol else 'global'}")
            else:
                logger.warning(f"| ⚠️  Data queue not initialized when news data received")
        except Exception as e:
            logger.error(f"| ❌ Error in news handler wrapper: {e}", exc_info=True)
    
    def _data_stream_worker(self, symbols: List[str], asset_types: Dict[str, AssetClass]):
        """Worker thread for running data streams."""
        loop = None
        self._worker_asset_types = asset_types
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
                
                # Ensure tables exist for all data types
                for symbol in symbols:
                    asset_type = asset_types.get(symbol, AssetClass.US_EQUITY)
                    if self._bars_handler:
                        await self._bars_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Bars table created/verified for {symbol}")
                    if self._quotes_handler:
                        await self._quotes_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Quotes table created/verified for {symbol}")
                    if self._trades_handler:
                        await self._trades_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Trades table created/verified for {symbol}")
                    if asset_type == AssetClass.CRYPTO and self._orderbooks_handler:
                        await self._orderbooks_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Orderbooks table created/verified for {symbol}")
                
                if self._news_handler:
                    await self._news_handler.ensure_table_exists(symbols[0] if symbols else "default")
                logger.info(f"| ✅ News table created/verified")
                
                # Subscribe to streams
                for symbol in symbols:
                    asset_type = asset_types.get(symbol, AssetClass.US_EQUITY)
                    
                    def create_handler(handler_wrapper_func, sym, atype):
                        async def handler(data):
                            if not self._data_stream_running:
                                return
                            try:
                                if self._event_loop and self._event_loop.is_running() and not self._event_loop.is_closed():
                                    asyncio.run_coroutine_threadsafe(
                                        handler_wrapper_func(data, atype, sym),
                                        self._event_loop
                                    )
                                else:
                                    logger.debug(f"| Event loop not available when handler called for {sym}")
                            except RuntimeError as e:
                                if "interpreter shutdown" in str(e) or "cannot schedule" in str(e).lower():
                                    logger.debug(f"| Cannot schedule coroutine (interpreter shutdown): {e}")
                                else:
                                    logger.warning(f"| ⚠️  Error scheduling coroutine for {sym}: {e}")
                            except Exception as e:
                                logger.debug(f"| Error in handler for {sym}: {e}")
                        return handler
                    
                    quotes_handler = create_handler(self._quotes_handler_wrapper, symbol, asset_type)
                    trades_handler = create_handler(self._trades_handler_wrapper, symbol, asset_type)
                    bars_handler = create_handler(self._bars_handler_wrapper, symbol, asset_type)
                    orderbooks_handler = create_handler(self._orderbooks_handler_wrapper, symbol, asset_type)
                    news_handler = create_handler(self._news_handler_wrapper, symbol, asset_type)
                    
                    if asset_type == AssetClass.CRYPTO:
                        self._crypto_data_stream.subscribe_quotes(quotes_handler, symbol)
                        self._crypto_data_stream.subscribe_trades(trades_handler, symbol)
                        self._crypto_data_stream.subscribe_bars(bars_handler, symbol)
                        self._crypto_data_stream.subscribe_orderbooks(orderbooks_handler, symbol)
                        logger.info(f"| 📡 Subscribed to crypto data (quotes, trades, bars, orderbooks) for {symbol}")
                    elif asset_type == AssetClass.US_EQUITY:
                        self._stock_data_stream.subscribe_quotes(quotes_handler, symbol)
                        self._stock_data_stream.subscribe_trades(trades_handler, symbol)
                        self._stock_data_stream.subscribe_bars(bars_handler, symbol)
                        logger.info(f"| 📡 Subscribed to stock data (quotes, trades, bars) for {symbol}")
                    
                    self._news_data_stream.subscribe_news(news_handler, symbol)
                    logger.info(f"| 📡 Subscribed to news data for {symbol}")
                
                logger.info(f"| ✅ All subscriptions completed for {len(symbols)} symbols")
                
                def run_crypto_stream():
                    try:
                        if self._crypto_data_stream:
                            logger.info("| 🚀 Starting crypto data stream...")
                            self._crypto_data_stream.run()
                    except Exception as e:
                        logger.error(f"| ❌ Error in crypto stream: {e}")
                
                def run_stock_stream():
                    try:
                        if self._stock_data_stream:
                            logger.info("| 🚀 Starting stock data stream...")
                            self._stock_data_stream.run()
                    except Exception as e:
                        logger.error(f"| ❌ Error in stock stream: {e}")
                
                def run_news_stream():
                    try:
                        if self._news_data_stream:
                            logger.info("| 🚀 Starting news data stream...")
                            self._news_data_stream.run()
                    except Exception as e:
                        logger.error(f"| ❌ Error in news stream: {e}")
                
                executor = None
                futures = []
                try:
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                    if self._crypto_data_stream:
                        futures.append(executor.submit(run_crypto_stream))
                    if self._stock_data_stream:
                        futures.append(executor.submit(run_stock_stream))
                    if self._news_data_stream:
                        futures.append(executor.submit(run_news_stream))
                    
                    logger.info(f"| ✅ All streams started in background threads")
                    
                    try:
                        while self._data_stream_running:
                            await asyncio.sleep(1.0)
                            if processor_task.done():
                                logger.warning("| ⚠️  Data processor task completed unexpectedly")
                                break
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"| ❌ Error in processor: {e}")
                    finally:
                        logger.info("| 🛑 Stopping streams...")
                        if self._crypto_data_stream:
                            try:
                                self._crypto_data_stream.stop()
                            except Exception as e:
                                logger.debug(f"| Error stopping crypto stream: {e}")
                        if self._stock_data_stream:
                            try:
                                self._stock_data_stream.stop()
                            except Exception as e:
                                logger.debug(f"| Error stopping stock stream: {e}")
                        if self._news_data_stream:
                            try:
                                self._news_data_stream.stop()
                            except Exception as e:
                                logger.debug(f"| Error stopping news stream: {e}")
                        
                        try:
                            await asyncio.sleep(0.1)
                        except:
                            pass
                        
                        if not processor_task.done():
                            try:
                                await self._data_queue.put(None)
                            except (RuntimeError, asyncio.CancelledError):
                                pass
                            try:
                                processor_task.cancel()
                            except:
                                pass
                        try:
                            await processor_task
                        except (asyncio.CancelledError, RuntimeError):
                            pass
                        
                        for future in futures:
                            try:
                                future.cancel()
                            except:
                                pass
                        if executor:
                            try:
                                executor.shutdown(wait=False, cancel_futures=True)
                            except (RuntimeError, Exception) as e:
                                logger.debug(f"| Executor shutdown error (expected during shutdown): {e}")
                except Exception as e:
                    logger.error(f"| ❌ Error in stream executor: {e}")
                    if executor:
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except:
                            pass
            
            loop.run_until_complete(setup_and_run())
            
        except KeyboardInterrupt:
            logger.info("| 🛑 Data stream stopped by user")
            self._data_stream_running = False
        except Exception as e:
            logger.error(f"| ❌ Error in data stream worker: {e}")
            self._data_stream_running = False
        finally:
            self._data_stream_running = False
            self._event_loop = None
            if loop:
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        try:
                            task.cancel()
                        except:
                            pass
                    if pending:
                        try:
                            loop.run_until_complete(asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=1.0
                            ))
                        except (asyncio.TimeoutError, RuntimeError):
                            pass
                except Exception as e:
                    logger.debug(f"| Error cancelling tasks: {e}")
                try:
                    loop.close()
                except Exception as e:
                    logger.debug(f"| Error closing loop: {e}")
    
    def start(self, symbols: List[str], asset_types: Optional[Dict[str, AssetClass]] = None) -> None:
        """Start data stream.
        
        Args:
            symbols: List of symbols to subscribe to
            asset_types: Optional dictionary mapping symbol to asset class
        """
        if self._data_stream_running:
            logger.warning("| ⚠️  Data stream is already running")
            return
        
        # Determine asset types if not provided
        if asset_types is None:
            asset_types = {}
        
        for symbol in symbols:
            if symbol not in self.symbols:
                logger.warning(f"| ⚠️  Symbol {symbol} not found in symbols list. Trying to determine asset class from symbol format...")
                if "/" in symbol:
                    asset_types[symbol] = AssetClass.CRYPTO
                    logger.info(f"| 📝 Detected {symbol} as CRYPTO based on symbol format")
                else:
                    asset_types[symbol] = AssetClass.US_EQUITY
                    logger.info(f"| 📝 Detected {symbol} as US_EQUITY based on symbol format")
            else:
                asset_types[symbol] = self.symbols[symbol]['asset_class']
        
        self._data_stream_running = True
        
        self._data_stream_thread = threading.Thread(
            target=self._data_stream_worker,
            args=(symbols, asset_types),
            daemon=True
        )
        self._data_stream_thread.start()
        logger.info(f"| 🚀 Started data stream for {len(symbols)} symbols (non-blocking)")
    
    def stop(self) -> None:
        """Stop data stream."""
        if not self._data_stream_running:
            logger.warning("| ⚠️  Data stream is not running")
            return
        
        logger.info("| 🛑 Stopping data stream...")
        self._data_stream_running = False
        
        if self._crypto_data_stream:
            try:
                self._crypto_data_stream.stop()
            except Exception as e:
                logger.debug(f"| Error stopping crypto stream: {e}")
        
        if self._stock_data_stream:
            try:
                self._stock_data_stream.stop()
            except Exception as e:
                logger.debug(f"| Error stopping stock stream: {e}")
        
        if self._news_data_stream:
            try:
                self._news_data_stream.stop()
            except Exception as e:
                logger.debug(f"| Error stopping news stream: {e}")
        
        if self._data_stream_thread and self._data_stream_thread.is_alive():
            try:
                self._data_stream_thread.join(timeout=5)
                if self._data_stream_thread.is_alive():
                    logger.warning("| ⚠️  Data stream thread did not finish within timeout")
            except Exception as e:
                logger.debug(f"| Error joining thread: {e}")
        
        logger.info("| 🛑 Data stream stopped")

