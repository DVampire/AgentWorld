"""Alpaca trading service implementation using alpaca-py."""
import threading
import asyncio
from typing import Optional, Union, List, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)
import concurrent.futures

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest as AlpacaGetAssetsRequest,
    MarketOrderRequest,
    GetOrdersRequest as AlpacaGetOrdersRequest,
)
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import (StockHistoricalDataClient,
                                    CryptoHistoricalDataClient,
                                    NewsClient,
                                    OptionHistoricalDataClient)
from alpaca.data.live import (CryptoDataStream,
                              StockDataStream,
                              NewsDataStream, 
                              OptionDataStream)
from alpaca.common.exceptions import APIError
from pydantic import BaseModel

from src.logger import logger
from src.environments.protocol.types import ActionResult
from src.environments.alpacaentry.types import (
    GetAccountRequest,
    GetAssetsRequest,
    GetPositionsRequest,
    GetDataRequest,
    CreateOrderRequest,
    GetOrdersRequest,
    GetOrderRequest,
    CancelOrderRequest,
    CancelAllOrdersRequest,
)
from src.environments.alpacaentry.exceptions import (
    AlpacaError,
    AuthenticationError,
)
from src.environments.alpacaentry.bars import BarsHandler
from src.environments.alpacaentry.quotes import QuotesHandler
from src.environments.alpacaentry.trades import TradesHandler
from src.environments.alpacaentry.orderbooks import OrderbooksHandler
from src.environments.alpacaentry.news import NewsHandler
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environments.alpacaentry.types import DataStreamType
from src.utils import assemble_project_path

class AccountInfo(BaseModel):
    """Alpaca account model."""
    api_key: str
    secret_key: str
    name: str

class AlpacaService:
    """Alpaca paper trading service using alpaca-py."""

    def __init__(
        self,
        base_dir: Union[str, Path],
        accounts: List[Dict[str, str]],
        live: bool = False,
        auto_start_data_stream: bool = True,
        symbol: Optional[Union[str, List[str]]] = None,
        data_type: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize Alpaca paper trading service.
        
        Args:
            base_dir: Base directory for Alpaca operations
            accounts: Dictionary of accounts, each containing API key and secret key
            live: Whether to use live trading
            
            accounts = [
                {
                    "name": "Account 1",
                    "api_key": "api_key_1",
                    "secret_key": "secret_key_1",
                },
                {
                    "name": "Account 2",
                    "api_key": "api_key_2",
                    "secret_key": "secret_key_2",
                }
            ]
        """
        self.base_dir = Path(assemble_project_path(base_dir))
        
        self.auto_start_data_stream = auto_start_data_stream
        
        self.default_account = AccountInfo(**accounts[0])
        self.accounts: Dict[str, AccountInfo] = {
           account["name"]: AccountInfo(**account) for account in accounts
        }
        self.live = live
        
        self.symbol = symbol
        self.data_type = data_type
        
        self._trading_clients: Dict[str, TradingClient] = None
        
        self._stock_data_client: Optional[StockHistoricalDataClient] = None
        self._crypto_data_client: Optional[CryptoHistoricalDataClient] = None
        self._news_client: Optional[NewsClient] = None
        self._option_data_client: Optional[OptionHistoricalDataClient] = None
        
        self._crypto_data_stream: Optional[CryptoDataStream] = None
        self._stock_data_stream: Optional[StockDataStream] = None
        self._news_data_stream: Optional[NewsDataStream] = None
        self._option_data_stream: Optional[OptionDataStream] = None
        
        # Initialize data handlers
        self._bars_handler: Optional[BarsHandler] = None
        self._quotes_handler: Optional[QuotesHandler] = None
        self._orderbooks_handler: Optional[OrderbooksHandler] = None
        self._trades_handler: Optional[TradesHandler] = None
        self._news_handler: Optional[NewsHandler] = None
        
        self._data_queue: Optional[asyncio.Queue] = None
        self._data_semaphore: Optional[asyncio.Semaphore] = None
        self._data_stream_running: bool = False
        self._data_stream_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._max_concurrent_writes: int = 10 # Max concurrent database writes

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the Alpaca paper trading service.
        
        Args:
            auto_start_data_stream: If True, automatically start data stream after initialization
        """
        try:
            # Initialize trading client (paper trading only)
            self._trading_clients = {
                account.name: TradingClient(
                    api_key=account.api_key,
                    secret_key=account.secret_key,
                    paper=not self.live
                ) for account in self.accounts.values()
            }
            self._default_trading_client = self._trading_clients[self.default_account.name]
            
            # Initialize data client
            self._stock_data_client = StockHistoricalDataClient(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key
            )
            
            self._crypto_data_client = CryptoHistoricalDataClient(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key
            )
            
            self._news_client = NewsClient(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key
            )
            
            self._option_data_client = OptionHistoricalDataClient(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key
            )
            
            self._crypto_data_stream = CryptoDataStream(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key,
                raw_data=False
            )
            
            self._stock_data_stream = StockDataStream(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key,
                raw_data=False
            )
            
            self._news_data_stream = NewsDataStream(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key,
                raw_data=False
            )
            
            self._option_data_stream = OptionDataStream(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key,
                raw_data=False
            )
            
            # Test connection by getting account info
            for account_name, account in self.accounts.items():
                account = self._trading_clients[account_name].get_account()
                logger.info(f"| 📝 Connected to Alpaca paper trading account: {account.account_number}")
            
            self.symbols = {}
            # Stock Symbols
            stock_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY))
            stock_symbols = stock_symbols.extra["assets"]
            self.symbols.update({symbol['symbol']: symbol for symbol in stock_symbols})
            logger.info(f"| 📝 Found {len(stock_symbols)} stock symbols.")
            
            # Crypto Symbols
            crypto_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.CRYPTO))
            crypto_symbols = crypto_symbols.extra["assets"]
            self.symbols.update({symbol['symbol']: symbol for symbol in crypto_symbols})
            logger.info(f"| 📝 Found {len(crypto_symbols)} crypto symbols.")
            
            # Perpetual Futures Crypto Symbols
            perpetual_futures_crypto_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.CRYPTO_PERP))
            perpetual_futures_crypto_symbols = perpetual_futures_crypto_symbols.extra["assets"]
            self.symbols.update({symbol['symbol']: symbol for symbol in perpetual_futures_crypto_symbols})
            logger.info(f"| 📝 Found {len(perpetual_futures_crypto_symbols)} perpetual futures crypto symbols.")
            
            # Option Symbols
            option_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_OPTION))
            option_symbols = option_symbols.extra["assets"]
            self.symbols.update({symbol['symbol']: symbol for symbol in option_symbols})
            logger.info(f"| 📝 Found {len(option_symbols)} option symbols.")
            
            logger.info(f"| 📝 Found {len(self.symbols)} total symbols.")
            logger.info(f"| 📝 Symbols: {', '.join([symbol for symbol in self.symbols.keys()])}")
            
            self.database_base_dir = self.base_dir / "database"
            self.database_base_dir.mkdir(parents=True, exist_ok=True)
            self.database_service = DatabaseService(self.database_base_dir)
            await self.database_service.connect()
            
            # Initialize data handlers
            self._bars_handler = BarsHandler(self.database_service)
            self._quotes_handler = QuotesHandler(self.database_service)
            self._trades_handler = TradesHandler(self.database_service)
            self._orderbooks_handler = OrderbooksHandler(self.database_service)
            self._news_handler = NewsHandler(self.database_service)
            
            # Optionally start data stream automatically
            if self.auto_start_data_stream and self.symbol:
                # Normalize symbol to list
                symbols_list = self.symbol if isinstance(self.symbol, list) else [self.symbol]
                self.start_data_stream(symbols_list)
                logger.info(f"| 📡 Auto-started data stream for {len(symbols_list)} symbols: {symbols_list}")
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Invalid Alpaca credentials: {e}")
            raise AlpacaError(f"Failed to initialize Alpaca service: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to initialize Alpaca service: {e}.")

    async def cleanup(self) -> None:
        """Cleanup the Alpaca service."""
        # Stop data stream first to ensure proper cleanup
        if self._data_stream_running:
            logger.info("| 🛑 Stopping data stream during cleanup...")
            self.stop_data_stream()
            # Wait a bit for threads to finish
            import time
            time.sleep(0.5)
        
        self._trading_clients = None
        self._default_trading_client = None
        
        self._stock_data_client = None
        self._crypto_data_client = None
        self._news_client = None
        self._option_data_client = None
        
        self._crypto_data_stream = None
        self._stock_data_stream = None
        self._news_data_stream = None
        self._option_data_stream = None
        
        self._trades_handler = None
        self._quotes_handler = None
        self._bars_handler = None
        self._orderbooks_handler = None
        self._news_handler = None

    # Account methods
    async def get_account(self, request: GetAccountRequest) -> ActionResult:
        """Get account information."""
        try:
            account = self._trading_clients[request.account_name].get_account()
            
            account_info = {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "buying_power": account.buying_power,
                "cash": account.cash,
                "portfolio_value": account.portfolio_value,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "created_at": account.created_at,
                "trade_suspended_by_user": account.trade_suspended_by_user,
                "multiplier": account.multiplier,
                "shorting_enabled": account.shorting_enabled,
                "equity": account.equity,
                "last_equity": account.last_equity,
                "long_market_value": account.long_market_value,
                "short_market_value": account.short_market_value,
                "initial_margin": account.initial_margin,
                "maintenance_margin": account.maintenance_margin,
                "last_maintenance_margin": account.last_maintenance_margin,
                "sma": account.sma,
                "daytrade_count": account.daytrade_count
            }
            
            return ActionResult(
                success=True,
                message="Account information retrieved successfully.",
                extra={"account": account_info}
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            raise AlpacaError(f"Failed to get account: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to get account: {e}.")
        
    async def get_assets(self, request: GetAssetsRequest) -> ActionResult:
        """Get assets information."""
        try:
            # Convert string parameters to Alpaca enums if provided
            alpaca_status = None
            if request.status:
                try:
                    # Try to convert string to AssetStatus enum
                    if isinstance(request.status, str):
                        # Try direct conversion first
                        try:
                            alpaca_status = AssetStatus(request.status)
                        except ValueError:
                            # Try case-insensitive match
                            for status in AssetStatus:
                                if status.value.upper() == request.status.upper():
                                    alpaca_status = status
                                    break
                    else:
                        alpaca_status = request.status
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to convert status '{request.status}' to AssetStatus: {e}")
            
            alpaca_asset_class = None
            if request.asset_class:
                try:
                    # Try to convert string to AssetClass enum
                    if isinstance(request.asset_class, str):
                        # Try direct conversion first
                        try:
                            alpaca_asset_class = AssetClass(request.asset_class)
                        except ValueError:
                            # Try case-insensitive match
                            for asset_class in AssetClass:
                                if asset_class.value.upper() == request.asset_class.upper():
                                    alpaca_asset_class = asset_class
                                    break
                    else:
                        alpaca_asset_class = request.asset_class
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to convert asset_class '{request.asset_class}' to AssetClass: {e}")
            
            # Create Alpaca SDK request object
            alpaca_request = None
            if alpaca_status or alpaca_asset_class:
                alpaca_request = AlpacaGetAssetsRequest(
                    status=alpaca_status,
                    asset_class=alpaca_asset_class
                )
            
            assets = self._default_trading_client.get_all_assets(filter=alpaca_request)
            
            alpaca_assets = []
            for asset in assets:
                alpaca_assets.append({
                    "id": asset.id,
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "asset_class": asset.asset_class,
                    "exchange": asset.exchange,
                    "status": asset.status,
                    "tradable": asset.tradable,
                    "marginable": asset.marginable,
                    "shortable": asset.shortable,
                    "easy_to_borrow": asset.easy_to_borrow,
                    "fractionable": asset.fractionable
                })
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(alpaca_assets)} assets.",
                extra={"assets": alpaca_assets}
            )
            
        except APIError as e:
            raise AlpacaError(f"Failed to get assets: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to get assets: {e}.")
    
    async def get_positions(self, request: GetPositionsRequest) -> ActionResult:
        """Get all positions."""
        try:
            positions = self._trading_clients[request.account_name].get_all_positions()
            
            alpaca_positions = []
            for position in positions:
                alpaca_positions.append({
                    "symbol": position.symbol,
                    "qty": str(position.qty),
                    "side": position.side,
                    "market_value": str(position.market_value),
                    "avg_entry_price": str(position.avg_entry_price),
                    "current_price": str(position.current_price),
                    "cost_basis": str(position.cost_basis),
                    "unrealized_pl": str(position.unrealized_pl),
                    "unrealized_plpc": str(position.unrealized_plpc),
                    "unrealized_intraday_pl": str(position.unrealized_intraday_pl),
                    "unrealized_intraday_plpc": str(position.unrealized_intraday_plpc),
                    "asset_id": position.asset_id,
                    "asset_class": position.asset_class,
                    "exchange": position.exchange,
                    "lastday_price": str(position.lastday_price)
                })
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(alpaca_positions)} positions.",
                extra={"positions": alpaca_positions}
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            raise AlpacaError(f"Failed to get positions: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to get positions: {e}.")
    
    
    async def _handle_data(self, data: Dict, 
                           symbol: str, 
                           asset_type: AssetClass, 
                           data_type: DataStreamType) -> None:
        """Handle incoming data and write to database with concurrency control.
        
        Args:
            data: Raw data from Alpaca stream
            symbol: Symbol name
            asset_type: Asset type ("crypto" or "stock")
            data_type: Data type ("quotes", "trades", "bars", "orderbooks")
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
                # Get data from queue with timeout
                item = await asyncio.wait_for(self._data_queue.get(), timeout=1.0)
                if item is None:  # Poison pill
                    logger.info("| 🛑 Data processor received poison pill, stopping...")
                    break
                
                # Queue item format: (data, symbol, asset_type, data_type_str)
                if len(item) == 4:
                    data, symbol, asset_type, data_type_str = item
                    # Convert string data_type to DataStreamType enum
                    data_type = DataStreamType(data_type_str)
                else:
                    logger.error(f"| ❌ Unexpected queue item format: {item}")
                    continue
                
                await self._handle_data(data, symbol, asset_type, data_type)
                self._data_queue.task_done()
                
            except asyncio.TimeoutError:
                # This is normal - just waiting for data
                continue
            except Exception as e:
                logger.error(f"| ❌ Error in data processor: {e}", exc_info=True)
    
    async def _quotes_handler_wrapper(self, data, asset_type: AssetClass, symbol: str):
        """Unified async handler for quotes data (crypto and stock).
        
        Args:
            data: Quotes data from Alpaca stream
            asset_type: Asset class (AssetClass)
            symbol: Symbol name (provided for verification)
        """
        try:
            data = data.model_dump()
            # Extract symbol from data or use provided symbol
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            # Use data_symbol for queue check (more robust)
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
        """Unified async handler for trades data (crypto and stock).
        
        Args:
            data: Trades data from Alpaca stream
            asset_type: Asset class (AssetClass)
            symbol: Symbol name (provided for verification)
        """
        try:
            data = data.model_dump()
            # Extract symbol from data or use provided symbol
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            # Use data_symbol for queue check (more robust)
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
        """Unified async handler for bars data (crypto and stock).
        
        Args:
            data: Bars data from Alpaca stream
            asset_type: Asset class (AssetClass)
            symbol: Symbol name (provided for verification)
        """
        try:
            data = data.model_dump()
            # Extract symbol from data or use provided symbol
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            # Use data_symbol for queue check (more robust)
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
        """Async handler for orderbooks data (crypto only).
        
        Args:
            data: Orderbooks data from Alpaca stream
            asset_type: Asset class (should be CRYPTO)
            symbol: Symbol name (provided for verification)
        """
        try:
            data = data.model_dump()
            # Extract symbol from data or use provided symbol
            data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol
            # Use data_symbol for queue check (more robust)
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
        """Async handler for news data.
        
        Args:
            data: News data from Alpaca stream
            asset_type: Asset class (AssetClass)
            symbol: Symbol name (provided for verification)
        """
        try:
            data = data.model_dump()
            # News may have symbols array or single symbol
            data_symbol = data.get("symbols", [])
            if data_symbol:
                # Use first symbol if multiple
                data_symbol = data_symbol[0] if isinstance(data_symbol, list) else data_symbol
            else:
                data_symbol = data.get("symbol", "")
            if not data_symbol:
                data_symbol = symbol if symbol else ""
            # News can be queued even without symbol (global news)
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
        # Store asset_types for use in processor
        self._worker_asset_types = asset_types
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            
            async def setup_and_run():
                # Ensure running flag is set
                self._data_stream_running = True
                
                # Database is already connected in initialize()
                # Initialize data queue and semaphore for concurrency control
                self._data_queue = asyncio.Queue(maxsize=1000)  # Buffer up to 1000 items
                self._data_semaphore = asyncio.Semaphore(self._max_concurrent_writes)
                
                logger.info(f"| ✅ Data queue initialized")
                
                # Start background data processor
                processor_task = asyncio.create_task(self._data_processor())
                logger.info(f"| ✅ Data processor started")
                
                # Ensure tables exist for all data types using handlers
                for symbol in symbols:
                    asset_type = asset_types.get(symbol, AssetClass.US_EQUITY)
                    # Ensure bars table exists
                    if self._bars_handler:
                        await self._bars_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Bars table created/verified for {symbol}")
                    # Ensure quotes table exists
                    if self._quotes_handler:
                        await self._quotes_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Quotes table created/verified for {symbol}")
                    # Ensure trades table exists
                    if self._trades_handler:
                        await self._trades_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Trades table created/verified for {symbol}")
                    # Ensure orderbooks table exists (crypto only)
                    if asset_type == AssetClass.CRYPTO and self._orderbooks_handler:
                        await self._orderbooks_handler.ensure_table_exists(symbol)
                        logger.info(f"| ✅ Orderbooks table created/verified for {symbol}")
                
                # Create news table (news is global, not per-symbol)
                if self._news_handler:
                    await self._news_handler.ensure_table_exists(symbol)
                logger.info(f"| ✅ News table created/verified")
                
                # Subscribe to streams
                # Alpaca SDK requires async handlers
                # Create async handlers that schedule operations to our event loop
                for symbol in symbols:
                    asset_type = asset_types.get(symbol, AssetClass.US_EQUITY)
                    
                    # Create async handlers with proper closure to capture symbol and asset_type
                    def create_handler(handler_wrapper_func, sym, atype):
                        async def handler(data):
                            # Schedule async operation in our event loop
                            # Check if event loop is still valid and running
                            if not self._data_stream_running:
                                # Stream is stopping, ignore new data
                                return
                            try:
                                if self._event_loop and self._event_loop.is_running() and not self._event_loop.is_closed():
                                    asyncio.run_coroutine_threadsafe(
                                        handler_wrapper_func(data, atype, sym),
                                        self._event_loop
                                    )
                                else:
                                    # Event loop not available, ignore (stream is stopping)
                                    logger.debug(f"| Event loop not available when handler called for {sym}")
                            except RuntimeError as e:
                                # RuntimeError can occur during interpreter shutdown
                                if "interpreter shutdown" in str(e) or "cannot schedule" in str(e).lower():
                                    logger.debug(f"| Cannot schedule coroutine (interpreter shutdown): {e}")
                                else:
                                    logger.warning(f"| ⚠️  Error scheduling coroutine for {sym}: {e}")
                            except Exception as e:
                                logger.debug(f"| Error in handler for {sym}: {e}")
                        return handler
                    
                    # Create handlers for this symbol
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
                
                # Run streams in separate threads (they are blocking)
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
                
                # Use executor as instance variable to avoid resource cleanup issues
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
                    
                    # Run processor in async while streams run in threads
                    # Keep the event loop running to process data
                    try:
                        # Wait for processor task (it runs in a loop)
                        # Use asyncio.sleep to keep the loop alive
                        while self._data_stream_running:
                            await asyncio.sleep(1.0)
                            # Check if processor task is still running
                            if processor_task.done():
                                logger.warning("| ⚠️  Data processor task completed unexpectedly")
                                break
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"| ❌ Error in processor: {e}")
                    finally:
                        # Stop streams FIRST to prevent new handlers from being called
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
                        
                        # Wait a bit for streams to fully stop before stopping processor
                        try:
                            await asyncio.sleep(0.1)
                        except:
                            pass
                        
                        # Stop processor
                        if not processor_task.done():
                            try:
                                await self._data_queue.put(None)  # Poison pill
                            except (RuntimeError, asyncio.CancelledError):
                                # Queue might be closed or event loop shutting down
                                pass
                            try:
                                processor_task.cancel()
                            except:
                                pass
                        try:
                            await processor_task
                        except (asyncio.CancelledError, RuntimeError):
                            pass
                        
                        # Wait for streams to finish and shutdown executor
                        for future in futures:
                            try:
                                future.cancel()
                            except:
                                pass
                        # Shutdown executor gracefully
                        if executor:
                            try:
                                executor.shutdown(wait=False, cancel_futures=True)
                            except (RuntimeError, Exception) as e:
                                # Ignore shutdown errors during interpreter shutdown
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
            # Clear event loop reference and stop accepting new handlers
            self._data_stream_running = False
            self._event_loop = None
            if loop:
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        try:
                            task.cancel()
                        except:
                            pass
                    # Wait for tasks to complete cancellation (with timeout)
                    if pending:
                        try:
                            loop.run_until_complete(asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=1.0
                            ))
                        except (asyncio.TimeoutError, RuntimeError):
                            # Ignore timeout/runtime errors during shutdown
                            pass
                except Exception as e:
                    logger.debug(f"| Error cancelling tasks: {e}")
                try:
                    loop.close()
                except Exception as e:
                    logger.debug(f"| Error closing loop: {e}")
    
    def start_data_stream(self, symbols: List[str], asset_types: Optional[Dict[str, AssetClass]] = None) -> None:
        """Start real-time data stream collection for given symbols.
        
        This method starts a background daemon thread that will collect real-time data
        from Alpaca streams and write it to the database. The thread runs in the background
        and will NOT block the main process.
        
        Args:
            symbols: List of symbols to subscribe to (e.g., ["BTC/USD", "AAPL"])
            asset_types: Optional dictionary mapping symbol to asset class (AssetClass)
                        If not provided, will be determined from symbol format
        """
        if self._data_stream_running:
            logger.warning("| ⚠️  Data stream is already running")
            return
        
        if not hasattr(self, 'database_service') or self.database_service is None:
            raise AlpacaError("Database service not initialized. Call initialize() first.")
        
        # Determine asset types if not provided
        asset_types = {}
        for symbol in symbols:
            if symbol not in self.symbols:
                logger.warning(f"| ⚠️  Symbol {symbol} not found in symbols list. Trying to determine asset class from symbol format...")
                # Try to determine asset class from symbol format
                if "/" in symbol:  # Crypto symbols typically contain "/"
                    asset_types[symbol] = AssetClass.CRYPTO
                    logger.info(f"| 📝 Detected {symbol} as CRYPTO based on symbol format")
                else:
                    asset_types[symbol] = AssetClass.US_EQUITY
                    logger.info(f"| 📝 Detected {symbol} as US_EQUITY based on symbol format")
            else:
                asset_types[symbol] = self.symbols[symbol]['asset_class']
        
        # Set running flag before starting thread
        self._data_stream_running = True
        
        # Start worker thread (daemon=True so it doesn't block main process)
        # The thread runs in background and will be cleaned up when main process exits
        self._data_stream_thread = threading.Thread(
            target=self._data_stream_worker,
            args=(symbols, asset_types),
            daemon=True  # Daemon thread so it doesn't block main process
        )
        self._data_stream_thread.start()
        logger.info(f"| 🚀 Started data stream for {len(symbols)} symbols (non-blocking)")
    
    def stop_data_stream(self) -> None:
        """Stop the data stream."""
        if not self._data_stream_running:
            logger.warning("| ⚠️  Data stream is not running")
            return
        
        logger.info("| 🛑 Stopping data stream...")
        self._data_stream_running = False
        
        # Stop streams first
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
        
        # Wait for thread to finish (with timeout)
        if self._data_stream_thread and self._data_stream_thread.is_alive():
            try:
                self._data_stream_thread.join(timeout=5)
                if self._data_stream_thread.is_alive():
                    logger.warning("| ⚠️  Data stream thread did not finish within timeout")
            except Exception as e:
                logger.debug(f"| Error joining thread: {e}")
        
        logger.info("| 🛑 Data stream stopped")
        
        
    async def _get_data_from_handler(
        self, 
        symbol: str, 
        data_type: DataStreamType, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Helper method to get data for a single symbol and data_type."""
        # Determine asset type from symbol
        if data_type == DataStreamType.QUOTES:
            return await self._quotes_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.TRADES:
            return await self._trades_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.BARS:
            return await self._bars_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.ORDERBOOKS:
            return await self._orderbooks_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.NEWS:
            return await self._news_handler.get_data(symbol, start_date, end_date, limit)
        else:
           raise ValueError(f"Invalid data type: {data_type}")
       
    async def get_data(self, request: GetDataRequest) -> ActionResult:
        """Get historical data from database.
        
        Args:
            request: GetDataRequest with symbol (str or list), data_type (str or list), 
                    optional start_date, end_date, and limit
            - If start_date and end_date are provided: returns data in that date range
            - If not provided: returns latest data (sorted by timestamp DESC)
            - limit: limits the number of records returned per symbol/data_type combination
            
        Returns:
            ActionResult with data organized by symbol in extra field:
            {
                "symbol1": {
                    "bars": [...],
                    "news": [...],
                    "quotes": [...]
                },
                "symbol2": {
                    "bars": [...],
                    "trades": [...]
                }
            }
        """
        try:
            if not hasattr(self, 'database_service') or self.database_service is None:
                raise AlpacaError("Database service not initialized. Call initialize() first.")
            
            # Ensure database is connected
            if not self.database_service._is_connected:
                await self.database_service.connect()
            
            # Normalize symbol and data_type to lists
            symbols = request.symbol if isinstance(request.symbol, list) else [request.symbol]
            data_types = request.data_type if isinstance(request.data_type, list) else [request.data_type]
            data_types = [DataStreamType(data_type) for data_type in data_types]
            
            # Organize data by symbol
            result_data: Dict[str, Dict[str, List[Dict]]] = {}
            total_rows = 0
            
            # Get data for each symbol and data_type combination
            for symbol in symbols:
                result_data[symbol] = {}
                
                for data_type in data_types:
                    logger.info(f"| 🔍 Getting {data_type.value} data for {symbol}...")
                    data = await self._get_data_from_handler(
                        symbol=symbol,
                        data_type=data_type,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        limit=request.limit
                    )
                    result_data[symbol][data_type.value] = data
                    total_rows += len(data)
                    logger.info(f"| ✅ Retrieved {len(data)} {data_type.value} records for {symbol}")
            
            # Build message
            symbol_str = ", ".join(symbols) if len(symbols) <= 10 else f"{len(symbols)} symbols"
            data_type_str = ", ".join([datatype.value for datatype in data_types]) if len(data_types) <= 10 else f"{len(data_types)} types"
            
            if request.start_date and request.end_date:
                message = f"Retrieved {total_rows} records ({data_type_str}) for {symbol_str} from {request.start_date} to {request.end_date}."
            else:
                message = f"Retrieved {total_rows} latest records ({data_type_str}) for {symbol_str}."
            
            return ActionResult(
                success=True,
                message=message,
                extra={
                    "data": result_data,
                    "symbols": symbols,
                    "data_types": data_types,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "row_count": total_rows
                }
            )
            
        except Exception as e:
            raise AlpacaError(f"Failed to get data: {e}.")
    
    # Order methods
    async def create_order(self, request: CreateOrderRequest) -> ActionResult:
        """Create a market order.
        
        Args:
            request: CreateOrderRequest with account_name, symbol, qty/notional, side, time_in_force
            
        Returns:
            ActionResult with order information
        """
        try:
            if request.qty is None and request.notional is None:
                raise AlpacaError("Either 'qty' or 'notional' must be provided")
            
            if request.qty is not None and request.notional is not None:
                raise AlpacaError("Cannot specify both 'qty' and 'notional'")
            
            # Determine asset class from symbol
            # Crypto symbols typically contain "/" (e.g., "BTC/USD")
            is_crypto = "/" in request.symbol or (hasattr(self, 'symbols') and 
                        request.symbol in self.symbols and 
                        self.symbols[request.symbol].get('asset_class') == AssetClass.CRYPTO)
            
            # Convert side string to OrderSide enum
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
            
            # Convert time_in_force string to TimeInForce enum
            # For crypto, only IOC and FOK are supported
            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "opg": TimeInForce.OPG,
                "cls": TimeInForce.CLS,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }
            requested_tif = request.time_in_force.lower()
            
            if is_crypto:
                # Crypto only supports IOC and FOK
                if requested_tif not in ["ioc", "fok"]:
                    # Default to IOC for crypto if invalid time_in_force is specified
                    logger.warning(f"| ⚠️  Crypto orders only support 'ioc' or 'fok' time_in_force. "
                                 f"'{request.time_in_force}' is not supported, using 'ioc' instead.")
                    time_in_force = TimeInForce.IOC
                else:
                    time_in_force = tif_map[requested_tif]
            else:
                # Stock supports all time_in_force options
                time_in_force = tif_map.get(requested_tif, TimeInForce.DAY)
            
            # Create market order request
            if request.qty is not None:
                order_request = MarketOrderRequest(
                    symbol=request.symbol,
                    qty=request.qty,
                    side=side,
                    time_in_force=time_in_force
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=request.symbol,
                    notional=request.notional,
                    side=side,
                    time_in_force=time_in_force
                )
            
            # Submit order
            trading_client = self._trading_clients[request.account_name]
            order = trading_client.submit_order(order_request)
            
            # Convert order to dictionary
            order_info = {
                "id": str(order.id),
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "asset_class": order.asset_class.value if hasattr(order.asset_class, 'value') else str(order.asset_class),
                "qty": str(order.qty) if order.qty else None,
                "notional": str(order.notional) if order.notional else None,
                "filled_qty": str(order.filled_qty) if order.filled_qty else "0",
                "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
                "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "time_in_force": order.time_in_force.value if hasattr(order.time_in_force, 'value') else str(order.time_in_force),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "submitted_at": str(order.submitted_at) if order.submitted_at else None,
                "filled_at": str(order.filled_at) if order.filled_at else None,
                "expired_at": str(order.expired_at) if order.expired_at else None,
                "canceled_at": str(order.canceled_at) if order.canceled_at else None,
                "failed_at": str(order.failed_at) if order.failed_at else None,
            }
            
            return ActionResult(
                success=True,
                message=f"Order {order.id} submitted successfully for {request.symbol} ({request.side} {request.qty or request.notional}).",
                extra={"order": order_info}
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            raise AlpacaError(f"Failed to create order: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to create order: {e}.")
    
    async def get_orders(self, request: GetOrdersRequest) -> ActionResult:
        """Get orders for an account.
        
        Args:
            request: GetOrdersRequest with account_name, status, limit, etc.
            
        Returns:
            ActionResult with list of orders
        """
        try:
            # Convert status string to OrderStatus enum or None
            status_filter = None
            if request.status and request.status != "all":
                status_map = {
                    "open": OrderStatus.OPEN,
                    "closed": OrderStatus.CLOSED,
                }
                status_filter = status_map.get(request.status.lower())
            
            # Create Alpaca GetOrdersRequest
            alpaca_request = AlpacaGetOrdersRequest(
                status=status_filter,
                limit=request.limit,
                after=request.after,
                until=request.until,
                direction=request.direction,
            )
            
            # Get orders
            trading_client = self._trading_clients[request.account_name]
            orders = trading_client.get_orders(alpaca_request)
            
            # Convert orders to list of dictionaries
            orders_list = []
            for order in orders:
                orders_list.append({
                    "id": str(order.id),
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "asset_class": order.asset_class.value if hasattr(order.asset_class, 'value') else str(order.asset_class),
                    "qty": str(order.qty) if order.qty else None,
                    "notional": str(order.notional) if order.notional else None,
                    "filled_qty": str(order.filled_qty) if order.filled_qty else "0",
                    "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
                    "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                    "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                    "time_in_force": order.time_in_force.value if hasattr(order.time_in_force, 'value') else str(order.time_in_force),
                    "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                    "submitted_at": str(order.submitted_at) if order.submitted_at else None,
                    "filled_at": str(order.filled_at) if order.filled_at else None,
                    "expired_at": str(order.expired_at) if order.expired_at else None,
                    "canceled_at": str(order.canceled_at) if order.canceled_at else None,
                    "failed_at": str(order.failed_at) if order.failed_at else None,
                })
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(orders_list)} orders.",
                extra={"orders": orders_list}
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            raise AlpacaError(f"Failed to get orders: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to get orders: {e}.")
    
    async def get_order(self, request: GetOrderRequest) -> ActionResult:
        """Get a specific order by ID.
        
        Args:
            request: GetOrderRequest with account_name and order_id
            
        Returns:
            ActionResult with order information
        """
        try:
            trading_client = self._trading_clients[request.account_name]
            order = trading_client.get_order_by_id(request.order_id)
            
            order_info = {
                "id": str(order.id),
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "asset_class": order.asset_class.value if hasattr(order.asset_class, 'value') else str(order.asset_class),
                "qty": str(order.qty) if order.qty else None,
                "notional": str(order.notional) if order.notional else None,
                "filled_qty": str(order.filled_qty) if order.filled_qty else "0",
                "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
                "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "time_in_force": order.time_in_force.value if hasattr(order.time_in_force, 'value') else str(order.time_in_force),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "submitted_at": str(order.submitted_at) if order.submitted_at else None,
                "filled_at": str(order.filled_at) if order.filled_at else None,
                "expired_at": str(order.expired_at) if order.expired_at else None,
                "canceled_at": str(order.canceled_at) if order.canceled_at else None,
                "failed_at": str(order.failed_at) if order.failed_at else None,
            }
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} retrieved successfully.",
                extra={"order": order_info}
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            if e.status_code == 404:
                from src.environments.alpacaentry.exceptions import NotFoundError
                raise NotFoundError(f"Order {request.order_id} not found: {e}")
            raise AlpacaError(f"Failed to get order: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to get order: {e}.")
    
    async def cancel_order(self, request: CancelOrderRequest) -> ActionResult:
        """Cancel an order.
        
        Args:
            request: CancelOrderRequest with account_name and order_id
            
        Returns:
            ActionResult indicating success or failure
        """
        try:
            trading_client = self._trading_clients[request.account_name]
            trading_client.cancel_order_by_id(request.order_id)
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} canceled successfully.",
                extra={"order_id": request.order_id}
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            if e.status_code == 404:
                from src.environments.alpacaentry.exceptions import NotFoundError
                raise NotFoundError(f"Order {request.order_id} not found: {e}")
            raise AlpacaError(f"Failed to cancel order: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to cancel order: {e}.")
    
    async def cancel_all_orders(self, request: CancelAllOrdersRequest) -> ActionResult:
        """Cancel all orders for an account.
        
        Args:
            request: CancelAllOrdersRequest with account_name
            
        Returns:
            ActionResult indicating success or failure
        """
        try:
            trading_client = self._trading_clients[request.account_name]
            trading_client.cancel_orders()
            
            return ActionResult(
                success=True,
                message="All orders canceled successfully.",
                extra={"account_name": request.account_name}
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            raise AlpacaError(f"Failed to cancel all orders: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to cancel all orders: {e}.")