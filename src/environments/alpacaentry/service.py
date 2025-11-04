"""Alpaca trading service implementation using alpaca-py."""
import threading
import asyncio
import json
from typing import Optional, Union, List, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)
import concurrent.futures

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest as AlpacaGetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
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

from src.environments.protocol.types import ActionResult
from src.environments.alpacaentry.types import (
    GetAccountRequest,
    GetAssetsRequest,
    GetPositionsRequest,
    GetDataRequest,
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
from src.logger import logger
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environments.alpacaentry.types import DataStreamType

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
        data_stream_symbols: Optional[List[str]] = None,
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
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
        
        self.auto_start_data_stream = auto_start_data_stream
        self.data_stream_symbols = data_stream_symbols
        
        self.default_account = AccountInfo(**accounts[0])
        self.accounts: Dict[str, AccountInfo] = {
           account["name"]: AccountInfo(**account) for account in accounts
        }
        self.live = live
        
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
        self._data_stream_running: bool = True
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
            data_stream_symbols: List of symbols to subscribe to if auto_start_data_stream is True
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
            if self.auto_start_data_stream and self.data_stream_symbols:
                # Start data stream in background (non-blocking)
                self.start_data_stream(self.data_stream_symbols)
                logger.info(f"| 📡 Auto-started data stream for {len(self.data_stream_symbols)} symbols")
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Invalid Alpaca credentials: {e}")
            raise AlpacaError(f"Failed to initialize Alpaca service: {e}.")
        except Exception as e:
            raise AlpacaError(f"Failed to initialize Alpaca service: {e}.")

    async def cleanup(self) -> None:
        """Cleanup the Alpaca service."""
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
                    "lastday_price": str(position.lastday_price),
                    "today_cost_basis": str(position.today_cost_basis),
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
    
    def _normalize_timestamp_in_record(self, record: Dict) -> Dict:
        """Normalize timestamp field in a database record to 'YYYY-MM-DD HH:MM:SS' format.
        
        Note: This method is kept for handling historical data only.
        New data should already be in "YYYY-MM-DD HH:MM:SS" format from _prepare_data_for_insert.
        """
        if 'timestamp' in record:
            timestamp_str = record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            record['timestamp'] = timestamp_str
        return record
    
    def _prepare_data_for_insert(self, 
                                 data: Dict, 
                                 symbol: str, 
                                 asset_type: AssetClass = AssetClass.US_EQUITY, 
                                 data_type: DataStreamType = DataStreamType.QUOTES) -> Dict:
        """Prepare data dictionary for database insertion.
        
        Args:
            data: Data dictionary from Alpaca stream
            symbol: Symbol name
            asset_type: Asset class (AssetClass)
            data_type: Data type (DataStreamType)
        """
        
        # Convert timestamp to "YYYY-MM-DD HH:MM:SS" format string
        data = self._normalize_timestamp_in_record(data)
        
        db_data = {
            "timestamp": data.get("timestamp"),
            "symbol": symbol,
        }
        
        if data_type == DataStreamType.QUOTES:
            # Support both raw_data format (single letters) and object format (full names)
            db_data["bid_price"] = data.get("bid_price") if "bid_price" in data else data.get("bp")
            db_data["bid_size"] = data.get("bid_size") if "bid_size" in data else data.get("bs")
            db_data["ask_price"] = data.get("ask_price") if "ask_price" in data else data.get("ap")
            db_data["ask_size"] = data.get("ask_size") if "ask_size" in data else data.get("as")
            if asset_type == AssetClass.US_EQUITY:
                db_data["tape"] = data.get("tape")
        
        elif data_type == DataStreamType.TRADES:
            # Support both raw_data format (single letters) and object format (full names)
            db_data["price"] = data.get("price") if "price" in data else data.get("p")
            db_data["size"] = data.get("size") if "size" in data else data.get("s")
            db_data["trade_id"] = data.get("trade_id") if "trade_id" in data else data.get("i")
            if asset_type == AssetClass.CRYPTO:
                db_data["taker_side"] = data.get("taker_side") if "taker_side" in data else data.get("tks")
            else:  # stock
                conditions = data.get("conditions") if "conditions" in data else data.get("c", [])
                db_data["conditions"] = str(conditions) if conditions else None
                db_data["tape"] = data.get("tape")
        
        elif data_type == DataStreamType.BARS:
            # Support both raw_data format (single letters) and object format (full names)
            db_data["open"] = data.get("open") if "open" in data else data.get("o")
            db_data["high"] = data.get("high") if "high" in data else data.get("h")
            db_data["low"] = data.get("low") if "low" in data else data.get("l")
            db_data["close"] = data.get("close") if "close" in data else data.get("c")
            db_data["volume"] = data.get("volume") if "volume" in data else data.get("v")
            db_data["trade_count"] = data.get("trade_count") if "trade_count" in data else data.get("n")
            db_data["vwap"] = data.get("vwap") if "vwap" in data else data.get("vw")
        
        elif data_type == DataStreamType.ORDERBOOKS:
            # Orderbooks contain arrays of bids and asks
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            db_data["bids"] = json.dumps(bids) if bids else None
            db_data["asks"] = json.dumps(asks) if asks else None
        
        elif data_type == DataStreamType.NEWS:
            # News data structure
            db_data["headline"] = data.get("headline")
            db_data["summary"] = data.get("summary")
            db_data["author"] = data.get("author")
            db_data["source"] = data.get("source")
            db_data["url"] = data.get("url")
            db_data["image_url"] = data.get("image_url")
            db_data["news_id"] = data.get("id")
            # News may have symbols array
            symbols = data.get("symbols", [])
            if symbols:
                db_data["symbol"] = ",".join(symbols)  # Join multiple symbols
            else:
                db_data["symbol"] = symbol  # Use provided symbol if available
        
        return db_data
    
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
                db_data = self._prepare_data_for_insert(data, symbol, asset_type, data_type)
                
                
                print(db_data)
                
                if data_type == DataStreamType.BARS:
                    await self._bars_handler.stream_insert(db_data, symbol)
                elif data_type == DataStreamType.QUOTES:
                    await self._quotes_handler.stream_insert(db_data, symbol)
                elif data_type == DataStreamType.TRADES:
                    await self._trades_handler.stream_insert(db_data, symbol)
                elif data_type == DataStreamType.ORDERBOOKS:
                    await self._orderbooks_handler.stream_insert(db_data, symbol)
                elif data_type == DataStreamType.NEWS:
                    await self._news_handler.stream_insert(db_data, symbol)
            except Exception as e:
                logger.error(f"Error in data handler: {e}")
    
    async def _data_processor(self) -> None:
        """Background task to process data from queue."""
        while self._data_stream_running:
            try:
                # Get data from queue with timeout
                item = await asyncio.wait_for(self._data_queue.get(), timeout=1.0)
                if item is None:  # Poison pill
                    break
                
                data, symbol, asset_type, data_type = item
                await self._handle_data(data, symbol, asset_type, data_type)
                self._data_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in data processor: {e}")
    
    async def _quotes_handler_wrapper(self, data, asset_type: AssetClass = AssetClass.CRYPTO):
        """Unified async handler for quotes data (crypto and stock).
        
        Args:
            data: Quotes data from Alpaca stream
            asset_type: Asset class (AssetClass)
        """
        data = data.model_dump()
        symbol = data.get("symbol", "")
        if symbol and self._data_queue:
            await self._data_queue.put((data, symbol, asset_type, "quotes"))
    
    async def _trades_handler_wrapper(self, data, asset_type: AssetClass = AssetClass.CRYPTO):
        """Unified async handler for trades data (crypto and stock).
        
        Args:
            data: Trades data from Alpaca stream
            asset_type: Asset class (AssetClass)
        """
        data = data.model_dump()
        symbol = data.get("symbol", "")
        if symbol and self._data_queue:
            await self._data_queue.put((data, symbol, asset_type, "trades"))
    
    async def _bars_handler_wrapper(self, data, asset_type: AssetClass = AssetClass.CRYPTO):
        """Unified async handler for bars data (crypto and stock).
        
        Args:
            data: Bars data from Alpaca stream
            asset_type: Asset class (AssetClass)
        """
        data = data.model_dump()
        symbol = data.get("symbol", "")
        if symbol and self._data_queue:
            await self._data_queue.put((data, symbol, asset_type, "bars"))
    
    async def _orderbooks_handler_wrapper(self, data, asset_type: AssetClass = AssetClass.CRYPTO):
        """Async handler for orderbooks data (crypto only).
        
        Args:
            data: Orderbooks data from Alpaca stream
            asset_type: Asset class (AssetClass)
        """
        data = data.model_dump()
        symbol = data.get("symbol", "")
        if symbol and self._data_queue:
            await self._data_queue.put((data, symbol, asset_type, "orderbooks"))
    
    async def _news_handler_wrapper(self, data, asset_type: AssetClass = AssetClass.CRYPTO):
        """Async handler for news data.
        
        Args:
            data: News data from Alpaca stream
            asset_type: Asset class (AssetClass)
        """
        # News may have multiple symbols or a single symbol
        data = data.model_dump()
        symbols = data.get("symbols", [])
        if symbols:
            # If multiple symbols, use the first one as the primary symbol
            symbol = symbols[0] if symbols else None
        else:
            symbol = data.get("symbol", None)  # Fallback to None if no symbol
        
        if symbol and self._data_queue:
            await self._data_queue.put((data, symbol, asset_type, "news"))
    
    def _data_stream_worker(self, symbols: List[str], asset_types: Dict[str, AssetClass]):
        """Worker thread for running data streams."""
        loop = None
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            
            async def setup_and_run():
                # Database is already connected in initialize()
                # Initialize data queue and semaphore for concurrency control (if needed for trades)
                self._data_queue = asyncio.Queue(maxsize=1000)  # Buffer up to 1000 items
                self._data_semaphore = asyncio.Semaphore(self._max_concurrent_writes)
                
                # Start background data processor (for trades, which still use queue)
                processor_task = asyncio.create_task(self._data_processor())
                
                # Ensure tables exist for all data types using handlers
                for symbol in symbols:
                    asset_type = asset_types.get(symbol, AssetClass.US_EQUITY)
                    # Ensure bars table exists
                    if self._bars_handler:
                        await self._bars_handler.ensure_table_exists(symbol, asset_type)
                        logger.info(f"| ✅ Bars table created/verified for {symbol}")
                    # Ensure quotes table exists
                    if self._quotes_handler:
                        await self._quotes_handler.ensure_table_exists(symbol, asset_type)
                        logger.info(f"| ✅ Quotes table created/verified for {symbol}")
                    # Ensure trades table exists
                    if self._trades_handler:
                        await self._trades_handler.ensure_table_exists(symbol, asset_type)
                        logger.info(f"| ✅ Trades table created/verified for {symbol}")
                    # Ensure orderbooks table exists (crypto only)
                    if asset_type == AssetClass.CRYPTO and self._orderbooks_handler:
                        await self._orderbooks_handler.ensure_table_exists(symbol, asset_type)
                        logger.info(f"| ✅ Orderbooks table created/verified for {symbol}")
                
                # Create news table (news is global, not per-symbol)
                if self._news_handler:
                    await self._news_handler.ensure_table_exists(symbol, asset_type)
                    logger.info(f"| ✅ News table created/verified")
                    
                # Subscribe to streams
                for symbol in symbols:
                    asset_type = asset_types.get(symbol, AssetClass.US_EQUITY)
                    
                    if asset_type == AssetClass.CRYPTO:
                        self._crypto_data_stream.subscribe_quotes(self._quotes_handler_wrapper, 
                                                                  symbol)
                        self._crypto_data_stream.subscribe_trades(self._trades_handler_wrapper, 
                                                                  symbol)
                        self._crypto_data_stream.subscribe_bars(self._bars_handler_wrapper,
                                                                symbol)
                        self._crypto_data_stream.subscribe_orderbooks(self._orderbooks_handler_wrapper, 
                                                                      symbol)
                    elif asset_type == AssetClass.US_EQUITY:
                        self._stock_data_stream.subscribe_quotes(self._quotes_handler_wrapper, 
                                                                 symbol)
                        self._stock_data_stream.subscribe_trades(self._trades_handler_wrapper, 
                                                                 symbol)
                        self._stock_data_stream.subscribe_bars(self._bars_handler_wrapper,
                                                               symbol)
                    
                    self._news_data_stream.subscribe_news(self._news_handler_wrapper,
                                                          symbol)
                
                # Run streams in separate threads (they are blocking)
                def run_crypto_stream():
                    if self._crypto_data_stream:
                        self._crypto_data_stream.run()
                
                def run_stock_stream():
                    if self._stock_data_stream:
                        self._stock_data_stream.run()
                
                def run_news_stream():
                    if self._news_data_stream:
                        self._news_data_stream.run()
                
                # Start streams in thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = []
                    if self._crypto_data_stream:
                        futures.append(executor.submit(run_crypto_stream))
                    if self._stock_data_stream:
                        futures.append(executor.submit(run_stock_stream))
                    if self._news_data_stream:
                        futures.append(executor.submit(run_news_stream))
                    
                    # Run processor in async while streams run in threads
                    try:
                        # Wait for processor (streams run in background threads)
                        await processor_task
                    except asyncio.CancelledError:
                        pass
                    finally:
                        # Stop processor
                        await self._data_queue.put(None)  # Poison pill
                        processor_task.cancel()
                        try:
                            await processor_task
                        except asyncio.CancelledError:
                            pass
                        
                        # Wait for streams to finish
                        for future in futures:
                            future.cancel()
            
            loop.run_until_complete(setup_and_run())
            
        except KeyboardInterrupt:
            logger.info("| 🛑 Data stream stopped by user")
            self._data_stream_running = False
        except Exception as e:
            logger.error(f"| ❌ Error in data stream worker: {e}")
            self._data_stream_running = False
        finally:
            if loop:
                loop.close()
    
    def start_data_stream(self, symbols: List[str], asset_types: Optional[Dict[str, AssetClass]] = None) -> None:
        """Start real-time data stream collection for given symbols.
        
        This method starts a blocking thread that will collect real-time data
        from Alpaca streams and write it to the database. The thread will block
        the main process until stopped.
        
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
            asset_types[symbol] = self.symbols[symbol]['asset_class']
        
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
        
        self._data_stream_running = False
        
        # Stop streams
        if self._crypto_data_stream:
            try:
                self._crypto_data_stream.stop()
            except:
                pass
        
        if self._stock_data_stream:
            try:
                self._stock_data_stream.stop()
            except:
                pass
        
        if self._news_data_stream:
            try:
                self._news_data_stream.stop()
            except:
                pass
        
        # Wait for thread to finish
        if self._data_stream_thread and self._data_stream_thread.is_alive():
            self._data_stream_thread.join(timeout=5)
        
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
            return self._quotes_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.TRADES:
            return self._trades_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.BARS:
            return self._bars_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.ORDERBOOKS:
            return self._orderbooks_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.NEWS:
            return self._news_handler.get_data(symbol, start_date, end_date, limit)
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
                    data = await self._get_data_from_handler(
                        symbol=symbol,
                        data_type=data_type,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        limit=request.limit
                    )
                    result_data[symbol][data_type.value] = data
                    total_rows += len(data)
            
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