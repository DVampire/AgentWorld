"""Alpaca trading service implementation using alpaca-py."""
import threading
import asyncio
from datetime import datetime
from typing import Optional, Union, List, Dict, Set
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)

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
)
from src.environments.alpacaentry.exceptions import (
    AlpacaError,
    AuthenticationError,
)
from src.logger import logger
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest

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
        
        # Data streaming state
        self._data_stream_thread: Optional[threading.Thread] = None
        self._data_stream_running = False
        self._subscribed_symbols: Set[str] = set()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._data_queue: Optional[asyncio.Queue] = None
        self._data_semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent_writes: int = 10  # Max concurrent database writes

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self, auto_start_data_stream: bool = False, data_stream_symbols: Optional[List[str]] = None) -> None:
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
                secret_key=self.default_account.secret_key
            )
            
            self._stock_data_stream = StockDataStream(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key
            )
            
            self._news_data_stream = NewsDataStream(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key
            )
            
            self._option_data_stream = OptionDataStream(
                api_key=self.default_account.api_key,
                secret_key=self.default_account.secret_key
            )
            
            # Test connection by getting account info
            for account_name, account in self.accounts.items():
                account = self._trading_clients[account_name].get_account()
                logger.info(f"| 📝 Connected to Alpaca paper trading account: {account.account_number}")
            
            symbols = []
            # Stock Symbols
            stock_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY))
            stock_symbols = stock_symbols.extra["assets"]
            symbols.extend(stock_symbols)
            logger.info(f"| 📝 Found {len(stock_symbols)} stock symbols.")
            
            # Crypto Symbols
            crypto_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.CRYPTO))
            crypto_symbols = crypto_symbols.extra["assets"]
            symbols.extend(crypto_symbols)
            logger.info(f"| 📝 Found {len(crypto_symbols)} crypto symbols.")
            
            # Perpetual Futures Crypto Symbols
            perpetual_futures_crypto_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.CRYPTO_PERP))
            perpetual_futures_crypto_symbols = perpetual_futures_crypto_symbols.extra["assets"]
            symbols.extend(perpetual_futures_crypto_symbols)
            logger.info(f"| 📝 Found {len(perpetual_futures_crypto_symbols)} perpetual futures crypto symbols.")
            
            # Option Symbols
            option_symbols = await self.get_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_OPTION))
            option_symbols = option_symbols.extra["assets"]
            symbols.extend(option_symbols)
            logger.info(f"| 📝 Found {len(option_symbols)} option symbols.")
            
            logger.info(f"| 📝 Found {len(symbols)} total symbols.")
            logger.info(f"| 📝 Symbols: {', '.join([symbol['symbol'] for symbol in symbols])}")
            
            self.database_base_dir = self.base_dir / "database"
            self.database_base_dir.mkdir(parents=True, exist_ok=True)
            self.database_service = DatabaseService(self.database_base_dir)
            
            # Optionally start data stream automatically
            if auto_start_data_stream and data_stream_symbols:
                # Start data stream in background (non-blocking)
                self.start_data_stream(data_stream_symbols)
                logger.info(f"| 📡 Auto-started data stream for {len(data_stream_symbols)} symbols")
            
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
    
    def _sanitize_table_name(self, symbol: str) -> str:
        """Sanitize symbol name to be used as table name."""
        # Replace invalid characters with underscore
        table_name = symbol.replace("/", "_").replace(".", "_").replace("-", "_")
        # Remove any other invalid characters
        table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
        return f"data_{table_name}"
    
    def _get_asset_class(self, symbol: str, asset_class: AssetClass) -> str:
        """Determine asset type (stock or crypto) from asset class."""
        if asset_class == AssetClass.CRYPTO or asset_class == AssetClass.CRYPTO_PERP:
            return "crypto"
        elif asset_class == AssetClass.US_EQUITY:
            return "stock"
        else:
            return "other"
    
    async def _ensure_table_exists(self, symbol: str, asset_type: str, data_type: str = "quotes") -> None:
        """Ensure table exists for a symbol and data type.
        
        Args:
            symbol: Symbol name
            asset_type: Asset type ("crypto" or "stock")
            data_type: Data type ("quotes", "trades", "bars", "orderbooks")
        """
        base_name = self._sanitize_table_name(symbol)
        table_name = f"{base_name}_{data_type}"
        
        # Check if table already exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if check_result.success and check_result.extra.get("data"):
            # Table exists
            return
        
        # Create table based on data type and asset type
        if data_type == "quotes":
            if asset_type == "crypto":
                columns = [
                    {"name": "id", "type": "INTEGER", "constraints": "AUTOINCREMENT"},
                    {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "bid_price", "type": "REAL"},
                    {"name": "bid_size", "type": "REAL"},
                    {"name": "ask_price", "type": "REAL"},
                    {"name": "ask_size", "type": "REAL"},
                    {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
                ]
            else:  # stock
                columns = [
                    {"name": "id", "type": "INTEGER", "constraints": "AUTOINCREMENT"},
                    {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "bid_price", "type": "REAL"},
                    {"name": "bid_size", "type": "REAL"},
                    {"name": "ask_price", "type": "REAL"},
                    {"name": "ask_size", "type": "REAL"},
                    {"name": "tape", "type": "TEXT"},
                    {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
                ]
        elif data_type == "trades":
            if asset_type == "crypto":
                columns = [
                    {"name": "id", "type": "INTEGER", "constraints": "AUTOINCREMENT"},
                    {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "price", "type": "REAL"},
                    {"name": "size", "type": "REAL"},
                    {"name": "trade_id", "type": "TEXT"},
                    {"name": "taker_side", "type": "TEXT"},
                    {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
                ]
            else:  # stock
                columns = [
                    {"name": "id", "type": "INTEGER", "constraints": "AUTOINCREMENT"},
                    {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "price", "type": "REAL"},
                    {"name": "size", "type": "REAL"},
                    {"name": "trade_id", "type": "TEXT"},
                    {"name": "conditions", "type": "TEXT"},
                    {"name": "tape", "type": "TEXT"},
                    {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
                ]
        elif data_type == "bars":
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "open", "type": "REAL"},
                {"name": "high", "type": "REAL"},
                {"name": "low", "type": "REAL"},
                {"name": "close", "type": "REAL"},
                {"name": "volume", "type": "REAL"},
                {"name": "trade_count", "type": "INTEGER"},
                {"name": "vwap", "type": "REAL"},
                {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            ]
        elif data_type == "orderbooks":
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "bids", "type": "TEXT"},  # JSON string for bid/ask arrays
                {"name": "asks", "type": "TEXT"},
                {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            ]
        elif data_type == "news":
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT"},
                {"name": "headline", "type": "TEXT"},
                {"name": "summary", "type": "TEXT"},
                {"name": "author", "type": "TEXT"},
                {"name": "source", "type": "TEXT"},
                {"name": "url", "type": "TEXT"},
                {"name": "image_url", "type": "TEXT"},
                {"name": "news_id", "type": "TEXT"},
                {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            ]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        create_request = CreateTableRequest(
            table_name=table_name,
            columns=columns,
            primary_key="id"
        )
        result = await self.database_service.create_table(create_request)
        if not result.success:
            logger.error(f"Failed to create table {table_name}: {result.message}")
            raise AlpacaError(f"Failed to create table {table_name}: {result.message}")
    
    def _prepare_data_for_insert(self, data: Dict, symbol: str, asset_type: str, data_type: str = "quotes") -> Dict:
        """Prepare data dictionary for database insertion.
        
        Args:
            data: Raw data from Alpaca stream
            symbol: Symbol name
            asset_type: Asset type ("crypto" or "stock")
            data_type: Data type ("quotes", "trades", "bars", "orderbooks")
        """
        import json
        
        db_data = {
            "timestamp": data.get("t", data.get("timestamp", datetime.utcnow().isoformat())),
            "symbol": symbol,
        }
        
        if data_type == "quotes":
            db_data["bid_price"] = data.get("bp")
            db_data["bid_size"] = data.get("bs")
            db_data["ask_price"] = data.get("ap")
            db_data["ask_size"] = data.get("as")
            if asset_type == "stock":
                db_data["tape"] = data.get("tape")
        
        elif data_type == "trades":
            db_data["price"] = data.get("p")
            db_data["size"] = data.get("s")
            db_data["trade_id"] = data.get("i")
            if asset_type == "crypto":
                db_data["taker_side"] = data.get("tks")
            else:  # stock
                db_data["conditions"] = str(data.get("c", [])) if data.get("c") else None
                db_data["tape"] = data.get("tape")
        
        elif data_type == "bars":
            db_data["open"] = data.get("o")
            db_data["high"] = data.get("h")
            db_data["low"] = data.get("l")
            db_data["close"] = data.get("c")
            db_data["volume"] = data.get("v")
            db_data["trade_count"] = data.get("n")
            db_data["vwap"] = data.get("vw")
        
        elif data_type == "orderbooks":
            # Orderbooks contain arrays of bids and asks
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            db_data["bids"] = json.dumps(bids) if bids else None
            db_data["asks"] = json.dumps(asks) if asks else None
        
        elif data_type == "news":
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
    
    async def _handle_data(self, data: Dict, symbol: str, asset_type: str, data_type: str = "quotes") -> None:
        """Handle incoming data and write to database with concurrency control.
        
        Args:
            data: Raw data from Alpaca stream
            symbol: Symbol name
            asset_type: Asset type ("crypto" or "stock")
            data_type: Data type ("quotes", "trades", "bars", "orderbooks")
        """
        async with self._data_semaphore:
            try:
                base_name = self._sanitize_table_name(symbol)
                table_name = f"{base_name}_{data_type}"
                db_data = self._prepare_data_for_insert(data, symbol, asset_type, data_type)
                
                insert_request = InsertRequest(
                    table_name=table_name,
                    data=db_data
                )
                result = await self.database_service.insert_data(insert_request)
                
                if not result.success:
                    logger.error(f"Failed to insert {data_type} data for {symbol}: {result.message}")
                
            except Exception as e:
                logger.error(f"Error handling {data_type} data for {symbol}: {e}")
    
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
    
    async def _crypto_quotes_handler(self, data):
        """Async handler for crypto quotes data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "crypto", "quotes"))
    
    async def _crypto_trades_handler(self, data):
        """Async handler for crypto trades data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "crypto", "trades"))
    
    async def _crypto_bars_handler(self, data):
        """Async handler for crypto bars data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "crypto", "bars"))
    
    async def _crypto_orderbooks_handler(self, data):
        """Async handler for crypto orderbooks data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "crypto", "orderbooks"))
    
    async def _stock_quotes_handler(self, data):
        """Async handler for stock quotes data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "stock", "quotes"))
    
    async def _stock_trades_handler(self, data):
        """Async handler for stock trades data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "stock", "trades"))
    
    async def _stock_bars_handler(self, data):
        """Async handler for stock bars data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "stock", "bars"))
    
    async def _stock_orderbooks_handler(self, data):
        """Async handler for stock orderbooks data."""
        symbol = data.get("S", "")
        if symbol:
            await self._data_queue.put((data, symbol, "stock", "orderbooks"))
    
    async def _news_handler(self, data):
        """Async handler for news data."""
        # News may have multiple symbols or a single symbol
        symbols = data.get("symbols", [])
        if symbols:
            # If multiple symbols, use the first one as the primary symbol
            symbol = symbols[0] if symbols else "news"
        else:
            symbol = data.get("S", "news")  # Fallback to "news" if no symbol
        
        # News data doesn't have asset_type, use "news" as asset_type
        await self._data_queue.put((data, symbol, "news", "news"))
    
    def _data_stream_worker(self, symbols: List[str], asset_types: Dict[str, str]):
        """Worker thread for running data streams."""
        loop = None
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            
            async def setup_and_run():
                await self.database_service.connect()
                
                # Initialize data queue and semaphore for concurrency control
                self._data_queue = asyncio.Queue(maxsize=1000)  # Buffer up to 1000 items
                self._data_semaphore = asyncio.Semaphore(self._max_concurrent_writes)
                
                # Start background data processor
                processor_task = asyncio.create_task(self._data_processor())
                
                # Separate symbols by type
                crypto_symbols = [s for s in symbols if asset_types.get(s) == "crypto"]
                stock_symbols = [s for s in symbols if asset_types.get(s) == "stock"]
                
                # Ensure tables exist for all data types
                for symbol in symbols:
                    asset_type = asset_types.get(symbol, "stock")
                    # Stock doesn't support orderbooks, only crypto does
                    data_types = ["quotes", "trades", "bars"]
                    if asset_type == "crypto":
                        data_types.append("orderbooks")
                    for data_type in data_types:
                        await self._ensure_table_exists(symbol, asset_type, data_type)
                
                # Create news table (news is global, not per-symbol)
                await self._ensure_table_exists("news", "news", "news")
                
                # Initialize and subscribe to crypto stream
                if crypto_symbols:
                    self._crypto_data_stream = CryptoDataStream(
                        api_key=self.default_account.api_key,
                        secret_key=self.default_account.secret_key
                    )
                    for symbol in crypto_symbols:
                        self._crypto_data_stream.subscribe_quotes(self._crypto_quotes_handler, symbol)
                        self._crypto_data_stream.subscribe_trades(self._crypto_trades_handler, symbol)
                        self._crypto_data_stream.subscribe_bars(self._crypto_bars_handler, symbol)
                        self._crypto_data_stream.subscribe_orderbooks(self._crypto_orderbooks_handler, symbol)
                        logger.info(f"| 📡 Subscribed to crypto data (quotes, trades, bars, orderbooks): {symbol}")
                
                # Initialize and subscribe to stock stream
                if stock_symbols:
                    self._stock_data_stream = StockDataStream(
                        api_key=self.default_account.api_key,
                        secret_key=self.default_account.secret_key
                    )
                    for symbol in stock_symbols:
                        self._stock_data_stream.subscribe_quotes(self._stock_quotes_handler, symbol)
                        self._stock_data_stream.subscribe_trades(self._stock_trades_handler, symbol)
                        self._stock_data_stream.subscribe_bars(self._stock_bars_handler, symbol)
                        logger.info(f"| 📡 Subscribed to stock data (quotes, trades, bars): {symbol}")
                
                # Initialize and subscribe to news stream
                self._news_data_stream = NewsDataStream(
                    api_key=self.default_account.api_key,
                    secret_key=self.default_account.secret_key
                )
                # Subscribe to news for all symbols
                for symbol in symbols:
                    self._news_data_stream.subscribe_news(self._news_handler, symbol)
                logger.info(f"| 📡 Subscribed to news data for {len(symbols)} symbols")
                
                # Run streams in separate threads (they are blocking)
                import concurrent.futures
                
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
                    if crypto_symbols:
                        futures.append(executor.submit(run_crypto_stream))
                    if stock_symbols:
                        futures.append(executor.submit(run_stock_stream))
                    # Always run news stream
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
    
    def start_data_stream(self, symbols: List[str], asset_types: Optional[Dict[str, str]] = None) -> None:
        """Start real-time data stream collection for given symbols.
        
        This method starts a blocking thread that will collect real-time data
        from Alpaca streams and write it to the database. The thread will block
        the main process until stopped.
        
        Args:
            symbols: List of symbols to subscribe to (e.g., ["BTC/USD", "AAPL"])
            asset_types: Optional dictionary mapping symbol to asset type ("stock" or "crypto")
                        If not provided, will be determined from symbol format
        """
        if self._data_stream_running:
            logger.warning("| ⚠️  Data stream is already running")
            return
        
        if not hasattr(self, 'database_service') or self.database_service is None:
            raise AlpacaError("Database service not initialized. Call initialize() first.")
        
        # Determine asset types if not provided
        if asset_types is None:
            asset_types = {}
            for symbol in symbols:
                if "/" in symbol:  # Crypto symbols typically have "/"
                    asset_types[symbol] = "crypto"
                else:
                    asset_types[symbol] = "stock"
        
        self._subscribed_symbols.update(symbols)
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
        