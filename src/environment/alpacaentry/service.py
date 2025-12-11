"""Alpaca trading service implementation using alpaca-py."""
import threading
import asyncio
from typing import Optional, Union, List, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)

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
from alpaca.common.exceptions import APIError
from pydantic import BaseModel

from src.logger import logger
from src.environment.types import ActionResult
from src.environment.alpacaentry.types import (
    AccountInfo,
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
from src.environment.alpacaentry.exceptions import (
    AlpacaError,
    AuthenticationError,
)
from src.environment.alpacaentry.bars import BarsHandler
from src.environment.alpacaentry.quotes import QuotesHandler
from src.environment.alpacaentry.trades import TradesHandler
from src.environment.alpacaentry.orderbooks import OrderbooksHandler
from src.environment.alpacaentry.news import NewsHandler
from src.environment.alpacaentry.producer import DataProducer
from src.environment.alpacaentry.consumer import DataConsumer
from src.environment.database.service import DatabaseService
from src.utils import assemble_project_path
from src.config import config


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
        
        # Initialize data handlers
        self._bars_handler: Optional[BarsHandler] = None
        self._quotes_handler: Optional[QuotesHandler] = None
        self._orderbooks_handler: Optional[OrderbooksHandler] = None
        self._trades_handler: Optional[TradesHandler] = None
        self._news_handler: Optional[NewsHandler] = None
        
        # Producer and Consumer
        self.data_producer: Optional[DataProducer] = None
        self.data_consumer: Optional[DataConsumer] = None
        
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
            self.base_dir = Path(assemble_project_path(self.base_dir))
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
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
            
            # Initialize Producer and Consumer
            self.data_producer = DataProducer(
                account=self.default_account,
                bars_handler=self._bars_handler,
                quotes_handler=self._quotes_handler,
                trades_handler=self._trades_handler,
                orderbooks_handler=self._orderbooks_handler,
                news_handler=self._news_handler,
                symbols=self.symbols,
                max_concurrent_writes=self._max_concurrent_writes
            )
            
            self.data_consumer = DataConsumer(
                bars_handler=self._bars_handler,
                quotes_handler=self._quotes_handler,
                trades_handler=self._trades_handler,
                orderbooks_handler=self._orderbooks_handler,
                news_handler=self._news_handler
            )
            
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
        if self.data_producer and self.data_producer._data_stream_running:
            logger.info("| 🛑 Stopping data stream during cleanup...")
            self.data_producer.stop()
            # Wait a bit for threads to finish
            import time
            time.sleep(0.5)
        
        self._trading_clients = None
        self._default_trading_client = None
        
        self._stock_data_client = None
        self._crypto_data_client = None
        self._news_client = None
        self._option_data_client = None
        
        self._trades_handler = None
        self._quotes_handler = None
        self._bars_handler = None
        self._orderbooks_handler = None
        self._news_handler = None
        
        self.data_producer = None
        self.data_consumer = None

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
    
    def start_data_stream(self, symbols: List[str], asset_types: Optional[Dict[str, AssetClass]] = None) -> None:
        """Start real-time data stream collection for given symbols.
        
        This method delegates to the DataProducer.
        
        Args:
            symbols: List of symbols to subscribe to (e.g., ["BTC/USD", "AAPL"])
            asset_types: Optional dictionary mapping symbol to asset class (AssetClass)
                        If not provided, will be determined from symbol format
        """
        if not self.data_producer:
            raise AlpacaError("Data producer not initialized. Call initialize() first.")
        self.data_producer.start(symbols, asset_types)
    
    def stop_data_stream(self) -> None:
        """Stop the data stream.
        
        This method delegates to the DataProducer.
        """
        if not self.data_producer:
            logger.warning("| ⚠️  Data producer not initialized")
            return
        self.data_producer.stop()
    
    async def get_data(self, request: GetDataRequest) -> ActionResult:
        """Get historical data from database.
        
        This method delegates to the DataConsumer.
        
        Args:
            request: GetDataRequest with symbol (str or list), data_type (str or list), 
                    optional start_date, end_date, and limit
            
        Returns:
            ActionResult with data organized by symbol in extra field
        """
        if not self.data_consumer:
            raise AlpacaError("Data consumer not initialized. Call initialize() first.")
        return await self.data_consumer.get_data(request)
    
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
                from src.environment.alpacaentry.exceptions import NotFoundError
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
                from src.environment.alpacaentry.exceptions import NotFoundError
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