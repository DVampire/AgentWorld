"""Binance trading service implementation using REST API clients."""
import asyncio
import json
from typing import Optional, Union, List, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.environments.binanceentry.spot_client import BinanceSpotClient
from src.environments.binanceentry.futures_client import BinanceFuturesClient

from src.logger import logger
from src.environments.protocol.types import ActionResult
from src.environments.binanceentry.types import (
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
    TradeType,
    OrderType,
)
from src.environments.binanceentry.exceptions import (
    BinanceError,
    AuthenticationError,
    NotFoundError,
    OrderError,
    InsufficientFundsError,
    InvalidSymbolError,
)
from src.environments.binanceentry.klines import KlinesHandler
from src.environments.binanceentry.producer import DataProducer
from src.environments.binanceentry.consumer import DataConsumer
from src.environments.database.service import DatabaseService
from src.utils import assemble_project_path
from src.config import config


class BinanceService:
    """Binance trading service using REST API clients.
    
    This service only handles cryptocurrency trading:
    - Spot trading
    - Perpetual futures trading
    
    Supports live trading and testnet via the 'live' parameter.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        accounts: List[Dict[str, str]],
        live: bool = False,
        default_trade_type: TradeType = TradeType.PERPETUAL,
        auto_start_data_stream: bool = False,
        symbol: Optional[Union[str, List[str]]] = None,
        data_type: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize Binance trading service.
        
        Args:
            base_dir: Base directory for Binance operations
            accounts: List of account dictionaries, each containing API key and secret key
            live: Whether to use live trading (True) or testnet (False)
            default_trade_type: Default trade type to use (SPOT or PERPETUAL). Default: PERPETUAL
            auto_start_data_stream: If True, automatically start data stream after initialization
            symbol: Optional symbol(s) to subscribe to
            data_type: Optional data type(s) to subscribe to
            
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
        self.testnet = not live
        self.default_trade_type = default_trade_type
        
        self.symbol = symbol
        self.data_type = data_type
        
        self._spot_clients: Dict[str, BinanceSpotClient] = {}
        self._futures_clients: Dict[str, BinanceFuturesClient] = {}
        
        self.symbols: Dict[str, Dict] = {}
        
        # Initialize data handlers
        self._klines_handler: Optional[KlinesHandler] = None
        
        # Producer and Consumer
        self.data_producer: Optional[DataProducer] = None
        self.data_consumer: Optional[DataConsumer] = None
        
        self._max_concurrent_writes: int = 10  # Max concurrent database writes
    
    def _get_spot_client(self, account_name: str) -> BinanceSpotClient:
        """Get or create spot client for an account (lazy initialization).
        
        Args:
            account_name: Account name
            
        Returns:
            BinanceSpotClient instance
        """
        if account_name not in self._spot_clients:
            account = self.accounts[account_name]
            self._spot_clients[account_name] = BinanceSpotClient(
                api_key=account.api_key,
                api_secret=account.secret_key,
                testnet=self.testnet
            )
        return self._spot_clients[account_name]
    
    def _get_futures_client(self, account_name: str) -> BinanceFuturesClient:
        """Get or create futures client for an account (lazy initialization).
        
        Args:
            account_name: Account name
            
        Returns:
            BinanceFuturesClient instance
        """
        if account_name not in self._futures_clients:
            account = self.accounts[account_name]
            self._futures_clients[account_name] = BinanceFuturesClient(
                api_key=account.api_key,
                api_secret=account.secret_key,
                testnet=self.testnet
            )
        return self._futures_clients[account_name]

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the Binance trading service."""
        try:
            self.base_dir = Path(assemble_project_path(self.base_dir))
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize default client based on default_trade_type
            for account_name, account in self.accounts.items():
                if self.default_trade_type == TradeType.PERPETUAL:
                    # Initialize futures client (default)
                    self._futures_clients[account_name] = BinanceFuturesClient(
                        api_key=account.api_key,
                        api_secret=account.secret_key,
                        testnet=self.testnet
                    )
                else:
                    # Initialize spot client (default)
                    self._spot_clients[account_name] = BinanceSpotClient(
                        api_key=account.api_key,
                        api_secret=account.secret_key,
                        testnet=self.testnet
                    )
                # Other client will be initialized lazily when needed
            
            # Test connection by getting default account info
            for account_name in self.accounts.keys():
                try:
                    if self.default_trade_type == TradeType.PERPETUAL:
                        account_info = await asyncio.to_thread(
                            self._futures_clients[account_name].get_account
                        )
                        logger.info(f"| 📝 Connected to Binance {'live' if self.live else 'testnet'} futures account: {account_name}")
                    else:
                        account_info = await asyncio.to_thread(
                            self._spot_clients[account_name].account
                        )
                        logger.info(f"| 📝 Connected to Binance {'live' if self.live else 'testnet'} spot account: {account_name}")
                except Exception as e:
                    logger.warning(f"| ⚠️  Failed to connect to {self.default_trade_type.value} account {account_name}: {e}")
            
            # Get available trading symbols (cryptocurrency only)
            await self._load_symbols()
            
            logger.info(f"| 📝 Found {len(self.symbols)} cryptocurrency symbols.")
            if len(self.symbols) > 0:
                logger.info(f"| 📝 Sample symbols: {', '.join(list(self.symbols.keys())[:10])}")
            
            # Initialize database
            self.database_base_dir = self.base_dir / "database"
            self.database_base_dir.mkdir(parents=True, exist_ok=True)
            self.database_service = DatabaseService(self.database_base_dir)
            await self.database_service.connect()
            
            # Initialize data handlers
            self._klines_handler = KlinesHandler(self.database_service)
            
            # Initialize Producer and Consumer
            self.data_producer = DataProducer(
                account=self.default_account,
                klines_handler=self._klines_handler,
                symbols=self.symbols,
                max_concurrent_writes=self._max_concurrent_writes,
                testnet=self.testnet,
                default_trade_type=self.default_trade_type
            )
            
            self.data_consumer = DataConsumer(
                klines_handler=self._klines_handler
            )
            
            # Optionally start data stream automatically (use default_trade_type)
            if self.auto_start_data_stream and self.symbol:
                # Normalize symbol to list
                symbols_list = self.symbol if isinstance(self.symbol, list) else [self.symbol]
                # Use default_trade_type for all symbols
                trade_types = {symbol: self.default_trade_type for symbol in symbols_list}
                self.start_data_stream(symbols_list, trade_types=trade_types)
                logger.info(f"| 📡 Auto-started data stream ({self.default_trade_type.value} WebSocket) for {len(symbols_list)} symbols: {symbols_list}")
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Invalid Binance credentials: {e}")
            raise BinanceError(f"Failed to initialize Binance service: {e}")

    async def _load_symbols(self) -> None:
        """Load available cryptocurrency trading symbols."""
        try:
            # Get futures trading symbols (default to futures)
            # Note: We still use spot exchange_info for symbol list as it's more comprehensive
            # But we initialize spot client lazily only when needed
            spot_client = self._get_spot_client(self.default_account.name)
            exchange_info = await asyncio.to_thread(
                spot_client.exchange_info
            )
            
            self.symbols = {}
            for symbol_info in exchange_info.get('symbols', []):
                symbol = symbol_info['symbol']
                if symbol_info.get('status') == 'TRADING':
                    self.symbols[symbol] = {
                        'symbol': symbol,
                        'baseAsset': symbol_info.get('baseAsset'),
                        'quoteAsset': symbol_info.get('quoteAsset'),
                        'status': symbol_info.get('status'),
                        'tradable': True,
                        'type': 'spot'
                    }
            
        except Exception as e:
            logger.warning(f"| ⚠️  Failed to load symbols: {e}")
            self.symbols = {}

    async def cleanup(self) -> None:
        """Cleanup the Binance service."""
        # Stop data stream first to ensure proper cleanup
        if self.data_producer and self.data_producer._data_stream_running:
            logger.info("| 🛑 Stopping data stream during cleanup...")
            self.data_producer.stop()
            # Wait a bit for threads to finish
            import time
            time.sleep(0.5)
        
        self._spot_clients = {}
        self._futures_clients = {}
        
        self._klines_handler = None
        
        self.data_producer = None
        self.data_consumer = None
        
        if hasattr(self, 'database_service'):
            await self.database_service.disconnect()
        
        self.symbols = {}

    # Account methods
    async def get_account(self, request: GetAccountRequest, trade_type: Optional[TradeType] = None) -> ActionResult:
        """Get account information.
        
        Args:
            request: GetAccountRequest with account_name
            trade_type: Optional trade type (SPOT or PERPETUAL). If None, uses default_trade_type.
        """
        try:
            # Use default trade type if not specified
            if trade_type is None:
                trade_type = self.default_trade_type
            
            if trade_type == TradeType.PERPETUAL:
                futures_client = self._get_futures_client(request.account_name)
                account_info = await asyncio.to_thread(futures_client.get_account)
                account_data = {
                    "assets": account_info.get('assets', []),
                    "total_wallet_balance": account_info.get('totalWalletBalance'),
                    "total_unrealized_profit": account_info.get('totalUnrealizedProfit'),
                    "available_balance": account_info.get('availableBalance'),
                    "trade_type": "perpetual",
                }
            else:
                spot_client = self._get_spot_client(request.account_name)
                account_info = await asyncio.to_thread(spot_client.account)
                account_data = {
                    "account_type": account_info.get('accountType'),
                    "balances": account_info.get('balances', []),
                    "permissions": account_info.get('permissions', []),
                    "trade_type": "spot",
                }
            
            return ActionResult(
                success=True,
                message=f"Account information retrieved successfully ({trade_type.value}).",
                extra={"account": account_data}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise BinanceError(f"Failed to get account: {e}")
        
    async def get_assets(self, request: GetAssetsRequest) -> ActionResult:
        """Get available cryptocurrency trading symbols."""
        try:
            # Return all symbols we loaded during initialization
            assets = list(self.symbols.values())
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(assets)} cryptocurrency symbols.",
                extra={"assets": assets}
            )
            
        except Exception as e:
            raise BinanceError(f"Failed to get assets: {e}")
    
    async def get_positions(self, request: GetPositionsRequest) -> ActionResult:
        """Get all positions.
        
        Args:
            request: GetPositionsRequest with account_name and optional trade_type.
                    If trade_type is None, uses default_trade_type.
        """
        try:
            # Use default trade type if not specified
            trade_type = request.trade_type if request.trade_type is not None else self.default_trade_type
            
            all_positions = []
            
            if trade_type == TradeType.SPOT:
                # Get spot positions (non-zero balances)
                spot_client = self._get_spot_client(request.account_name)
                account_info = await asyncio.to_thread(spot_client.account)
                balances = account_info.get('balances', [])
                
                for balance in balances:
                    free = float(balance.get('free', 0))
                    locked = float(balance.get('locked', 0))
                    total = free + locked
                    
                    if total > 0:
                        asset = balance.get('asset')
                        positions = {
                            "symbol": asset,
                            "asset": asset,
                            "free": str(free),
                            "locked": str(locked),
                            "total": str(total),
                            "trade_type": "spot",
                        }
                        all_positions.append(positions)
            else:
                # Get perpetual futures positions
                futures_client = self._get_futures_client(request.account_name)
                futures_positions = await asyncio.to_thread(
                    futures_client.get_position_risk
                )
                
                for position in futures_positions:
                    position_amt = float(position.get('positionAmt', 0))
                    if position_amt != 0:
                        positions = {
                            "symbol": position.get('symbol'),
                            "position_amt": str(position_amt),
                            "entry_price": position.get('entryPrice'),
                            "mark_price": position.get('markPrice'),
                            "unrealized_profit": position.get('unRealizedProfit'),
                            "leverage": position.get('leverage'),
                            "position_side": position.get('positionSide'),
                            "trade_type": "perpetual",
                        }
                        all_positions.append(positions)
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(all_positions)} positions ({trade_type.value}).",
                extra={"positions": all_positions}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise BinanceError(f"Failed to get positions: {e}")
    
    def start_data_stream(self, symbols: List[str], intervals: Optional[Dict[str, str]] = None, trade_types: Optional[Dict[str, TradeType]] = None) -> None:
        """Start real-time data stream collection for given symbols.
        
        This method delegates to the DataProducer.
        
        Args:
            symbols: List of symbols to subscribe to (e.g., ["BTCUSDT", "ETHUSDT"])
            intervals: Optional dictionary mapping symbol to interval (e.g., {"BTCUSDT": "1m", "ETHUSDT": "5m"})
            trade_types: Optional dictionary mapping symbol to trade type. If None, uses default_trade_type.
        """
        if not self.data_producer:
            raise BinanceError("Data producer not initialized. Call initialize() first.")
        # If trade_types is None, DataProducer will use default_trade_type for all symbols
        self.data_producer.start(symbols, intervals, trade_types)
    
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
            request: GetDataRequest with symbol (str or list), data_type,
                    optional start_date, end_date, and limit
            
        Returns:
            ActionResult with data organized by symbol in extra field
        """
        if not self.data_consumer:
            raise BinanceError("Data consumer not initialized. Call initialize() first.")
        return await self.data_consumer.get_data(request)
    
    # Order methods
    async def create_order(self, request: CreateOrderRequest) -> ActionResult:
        """Create an order (market order or perpetual futures order).
        
        Args:
            request: CreateOrderRequest with account_name, symbol, side, trade_type, order_type, qty, etc.
                    If trade_type is not specified in request, uses default_trade_type.
            
        Returns:
            ActionResult with order information
        """
        try:
            if request.qty is None:
                raise BinanceError("'qty' must be provided")
            
            if request.order_type == OrderType.LIMIT and request.price is None:
                raise BinanceError("'price' must be provided for LIMIT orders")
            
            # Validate symbol
            if request.symbol not in self.symbols:
                raise InvalidSymbolError(f"Symbol {request.symbol} not found or not tradable")
            
            # Use default trade type if not specified in request
            trade_type = request.trade_type if hasattr(request, 'trade_type') and request.trade_type else self.default_trade_type
            
            side = request.side.upper()
            order_result = None
            
            if trade_type == TradeType.SPOT:
                # Spot market order
                spot_client = self._get_spot_client(request.account_name)
                
                if request.order_type == OrderType.MARKET:
                    if side == "BUY":
                        order_result = await asyncio.to_thread(
                            spot_client.new_order,
                            symbol=request.symbol,
                            side="BUY",
                            type="MARKET",
                            quantity=str(request.qty)
                        )
                    else:  # SELL
                        order_result = await asyncio.to_thread(
                            spot_client.new_order,
                            symbol=request.symbol,
                            side="SELL",
                            type="MARKET",
                            quantity=str(request.qty)
                        )
                else:  # LIMIT order
                    order_result = await asyncio.to_thread(
                        spot_client.new_order,
                        symbol=request.symbol,
                        side=side,
                        type="LIMIT",
                        timeInForce=request.time_in_force,
                        quantity=str(request.qty),
                        price=str(request.price)
                    )
            
            elif request.trade_type == TradeType.PERPETUAL:
                # Set leverage if provided
                futures_client = self._futures_clients[request.account_name]
                
                # Check and set position mode if positionSide is specified
                # If positionSide is provided, we need hedge mode (dual-side position)
                if request.position_side:
                    try:
                        position_mode = await asyncio.to_thread(futures_client.get_position_mode)
                        is_hedge_mode = position_mode.get('dualSidePosition', False)
                        
                        if not is_hedge_mode:
                            # Switch to hedge mode to support positionSide
                            logger.info(f"| 🔄 Switching to hedge mode (dual-side position) to support positionSide={request.position_side}")
                            await asyncio.to_thread(futures_client.set_position_mode, dual_side_position=True)
                    except Exception as e:
                        logger.warning(f"| ⚠️  Could not check/set position mode: {e}. Proceeding with order creation.")
                
                # Set leverage if provided
                if request.leverage is not None:
                    await asyncio.to_thread(
                        futures_client.set_leverage,
                        symbol=request.symbol,
                        leverage=request.leverage
                    )
                
                # Create perpetual futures order
                order_result = await asyncio.to_thread(
                    futures_client.create_order,
                    symbol=request.symbol,
                    side=side,
                    type=request.order_type.value,
                    quantity=str(request.qty),
                    price=str(request.price) if request.order_type == OrderType.LIMIT else None,
                    timeInForce=request.time_in_force if request.order_type == OrderType.LIMIT else None,
                    positionSide=request.position_side
                )
            
            # Format order information
            order_info = {
                "order_id": str(order_result.get('orderId')),
                "client_order_id": order_result.get('clientOrderId'),
                "symbol": order_result.get('symbol'),
                "side": order_result.get('side'),
                "type": order_result.get('type'),
                "status": order_result.get('status'),
                "quantity": order_result.get('origQty') or order_result.get('quantity'),
                "price": order_result.get('price'),
                "executed_qty": order_result.get('executedQty', '0'),
                "trade_type": trade_type.value,
            }
            
            return ActionResult(
                success=True,
                message=f"Order {order_info['order_id']} submitted successfully for {request.symbol} ({side} {request.qty}).",
                extra={"order": order_info}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            if 'insufficient' in str(e).lower() or 'balance' in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            raise OrderError(f"Failed to create order: {e}")
    
    async def get_orders(self, request: GetOrdersRequest) -> ActionResult:
        """Get orders for an account.
        
        Args:
            request: GetOrdersRequest with account_name and optional trade_type.
                    If trade_type is None, uses default_trade_type.
        """
        try:
            # Use default trade type if not specified
            trade_type = request.trade_type if request.trade_type is not None else self.default_trade_type
            
            all_orders = []
            
            if trade_type == TradeType.SPOT:
                spot_client = self._get_spot_client(request.account_name)
                if request.symbol:
                    spot_orders = await asyncio.to_thread(
                        spot_client.get_orders,
                        symbol=request.symbol
                    )
                else:
                    spot_orders = []
                
                if request.limit:
                    spot_orders = spot_orders[:request.limit]
                
                for order in spot_orders:
                    if request.order_id is None or str(order.get('orderId')) == str(request.order_id):
                        order_info = {
                            "order_id": str(order.get('orderId')),
                            "client_order_id": order.get('clientOrderId'),
                            "symbol": order.get('symbol'),
                            "side": order.get('side'),
                            "type": order.get('type'),
                            "status": order.get('status'),
                            "quantity": order.get('origQty'),
                            "price": order.get('price'),
                            "executed_qty": order.get('executedQty', '0'),
                            "time": order.get('time'),
                            "update_time": order.get('updateTime'),
                            "trade_type": "spot",
                        }
                        all_orders.append(order_info)
            else:
                futures_client = self._get_futures_client(request.account_name)
                futures_orders = await asyncio.to_thread(
                    futures_client.get_all_orders,
                    symbol=request.symbol if request.symbol else None
                )
                
                if request.limit:
                    futures_orders = futures_orders[:request.limit]
                
                for order in futures_orders:
                    if request.order_id is None or str(order.get('orderId')) == str(request.order_id):
                        order_info = {
                            "order_id": str(order.get('orderId')),
                            "client_order_id": order.get('clientOrderId'),
                            "symbol": order.get('symbol'),
                            "side": order.get('side'),
                            "type": order.get('type'),
                            "status": order.get('status'),
                            "quantity": order.get('origQty') or order.get('quantity'),
                            "price": order.get('price'),
                            "executed_qty": order.get('executedQty', '0'),
                            "time": order.get('time'),
                            "update_time": order.get('updateTime'),
                            "trade_type": "perpetual",
                        }
                        all_orders.append(order_info)
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(all_orders)} orders ({trade_type.value}).",
                extra={"orders": all_orders}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise BinanceError(f"Failed to get orders: {e}")
    
    async def get_order(self, request: GetOrderRequest) -> ActionResult:
        """Get a specific order by ID.
        
        Args:
            request: GetOrderRequest with account_name, order_id, symbol, and trade_type.
                    If trade_type is not specified, uses default_trade_type.
        """
        try:
            # Use default trade type if not specified
            trade_type = request.trade_type if hasattr(request, 'trade_type') and request.trade_type else self.default_trade_type
            
            if trade_type == TradeType.SPOT:
                spot_client = self._get_spot_client(request.account_name)
                order = await asyncio.to_thread(
                    spot_client.get_order,
                    symbol=request.symbol,
                    orderId=request.order_id
                )
            else:  # PERPETUAL
                futures_client = self._get_futures_client(request.account_name)
                order = await asyncio.to_thread(
                    futures_client.get_order,
                    symbol=request.symbol,
                    orderId=request.order_id
                )
            
            order_info = {
                "order_id": str(order.get('orderId')),
                "client_order_id": order.get('clientOrderId'),
                "symbol": order.get('symbol'),
                "side": order.get('side'),
                "type": order.get('type'),
                "status": order.get('status'),
                "quantity": order.get('origQty') or order.get('quantity'),
                "price": order.get('price'),
                "executed_qty": order.get('executedQty', '0'),
                "time": order.get('time'),
                "update_time": order.get('updateTime'),
                "trade_type": trade_type.value,
            }
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} retrieved successfully ({trade_type.value}).",
                extra={"order": order_info}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            if "404" in str(e) or "-2013" in str(e):  # Order does not exist
                raise NotFoundError(f"Order {request.order_id} not found: {e}")
            raise BinanceError(f"Failed to get order: {e}")
    
    async def cancel_order(self, request: CancelOrderRequest) -> ActionResult:
        """Cancel an order.
        
        Args:
            request: CancelOrderRequest with account_name, order_id, symbol, and trade_type.
                    If trade_type is not specified, uses default_trade_type.
        """
        try:
            # Use default trade type if not specified
            trade_type = request.trade_type if hasattr(request, 'trade_type') and request.trade_type else self.default_trade_type
            
            if trade_type == TradeType.SPOT:
                spot_client = self._get_spot_client(request.account_name)
                result = await asyncio.to_thread(
                    spot_client.cancel_order,
                    symbol=request.symbol,
                    orderId=request.order_id
                )
            else:  # PERPETUAL
                futures_client = self._get_futures_client(request.account_name)
                result = await asyncio.to_thread(
                    futures_client.cancel_order,
                    symbol=request.symbol,
                    orderId=request.order_id
                )
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} canceled successfully ({trade_type.value}).",
                extra={"order_id": request.order_id, "result": result}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            if "404" in str(e) or "-2013" in str(e):  # Order does not exist
                raise NotFoundError(f"Order {request.order_id} not found: {e}")
            raise BinanceError(f"Failed to cancel order: {e}")
    
    async def cancel_all_orders(self, request: CancelAllOrdersRequest) -> ActionResult:
        """Cancel all orders for an account.
        
        Args:
            request: CancelAllOrdersRequest with account_name, optional symbol, and optional trade_type.
                    If trade_type is None, uses default_trade_type.
        """
        try:
            # Use default trade type if not specified
            trade_type = request.trade_type if request.trade_type is not None else self.default_trade_type
            
            canceled_orders = []
            
            if trade_type == TradeType.SPOT:
                spot_client = self._get_spot_client(request.account_name)
                if request.symbol:
                    result = await asyncio.to_thread(
                        spot_client.cancel_open_orders,
                        symbol=request.symbol
                    )
                    canceled_orders.append({"trade_type": "spot", "symbol": request.symbol, "result": result})
            else:
                futures_client = self._get_futures_client(request.account_name)
                if request.symbol:
                    result = await asyncio.to_thread(
                        futures_client.cancel_all_open_orders,
                        symbol=request.symbol
                    )
                    canceled_orders.append({"trade_type": "perpetual", "symbol": request.symbol, "result": result})
            
            return ActionResult(
                success=True,
                message=f"All orders canceled successfully ({trade_type.value}).",
                extra={"account_name": request.account_name, "canceled_orders": canceled_orders}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise BinanceError(f"Failed to cancel all orders: {e}")
