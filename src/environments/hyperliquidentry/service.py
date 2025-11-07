"""Hyperliquid trading service implementation using REST API clients."""
import asyncio
import json
from typing import Optional, Union, List, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.environments.hyperliquidentry.client import HyperliquidClient

from src.logger import logger
from src.environments.protocol.types import ActionResult
from src.environments.hyperliquidentry.types import (
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
from src.environments.hyperliquidentry.exceptions import (
    HyperliquidError,
    AuthenticationError,
    NotFoundError,
    OrderError,
    InsufficientFundsError,
    InvalidSymbolError,
)
from src.environments.database.service import DatabaseService
from src.environments.hyperliquidentry.candle import CandleHandler
from src.environments.hyperliquidentry.trades import TradesHandler
from src.environments.hyperliquidentry.l2book import L2BookHandler
from src.environments.hyperliquidentry.producer import DataProducer
from src.environments.hyperliquidentry.consumer import DataConsumer
from src.environments.hyperliquidentry.types import DataStreamType
from src.utils import assemble_project_path
from src.config import config


class HyperliquidService:
    """Hyperliquid trading service using REST API clients.
    
    This service handles perpetual futures trading on Hyperliquid:
    - Perpetual futures trading only (Hyperliquid doesn't have spot trading)
    
    Supports live trading and testnet via the 'live' parameter.
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        accounts: List[Dict[str, str]],
        live: bool = False,
        auto_start_data_stream: bool = False,
        symbol: Optional[Union[str, List[str]]] = None,
        data_type: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize Hyperliquid trading service.
        
        Args:
            base_dir: Base directory for Hyperliquid operations
            accounts: List of account dictionaries, each containing address and optional private_key
            live: Whether to use live trading (True) or testnet (False)
            auto_start_data_stream: If True, automatically start data stream after initialization
            symbol: Optional symbol(s) to subscribe to
            data_type: Optional data type(s) to subscribe to
            
            accounts = [
                {
                    "name": "Account 1",
                    "address": "0x...",  # Wallet address
                    "private_key": "0x...",  # Optional, required for trading
                },
                {
                    "name": "Account 2",
                    "address": "0x...",
                    "private_key": "0x...",
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
        
        self.symbol = symbol
        self.data_type = data_type
        
        self._clients: Dict[str, HyperliquidClient] = {}
        
        self.symbols: Dict[str, Dict] = {}
        
        # Initialize database
        self.database_base_dir = self.base_dir / "database"
        self.database_base_dir.mkdir(parents=True, exist_ok=True)
        self.database_service: Optional[DatabaseService] = None
        
        # Initialize data handlers
        self._candle_handler: Optional[CandleHandler] = None
        self._trades_handler: Optional[TradesHandler] = None
        self._l2book_handler: Optional[L2BookHandler] = None
        
        # Producer and Consumer
        self.data_producer: Optional[DataProducer] = None
        self.data_consumer: Optional[DataConsumer] = None
        
        self._max_concurrent_writes: int = 10  # Max concurrent database writes
    
    def _get_client(self, account_name: str) -> HyperliquidClient:
        """Get or create client for an account (lazy initialization).
        
        Args:
            account_name: Account name
            
        Returns:
            HyperliquidClient instance
        """
        if account_name not in self._clients:
            account = self.accounts[account_name]
            self._clients[account_name] = HyperliquidClient(
                wallet_address=account.address,  # Wallet address
                private_key=account.private_key if account.private_key else None,
                testnet=self.testnet
            )
        return self._clients[account_name]
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize the Hyperliquid trading service."""
        try:
            self.base_dir = Path(assemble_project_path(self.base_dir))
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize default client
            for account_name, account in self.accounts.items():
                self._clients[account_name] = HyperliquidClient(
                    wallet_address=account.address,  # Wallet address
                    private_key=account.private_key if account.private_key else None,
                    testnet=self.testnet
                )
            
            # Test connection by getting default account info
            for account_name in self.accounts.keys():
                try:
                    account_info = await asyncio.to_thread(
                        self._clients[account_name].get_account
                    )
                    logger.info(f"| 📝 Connected to Hyperliquid {'live' if self.live else 'testnet'} account: {account_name}")
                except Exception as e:
                    logger.warning(f"| ⚠️  Failed to connect to account {account_name}: {e}")
            
            # Get available trading symbols
            await self._load_symbols()
            
            logger.info(f"| 📝 Found {len(self.symbols)} symbols.")
            if len(self.symbols) > 0:
                logger.info(f"| 📝 Sample symbols: {', '.join(list(self.symbols.keys())[:10])}")
            
            # Initialize database
            self.database_service = DatabaseService(self.database_base_dir)
            await self.database_service.connect()
            
            # Initialize data handlers
            self._candle_handler = CandleHandler(self.database_service)
            self._trades_handler = TradesHandler(self.database_service)
            self._l2book_handler = L2BookHandler(self.database_service)
            
            # Initialize producer and consumer
            self.data_producer = DataProducer(
                account=self.default_account,
                candle_handler=self._candle_handler,
                trades_handler=self._trades_handler,
                l2book_handler=self._l2book_handler,
                symbols=self.symbols,
                max_concurrent_writes=self._max_concurrent_writes,
                testnet=self.testnet
            )
            
            self.data_consumer = DataConsumer(
                candle_handler=self._candle_handler,
                trades_handler=self._trades_handler,
                l2book_handler=self._l2book_handler
            )
            
            # Auto-start data stream if requested
            if self.auto_start_data_stream and self.symbol:
                symbols = self.symbol if isinstance(self.symbol, list) else [self.symbol]
                data_types = None
                if self.data_type:
                    if isinstance(self.data_type, list):
                        data_types = [DataStreamType(dt) for dt in self.data_type]
                    else:
                        data_types = [DataStreamType(self.data_type)]
                self.start_data_stream(symbols, data_types)
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Invalid Hyperliquid credentials: {e}")
            raise HyperliquidError(f"Failed to initialize Hyperliquid service: {e}")
    
    async def _load_symbols(self) -> None:
        """Load available trading symbols."""
        try:
            # Get exchange info
            client = self._get_client(self.default_account.name)
            exchange_info = await asyncio.to_thread(
                client.get_exchange_info
            )
            
            self.symbols = {}
            # Parse exchange info to extract symbols
            # This depends on Hyperliquid's actual response structure
            if isinstance(exchange_info, dict):
                # Assuming exchange_info contains a list of symbols or coins
                coins = exchange_info.get('universe', [])
                for coin_info in coins:
                    if isinstance(coin_info, dict):
                        symbol = coin_info.get('name', '')
                    else:
                        symbol = str(coin_info)
                    
                    if symbol:
                        self.symbols[symbol] = {
                            'symbol': symbol,
                            'baseAsset': symbol,
                            'quoteAsset': 'USD',  # Hyperliquid uses USD as quote
                            'status': 'TRADING',
                            'tradable': True,
                            'type': 'perpetual'
                        }
            elif isinstance(exchange_info, list):
                for coin_info in exchange_info:
                    if isinstance(coin_info, dict):
                        symbol = coin_info.get('name', '')
                    else:
                        symbol = str(coin_info)
                    
                    if symbol:
                        self.symbols[symbol] = {
                            'symbol': symbol,
                            'baseAsset': symbol,
                            'quoteAsset': 'USD',
                            'status': 'TRADING',
                            'tradable': True,
                            'type': 'perpetual'
                        }
            
        except Exception as e:
            logger.warning(f"| ⚠️  Failed to load symbols: {e}")
            self.symbols = {}
    
    async def cleanup(self) -> None:
        """Cleanup the Hyperliquid service."""
        # Stop data stream if running
        if self.data_producer:
            self.data_producer.stop()
        
        self._clients = {}
        
        if hasattr(self, 'database_service') and self.database_service:
            await self.database_service.disconnect()
        
        self.symbols = {}
        
        # Clear handlers
        self._candle_handler = None
        self._trades_handler = None
        self._l2book_handler = None
        self.data_producer = None
        self.data_consumer = None
    
    # Account methods
    async def get_account(self, request: GetAccountRequest) -> ActionResult:
        """Get account information.
        
        Args:
            request: GetAccountRequest with account_name
        """
        try:
            client = self._get_client(request.account_name)
            account_info = await asyncio.to_thread(client.get_account)
            
            # Format account data
            account_data = {
                "wallet_address": client.wallet_address,
                "margin_summary": account_info.get('marginSummary', {}),
                "asset_positions": account_info.get('assetPositions', []),
                "trade_type": "perpetual",
            }
            
            return ActionResult(
                success=True,
                message=f"Account information retrieved successfully.",
                extra={"account": account_data}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise HyperliquidError(f"Failed to get account: {e}")
    
    async def get_assets(self, request: GetAssetsRequest) -> ActionResult:
        """Get available trading symbols."""
        try:
            # Return all symbols we loaded during initialization
            assets = list(self.symbols.values())
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(assets)} symbols.",
                extra={"assets": assets}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to get assets: {e}")
    
    async def get_positions(self, request: GetPositionsRequest) -> ActionResult:
        """Get all positions.
        
        Args:
            request: GetPositionsRequest with account_name
        """
        try:
            client = self._get_client(request.account_name)
            positions = await asyncio.to_thread(client.get_positions)
            
            all_positions = []
            for position in positions:
                if isinstance(position, dict):
                    position_amt = float(position.get('position', {}).get('coin', 0) or 0)
                    if position_amt != 0:
                        all_positions.append({
                            "symbol": position.get('position', {}).get('coin', ''),
                            "position_amt": str(position_amt),
                            "entry_price": position.get('position', {}).get('entryPx', '0'),
                            "mark_price": position.get('position', {}).get('markPx', '0'),
                            "unrealized_profit": position.get('position', {}).get('unrealizedPnl', '0'),
                            "leverage": position.get('position', {}).get('leverage', {}).get('value', '1'),
                            "trade_type": "perpetual",
                        })
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(all_positions)} positions.",
                extra={"positions": all_positions}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise HyperliquidError(f"Failed to get positions: {e}")
    
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
            raise HyperliquidError("Data consumer not initialized. Call initialize() first.")
        return await self.data_consumer.get_data(request)
    
    def start_data_stream(
        self,
        symbols: List[str],
        data_types: Optional[List[DataStreamType]] = None
    ) -> None:
        """Start real-time data stream collection for given symbols.
        
        This method delegates to the DataProducer.
        
        Args:
            symbols: List of symbols to subscribe to (e.g., ["BTC", "ETH"])
            data_types: Optional list of data types to subscribe to (default: all types)
        """
        if not self.data_producer:
            raise HyperliquidError("Data producer not initialized. Call initialize() first.")
        self.data_producer.start(symbols, data_types)
    
    def stop_data_stream(self) -> None:
        """Stop the data stream.
        
        This method delegates to the DataProducer.
        """
        if not self.data_producer:
            logger.warning("| ⚠️  Data producer not initialized")
            return
        self.data_producer.stop()
    
    # Order methods
    async def create_order(self, request: CreateOrderRequest) -> ActionResult:
        """Create an order (perpetual futures order).
        
        Args:
            request: CreateOrderRequest with account_name, symbol, side, order_type, qty, etc.
            
        Returns:
            ActionResult with order information
        """
        try:
            if request.qty is None:
                raise HyperliquidError("'qty' must be provided")
            
            if request.order_type == OrderType.LIMIT and request.price is None:
                raise HyperliquidError("'price' must be provided for LIMIT orders")
            
            # Validate symbol
            if request.symbol not in self.symbols:
                raise InvalidSymbolError(f"Symbol {request.symbol} not found or not tradable")
            
            client = self._get_client(request.account_name)
            
            # Convert side to Hyperliquid format
            side = "B" if request.side.lower() == "buy" else "A"
            
            # Create order
            order_result = await asyncio.to_thread(
                client.create_order,
                symbol=request.symbol,
                side=side,
                order_type=request.order_type.value,
                size=request.qty,
                price=request.price,
                reduce_only=request.reduce_only or False
            )
            
            # Format order information
            order_info = {
                "order_id": str(order_result.get('status', {}).get('resting', {}).get('oid', 'N/A')),
                "symbol": request.symbol,
                "side": request.side,
                "type": request.order_type.value,
                "status": "submitted",
                "quantity": str(request.qty),
                "price": str(request.price) if request.price else None,
                "trade_type": "perpetual",
            }
            
            return ActionResult(
                success=True,
                message=f"Order {order_info['order_id']} submitted successfully for {request.symbol} ({request.side} {request.qty}).",
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
            request: GetOrdersRequest with account_name and optional symbol
        """
        try:
            client = self._get_client(request.account_name)
            open_orders = await asyncio.to_thread(
                client.get_open_orders,
                symbol=request.symbol
            )
            
            all_orders = []
            for order in open_orders:
                if request.order_id is None or str(order.get('oid')) == str(request.order_id):
                    order_info = {
                        "order_id": str(order.get('oid', 'N/A')),
                        "symbol": order.get('coin', 'N/A'),
                        "side": "buy" if order.get('side') == 'B' else "sell",
                        "type": order.get('orderType', 'N/A'),
                        "status": "open",
                        "quantity": str(order.get('sz', '0')),
                        "price": str(order.get('limitPx', '0')) if order.get('limitPx') else None,
                        "trade_type": "perpetual",
                    }
                    all_orders.append(order_info)
            
            if request.limit:
                all_orders = all_orders[:request.limit]
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(all_orders)} orders.",
                extra={"orders": all_orders}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise HyperliquidError(f"Failed to get orders: {e}")
    
    async def get_order(self, request: GetOrderRequest) -> ActionResult:
        """Get a specific order by ID.
        
        Args:
            request: GetOrderRequest with account_name, order_id, symbol
        """
        try:
            client = self._get_client(request.account_name)
            order = await asyncio.to_thread(
                client.get_order,
                order_id=request.order_id,
                symbol=request.symbol
            )
            
            order_info = {
                "order_id": str(order.get('oid', 'N/A')),
                "symbol": order.get('coin', 'N/A'),
                "side": "buy" if order.get('side') == 'B' else "sell",
                "type": order.get('orderType', 'N/A'),
                "status": "open" if order.get('status') == 'open' else "filled",
                "quantity": str(order.get('sz', '0')),
                "price": str(order.get('limitPx', '0')) if order.get('limitPx') else None,
                "trade_type": "perpetual",
            }
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} retrieved successfully.",
                extra={"order": order_info}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            if "404" in str(e) or "not found" in str(e).lower():
                raise NotFoundError(f"Order {request.order_id} not found: {e}")
            raise HyperliquidError(f"Failed to get order: {e}")
    
    async def cancel_order(self, request: CancelOrderRequest) -> ActionResult:
        """Cancel an order.
        
        Args:
            request: CancelOrderRequest with account_name, order_id, symbol
        """
        try:
            client = self._get_client(request.account_name)
            result = await asyncio.to_thread(
                client.cancel_order,
                order_id=request.order_id,
                symbol=request.symbol
            )
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} canceled successfully.",
                extra={"order_id": request.order_id, "result": result}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            if "404" in str(e) or "not found" in str(e).lower():
                raise NotFoundError(f"Order {request.order_id} not found: {e}")
            raise HyperliquidError(f"Failed to cancel order: {e}")
    
    async def cancel_all_orders(self, request: CancelAllOrdersRequest) -> ActionResult:
        """Cancel all orders for an account.
        
        Args:
            request: CancelAllOrdersRequest with account_name, optional symbol
        """
        try:
            client = self._get_client(request.account_name)
            result = await asyncio.to_thread(
                client.cancel_all_orders,
                symbol=request.symbol
            )
            
            return ActionResult(
                success=True,
                message=f"All orders canceled successfully.",
                extra={"account_name": request.account_name, "result": result}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise HyperliquidError(f"Failed to cancel all orders: {e}")

