"""Hyperliquid trading service implementation using REST API clients."""
import asyncio
import time
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
    GetExchangeInfoRequest,
    GetSymbolInfoRequest,
    GetAssetsRequest,
    GetPositionsRequest,
    GetDataRequest,
    CreateOrderRequest,
    GetOrdersRequest,
    GetOrderRequest,
    CancelOrderRequest,
    CancelAllOrdersRequest,
    CloseOrderRequest,
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
from src.environments.database.types import QueryRequest, SelectRequest
from src.environments.hyperliquidentry.candle import CandleHandler
from src.environments.hyperliquidentry.types import DataStreamType
from src.utils import assemble_project_path

class OnlineHyperliquidService:
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
        self.candle_handler: Optional[CandleHandler] = None
        self.indicators_name: List[str] = []
        
        # Background candle polling task
        self._candle_stream_task: Optional[asyncio.Task] = None
        self._candle_stream_running: bool = False
        self._candle_stream_symbols: List[str] = []
        self._candle_stream_lock = asyncio.Lock()
        
        self._max_concurrent_writes: int = 10  # Max concurrent database writes
        self._max_historical_data_points: int = 120 # 120 minutes = 2 hours
    
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
            
            # Step 1: Initialize accounts
            await self._initialize_account()
            
            # Step 2: Get available trading symbols
            await self._load_symbols()
            
            # Step 3: Initialize database
            await self._initialize_database()
            
            # Step 4: Initialize data handlers
            await self._initialize_data_handlers()
            
            # Auto-start data stream if requested
            if self.auto_start_data_stream and self.symbol:
                symbols = self.symbol if isinstance(self.symbol, list) else [self.symbol]
                logger.info(f"| 📡 Auto-starting candle polling for {len(symbols)} symbols: {symbols}")
                await self.start_data_stream(symbols)
                logger.info(f"| ✅ Candle polling started successfully")
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Invalid Hyperliquid credentials: {e}")
            raise HyperliquidError(f"Failed to initialize Hyperliquid service: {e}")
        
    async def _initialize_account(self) -> None:
        """Initialize accounts."""
        for account_name, account in self.accounts.items():
            self._clients[account_name] = HyperliquidClient(
                wallet_address=account.address,  # Wallet address
                private_key=account.private_key if account.private_key else None,
                testnet=self.testnet
            )
        
        # Test connection by getting default account info
        for account_name in self.accounts.keys():
            try:
                account_info = await self._clients[account_name].get_account()
                logger.info(f"| 📝 Connected to Hyperliquid {'live' if self.live else 'testnet'} account: {account_name}")
            except Exception as e:
                logger.warning(f"| ⚠️  Failed to connect to account {account_name}: {e}")
                
    async def _initialize_database(self) -> None:
        """Initialize database."""
        self.database_service = DatabaseService(self.database_base_dir)
        await self.database_service.connect()
        
    async def _initialize_data_handlers(self) -> None:
        """Initialize data handlers."""
        self.candle_handler = CandleHandler(self.database_service)
        self.indicators_name = await self.candle_handler.get_indicators_name()
        
        # Get symbol data from client
        client = self._get_client(self.default_account.name)
        symbols = self.symbol if isinstance(self.symbol, list) else [self.symbol]
        
        now_time = int(time.time() * 1000)
        start_time = int(now_time - self._max_historical_data_points * 60 * 1000) # 120 minutes = 2 hours ago
        end_time = int(now_time)
        
        for symbol in symbols:
            symbol_data = await client.get_symbol_data(symbol, start_time=start_time, end_time=end_time)
            result = await self.candle_handler.full_insert(symbol_data, symbol)
            if result["success"]:
                logger.info(f"| ✅ Inserted {len(symbol_data)} candles for {symbol}")
            else:
                logger.warning(f"| ⚠️  Failed to insert candles for {symbol}: {result['message']}")
                
                
    async def _load_symbols(self) -> None:
        """Load available trading symbols."""
        try:
            # Get exchange info
            client = self._get_client(self.default_account.name)
            exchange_info = await client.get_exchange_info()
            
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
        # Stop candle polling if running
        if self._candle_stream_task:
            await self.stop_data_stream()
        
        self._clients = {}
        
        if hasattr(self, 'database_service') and self.database_service:
            await self.database_service.disconnect()
        
        self.symbols = {}
        
        # Clear handlers
        self.candle_handler = None
        self.indicators_name = []
        
        self._candle_stream_task = None
        self._candle_stream_running = False
        self._candle_stream_symbols = []
        
    # Get Exchange Info
    async def get_exchange_info(self, request: GetExchangeInfoRequest) -> ActionResult:
        """Get exchange information including available symbols.
        
        Args:
            request: GetExchangeInfoRequest with optional account_name
            
        Returns:
            ActionResult with exchange information
        """
        try:
            client = self._get_client(self.default_account.name)
            exchange_info = await client.get_exchange_info()
            return ActionResult(
                success=True,
                message=f"Exchange information retrieved successfully.",
                extra={"exchange_info": exchange_info}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to get exchange info: {e}")
    
    # Account methods
    async def get_account(self, request: GetAccountRequest) -> ActionResult:
        """Get account information.
        
        Args:
            request: GetAccountRequest with account_name
        """
        try:
            client = self._get_client(request.account_name)
            account_info = await client.get_account()
            
            # Format account data
            account_data = {
                "margin_summary": account_info.get('marginSummary', {}),
                "cross_margin_summary": account_info.get('crossMarginSummary', {}),
                "cross_maintenance_margin_used": account_info.get('crossMarginSummary', {}).get('totalMarginUsed', 0),
                "withdrawable": account_info.get('withdrawable', 0),
                "asset_positions": account_info.get('assetPositions', []),
                "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
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
    
    # Get Symbol Info
    async def get_symbol_info(self, request: GetSymbolInfoRequest) -> ActionResult:
        """Get symbol information for a specific symbol.
        
        Args:
            request: GetSymbolInfoRequest with symbol name
            
        Returns:
            ActionResult with symbol information
        """
        try:
            client = self._get_client(self.default_account.name)
            symbol_info = await client.get_symbol_info(request.symbol)
            
            return ActionResult(
                success=True,
                message=f"Symbol information retrieved successfully for {request.symbol}.",
                extra={"symbol_info": symbol_info}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to get symbol info: {e}")
    
    async def get_positions(self, request: GetPositionsRequest) -> ActionResult:
        """Get all positions.
        
        Args:
            request: GetPositionsRequest with account_name
        """
        try:
            client = self._get_client(request.account_name)
            positions = await client.get_positions()
            
            all_positions = []
            for position in positions:
                if isinstance(position, dict):
                    pos_data = position.get('position', {})
                    # Get position size (szi) - this is the actual position amount
                    # szi is a string representation of the position size
                    szi_str = pos_data.get('szi', '0')
                    try:
                        position_amt = float(szi_str) if szi_str else 0.0
                    except (ValueError, TypeError):
                        position_amt = 0.0
                        
                    symbol_data = await client.get_symbol_data(pos_data.get('coin', ''))
                    if symbol_data:
                        last_price = symbol_data[-1].get('c', '0')
                    else:
                        last_price = '0'
                    
                    # Only include positions with non-zero size
                    if position_amt != 0:
                        all_positions.append({
                            "symbol": pos_data.get('coin', ''),
                            "position_amt": str(position_amt),
                            "entry_price": pos_data.get('entryPx', '0'),
                            "mark_price": last_price,
                            "return_on_equity": pos_data.get('returnOnEquity', '0'),
                            "unrealized_profit": pos_data.get('unrealizedPnl', '0'),
                            "leverage": pos_data.get('leverage', {}).get('value', '1') if isinstance(pos_data.get('leverage'), dict) else '1',
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
        
        This method delegates to the CandleHandler directly.
        
        Args:
            request: GetDataRequest with symbol (str or list), data_type,
                    optional start_date, end_date, and limit
            
        Returns:
            ActionResult with data organized by symbol in extra field
        """
        if not self.candle_handler:
            raise HyperliquidError("Candle handler not initialized. Call initialize() first.")
        
        if not request.symbol:
            raise HyperliquidError("Symbol must be provided to get data.")
        
        try:
            symbols = request.symbol if isinstance(request.symbol, list) else [request.symbol]
            data_type = DataStreamType(request.data_type)
            
            if data_type != DataStreamType.CANDLE:
                raise HyperliquidError(f"Unsupported data type {data_type.value}. Only candle data is available.")
            
            result_data: Dict[str, Dict[str, List[Dict]]] = {}
            total_rows = 0
            
            for symbol in symbols:
                logger.info(f"| 🔍 Getting {data_type.value} data for {symbol}...")
                data = await self.candle_handler.get_data(
                    symbol=symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    limit=request.limit
                )
                
                result_data[symbol] = data
                total_rows += len(data.get("candles", [])) + len(data.get("indicators", []))
            
            symbol_str = ", ".join(symbols) if len(symbols) <= 10 else f"{len(symbols)} symbols"
            if request.start_date and request.end_date:
                message = f"Retrieved {total_rows} records ({data_type.value}) for {symbol_str} from {request.start_date} to {request.end_date}."
            else:
                message = f"Retrieved {total_rows} latest records ({data_type.value}) for {symbol_str}."
            
            return ActionResult(
                success=True,
                message=message,
                extra={
                    "data": result_data,
                    "symbols": symbols,
                    "data_type": data_type.value,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "row_count": total_rows
                }
            )
        except Exception as e:
            raise HyperliquidError(f"Failed to get data: {e}")
    
    async def _sleep_until_start(self) -> None:
        """Wait until the next minute boundary for minute-level trading.
        
        This ensures we get complete minute kline data by waiting until the start
        of the next minute before fetching data.
        """
        current_ts = time.time()
        seconds_since_minute = current_ts % 60
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_ts))
        
        if seconds_since_minute <= 1e-3:
            logger.debug(f"| ✅ Already at minute boundary (current: {timestamp_str})")
            return
        
        wait_time = 60 - seconds_since_minute
        logger.debug(f"| ⏳ Waiting {wait_time:.2f} seconds until next minute boundary (current: {timestamp_str})...")
        await asyncio.sleep(wait_time)
        
        final_ts = time.time()
        final_timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(final_ts))
        logger.debug(f"| ✅ Reached minute boundary (current: {final_timestamp_str})")
    
    async def _ingest_latest_candles(self) -> None:
        """Fetch the latest closed 1m candle for all tracked symbols and store it."""
        async with self._candle_stream_lock:
            symbols_snapshot = list(self._candle_stream_symbols)
        
        if not symbols_snapshot:
            logger.debug("| ⚠️  Candle polling has no symbols to process.")
            return
        
        if not self.candle_handler:
            logger.warning("| ⚠️  Candle handler missing while polling; skipping this cycle.")
            return
        
        client = self._get_client(self.default_account.name)
        now_ms = int(time.time() * 1000)
        start_time = now_ms - 60 * 1000
        end_time = now_ms
        
        for symbol in symbols_snapshot:
            try:
                symbol_data = await client.get_symbol_data(symbol, start_time=start_time, end_time=end_time)
            except Exception as e:
                logger.warning(f"| ⚠️  Failed to fetch candles for {symbol}: {e}")
                continue
            
            if not symbol_data:
                logger.debug(f"| ⚠️  No candle data returned for {symbol} in the last minute.")
                continue
            
            latest_candle = symbol_data[-1] if len(symbol_data) == 1 else symbol_data[-2]
            
            try:
                result = await self.candle_handler.stream_insert(latest_candle, symbol)
            except Exception as insert_error:
                logger.error(f"| ❌ Failed to insert candle for {symbol}: {insert_error}", exc_info=True)
                continue
            
            success = result.get("success") if isinstance(result, dict) else getattr(result, "success", False)
            if success:
                logger.info(f"| ✅ Inserted candle for {symbol} at timestamp {latest_candle.get('t')}")
            else:
                message = result.get("message") if isinstance(result, dict) else getattr(result, "message", "")
                logger.warning(f"| ⚠️  Candle insert reported failure for {symbol}: {message}")
    
    async def _run_candle_stream(self) -> None:
        """Background task that aligns to minute boundaries and ingests candles."""
        logger.info("| 🔄 Candle polling task started.")
        try:
            await self._sleep_until_start()
            while self._candle_stream_running:
                await self._ingest_latest_candles()
                await self._sleep_until_start()
        except asyncio.CancelledError:
            logger.info("| ⏹️  Candle polling task cancelled.")
        except Exception as e:
            logger.error(f"| ❌ Candle polling encountered an error: {e}", exc_info=True)
        finally:
            self._candle_stream_running = False
            async with self._candle_stream_lock:
                self._candle_stream_task = None
            logger.info("| ✅ Candle polling task stopped.")
    
    async def start_data_stream(
        self,
        symbols: List[str],
        data_types: Optional[List[DataStreamType]] = None
    ) -> None:
        """Start the coroutine-based candle polling loop for given symbols."""
        if not self.candle_handler:
            raise HyperliquidError("Candle handler not initialized. Call initialize() first.")
        
        if not symbols:
            raise HyperliquidError("At least one symbol is required to start the data stream.")
        
        normalized_symbols = []
        for symbol in symbols:
            if symbol:
                normalized_symbols.append(symbol.upper())
        # Remove duplicates while preserving order
        normalized_symbols = list(dict.fromkeys(normalized_symbols))
        
        if not normalized_symbols:
            raise HyperliquidError("No valid symbols provided to start the data stream.")
        
        if data_types:
            unsupported = [dt for dt in data_types if dt != DataStreamType.CANDLE]
            if unsupported:
                raise HyperliquidError(f"Unsupported data types requested: {[dt.value for dt in unsupported]}. Only candle data is available.")
        
        async with self._candle_stream_lock:
            self._candle_stream_symbols = normalized_symbols
            if self._candle_stream_task and not self._candle_stream_task.done():
                logger.info(f"| 🔁 Candle polling already running. Updated symbols: {normalized_symbols}")
                return
            
            self._candle_stream_running = True
            self._candle_stream_task = asyncio.create_task(self._run_candle_stream())
            logger.info(f"| 🚀 Candle polling scheduled for symbols: {normalized_symbols}")
    
    async def stop_data_stream(self) -> None:
        """Stop the data stream.
        
        """
        async with self._candle_stream_lock:
            task = self._candle_stream_task
            if not task:
                logger.warning("| ⚠️  Candle polling task is not running.")
                return
            
            self._candle_stream_running = False
            task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            logger.debug("| ⏹️  Candle polling task cancelled.")
        finally:
            async with self._candle_stream_lock:
                if self._candle_stream_task is task:
                    self._candle_stream_task = None
                    self._candle_stream_symbols = []
    
    # Order methods
    async def create_order(self, request: CreateOrderRequest) -> ActionResult:
        """Create an order (perpetual futures order) with optional stop loss and take profit.
        
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
            
            # Create order via client
            order_result = await client.create_order(
                symbol=request.symbol,
                side=side,
                order_type=request.order_type.value,
                size=request.qty,
                price=request.price,
                leverage=request.leverage,
                stop_loss_price=request.stop_loss_price,
                take_profit_price=request.take_profit_price
            )
            
            # Parse main order result
            main_order = order_result.get('main_order', {})
            if main_order.get('status') != 'ok':
                error_msg = main_order.get('error', 'Order failed')
                raise OrderError(f"Order failed: {error_msg}")
            
            # Extract order ID and status from response
            response = main_order.get('response', {})
            data = response.get('data', {})
            statuses = data.get('statuses', [])
            
            order_id = 'N/A'
            order_status = "submitted"
            if statuses:
                status = statuses[0]
                if 'filled' in status:
                    order_id = str(status['filled'].get('oid', 'N/A'))
                    order_status = "filled"
                elif 'resting' in status:
                    order_id = str(status['resting'].get('oid', 'N/A'))
                    order_status = "submitted"
                elif 'error' in status:
                    raise OrderError(f"Order failed: {status.get('error', 'Unknown error')}")
            
            # Build order info
            order_info = {
                "order_id": order_id,
                "order_status": order_status,
                "symbol": request.symbol,
                "side": request.side,
                "order_type": request.order_type.value,
                "quantity": request.qty,
                "price": request.price,
                "main_order": main_order,
            }
            
            # Add TP/SL info if present
            if order_result.get('stop_loss_order'):
                order_info["stop_loss_order"] = order_result['stop_loss_order']
            if order_result.get('take_profit_order'):
                order_info["take_profit_order"] = order_result['take_profit_order']
            if order_result.get('stop_loss_error'):
                order_info["stop_loss_error"] = order_result['stop_loss_error']
            if order_result.get('take_profit_error'):
                order_info["take_profit_error"] = order_result['take_profit_error']
            
            message = f"Order {order_id} {order_status} for {request.symbol} ({request.side} {request.qty})"
            
            return ActionResult(
                success=True,
                message=message,
                extra={"order_info": order_info}
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
            orders_list = await client.get_orders()
            open_orders = [o for o in orders_list if o.get("coin") == request.symbol] if request.symbol else orders_list
            
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
            all_orders = await client.get_orders()
            # Find order by ID and symbol
            order = None
            for o in all_orders:
                if str(o.get('oid')) == str(request.order_id) and o.get('coin') == request.symbol:
                    order = o
                    break
            
            if not order:
                raise NotFoundError(f"Order {request.order_id} not found for symbol {request.symbol}")
            
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
            # Get symbol info
            symbol_info = await client.get_symbol_info(request.symbol)
            # Convert order_id to int if it's a string
            order_id_int = int(request.order_id) if isinstance(request.order_id, str) else request.order_id
            result = await client.cancel_order(symbol_info=symbol_info, order_id=order_id_int)
            
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
            result = await client.cancel_all_orders(symbol=request.symbol)
            
            return ActionResult(
                success=True,
                message=f"All orders canceled successfully.",
                extra={"account_name": request.account_name, "result": result}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            raise HyperliquidError(f"Failed to cancel all orders: {e}")
    
    async def close_order(self, request: CloseOrderRequest) -> ActionResult:
        """Close a position (reduce-only order).
        
        Args:
            request: CloseOrderRequest with account_name, symbol, side, size, order_type, optional price
            
        Returns:
            ActionResult with close order information
        """
        try:
            # Validate symbol
            if request.symbol not in self.symbols:
                raise InvalidSymbolError(f"Symbol {request.symbol} not found or not tradable")
            
            client = self._get_client(request.account_name)
            
            # Convert side to Hyperliquid format
            side = "B" if request.side.lower() == "buy" else "A"
            
            # Close position
            close_result = await client.close_order(
                symbol=request.symbol,
                side=side,
                size=request.size if request.size else None,
                order_type=request.order_type.value,
                price=request.price
            )
            
            # Parse close order result
            close_order_data = close_result.get('close_order', {})
            order_id = 'N/A'
            order_status = "submitted"
            error_message = None
            
            if isinstance(close_order_data, dict):
                if close_order_data.get('status') == 'ok':
                    response = close_order_data.get('response', {})
                    if response.get('type') == 'order':
                        data = response.get('data', {})
                        statuses = data.get('statuses', [])
                        if statuses:
                            status = statuses[0]
                            if 'resting' in status:
                                order_id = str(status['resting'].get('oid', 'N/A'))
                                order_status = "submitted"
                            elif 'filled' in status:
                                order_id = str(status['filled'].get('oid', 'N/A'))
                                order_status = "filled"
                            elif 'error' in status:
                                error_message = status.get('error', 'Unknown error')
                                order_status = "failed"
                elif 'error' in close_order_data:
                    error_message = close_order_data.get('error', 'Unknown error')
                    order_status = "failed"
            
            if error_message:
                raise OrderError(f"Close order failed: {error_message}")
            
            # Format close order information
            close_order_info = {
                "order_id": order_id,
                "symbol": request.symbol,
                "side": request.side,
                "type": request.order_type.value,
                "status": order_status,
                "quantity": str(request.size),
                "price": str(request.price) if request.price else None,
                "trade_type": "perpetual",
            }
            
            return ActionResult(
                success=True,
                message=f"Close order {order_id} submitted successfully for {request.symbol} ({request.side} {request.size}).",
                extra={"close_order": close_order_info}
            )
            
        except Exception as e:
            if "401" in str(e) or "Invalid" in str(e):
                raise AuthenticationError(f"Authentication failed: {e}")
            if 'insufficient' in str(e).lower() or 'balance' in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            raise OrderError(f"Failed to close order: {e}")


class OfflineHyperliquidService:
    """Offline Hyperliquid trading service using cached database data.
    
    This service simulates trading using historical data from the database:
    - No real API connections
    - All account, position, and order data is maintained locally
    - Order execution is simulated based on historical prices from database
    - Supports perpetual futures trading simulation
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        accounts: List[Dict[str, Union[str, float]]],
        live: bool = False,
        symbol: Optional[Union[str, List[str]]] = None,
        data_type: Optional[Union[str, List[str]]] = None,
        initial_balance: float = 500.0,
    ):
        """Initialize offline Hyperliquid trading service.
        
        Args:
            base_dir: Base directory for Hyperliquid operations (should contain database)
            accounts: List of account dictionaries, each containing name and optional initial_balance
            symbol: Optional symbol(s) to work with
            initial_balance: Default initial balance for accounts (if not specified per account)
            
            accounts = [
                {
                    "name": "Account 1",
                    "initial_balance": 10000.0,  # Optional, defaults to initial_balance parameter
                },
                {
                    "name": "Account 2",
                    "initial_balance": 5000.0,
                }
            ]
        """
        self.base_dir = Path(assemble_project_path(base_dir))
        self.live = live
        self.data_type = data_type
        
        # Initialize accounts with balances
        self.accounts: Dict[str, Dict[str, Union[str, float]]] = {}
        self.account_balances: Dict[str, float] = {}
        self.default_account_name = accounts[0]["name"] if accounts else "default"
        
        for account in accounts:
            account_name = account["name"]
            self.accounts[account_name] = account
            self.account_balances[account_name] = account.get("initial_balance", initial_balance)
        
        self.symbol = symbol
        self.symbols: Dict[str, Dict] = {}
        
        # Initialize database
        self.database_base_dir = self.base_dir / "database"
        self.database_base_dir.mkdir(parents=True, exist_ok=True)
        self.database_service: Optional[DatabaseService] = None
        
        # Initialize data handlers
        self.candle_handler: Optional[CandleHandler] = None
        self.indicators_name: List[str] = []
        
        # Local state: positions and orders
        # positions: Dict[account_name][symbol] -> position info
        self.positions: Dict[str, Dict[str, Dict]] = {account_name: {} for account_name in self.accounts.keys()}
        
        # orders: Dict[account_name] -> List[order_info]
        self.orders: Dict[str, List[Dict]] = {account_name: [] for account_name in self.accounts.keys()}
        
        # Order ID counter
        self._order_id_counter: int = 1
        
        # Time index management for backtest simulation
        # current_index: Dict[symbol] -> current index in historical data
        self._current_index: Dict[str, int] = {}
        
        # total_data_count: Dict[symbol] -> total number of data points available
        self._total_data_count: Dict[str, int] = {}
        
        # timestamps: Dict[symbol] -> List of timestamps ordered by index
        self._timestamps: Dict[str, List[int]] = {}
        
        # Max historical data points (same as online service: 120 minutes = 2 hours)
        # Initial index should start from this point to match online service behavior
        self._max_historical_data_points: int = 120
        
        # Trading fee rate: 0.045% = 0.00045
        self._trading_fee_rate: float = 0.00045
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize the offline Hyperliquid trading service."""
        try:
            self.base_dir = Path(assemble_project_path(self.base_dir))
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Initialize database
            await self._initialize_database()
            
            # Step 2: Load available symbols from database
            await self._load_symbols()
            
            # Step 3: Initialize data handlers
            await self._initialize_data_handlers()
            
            # Step 4: Initialize time indices for symbols
            await self._initialize_time_indices()
            
            logger.info(f"| ✅ Offline Hyperliquid service initialized with {len(self.accounts)} account(s)")
            
        except Exception as e:
            raise HyperliquidError(f"Failed to initialize offline Hyperliquid service: {e}")
    
    async def _initialize_database(self) -> None:
        """Initialize database."""
        self.database_service = DatabaseService(self.database_base_dir)
        await self.database_service.connect()
    
    async def _initialize_data_handlers(self) -> None:
        """Initialize data handlers."""
        self.candle_handler = CandleHandler(self.database_service)
        self.indicators_name = await self.candle_handler.get_indicators_name()
    
    async def _initialize_time_indices(self) -> None:
        """Initialize time indices for all symbols from database.
        
        This loads all timestamps from the database and initializes the current index
        to start from _max_historical_data_points (120 minutes = 2 hours) to match
        online service behavior, which pre-caches 2 hours of historical data.
        """
        if not self.candle_handler:
            return
        
        symbols_to_init = []
        if self.symbol:
            symbols_to_init = self.symbol if isinstance(self.symbol, list) else [self.symbol]
        else:
            symbols_to_init = list(self.symbols.keys())
        
        for symbol in symbols_to_init:
            try:
                # Get all timestamps from database, ordered by timestamp
                symbol_upper = symbol.upper()
                base_name = self.candle_handler._sanitize_table_name(symbol_upper)
                candle_table_name = f"{base_name}_candle"
                
                query = f"SELECT timestamp FROM {candle_table_name} WHERE symbol = ? ORDER BY timestamp ASC"
                result = await self.database_service.execute_query(
                    QueryRequest(query=query, parameters=(symbol_upper,))
                )
                
                if result.success and result.extra.get("data"):
                    timestamps = [row["timestamp"] for row in result.extra["data"]]
                    self._timestamps[symbol] = timestamps
                    total_count = len(timestamps)
                    self._total_data_count[symbol] = total_count
                    
                    # Initialize index to start from _max_historical_data_points (2 hours)
                    # This matches online service behavior which pre-caches 2 hours of data
                    initial_index = min(self._max_historical_data_points, total_count)
                    self._current_index[symbol] = initial_index
                    
                    logger.info(f"| 📅 Initialized time index for {symbol}: {total_count} data points, starting from index {initial_index} (2 hours)")
                else:
                    logger.warning(f"| ⚠️  No data found for {symbol} in database")
                    self._timestamps[symbol] = []
                    self._current_index[symbol] = 0
                    self._total_data_count[symbol] = 0
            except Exception as e:
                logger.warning(f"| ⚠️  Failed to initialize time index for {symbol}: {e}")
                self._timestamps[symbol] = []
                self._current_index[symbol] = 0
                self._total_data_count[symbol] = 0
    
    async def _load_symbols(self) -> None:
        """Load available trading symbols from database."""
        try:
            # Query database to find all symbol tables
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'data_%_candle'"
            result = await self.database_service.execute_query(
                QueryRequest(query=query)
            )
            
            if result.success and result.extra.get("data"):
                tables = result.extra["data"]
                for table_row in tables:
                    table_name = table_row.get("name", "")
                    # Extract symbol from table name: data_{SYMBOL}_candle
                    if table_name.startswith("data_") and table_name.endswith("_candle"):
                        symbol = table_name[5:-7]  # Remove "data_" prefix and "_candle" suffix
                        symbol = symbol.replace("_", "/")  # Restore original symbol format if needed
                        
                        self.symbols[symbol] = {
                            'symbol': symbol,
                            'baseAsset': symbol,
                            'quoteAsset': 'USD',
                            'status': 'TRADING',
                            'tradable': True,
                            'type': 'perpetual'
                        }
            
            logger.info(f"| 📊 Loaded {len(self.symbols)} symbols from database")
            
        except Exception as e:
            logger.warning(f"| ⚠️  Failed to load symbols: {e}")
            self.symbols = {}
    
    async def cleanup(self) -> None:
        """Cleanup the offline Hyperliquid service."""
        if hasattr(self, 'database_service') and self.database_service:
            await self.database_service.disconnect()
        
        self.symbols = {}
        self.candle_handler = None
        self.indicators_name = []
        
        # Cleanup time indices
        self._current_index = {}
        self._total_data_count = {}
        self._timestamps = {}
    
    async def _get_latest_price(self, symbol: str) -> float:
        """Get current price for a symbol based on current time index.
        
        In offline backtest mode, this returns the price at the current index,
        not the latest price in the database.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Current close price at current index
            
        Raises:
            HyperliquidError: If price cannot be retrieved
        """
        if not self.candle_handler:
            raise HyperliquidError("Candle handler not initialized")
        
        # Initialize index if not exists
        if symbol not in self._current_index:
            await self._initialize_symbol_index(symbol)
        
        current_idx = self._current_index.get(symbol, 0)
        total_count = self._total_data_count.get(symbol, 0)
        
        if current_idx >= total_count:
            raise HyperliquidError(f"No more price data available for {symbol} (index {current_idx}/{total_count})")
        
        # Get timestamp at current index
        timestamps = self._timestamps.get(symbol, [])
        if not timestamps or current_idx >= len(timestamps):
            raise HyperliquidError(f"No timestamp data available for {symbol} at index {current_idx}")
        
        target_timestamp = timestamps[current_idx]
        
        # Query database for candle at this timestamp
        symbol_upper = symbol.upper()
        base_name = self.candle_handler._sanitize_table_name(symbol_upper)
        candle_table_name = f"{base_name}_candle"
        
        query = f"SELECT close FROM {candle_table_name} WHERE symbol = ? AND timestamp = ?"
        result = await self.database_service.execute_query(
            QueryRequest(query=query, parameters=(symbol_upper, target_timestamp))
        )
        
        if not result.success or not result.extra.get("data"):
            raise HyperliquidError(f"No price data available for {symbol} at timestamp {target_timestamp}")
        
        candle_data = result.extra["data"]
        if not candle_data:
            raise HyperliquidError(f"No price data available for {symbol} at timestamp {target_timestamp}")
        
        close_price = candle_data[0].get("close")
        
        if close_price is None:
            raise HyperliquidError(f"Invalid price data for {symbol}")
        
        return float(close_price)
    
    async def _update_position_pnl(self, account_name: str, symbol: str) -> None:
        """Update unrealized PnL for a position based on current market price.
        
        Args:
            account_name: Account name
            symbol: Symbol name
        """
        if account_name not in self.positions or symbol not in self.positions[account_name]:
            return
        
        try:
            current_price = await self._get_latest_price(symbol)
            position = self.positions[account_name][symbol]
            
            entry_price = float(position.get("entry_price", 0))
            position_amt = float(position.get("position_amt", 0))
            
            if position_amt != 0:
                # Calculate unrealized PnL
                if position_amt > 0:  # Long position
                    unrealized_pnl = (current_price - entry_price) * position_amt
                else:  # Short position
                    unrealized_pnl = (entry_price - current_price) * abs(position_amt)
                
                position["unrealized_profit"] = str(unrealized_pnl)
                position["mark_price"] = str(current_price)
                
                # Calculate return on equity
                leverage = float(position.get("leverage", 1))
                margin_used = abs(position_amt) * entry_price / leverage
                if margin_used > 0:
                    roe = (unrealized_pnl / margin_used) * 100
                    position["return_on_equity"] = str(roe)
        except Exception as e:
            logger.warning(f"| ⚠️  Failed to update PnL for {account_name}/{symbol}: {e}")
    
    # Get Exchange Info
    async def get_exchange_info(self, request: GetExchangeInfoRequest) -> ActionResult:
        """Get exchange information including available symbols.
        
        Args:
            request: GetExchangeInfoRequest
            
        Returns:
            ActionResult with exchange information
        """
        try:
            exchange_info = {
                "universe": [
                    {"name": symbol, **info} for symbol, info in self.symbols.items()
                ]
            }
            
            return ActionResult(
                success=True,
                message=f"Exchange information retrieved successfully.",
                extra={"exchange_info": exchange_info}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to get exchange info: {e}")
    
    # Account methods
    async def get_account(self, request: GetAccountRequest) -> ActionResult:
        """Get account information.
        
        Args:
            request: GetAccountRequest with account_name
        """
        try:
            if request.account_name not in self.account_balances:
                raise HyperliquidError(f"Account {request.account_name} not found")
            
            balance = self.account_balances[request.account_name]
            
            # Calculate total margin used and positions value
            total_margin_used = 0.0
            total_unrealized_pnl = 0.0
            
            if request.account_name in self.positions:
                for symbol, position in self.positions[request.account_name].items():
                    await self._update_position_pnl(request.account_name, symbol)
                    position_amt = float(position.get("position_amt", 0))
                    entry_price = float(position.get("entry_price", 0))
                    leverage = float(position.get("leverage", 1))
                    
                    if position_amt != 0:
                        margin_used = abs(position_amt) * entry_price / leverage
                        total_margin_used += margin_used
                        total_unrealized_pnl += float(position.get("unrealized_profit", 0))
            
            account_equity = balance + total_unrealized_pnl
            withdrawable = account_equity - total_margin_used
            
            # Format account data to match online service format exactly
            # Online service uses the raw API response structure
            account_data = {
                "margin_summary": {
                    "accountValue": str(account_equity),
                    "totalMarginUsed": str(total_margin_used),
                    "totalNtlPos": str(total_unrealized_pnl),
                    "totalRawUsd": str(balance),  # Add totalRawUsd to match online format
                },
                "cross_margin_summary": {
                    "accountValue": str(account_equity),
                    "totalNtlPos": str(total_unrealized_pnl),
                    "totalRawUsd": str(balance),
                    "totalMarginUsed": str(total_margin_used),
                },
                "cross_maintenance_margin_used": str(total_margin_used),
                "withdrawable": str(max(0, withdrawable)),
                "asset_positions": [
                    {
                        "type": "oneWay",  # Match online format
                        "position": {
                            "coin": symbol,
                            "szi": str(position.get("position_amt", "0")),
                            "entryPx": position.get("entry_price", "0"),
                            "unrealizedPnl": position.get("unrealized_profit", "0"),
                            "returnOnEquity": position.get("return_on_equity", "0"),
                            "leverage": {"value": position.get("leverage", "1")},
                        }
                    }
                    for symbol, position in self.positions.get(request.account_name, {}).items()
                    if float(position.get("position_amt", 0)) != 0
                ],
                "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                "trade_type": "perpetual",
            }
            
            return ActionResult(
                success=True,
                message=f"Account information retrieved successfully.",
                extra={"account": account_data}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to get account: {e}")
    
    # Get Symbol Info
    async def get_symbol_info(self, request: GetSymbolInfoRequest) -> ActionResult:
        """Get symbol information for a specific symbol.
        
        Args:
            request: GetSymbolInfoRequest with symbol name
            
        Returns:
            ActionResult with symbol information
        """
        try:
            if request.symbol not in self.symbols:
                raise InvalidSymbolError(f"Symbol {request.symbol} not found")
            
            # Get latest price
            latest_price = await self._get_latest_price(request.symbol)
            
            symbol_info = {
                **self.symbols[request.symbol],
                "latest_price": latest_price,
            }
            
            return ActionResult(
                success=True,
                message=f"Symbol information retrieved successfully for {request.symbol}.",
                extra={"symbol_info": symbol_info}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to get symbol info: {e}")
    
    async def get_positions(self, request: GetPositionsRequest) -> ActionResult:
        """Get all positions.
        
        Args:
            request: GetPositionsRequest with account_name
        """
        try:
            if request.account_name not in self.positions:
                return ActionResult(
                    success=True,
                    message="Retrieved 0 positions.",
                    extra={"positions": []}
                )
            
            all_positions = []
            for symbol, position in self.positions[request.account_name].items():
                await self._update_position_pnl(request.account_name, symbol)
                
                position_amt = float(position.get("position_amt", 0))
                if position_amt != 0:
                    all_positions.append({
                        "symbol": symbol,
                        "position_amt": str(position_amt),
                        "entry_price": position.get("entry_price", "0"),
                        "mark_price": position.get("mark_price", "0"),
                        "return_on_equity": position.get("return_on_equity", "0"),
                        "unrealized_profit": position.get("unrealized_profit", "0"),
                        "leverage": position.get("leverage", "1"),
                        "trade_type": "perpetual",
                    })
            
            return ActionResult(
                success=True,
                message=f"Retrieved {len(all_positions)} positions.",
                extra={"positions": all_positions}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to get positions: {e}")
    
    async def get_data(self, request: GetDataRequest) -> ActionResult:
        """Get historical data from database based on current time index.
        
        In offline backtest mode, this method retrieves data sequentially from the database:
        - Current index points to the current time point (e.g., index 1 = 10:02:00)
        - Retrieves historical data from (current_index - limit + 1) to current_index (inclusive)
        - After retrieval, advances index by 1 for next call
        
        Example:
            Timestamps: [10:01:00 (idx 0), 10:02:00 (idx 1), 10:03:00 (idx 2)]
            - First call (index=0, limit=30): Get data from max(0, 0-30+1)=0 to 0+1=1, then advance index to 1
            - Second call (index=1, limit=30): Get data from max(0, 1-30+1)=0 to 1+1=2, then advance index to 2
        
        Args:
            request: GetDataRequest with symbol (str or list), data_type,
                    optional limit (number of historical records to retrieve, default: all available)
                    Note: start_date and end_date are ignored in offline mode
            
        Returns:
            ActionResult with data organized by symbol in extra field
        """
        if not self.candle_handler:
            raise HyperliquidError("Candle handler not initialized. Call initialize() first.")
        
        if not request.symbol:
            raise HyperliquidError("Symbol must be provided to get data.")
        
        try:
            symbols = request.symbol if isinstance(request.symbol, list) else [request.symbol]
            data_type = DataStreamType(request.data_type)
            
            if data_type != DataStreamType.CANDLE:
                raise HyperliquidError(f"Unsupported data type {data_type.value}. Only candle data is available.")
            
            result_data: Dict[str, Dict[str, List[Dict]]] = {}
            total_rows = 0
            
            for symbol in symbols:
                # Initialize index if not exists
                if symbol not in self._current_index:
                    await self._initialize_symbol_index(symbol)
                
                current_idx = self._current_index.get(symbol, 0)
                total_count = self._total_data_count.get(symbol, 0)
                
                if current_idx >= total_count:
                    logger.warning(f"| ⚠️  No more data available for {symbol} (index {current_idx}/{total_count})")
                    result_data[symbol] = {"candles": [], "indicators": []}
                    continue
                
                # Determine how many historical records to retrieve
                # If limit is specified, get data from (current_idx - limit + 1) to (current_idx + 1)
                # If limit is not specified, get all data from 0 to (current_idx + 1)
                if request.limit:
                    # Get historical data: from max(0, current_idx - limit + 1) to current_idx + 1
                    start_idx = max(0, current_idx - request.limit + 1)
                    end_idx = min(current_idx + 1, total_count)
                else:
                    # Get all data from beginning to current_idx + 1
                    start_idx = 0
                    end_idx = min(current_idx + 1, total_count)
                
                # Get timestamps for the range
                timestamps = self._timestamps.get(symbol, [])
                if not timestamps or start_idx >= len(timestamps):
                    result_data[symbol] = {"candles": [], "indicators": []}
                    continue
                
                target_timestamps = timestamps[start_idx:end_idx]
                
                if not target_timestamps:
                    result_data[symbol] = {"candles": [], "indicators": []}
                    continue
                
                logger.info(f"| 🔍 Getting {data_type.value} data for {symbol} from index {start_idx} to {end_idx-1} (current_idx={current_idx}, {len(target_timestamps)} records)...")
                
                # Query database for candles with these timestamps
                symbol_upper = symbol.upper()
                base_name = self.candle_handler._sanitize_table_name(symbol_upper)
                candle_table_name = f"{base_name}_candle"
                indicators_table_name = f"{base_name}_indicators"
                
                # Build query for candles
                placeholders = ','.join(['?' for _ in target_timestamps])
                candle_query = f"SELECT * FROM {candle_table_name} WHERE symbol = ? AND timestamp IN ({placeholders}) ORDER BY timestamp ASC"
                candle_result = await self.database_service.execute_query(
                    QueryRequest(query=candle_query, parameters=(symbol_upper, *target_timestamps))
                )
                
                candles = []
                if candle_result.success and candle_result.extra.get("data"):
                    candles = candle_result.extra["data"]
                
                # Query database for indicators with these timestamps
                # Use the same approach as candles: use IN clause with exact timestamps
                indicators = []
                try:
                    if target_timestamps:
                        # Use IN clause with exact timestamps (same as candles query)
                        indicator_query = f"SELECT * FROM {indicators_table_name} WHERE symbol = ? AND timestamp IN ({placeholders}) ORDER BY timestamp ASC"
                        indicator_result = await self.database_service.execute_query(
                            QueryRequest(query=indicator_query, parameters=(symbol_upper, *target_timestamps))
                        )
                        
                        if indicator_result.success and indicator_result.extra.get("data"):
                            indicators = indicator_result.extra["data"]
                            logger.info(f"| ✅ Retrieved {len(indicators)} indicators for {symbol} (out of {len(target_timestamps)} requested timestamps)")
                        else:
                            logger.warning(f"| ⚠️  No indicators data found for {symbol} at specified timestamps. Query success: {indicator_result.success}")
                            if hasattr(indicator_result, 'message'):
                                logger.warning(f"| ⚠️  Query message: {indicator_result.message}")
                    else:
                        logger.debug(f"| ⚠️  No timestamps to query indicators for {symbol}")
                except Exception as e:
                    logger.warning(f"| ⚠️  Failed to query indicators for {symbol}: {e}", exc_info=True)
                
                result_data[symbol] = {
                    "candles": candles,
                    "indicators": indicators
                }
                total_rows += len(candles) + len(indicators)
                
                # Advance index by 1 for next call
                self._current_index[symbol] = current_idx + 1
                logger.debug(f"| 📍 Updated index for {symbol}: {current_idx} -> {current_idx + 1}")
            
            symbol_str = ", ".join(symbols) if len(symbols) <= 10 else f"{len(symbols)} symbols"
            message = f"Retrieved {total_rows} records ({data_type.value}) for {symbol_str} from current index."
            
            return ActionResult(
                success=True,
                message=message,
                extra={
                    "data": result_data,
                    "symbols": symbols,
                    "data_type": data_type.value,
                    "row_count": total_rows
                }
            )
        except Exception as e:
            raise HyperliquidError(f"Failed to get data: {e}")
    
    async def _initialize_symbol_index(self, symbol: str) -> None:
        """Initialize time index for a specific symbol.
        
        Initializes the current index to start from _max_historical_data_points (120 minutes = 2 hours)
        to match online service behavior, which pre-caches 2 hours of historical data.
        
        Args:
            symbol: Symbol name
        """
        try:
            symbol_upper = symbol.upper()
            base_name = self.candle_handler._sanitize_table_name(symbol_upper)
            candle_table_name = f"{base_name}_candle"
            
            query = f"SELECT timestamp FROM {candle_table_name} WHERE symbol = ? ORDER BY timestamp ASC"
            result = await self.database_service.execute_query(
                QueryRequest(query=query, parameters=(symbol_upper,))
            )
            
            if result.success and result.extra.get("data"):
                timestamps = [row["timestamp"] for row in result.extra["data"]]
                self._timestamps[symbol] = timestamps
                total_count = len(timestamps)
                self._total_data_count[symbol] = total_count
                
                # Initialize index to start from _max_historical_data_points (2 hours)
                # This matches online service behavior which pre-caches 2 hours of data
                initial_index = min(self._max_historical_data_points, total_count)
                self._current_index[symbol] = initial_index
                
                logger.info(f"| 📅 Initialized time index for {symbol}: {total_count} data points, starting from index {initial_index} (2 hours)")
            else:
                logger.warning(f"| ⚠️  No data found for {symbol} in database")
                self._timestamps[symbol] = []
                self._current_index[symbol] = 0
                self._total_data_count[symbol] = 0
        except Exception as e:
            logger.warning(f"| ⚠️  Failed to initialize time index for {symbol}: {e}")
            self._timestamps[symbol] = []
            self._current_index[symbol] = 0
            self._total_data_count[symbol] = 0
    
    async def reset_time_index(self, symbol: Optional[str] = None) -> None:
        """Reset time index to beginning for symbol(s).
        
        Args:
            symbol: Symbol name to reset. If None, resets all symbols.
        """
        if symbol:
            if symbol in self._current_index:
                self._current_index[symbol] = 0
                logger.info(f"| 🔄 Reset time index for {symbol} to 0")
        else:
            for sym in self._current_index.keys():
                self._current_index[sym] = 0
            logger.info(f"| 🔄 Reset time indices for all symbols to 0")
    
    def get_current_index(self, symbol: str) -> int:
        """Get current time index for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Current index (0-based)
        """
        return self._current_index.get(symbol, 0)
    
    def get_total_data_count(self, symbol: str) -> int:
        """Get total number of data points available for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Total number of data points
        """
        return self._total_data_count.get(symbol, 0)
    
    def is_data_available(self, symbol: str) -> bool:
        """Check if more data is available for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            True if more data is available, False otherwise
        """
        current_idx = self._current_index.get(symbol, 0)
        total_count = self._total_data_count.get(symbol, 0)
        return current_idx < total_count
    
    async def _execute_order(self, account_name: str, order_info: Dict) -> Dict:
        """Execute an order based on current market price.
        
        Args:
            account_name: Account name
            order_info: Order information dictionary
            
        Returns:
            Execution result dictionary
        """
        symbol = order_info["symbol"]
        side = order_info["side"]
        qty = float(order_info["quantity"])
        order_type = order_info["order_type"]
        price = float(order_info.get("price", 0)) if order_info.get("price") else None
        
        # Get execution price
        if order_type == OrderType.MARKET.value:
            execution_price = await self._get_latest_price(symbol)
        else:  # LIMIT order
            if price is None:
                raise OrderError("Price must be provided for LIMIT orders")
            execution_price = price
        
        # Calculate cost (for perpetual futures, cost = qty * price)
        cost = qty * execution_price
        
        # Calculate trading fee: fee = qty * execution_price * fee_rate
        trading_fee = cost * self._trading_fee_rate
        
        # Check if account has sufficient balance
        # For perpetual futures, we need margin = cost / leverage
        leverage = float(order_info.get("leverage", 1))
        required_margin = cost / leverage
        
        # Calculate current margin used
        current_margin_used = 0.0
        if account_name in self.positions:
            for pos_symbol, position in self.positions[account_name].items():
                pos_amt = float(position.get("position_amt", 0))
                pos_entry = float(position.get("entry_price", 0))
                pos_leverage = float(position.get("leverage", 1))
                if pos_amt != 0:
                    current_margin_used += abs(pos_amt) * pos_entry / pos_leverage
        
        available_balance = self.account_balances[account_name] - current_margin_used
        
        # Check if account has sufficient balance for margin + trading fee
        total_required = required_margin + trading_fee
        if total_required > available_balance:
            raise InsufficientFundsError(f"Insufficient funds. Required margin: {required_margin}, Trading fee: {trading_fee}, Available: {available_balance}")
        
        # Deduct trading fee from account balance
        self.account_balances[account_name] -= trading_fee
        logger.debug(f"| 💰 Deducted trading fee {trading_fee:.6f} for {account_name} ({qty} {symbol} @ {execution_price})")
        
        # Update position
        if account_name not in self.positions:
            self.positions[account_name] = {}
        
        if symbol not in self.positions[account_name]:
            self.positions[account_name][symbol] = {
                "position_amt": "0",
                "entry_price": "0",
                "mark_price": str(execution_price),
                "unrealized_profit": "0",
                "return_on_equity": "0",
                "leverage": str(int(leverage)),
            }
        
        position = self.positions[account_name][symbol]
        current_position_amt = float(position["position_amt"])
        
        # Update position based on side
        if side.lower() == "buy":
            # Opening long or closing short
            if current_position_amt < 0:
                # Closing short position
                close_qty = min(abs(current_position_amt), qty)
                # Calculate realized PnL
                entry_price = float(position["entry_price"])
                realized_pnl = (entry_price - execution_price) * close_qty
                self.account_balances[account_name] += realized_pnl
                
                # Update position
                new_position_amt = current_position_amt + close_qty
                if abs(new_position_amt) < 1e-8:  # Position closed
                    position["position_amt"] = "0"
                    position["entry_price"] = "0"
                else:
                    position["position_amt"] = str(new_position_amt)
                    # Average entry price for remaining position
                    remaining_qty = abs(new_position_amt)
                    position["entry_price"] = str(entry_price)
            else:
                # Opening or increasing long position
                if current_position_amt == 0:
                    position["entry_price"] = str(execution_price)
                    position["position_amt"] = str(qty)
                else:
                    # Average entry price
                    total_cost = current_position_amt * float(position["entry_price"]) + qty * execution_price
                    total_qty = current_position_amt + qty
                    position["entry_price"] = str(total_cost / total_qty)
                    position["position_amt"] = str(total_qty)
        else:  # sell
            # Opening short or closing long
            if current_position_amt > 0:
                # Closing long position
                close_qty = min(current_position_amt, qty)
                # Calculate realized PnL
                entry_price = float(position["entry_price"])
                realized_pnl = (execution_price - entry_price) * close_qty
                self.account_balances[account_name] += realized_pnl
                
                # Update position
                new_position_amt = current_position_amt - close_qty
                if abs(new_position_amt) < 1e-8:  # Position closed
                    position["position_amt"] = "0"
                    position["entry_price"] = "0"
                else:
                    position["position_amt"] = str(new_position_amt)
                    position["entry_price"] = str(entry_price)
            else:
                # Opening or increasing short position
                if current_position_amt == 0:
                    position["entry_price"] = str(execution_price)
                    position["position_amt"] = str(-qty)
                else:
                    # Average entry price
                    total_cost = abs(current_position_amt) * float(position["entry_price"]) + qty * execution_price
                    total_qty = abs(current_position_amt) + qty
                    position["entry_price"] = str(total_cost / total_qty)
                    position["position_amt"] = str(-total_qty)
        
        # Update mark price
        position["mark_price"] = str(execution_price)
        position["leverage"] = str(int(leverage))
        
        return {
            "execution_price": execution_price,
            "filled_qty": qty,
            "status": "filled",
            "trading_fee": trading_fee
        }
    
    # Order methods
    async def create_order(self, request: CreateOrderRequest) -> ActionResult:
        """Create an order (simulated execution based on database prices).
        
        In offline mode:
        - Main order is executed immediately (filled)
        - Stop loss and take profit orders are created as guard orders (open/pending)
        - Only guard orders are stored in self.orders (main order is not stored as it's filled)
        
        Args:
            request: CreateOrderRequest with account_name, symbol, side, order_type, qty, etc.
            
        Returns:
            ActionResult with order information including main order and guard orders
        """
        try:
            if request.qty is None:
                raise HyperliquidError("'qty' must be provided")
            
            if request.order_type == OrderType.LIMIT and request.price is None:
                raise HyperliquidError("'price' must be provided for LIMIT orders")
            
            # Validate symbol
            if request.symbol not in self.symbols:
                raise InvalidSymbolError(f"Symbol {request.symbol} not found or not tradable")
            
            # Create main order record (will be filled immediately)
            main_order_id = f"MAIN-{self._order_id_counter}"
            self._order_id_counter += 1
            
            main_order_info = {
                "order_id": main_order_id,
                "symbol": request.symbol,
                "side": request.side,
                "order_type": request.order_type.value,
                "quantity": str(request.qty),
                "price": str(request.price) if request.price else None,
                "leverage": str(request.leverage) if request.leverage else "1",
                "status": "pending",
                "timestamp": int(time.time() * 1000),
            }
            
            # Execute main order immediately (simulated)
            try:
                execution_result = await self._execute_order(request.account_name, main_order_info)
                main_order_info["status"] = "filled"
                main_order_info["execution_price"] = str(execution_result["execution_price"])
                main_order_info["filled_qty"] = str(execution_result["filled_qty"])
            except InsufficientFundsError:
                main_order_info["status"] = "rejected"
                raise
            except Exception as e:
                main_order_info["status"] = "failed"
                main_order_info["error"] = str(e)
                raise OrderError(f"Order execution failed: {e}")
            
            # Build response with main order info
            order_info = {
                "order_id": main_order_id,
                "order_status": "filled",
                "symbol": request.symbol,
                "side": request.side,
                "order_type": request.order_type.value,
                "quantity": request.qty,
                "price": request.price,
                "main_order": main_order_info,
            }
            
            # Create guard orders (stop loss and take profit) - these are open/pending orders
            stop_loss_order = None
            take_profit_order = None
            stop_loss_error = None
            take_profit_error = None
            
            # Create stop loss order if specified
            if request.stop_loss_price:
                try:
                    # Determine stop loss side (opposite of main order side)
                    sl_side = "sell" if request.side.lower() == "buy" else "buy"
                    
                    sl_order_id = f"SL-{self._order_id_counter}"
                    self._order_id_counter += 1
                    
                    sl_order = {
                        "order_id": sl_order_id,
                        "symbol": request.symbol,
                        "side": sl_side,
                        "order_type": OrderType.LIMIT.value,  # Stop loss is typically a limit order
                        "quantity": str(request.qty),
                        "price": str(request.stop_loss_price),
                        "leverage": str(request.leverage) if request.leverage else "1",
                        "status": "open",  # Guard order is open/pending
                        "timestamp": int(time.time() * 1000),
                        "guard_type": "stop_loss",
                        "main_order_id": main_order_id,
                    }
                    
                    # Add guard order to orders list (only open orders are stored)
                    if request.account_name not in self.orders:
                        self.orders[request.account_name] = []
                    self.orders[request.account_name].append(sl_order)
                    
                    stop_loss_order = {
                        "order_id": sl_order_id,
                        "status": "open",
                        "symbol": request.symbol,
                        "side": sl_side,
                        "price": str(request.stop_loss_price),
                        "quantity": str(request.qty),
                    }
                except Exception as e:
                    stop_loss_error = str(e)
                    logger.warning(f"| ⚠️  Failed to create stop loss order: {e}")
            
            # Create take profit order if specified
            if request.take_profit_price:
                try:
                    # Determine take profit side (opposite of main order side)
                    tp_side = "sell" if request.side.lower() == "buy" else "buy"
                    
                    tp_order_id = f"TP-{self._order_id_counter}"
                    self._order_id_counter += 1
                    
                    tp_order = {
                        "order_id": tp_order_id,
                        "symbol": request.symbol,
                        "side": tp_side,
                        "order_type": OrderType.LIMIT.value,  # Take profit is typically a limit order
                        "quantity": str(request.qty),
                        "price": str(request.take_profit_price),
                        "leverage": str(request.leverage) if request.leverage else "1",
                        "status": "open",  # Guard order is open/pending
                        "timestamp": int(time.time() * 1000),
                        "guard_type": "take_profit",
                        "main_order_id": main_order_id,
                    }
                    
                    # Add guard order to orders list (only open orders are stored)
                    if request.account_name not in self.orders:
                        self.orders[request.account_name] = []
                    self.orders[request.account_name].append(tp_order)
                    
                    take_profit_order = {
                        "order_id": tp_order_id,
                        "status": "open",
                        "symbol": request.symbol,
                        "side": tp_side,
                        "price": str(request.take_profit_price),
                        "quantity": str(request.qty),
                    }
                except Exception as e:
                    take_profit_error = str(e)
                    logger.warning(f"| ⚠️  Failed to create take profit order: {e}")
            
            # Add guard order info to response
            if stop_loss_order:
                order_info["stop_loss_order"] = stop_loss_order
            if take_profit_order:
                order_info["take_profit_order"] = take_profit_order
            if stop_loss_error:
                order_info["stop_loss_error"] = stop_loss_error
            if take_profit_error:
                order_info["take_profit_error"] = take_profit_error
            
            message = f"Order {main_order_id} filled for {request.symbol} ({request.side} {request.qty})"
            
            return ActionResult(
                success=True,
                message=message,
                extra={"order_info": order_info}
            )
            
        except Exception as e:
            if isinstance(e, (InsufficientFundsError, InvalidSymbolError, OrderError)):
                raise
            raise OrderError(f"Failed to create order: {e}")
    
    async def get_orders(self, request: GetOrdersRequest) -> ActionResult:
        """Get open/pending orders for an account.
        
        In offline mode, this returns only guard orders (stop loss and take profit orders)
        that are still open/pending. Main orders are not returned as they are immediately filled.
        
        Args:
            request: GetOrdersRequest with account_name and optional symbol
        """
        try:
            if request.account_name not in self.orders:
                return ActionResult(
                    success=True,
                    message="Retrieved 0 orders.",
                    extra={"orders": []}
                )
            
            all_orders = []
            for order in self.orders[request.account_name]:
                # Only return open/pending orders (guard orders)
                # Main orders are filled immediately and not stored in self.orders
                order_status = order.get("status", "").lower()
                if order_status not in ["open", "pending"]:
                    continue
                
                if request.symbol and order.get("symbol") != request.symbol:
                    continue
                if request.order_id and str(order.get("order_id")) != str(request.order_id):
                    continue
                
                order_info = {
                    "order_id": str(order.get("order_id", "N/A")),
                    "symbol": order.get("symbol", "N/A"),
                    "side": order.get("side", "N/A"),
                    "type": order.get("order_type", "N/A"),
                    "status": "open",  # Guard orders are always open
                    "quantity": str(order.get("quantity", "0")),
                    "price": order.get("price"),
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
            raise HyperliquidError(f"Failed to get orders: {e}")
    
    async def get_order(self, request: GetOrderRequest) -> ActionResult:
        """Get a specific guard order by ID.
        
        In offline mode, this can only retrieve guard orders (stop loss/take profit)
        as main orders are filled immediately and not stored.
        
        Args:
            request: GetOrderRequest with account_name, order_id, symbol
        """
        try:
            if request.account_name not in self.orders:
                raise NotFoundError(f"Order {request.order_id} not found")
            
            order = None
            for o in self.orders[request.account_name]:
                if str(o.get("order_id")) == str(request.order_id) and o.get("symbol") == request.symbol:
                    order = o
                    break
            
            if not order:
                raise NotFoundError(f"Order {request.order_id} not found for symbol {request.symbol}")
            
            order_info = {
                "order_id": str(order.get("order_id", "N/A")),
                "symbol": order.get("symbol", "N/A"),
                "side": order.get("side", "N/A"),
                "type": order.get("order_type", "N/A"),
                "status": "open",  # Guard orders are always open
                "quantity": str(order.get("quantity", "0")),
                "price": order.get("price"),
                "trade_type": "perpetual",
            }
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} retrieved successfully.",
                extra={"order": order_info}
            )
            
        except Exception as e:
            if isinstance(e, NotFoundError):
                raise
            raise HyperliquidError(f"Failed to get order: {e}")
    
    async def cancel_order(self, request: CancelOrderRequest) -> ActionResult:
        """Cancel a guard order (stop loss or take profit order).
        
        In offline mode, this cancels guard orders that are still open/pending.
        Main orders cannot be cancelled as they are filled immediately.
        
        Args:
            request: CancelOrderRequest with account_name, order_id, symbol
        """
        try:
            if request.account_name not in self.orders:
                raise NotFoundError(f"Order {request.order_id} not found")
            
            order = None
            for o in self.orders[request.account_name]:
                if str(o.get("order_id")) == str(request.order_id) and o.get("symbol") == request.symbol:
                    order = o
                    break
            
            if not order:
                raise NotFoundError(f"Order {request.order_id} not found for symbol {request.symbol}")
            
            order_status = order.get("status", "").lower()
            if order_status not in ["open", "pending"]:
                raise OrderError(f"Cannot cancel order {request.order_id}: order is already {order.get('status')}")
            
            order["status"] = "cancelled"
            
            return ActionResult(
                success=True,
                message=f"Order {request.order_id} canceled successfully.",
                extra={"order_id": request.order_id}
            )
            
        except Exception as e:
            if isinstance(e, (NotFoundError, OrderError)):
                raise
            raise HyperliquidError(f"Failed to cancel order: {e}")
    
    async def cancel_all_orders(self, request: CancelAllOrdersRequest) -> ActionResult:
        """Cancel all open/pending guard orders for an account.
        
        In offline mode, this cancels all guard orders (stop loss/take profit) that are still open.
        Main orders cannot be cancelled as they are filled immediately.
        
        Args:
            request: CancelAllOrdersRequest with account_name, optional symbol
        """
        try:
            if request.account_name not in self.orders:
                return ActionResult(
                    success=True,
                    message="No orders to cancel.",
                    extra={"account_name": request.account_name, "cancelled_count": 0}
                )
            
            cancelled_count = 0
            for order in self.orders[request.account_name]:
                if request.symbol and order.get("symbol") != request.symbol:
                    continue
                order_status = order.get("status", "").lower()
                if order_status in ["open", "pending"]:
                    order["status"] = "cancelled"
                    cancelled_count += 1
            
            return ActionResult(
                success=True,
                message=f"All orders canceled successfully.",
                extra={"account_name": request.account_name, "cancelled_count": cancelled_count}
            )
            
        except Exception as e:
            raise HyperliquidError(f"Failed to cancel all orders: {e}")
    
    async def close_order(self, request: CloseOrderRequest) -> ActionResult:
        """Close a position (reduce-only order).
        
        Args:
            request: CloseOrderRequest with account_name, symbol, side, size, order_type, optional price
            
        Returns:
            ActionResult with close order information
        """
        try:
            # Validate symbol
            if request.symbol not in self.symbols:
                raise InvalidSymbolError(f"Symbol {request.symbol} not found or not tradable")
            
            if request.account_name not in self.positions:
                raise OrderError(f"No position found for {request.symbol}")
            
            if request.symbol not in self.positions[request.account_name]:
                raise OrderError(f"No position found for {request.symbol}")
            
            position = self.positions[request.account_name][request.symbol]
            current_position_amt = float(position.get("position_amt", 0))
            
            if current_position_amt == 0:
                raise OrderError(f"No position to close for {request.symbol}")
            
            # Determine close side and size
            if current_position_amt > 0:  # Long position
                close_side = "sell"
                max_close_size = current_position_amt
            else:  # Short position
                close_side = "buy"
                max_close_size = abs(current_position_amt)
            
            if request.side.lower() != close_side:
                raise OrderError(f"Cannot close {request.symbol} position: expected {close_side}, got {request.side}")
            
            close_size = request.size if request.size else max_close_size
            if close_size > max_close_size:
                close_size = max_close_size
            
            # Create close order
            close_order_request = CreateOrderRequest(
                account_name=request.account_name,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                qty=close_size,
                price=request.price,
                leverage=1,  # Not used for closing
            )
            
            result = await self.create_order(close_order_request)
            
            # Check if position is fully closed after the close order
            # If so, cancel all guard orders (stop loss/take profit) for this symbol
            position_fully_closed = False
            if request.account_name in self.positions:
                position = self.positions[request.account_name].get(request.symbol)
                if position:
                    remaining_position_amt = float(position.get("position_amt", 0))
                    # If position is fully closed (or almost closed)
                    if abs(remaining_position_amt) < 1e-8:
                        position_fully_closed = True
                elif request.symbol not in self.positions[request.account_name]:
                    # Position was deleted (fully closed)
                    position_fully_closed = True
            
            # Cancel all open guard orders for this symbol if position is fully closed
            if position_fully_closed:
                cancelled_guard_orders = 0
                if request.account_name in self.orders:
                    for order in self.orders[request.account_name]:
                        if (order.get("symbol") == request.symbol and 
                            order.get("status", "").lower() in ["open", "pending"]):
                            order["status"] = "cancelled"
                            cancelled_guard_orders += 1
                
                if cancelled_guard_orders > 0:
                    logger.info(f"| ✅ Cancelled {cancelled_guard_orders} guard orders for {request.symbol} after position closed")
            
            return ActionResult(
                success=True,
                message=f"Close order submitted successfully for {request.symbol} ({request.side} {close_size}).",
                extra={"close_order": result.extra.get("order_info", {})}
            )
            
        except Exception as e:
            if isinstance(e, (InvalidSymbolError, OrderError)):
                raise
            raise OrderError(f"Failed to close order: {e}")
