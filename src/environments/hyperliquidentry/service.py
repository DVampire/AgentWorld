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
from src.environments.hyperliquidentry.candle import CandleHandler
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
            if request.size is None or request.size <= 0:
                raise HyperliquidError("'size' must be provided and greater than 0")
            
            if request.order_type == OrderType.LIMIT and request.price is None:
                raise HyperliquidError("'price' must be provided for LIMIT orders")
            
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
                size=request.size,
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

