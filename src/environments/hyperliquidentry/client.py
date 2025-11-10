"""Hyperliquid Python SDK client implementation."""

from typing import Dict, Optional, Any, List
import logging

# Hyperliquid Python SDK
try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils.signing import OrderType, LimitOrderType
    from eth_account import Account
    HYPERLIQUID_SDK_AVAILABLE = True
except ImportError:
    HYPERLIQUID_SDK_AVAILABLE = False
    logging.warning("| ⚠️  hyperliquid-python-sdk not available. Install with: pip install hyperliquid-python-sdk")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HyperliquidClient:
    """Hyperliquid client using official Python SDK."""

    def __init__(
        self,
        wallet_address: str,
        private_key: Optional[str] = None,
        testnet: bool = False
    ):
        """Initialize Hyperliquid client.

        Args:
            wallet_address: Hyperliquid wallet address
            private_key: Private key for signing requests (optional, can be provided later)
            testnet: Whether to use testnet (True) or mainnet (False)
        """
        if not HYPERLIQUID_SDK_AVAILABLE:
            raise ImportError("hyperliquid-python-sdk is not installed. Install with: pip install hyperliquid-python-sdk")
        
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.testnet = testnet
        
        # Initialize SDK components
        # Info doesn't need authentication
        self.info = Info(base_url="https://api.hyperliquid-testnet.xyz" if testnet else "https://api.hyperliquid.xyz")
        
        # Exchange needs account for signed operations
        self.exchange = None
        if private_key:
            self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize Exchange object with account."""
        if not self.private_key:
            logger.warning("| ⚠️  No private key provided. Exchange operations will not be available.")
            return
        
        try:
            account = Account.from_key(self.private_key)
            base_url = "https://api.hyperliquid-testnet.xyz" if self.testnet else "https://api.hyperliquid.xyz"
            self.exchange = Exchange(account, base_url=base_url)
            logger.info("| ✅ Exchange initialized successfully")
        except Exception as e:
            logger.error(f"| ❌ Failed to initialize Exchange: {e}")
            raise

    def set_private_key(self, private_key: str):
        """Set private key and initialize Exchange."""
        self.private_key = private_key
        self._initialize_exchange()

    # -------------------------- EXCHANGE INFO --------------------------
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including available symbols."""
        try:
            meta = self.info.meta()
            return meta
        except Exception as e:
            logger.error(f"| ❌ Failed to get exchange info: {e}")
            raise Exception(f"Failed to get exchange info: {e}")

    def _get_asset_index(self, symbol: str) -> int:
        """Get asset index for a symbol."""
        exchange_info = self.get_exchange_info()
        universe = exchange_info.get("universe", [])
        for idx, coin_info in enumerate(universe):
            if isinstance(coin_info, dict):
                coin_name = coin_info.get("name", "")
            else:
                coin_name = str(coin_info)
            if coin_name.upper() == symbol.upper():
                return idx
        raise Exception(f"Symbol {symbol} not found in exchange info")

    # -------------------------- USER STATE --------------------------
    def get_user_state(self) -> Dict[str, Any]:
        """Get user account state."""
        try:
            state = self.info.user_state(self.wallet_address)
            return state
        except Exception as e:
            logger.error(f"| ❌ Failed to get user state: {e}")
            raise Exception(f"Failed to get user state: {e}")

    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        return self.get_user_state()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions."""
        user_state = self.get_user_state()
        return user_state.get("assetPositions", [])

    # -------------------------- CREATE ORDER --------------------------
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "Market",
        size: float = None,
        price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "Gtc",
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new order (market or limit) with optional stop loss and take profit.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            side: Order side ('B' or 'BUY' for buy, 'A' or 'SELL' for sell)
            order_type: Order type ('Market' or 'Limit'). Default: 'Market'
            size: Order size (in base units, e.g., 0.1 BTC)
            price: Order price (required for Limit orders, ignored for Market orders)
            reduce_only: Whether this is a reduce-only order. Default: False
            time_in_force: Time in force for limit orders ('Gtc', 'Ioc', 'Alo'). Default: 'Gtc'
            stop_loss_price: Optional stop loss trigger price. If provided, creates a stop loss order after main order.
            take_profit_price: Optional take profit trigger price. If provided, creates a take profit order after main order.
            **kwargs: Additional order parameters

        Returns:
            Order information dictionary with main order and optional stop loss/take profit orders

        Raises:
            Exception: If private key is not provided or order creation fails
        """
        if not self.exchange:
            raise Exception("Private key required for order creation. Call set_private_key() first.")

        if size is None or size <= 0:
            raise Exception("Order size must be provided and greater than 0")

        if order_type == "Limit" and price is None:
            raise Exception("Price is required for Limit orders")

        # Convert side to boolean
        is_buy = side.upper() in ["B", "BUY"]
        
        # Build order according to Hyperliquid SDK format
        # For perpetual futures:
        # - Market orders: use market_open() which calculates slippage price automatically
        # - Limit orders: use order() with specified price
        if order_type == "Limit":
            # Limit order for perpetual futures
            order_type_obj = OrderType(limit=LimitOrderType(tif=time_in_force))
            order_result = self.exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type=order_type_obj,
                reduce_only=reduce_only
            )
        else:
            # Market order for perpetual futures
            if reduce_only:
                # For reduce-only orders, use market_close() which closes positions
                order_result = self.exchange.market_close(
                    coin=symbol,
                    sz=size,
                    px=None,  # Let SDK calculate market price with slippage
                    slippage=0.05,  # Default 5% slippage
                    cloid=None,
                    builder=None
                )
            else:
                # For opening positions, use market_open() which handles slippage
                # market_open calculates aggressive market price with slippage and uses IoC
                order_result = self.exchange.market_open(
                    name=symbol,
                    is_buy=is_buy,
                    sz=size,
                    px=None,  # Let SDK calculate market price with slippage
                    slippage=0.05,  # Default 5% slippage
                    cloid=None,
                    builder=None
                )

        logger.info(f"| 📝 Created {order_type} order: {side} {size} {symbol} at {price if price else 'market'}")
        logger.debug(f"| 🔍 Order result: {order_result}")
        
        # Prepare result with main order
        result = {
            "main_order": order_result,
            "stop_loss_order": None,
            "take_profit_order": None
        }
        
        # Create stop loss and take profit orders if provided
        # Only create these for opening positions (not reduce_only)
        if not reduce_only:
            # Create stop loss order if provided
            if stop_loss_price is not None and stop_loss_price > 0:
                try:
                    # Stop loss: opposite side to close position
                    # For long positions (buy), stop loss is sell at lower price
                    # For short positions (sell), stop loss is buy at higher price
                    stop_loss_side = "A" if is_buy else "B"  # Opposite side
                    stop_loss_result = self._create_trigger_order(
                        symbol=symbol,
                        side=stop_loss_side,
                        size=size,
                        trigger_price=stop_loss_price,
                        order_type="StopLoss"
                    )
                    result["stop_loss_order"] = stop_loss_result
                    logger.info(f"| 🛑 Created stop loss order: {stop_loss_price} for {symbol}")
                except Exception as e:
                    logger.warning(f"| ⚠️  Failed to create stop loss order: {e}")
                    result["stop_loss_error"] = str(e)
            
            # Create take profit order if provided
            if take_profit_price is not None and take_profit_price > 0:
                try:
                    # Take profit: opposite side to close position
                    # For long positions (buy), take profit is sell at higher price
                    # For short positions (sell), take profit is buy at lower price
                    take_profit_side = "A" if is_buy else "B"  # Opposite side
                    take_profit_result = self._create_trigger_order(
                        symbol=symbol,
                        side=take_profit_side,
                        size=size,
                        trigger_price=take_profit_price,
                        order_type="TakeProfit"
                    )
                    result["take_profit_order"] = take_profit_result
                    logger.info(f"| 🎯 Created take profit order: {take_profit_price} for {symbol}")
                except Exception as e:
                    logger.warning(f"| ⚠️  Failed to create take profit order: {e}")
                    result["take_profit_error"] = str(e)
        
        return result
    
    def _create_trigger_order(
        self,
        symbol: str,
        side: str,
        size: float,
        trigger_price: float,
        order_type: str = "StopLoss"
    ) -> Dict[str, Any]:
        """Create a trigger order (stop loss or take profit).
        
        Args:
            symbol: Trading symbol
            side: Order side ('A' for sell, 'B' for buy)
            size: Order size
            trigger_price: Trigger price for the conditional order
            order_type: Order type ('StopLoss' or 'TakeProfit')
            
        Returns:
            Order result dictionary
        """
        if not self.exchange:
            raise Exception("Private key required for trigger order creation.")
        
        # Convert side to boolean
        is_buy = side.upper() in ["B", "BUY"]
        
        # For Hyperliquid, we use trigger orders (conditional orders)
        # These are limit orders that trigger when price reaches trigger_price
        # We use reduce_only=True to close positions
        try:
            # Try to use trigger order if SDK supports it
            # Otherwise, create a limit order with reduce_only
            order_type_obj = OrderType(limit=LimitOrderType(tif="Gtc"))
            trigger_order_result = self.exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=size,
                limit_px=trigger_price,
                order_type=order_type_obj,
                reduce_only=True
            )
            return trigger_order_result
        except Exception as e:
            # If trigger order creation fails, log and re-raise
            logger.error(f"| ❌ Failed to create {order_type} trigger order: {e}")
            raise Exception(f"Failed to create {order_type} trigger order: {e}")

    # -------------------------- ORDER INFO --------------------------
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        user_state = self.get_user_state()
        open_orders = user_state.get("openOrders", [])
        if symbol:
            open_orders = [o for o in open_orders if o.get("coin") == symbol]
        return open_orders

    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status."""
        open_orders = self.get_open_orders(symbol)
        for order in open_orders:
            if str(order.get("oid")) == str(order_id):
                return order
        fills = self.get_user_fills()
        for fill in fills:
            if str(fill.get("oid")) == str(order_id):
                return fill
        raise Exception(f"Order {order_id} not found")

    def get_user_fills(self) -> List[Dict[str, Any]]:
        """Get user fill history."""
        try:
            fills = self.info.user_fills(self.wallet_address)
            return fills
        except Exception as e:
            logger.error(f"| ❌ Failed to get user fills: {e}")
            raise Exception(f"Failed to get user fills: {e}")

    # -------------------------- CANCEL ORDER --------------------------
    def cancel_order(self, asset_index: int, order_id: int) -> Dict[str, Any]:
        """Cancel an order.

        Args:
            asset_index: Asset index
            order_id: Order ID

        Returns:
            Cancellation result dictionary
        """
        if not self.exchange:
            raise Exception("Private key required for order cancellation. Call set_private_key() first.")

        try:
            result = self.exchange.cancel(coin_idx=asset_index, oid=order_id)
            logger.info(f"| 🗑️  Cancelled order {order_id} for asset {asset_index}")
            return result
        except Exception as e:
            logger.error(f"| ❌ Failed to cancel order: {e}")
            raise Exception(f"Failed to cancel order: {e}")

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel all open orders.

        Args:
            symbol: Optional symbol to cancel orders for. If None, cancels all orders.

        Returns:
            Cancellation result dictionary
        """
        if not self.exchange:
            raise Exception("Private key required for order cancellation. Call set_private_key() first.")

        open_orders = self.get_open_orders(symbol)
        if not open_orders:
            return {"status": "ok", "message": "No orders to cancel"}

        try:
            # Cancel all orders
            cancels = []
            for order in open_orders:
                asset_index = self._get_asset_index(order.get("coin", ""))
                order_id = order.get("oid")
                if asset_index is not None and order_id is not None:
                    cancels.append({"a": asset_index, "o": order_id})

            if not cancels:
                return {"status": "ok", "message": "No valid orders to cancel"}

            # Use SDK's cancel method - may need to cancel one by one or use batch cancel
            # Check SDK documentation for batch cancel support
            results = []
            for cancel in cancels:
                try:
                    result = self.exchange.cancel(coin_idx=cancel["a"], oid=cancel["o"])
                    results.append(result)
                except Exception as e:
                    logger.warning(f"| ⚠️  Failed to cancel order {cancel['o']}: {e}")
                    results.append({"error": str(e), "order_id": cancel["o"]})

            logger.info(f"| 🗑️  Cancelled {len(results)} orders")
            return {"status": "ok", "results": results}
        except Exception as e:
            logger.error(f"| ❌ Failed to cancel all orders: {e}")
            raise Exception(f"Failed to cancel all orders: {e}")
