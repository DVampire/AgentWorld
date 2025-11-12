"""Hyperliquid Python SDK client implementation."""

from typing import Dict, Optional, Any, List
import logging

# Hyperliquid Python SDK
try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
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
        price: Optional[float] = None, # only for limit orders
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a market order for perpetual futures with optional stop loss and take profit.
        
        For market orders with TP/SL, uses bulk_orders to submit all orders at once.
        For market orders without TP/SL, uses market_open for better slippage handling.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            side: Order side ('B' or 'BUY' for buy, 'A' or 'SELL' for sell)
            order_type: Order type ('Market' or 'Limit'). Default: 'Market'
            size: Order size (in base units, e.g., 0.1 BTC)
            price: Order price (ignored for Market orders)
            stop_loss_price: Optional stop loss trigger price
            take_profit_price: Optional take profit trigger price

        Returns:
            Order information dictionary

        Raises:
            Exception: If private key is not provided or order creation fails
        """
        if not self.exchange:
            raise Exception("Private key required for order creation. Call set_private_key() first.")

        if size is None or size <= 0:
            raise Exception("Order size must be provided and greater than 0")

        # Convert side to boolean
        is_buy = side.upper() in ["B", "BUY"]
        
        if order_type == "Limit":
            if price is None:
                raise Exception("Price must be provided for limit orders")
            open_order_result = self.exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=False
            )
            logger.info(f"| 📝 Limit order created: {symbol} {'LONG' if is_buy else 'SHORT'} {size} @ {price}")
        elif order_type == "Market":
            open_order_result = self.exchange.market_open(
                name=symbol,
                is_buy=is_buy,
                sz=size,
                px=None,
                slippage=0.05,
                cloid=None,
                builder=None,
            )
            logger.info(f"| 📝 Market order created: {symbol} {'LONG' if is_buy else 'SHORT'} {size}")
            
        result = {
            "open_order": open_order_result,
            "stop_loss_order": None,
            "take_profit_order": None
        }
        
        if not stop_loss_price and not take_profit_price:
            return result
        
        # Determine order sides for TP/SL
        # For LONG: main order is BUY, close orders are SELL
        # For SHORT: main order is SELL, close orders are BUY
        close_is_buy = not is_buy
        
        # Helper function to round prices to avoid precision issues
        def round_price(price: float) -> float:
            """Round price to avoid float_to_wire precision errors."""
            return round(float(f"{price:.8f}"), 8)
        
        # Create take profit order if provided
        if take_profit_price:
            try:
                tp_price = round_price(take_profit_price)
                tp_order_result = self.exchange.order(
                    name=symbol,
                    is_buy=close_is_buy,
                    sz=size,
                    limit_px=tp_price,
                    order_type={"trigger": {"isMarket": False, "triggerPx": tp_price, "tpsl": "tp"}},
                    reduce_only=True
                )
                result["take_profit_order"] = tp_order_result
                logger.info(f"| 🎯 Take profit order created at {tp_price}")
            except Exception as e:
                logger.warning(f"| ⚠️  Failed to create take profit order: {e}")
                result["take_profit_error"] = str(e)
        
        # Create stop loss order if provided
        if stop_loss_price:
            try:
                sl_price = round_price(stop_loss_price)
                sl_order_result = self.exchange.order(
                    name=symbol,
                    is_buy=close_is_buy,
                    sz=size,
                    limit_px=sl_price,
                    order_type={"trigger": {"isMarket": False, "triggerPx": sl_price, "tpsl": "sl"}},
                    reduce_only=True
                )
                result["stop_loss_order"] = sl_order_result
                logger.info(f"| 🛡️  Stop loss order created at {sl_price}")
            except Exception as e:
                logger.warning(f"| ⚠️  Failed to create stop loss order: {e}")
                result["stop_loss_error"] = str(e)
        
        return result
    
    # -------------------------- Close Order -------------------------
    def close_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "Market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Close a position (reduce-only order).
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            side: Order side to close position ('B' or 'BUY' to close SHORT, 'A' or 'SELL' to close LONG)
            size: Position size to close (in base units, e.g., 0.1 BTC)
            order_type: Order type ('Market' or 'Limit'). Default: 'Market'
            price: Order price (required for Limit orders, ignored for Market orders)
            
        Returns:
            Order result dictionary
            
        Raises:
            Exception: If private key is not provided or order creation fails
        """
        if not self.exchange:
            raise Exception("Private key required for order closing. Call set_private_key() first.")
        
        if size is None or size <= 0:
            raise Exception("Size must be provided and greater than 0")
        
        if order_type == "Limit" and price is None:
            raise Exception("Price must be provided for limit orders")
        
        # Convert side to boolean
        # For closing LONG position: side should be SELL (is_buy=False)
        # For closing SHORT position: side should be BUY (is_buy=True)
        is_buy = side.upper() in ["B", "BUY"]
        
        if order_type == "Limit":
            # Use order() with reduce_only=True for limit orders
            # For limit orders, we need to manually specify the direction
            # LONG position: close with SELL (is_buy=False)
            # SHORT position: close with BUY (is_buy=True)
            close_order_result = self.exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=True
            )
            logger.info(f"| 🗑️  Limit close order created: {symbol} {'BUY' if is_buy else 'SELL'} {size} @ {price}")
        
        elif order_type == "Market":
            # Use market_close for market orders (handles slippage automatically)
            # Note: market_close automatically determines direction by checking position:
            # - If szi < 0 (SHORT), it uses BUY to close
            # - If szi > 0 (LONG), it uses SELL to close
            # So the side parameter is ignored for Market orders, but kept for consistency
            close_order_result = self.exchange.market_close(
                coin=symbol,
                sz=size,
                px=None,
                slippage=0.05,
                cloid=None,
                builder=None
            )
            logger.info(f"| 🗑️  Market close order created: {symbol} (auto-determined direction)")
            
        return {"close_order": close_order_result}
        

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
