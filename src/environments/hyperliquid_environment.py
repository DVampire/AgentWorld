"""Hyperliquid Trading Environment for AgentWorld - provides Hyperliquid trading operations as an environment."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(verbose=True)
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Type, List, Union
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol import ecp
from src.environments.hyperliquidentry.service import HyperliquidService
from src.environments.hyperliquidentry.exceptions import (
    AuthenticationError,
)
from src.environments.hyperliquidentry.types import (
    GetAccountRequest,
    GetAssetsRequest,
    GetPositionsRequest,
    GetDataRequest,
    CreateOrderRequest,
    GetOrderRequest,
    CancelOrderRequest,
    CancelAllOrdersRequest,
    CloseOrderRequest,
    TradeType,
)
from src.utils import dedent, assemble_project_path

@ecp.environment()
class HyperliquidEnvironment(BaseEnvironment):
    """Hyperliquid Trading Environment that provides Hyperliquid trading operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="hyperliquid", description="The name of the Hyperliquid trading environment.")
    type: str = Field(default="Hyperliquid Trading", description="The type of the Hyperliquid trading environment.")
    description: str = Field(default="Hyperliquid trading environment for real-time data and trading operations", description="The description of the Hyperliquid trading environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the Hyperliquid trading environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the Hyperliquid trading environment including account information, positions, and market data.",
        }
    }, description="The metadata of the Hyperliquid trading environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        account_name: str = None,
        symbol: Optional[Union[str, List[str]]] = None,
        data_type: Optional[Union[str, List[str]]] = None,
        hyperliquid_service: HyperliquidService = None,
        **kwargs
    ):
        """
        Initialize the Hyperliquid trading environment.
        
        Args:
            base_dir (str): Base directory for Hyperliquid operations
            account_name (str): Account name to use
            symbol (str or List[str]): Symbol(s) to trade (e.g., 'BTC', ['BTC', 'ETH'])
            data_type (str or List[str]): Data type(s) to retrieve (default: 'candle')
            hyperliquid_service (HyperliquidService): Hyperliquid service instance
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.account_name = account_name
        self.symbol = symbol
        self.data_type = data_type
        self.hyperliquid_service = hyperliquid_service
        
        # Performance tracking
        self.account_value = None
        self.initial_account_value = None
        self.max_account_value = None
        self.max_drawdown = 0.0
        self.total_profit = 0.0
    
    async def initialize(self) -> None:
        """Initialize the Hyperliquid trading environment."""
        logger.info(f"| 🚀 Hyperliquid Trading Environment initialized at: {self.base_dir}")
    
    async def cleanup(self) -> None:
        """Cleanup the Hyperliquid trading environment."""
        logger.info("| 🧹 Hyperliquid Trading Environment cleanup completed")
        
    async def _calculate_matrics(self, account_value: float) -> Dict[str, Any]:
        """Calculate real-time performance metrics including profit and max drawdown.
        
        Args:
            account_value: Current account value
            
        Returns:
            Dictionary containing performance metrics
        """
        # Ensure account_value is float
        try:
            account_value = float(account_value)
        except (ValueError, TypeError):
            logger.warning(f"| ⚠️  Invalid account_value type: {type(account_value)}, value: {account_value}")
            account_value = 0.0
        
        # Initialize on first call
        if self.initial_account_value is None:
            self.initial_account_value = account_value
            self.account_value = account_value
            self.max_account_value = account_value
            logger.info(f"| 📊 Initial account value set: ${account_value:,.2f}")
            
        pre_account_value = self.account_value
        current_account_value = account_value
        
        # Update current account value
        self.account_value = current_account_value
        
        # Calculate profit (absolute and percentage)
        self.total_profit = current_account_value - self.initial_account_value
        profit_percentage = (self.total_profit / self.initial_account_value * 100) if self.initial_account_value > 0 else 0.0
        
        # Update max account value (peak)
        if current_account_value > self.max_account_value:
            self.max_account_value = current_account_value
        
        # Calculate current drawdown from peak
        current_drawdown = 0.0
        current_drawdown_percentage = 0.0
        if self.max_account_value > 0:
            current_drawdown = self.max_account_value - current_account_value
            current_drawdown_percentage = (current_drawdown / self.max_account_value * 100)
        
        # Update max drawdown
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate max drawdown percentage
        max_drawdown_percentage = (self.max_drawdown / self.max_account_value * 100) if self.max_account_value > 0 else 0.0
        
        # Calculate period return (since last update)
        period_return = current_account_value - pre_account_value
        period_return_percentage = (period_return / pre_account_value * 100) if pre_account_value > 0 else 0.0
        
        metrics = {
            "current_account_value": current_account_value,
            "initial_account_value": self.initial_account_value,
            "max_account_value": self.max_account_value,
            "total_profit": self.total_profit,
            "profit_percentage": profit_percentage,
            "current_drawdown": current_drawdown,
            "current_drawdown_percentage": current_drawdown_percentage,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_percentage": max_drawdown_percentage,
            "period_return": period_return,
            "period_return_percentage": period_return_percentage,
        }
        
        return metrics
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            A string containing detailed account information including balances and positions.
        """
        try:
            request = GetAccountRequest(account_name=self.account_name)
            result = await self.hyperliquid_service.get_account(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            account = result.extra["account"]
            
            # Convert account_value to float to avoid type comparison errors
            account_value_raw = account.get("margin_summary", {}).get("accountValue", 0)
            try:
                account_value = float(account_value_raw)
            except (ValueError, TypeError):
                account_value = 0.0
            
            asset_positions = json.dumps(account.get("asset_positions", []), indent=4)
            
            # Calculate performance metrics
            metrics = None
            if account_value > 0:
                metrics = await self._calculate_matrics(account_value)
            
            # Build result text with metrics
            metrics_text = ""
            if metrics:
                metrics_text = dedent(f"""
                Performance Metrics:
                Total Profit: ${metrics['total_profit']:,.2f} ({metrics['profit_percentage']:+.2f}%)
                Current Drawdown: ${metrics['current_drawdown']:,.2f} ({metrics['current_drawdown_percentage']:.2f}%)
                Max Drawdown: ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_percentage']:.2f}%)
                """)
            
            result_text = dedent(f"""
                Account Information:
                Timestamp: {account.get("time", "N/A")}
                Account Value: ${account_value:,.2f}
                Asset Positions: {asset_positions}
                {metrics_text}
                """)
            
            extra = {
                "account_value": account_value,
                "asset_positions": asset_positions,
                "account": account,
                "time": account.get("time", "N/A"),
            }
            
            if metrics:
                extra["metrics"] = metrics
            
            return {
                "success": True,
                "message": result_text,
                "extra": extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get account information: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_assets(self, status: Optional[str] = None, asset_class: Optional[str] = None) -> Dict[str, Any]:
        """Get all assets information.
        
        Returns:
            A string containing detailed assets information including symbols and status.
        """
        try:
            from src.environments.hyperliquidentry.types import GetExchangeInfoRequest
            request = GetExchangeInfoRequest()
            result = await self.hyperliquid_service.get_exchange_info(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            exchange_info = result.extra["exchange_info"]
            universe = exchange_info.get("universe", [])
            symbols = [{"symbol": coin.get("name", "") if isinstance(coin, dict) else str(coin)} for coin in universe]
            
            result_text = dedent(f"""
                {len(symbols)} symbols found, list of symbols:
                {", ".join([symbol["symbol"] for symbol in symbols])}
                """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": {"symbols": symbols, "exchange_info": exchange_info}
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get assets information: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_positions(self, trade_type: Optional[TradeType] = TradeType.PERPETUAL) -> Dict[str, Any]:
        """Get all open positions.
        
        Args:
            trade_type: Trade type to filter by (default: PERPETUAL for futures positions)
        
        Returns:
            A string containing detailed positions information including symbols, quantities, entry prices, and unrealized P&L.
        """
        try:
            request = GetPositionsRequest(account_name=self.account_name, trade_type=trade_type)
            result = await self.hyperliquid_service.get_positions(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            positions = result.extra["positions"]
            
            if len(positions) == 0:
                result_text = "No open positions."
            else:
                position_lines = []
                for pos in positions:
                    try:
                        symbol = pos.get("symbol", "N/A")
                        position_amt = float(pos.get("position_amt", 0))
                        entry_price = float(pos.get("entry_price", 0))
                        mark_price = float(pos.get("mark_price", 0))
                        unrealized_profit = float(pos.get("unrealized_profit", 0))
                        leverage = pos.get("leverage", "N/A")
                        trade_type_str = pos.get("trade_type", "N/A")
                        
                        position_lines.append(
                            f"  {symbol} ({trade_type_str}): {position_amt:+.6f} @ Entry: {entry_price}, "
                            f"Mark: {mark_price}, Leverage: {leverage}x, "
                            f"P&L: {unrealized_profit:.6f}"
                        )
                    except (ValueError, TypeError, KeyError):
                        # Fallback to string representation if conversion fails
                        position_lines.append(
                            f"  {pos.get('symbol', 'N/A')}: {pos.get('position_amt', 'N/A')} "
                            f"(P&L: {pos.get('unrealized_profit', 'N/A')})"
                        )
                
                result_text = dedent(f"""
                    {len(positions)} open position(s):
                    {chr(10).join(position_lines)}
                    """)
            
            extra = result.extra
            return {
                "success": True,
                "message": result_text,
                "extra": extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get positions information: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get historical candle data from database.
        
        Args:
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-01 00:00:00'). If not provided, returns latest data.
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-31 23:59:59'). If not provided, returns latest data.
            limit: Optional maximum number of rows to return per symbol
            
        Returns:
            Dictionary with success, message, and extra containing the data organized by symbol
        """
        try:
            request = GetDataRequest(
                symbol=self.symbol,
                data_type="candle",
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            result = await self.hyperliquid_service.get_data(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": result.extra
                }
            
            # Data is organized by symbol: {symbol: {data_type: [...]}}
            data = result.extra.get("data", {})
            symbols = result.extra.get("symbols", [])
            data_types = result.extra.get("data_type", "candle")
            row_count = result.extra.get("row_count", 0)
            
            # Format message
            result_message = result.message
            
            # Build a summary message showing data structure
            if isinstance(data, dict) and len(data) > 0:
                summary_lines = []
                for sym, type_data in data.items():
                    type_summary = []
                    for dt, records in type_data.items():
                        if records:
                            type_summary.append(f"{dt}: {len(records)} records")
                    if type_summary:
                        summary_lines.append(f"  {sym}: {', '.join(type_summary)}")
                
                if summary_lines:
                    result_message += f"\n\nData summary:\n" + "\n".join(summary_lines)
            
            return {
                "success": True,
                "message": result_message,
                "extra": {
                    "data": data,
                    "symbols": symbols,
                    "data_type": data_types,
                    "start_date": start_date,
                    "end_date": end_date,
                    "row_count": row_count
                }
            }
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return {
                "success": False,
                "message": f"Failed to get data: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = "Market",
        leverage: Optional[int] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create an order (perpetual futures order) with automatic stop loss and take profit protection.
        
        For each LONG or SHORT order (when reduce_only=False), this method will create:
        1. Main order: Opens the position at market or limit price
        2. Stop loss order (if stop_loss_price provided): Exchange trigger order to close position at stop loss price
        3. Take profit order (if take_profit_price provided): Exchange trigger order to close position at take profit price
        
        Both stop loss and take profit are reduce-only limit orders submitted to the exchange.
        They will automatically execute when the trigger price is reached to close the position.
        
        Examples:
            LONG position at $100,000:
            - Stop loss at $97,000 (3% below entry, protects against downside)
            - Take profit at $106,000 (6% above entry, locks in profit)
            
            SHORT position at $100,000:
            - Stop loss at $103,000 (3% above entry, protects against upside)
            - Take profit at $94,000 (6% below entry, locks in profit)
        
        Args:
            symbol(str): Symbol to trade (e.g., 'BTC', 'ETH')
            side(str): Order side: 'buy' or 'sell'
            qty(Optional[float]): Quantity to trade
            price(Optional[float]): Price for limit orders (required for LIMIT order type)
            order_type(str): Order type - 'Market' (default) or 'Limit'
            leverage(Optional[int]): Leverage for perpetual futures (optional)
            reduce_only(bool): Whether this is a reduce-only order (for closing positions)
            time_in_force(str): Time in force for limit orders - 'Gtc', 'Ioc', 'Alo' (default: 'Gtc')
            stop_loss_price(Optional[float]): Stop loss trigger price (absolute price, not percentage).
                For LONG: should be below entry price. For SHORT: should be above entry price.
            take_profit_price(Optional[float]): Take profit trigger price (absolute price, not percentage).
                For LONG: should be above entry price. For SHORT: should be below entry price.
            
        Returns:
            Dictionary with success, message, and order information including stop_loss_order and take_profit_order status
        """
        try:
            from src.environments.hyperliquidentry.types import OrderType
            
            order_type_enum = OrderType.MARKET if order_type == "Market" else OrderType.LIMIT
            
            request = CreateOrderRequest(
                account_name=self.account_name,
                symbol=symbol,
                side=side,
                trade_type=TradeType.PERPETUAL,
                order_type=order_type_enum,
                qty=qty,
                price=price,
                leverage=leverage,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            result = await self.hyperliquid_service.create_order(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            order_info = result.extra["order_info"]
            result_text = dedent(f"""
                Order submitted successfully:
                Order ID: {order_info["order_id"]}
                Symbol: {order_info["symbol"]}
                Side: {order_info["side"]}
                Quantity: {order_info["quantity"]}
                Status: {order_info["order_status"]}
                Order Type: {order_info["order_type"]}
                """)
            
            # Add stop loss/take profit info if available
            if order_info.get("stop_loss_order"):
                result_text += f"\nStop Loss Order: Created"
            if order_info.get("take_profit_order"):
                result_text += f"\nTake Profit Order: Created"
            if order_info.get("stop_loss_error"):
                result_text += f"\nStop Loss Error: {order_info['stop_loss_error']}"
            if order_info.get("take_profit_error"):
                result_text += f"\nTake Profit Error: {order_info['take_profit_error']}"
            
            return {
                "success": True,
                "message": result_text,
                "extra": result.extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def close_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "Market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Close a position (reduce-only order).
        
        Args:
            symbol: Symbol to close position for (e.g., 'BTC', 'ETH')
            side: Order side to close position:
                - 'sell' or 'A': Close LONG position (sell to close)
                - 'buy' or 'B': Close SHORT position (buy to close)
            size: Position size to close (in base units, e.g., 0.1 BTC)
            order_type: Order type - 'Market' (default) or 'Limit'
                - 'Market': Uses market_close which automatically determines direction by checking position
                - 'Limit': Requires manual side specification (side parameter is used)
            price: Order price (required for LIMIT order type, ignored for Market orders)
            
        Returns:
            Dictionary with success, message, and close order information
            
        Note:
            For Market orders, the side parameter is kept for consistency but the actual direction
            is automatically determined by checking the position (szi < 0 for SHORT uses BUY, szi > 0 for LONG uses SELL).
            For Limit orders, the side parameter is required and used to specify the direction.
        """
        try:
            from src.environments.hyperliquidentry.types import OrderType
            
            order_type_enum = OrderType.MARKET if order_type == "Market" else OrderType.LIMIT
            
            request = CloseOrderRequest(
                account_name=self.account_name,
                symbol=symbol,
                side=side,
                size=size,
                order_type=order_type_enum,
                price=price,
                trade_type=TradeType.PERPETUAL
            )
            result = await self.hyperliquid_service.close_order(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            close_order = result.extra.get("close_order", {})
            result_text = dedent(f"""
                Close order submitted successfully:
                Order ID: {close_order.get("order_id", "N/A")}
                Symbol: {close_order.get("symbol")}
                Side: {close_order.get("side")}
                Quantity: {close_order.get("quantity")}
                Status: {close_order.get("status")}
                Order Type: {close_order.get("type")}
                Trade Type: {close_order.get("trade_type")}
                """)
            
            if close_order.get("price"):
                result_text += f"\nPrice: {close_order['price']}"
            
            return {
                "success": True,
                "message": result_text,
                "extra": result.extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to close order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_orders(
        self,
        trade_type: Optional[TradeType] = TradeType.PERPETUAL,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
        order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get orders for an account.
        
        Args:
            trade_type: Trade type to filter by (default: PERPETUAL for futures orders)
            symbol: Optional symbol to filter by
            limit: Optional maximum number of orders to return
            order_id: Optional order ID to filter by
            
        Returns:
            Dictionary with success, message, and list of orders
        """
        try:
            from src.environments.hyperliquidentry.types import GetOrdersRequest
            
            request = GetOrdersRequest(
                account_name=self.account_name,
                trade_type=trade_type,
                symbol=symbol,
                limit=limit,
                order_id=order_id
            )
            result = await self.hyperliquid_service.get_orders(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            orders = result.extra["orders"]
            
            if len(orders) == 0:
                result_text = f"No {trade_type.value} orders found."
            else:
                order_lines = []
                for order in orders:
                    qty_display = order.get("quantity", "N/A")
                    
                    order_lines.append(
                        f"  {order['symbol']}: {order['side']} {qty_display} "
                        f"(Status: {order['status']}, Trade Type: {order.get('trade_type', 'N/A')})"
                    )
                
                result_text = dedent(f"""
                    {len(orders)} order(s) found:
                    {chr(10).join(order_lines)}
                    """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": result.extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get orders: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_order(self, order_id: str, symbol: str, trade_type: TradeType = TradeType.PERPETUAL) -> Dict[str, Any]:
        """Get a specific order by ID.
        
        Args:
            order_id: Order ID
            symbol: Symbol
            trade_type: Trade type (default: PERPETUAL for futures order)
            
        Returns:
            Dictionary with success, message, and order information
        """
        try:
            request = GetOrderRequest(
                account_name=self.account_name,
                order_id=order_id,
                symbol=symbol,
                trade_type=trade_type
            )
            result = await self.hyperliquid_service.get_order(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            order = result.extra["order"]
            qty_display = order.get("quantity", "N/A")
            
            result_text = dedent(f"""
                Order Information:
                Order ID: {order["order_id"]}
                Symbol: {order["symbol"]}
                Side: {order["side"]}
                Quantity: {qty_display}
                Status: {order["status"]}
                Order Type: {order["type"]}
                Trade Type: {order.get("trade_type", "N/A")}
                """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": result.extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def cancel_order(self, order_id: str, symbol: str, trade_type: TradeType = TradeType.PERPETUAL) -> Dict[str, Any]:
        """Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Symbol
            trade_type: Trade type (default: PERPETUAL for futures order)
            
        Returns:
            Dictionary with success or failure message
        """
        try:
            request = CancelOrderRequest(
                account_name=self.account_name,
                order_id=order_id,
                symbol=symbol,
                trade_type=trade_type
            )
            result = await self.hyperliquid_service.cancel_order(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            return {
                "success": True,
                "message": result.message,
                "extra": result.extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def cancel_all_orders(self, symbol: Optional[str] = None, trade_type: TradeType = TradeType.PERPETUAL) -> Dict[str, Any]:
        """Cancel all orders for an account.
        
        Args:
            symbol: Optional symbol to cancel orders for
            trade_type: Trade type (default: PERPETUAL for futures orders)
        """
        try:
            request = CancelAllOrdersRequest(
                account_name=self.account_name,
                symbol=symbol,
                trade_type=trade_type
            )
            result = await self.hyperliquid_service.cancel_all_orders(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            return {
                "success": True,
                "message": result.message,
                "extra": result.extra
            }
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel all orders: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    @ecp.action(name="step",
                type="Hyperliquid Trading",
                description=dedent("""
                    Step the trading environment for perpetual futures trading.
                    Example:
                        - SHORT: {"symbol": "BTC", "action": "SHORT", "qty": 0.01, "leverage": 10, "stop_loss_price": 100000, "take_profit_price": 110000}
                        - LONG: {"symbol": "BTC", "action": "LONG", "qty": 0.01, "leverage": 10, "stop_loss_price": 90000, "take_profit_price": 110000}
                        - CLOSE_LONG: {"symbol": "BTC", "action": "CLOSE_LONG"}
                        - CLOSE_SHORT: {"symbol": "BTC", "action": "CLOSE_SHORT"}
                        - HOLD: {"symbol": "BTC", "action": "HOLD"}
                    """)
                )
    async def step(self, 
                   symbol: str = "BTC", 
                   action: str = "HOLD",  # LONG, SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
                   qty: float = 0.0,
                   leverage: Optional[int] = 10,
                   stop_loss_price: Optional[float] = None,
                   take_profit_price: Optional[float] = None,
                   ) -> Dict[str, Any]:
        """Step the trading environment for perpetual futures trading.
        
        Hyperliquid Perpetual Futures Trading Rules:
        ┌──────────────┬─────────────────────────────────────────────────────────────┐
        │ Action       │ Description                                                 │
        ├──────────────┼─────────────────────────────────────────────────────────────┤
        │ LONG         │ Open long position + create stop loss & take profit orders │
        │ CLOSE_LONG   │ Close long position (market order)                         │
        │ SHORT        │ Open short position + create stop loss & take profit orders│
        │ CLOSE_SHORT  │ Close short position (market order)                        │
        │ HOLD         │ Do nothing                                                  │
        └──────────────┴─────────────────────────────────────────────────────────────┘
        
        Important: For LONG and SHORT actions, if stop_loss_price and take_profit_price are provided,
        two additional reduce-only orders will be automatically created on the exchange:
        1. Stop loss order: Closes position if price moves against you (limits loss)
        2. Take profit order: Closes position if price reaches target (locks in profit)
        
        These are exchange orders that execute automatically, protecting your position even if
        the program stops running.
        
        Examples:
            LONG BTC at $100,000 with stop_loss_price=$97,000, take_profit_price=$106,000
            → Creates 3 orders:
              1. BUY 0.01 BTC at market (main order)
              2. SELL 0.01 BTC at $97,000 (stop loss, reduce-only)
              3. SELL 0.01 BTC at $106,000 (take profit, reduce-only)
            
            SHORT ETH at $3,000 with stop_loss_price=$3,090, take_profit_price=$2,820
            → Creates 3 orders:
              1. SELL 0.1 ETH at market (main order)
              2. BUY 0.1 ETH at $3,090 (stop loss, reduce-only)
              3. BUY 0.1 ETH at $2,820 (take profit, reduce-only)
        
        Args:
            symbol (str): Symbol to trade (e.g., 'BTC', 'ETH')
            action (str): Trading action for perpetual futures:
                - 'LONG': Open long position (with optional stop loss & take profit)
                - 'SHORT': Open short position (with optional stop loss & take profit)
                - 'CLOSE_LONG': Close long position (market order, no stop loss/take profit)
                - 'CLOSE_SHORT': Close short position (market order, no stop loss/take profit)
                - 'HOLD': Do nothing (default)
            qty (float): Quantity to trade.
            leverage (Optional[int]): Leverage for perpetual futures.
            stop_loss_price (Optional[float]): Stop loss trigger price (absolute price, not percentage). Must be provided for LONG and SHORT actions.
            take_profit_price (Optional[float]): Take profit trigger price (absolute price, not percentage). Must be provided for LONG and SHORT actions.
        Returns:
            Dictionary with success, message, and order information
        """
        
        action = action.upper()
        try:
            if action == "HOLD":
                result = {
                    "success": True,
                    "message": "HOLD action performed successfully. No order submitted.",
                    "extra": {}
                }
            elif action == "LONG":
                # Open long position: BUY, reduce_only=False
                result = await self.create_order(
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    order_type="Market",
                    leverage=leverage,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
            elif action == "SHORT":
                # Open short position: SELL, reduce_only=False
                result = await self.create_order(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    order_type="Market",
                    leverage=leverage,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
            elif action == "CLOSE_LONG":
                # Close long position: SELL to close LONG position
                # Note: For Market orders, market_close automatically determines direction,
                # but we specify side="sell" for consistency (LONG position requires SELL to close)
                result = await self.close_order(
                    symbol=symbol,
                    side="sell",  # SELL to close LONG position
                    size=qty,
                    order_type="Market"
                )
            elif action == "CLOSE_SHORT":
                # Close short position: BUY to close SHORT position
                # Note: For Market orders, market_close automatically determines direction,
                # but we specify side="buy" for consistency (SHORT position requires BUY to close)
                result = await self.close_order(
                    symbol=symbol,
                    side="buy",  # BUY to close SHORT position
                    size=qty,
                    order_type="Market"
                )
            else:
                result = {
                    "success": False,
                    "message": f"Invalid action: {action}. Must be LONG, SHORT, CLOSE_LONG, CLOSE_SHORT, or HOLD",
                    "extra": {"error": f"Invalid action: {action}"}
                }
                
            # Get account information and calculate metrics
            account_request = GetAccountRequest(account_name=self.account_name)
            account_result = await self.hyperliquid_service.get_account(account_request)
            result["extra"].update(account_result.extra)
            
            # Calculate performance metrics
            account_value_raw = account_result.extra.get("account", {}).get("margin_summary", {}).get("accountValue", 0)
            try:
                account_value = float(account_value_raw)
            except (ValueError, TypeError):
                account_value = 0.0
            
            if account_value > 0:
                metrics = await self._calculate_matrics(account_value)
                result["extra"]["metrics"] = metrics
            
            return result
        
        except AuthenticationError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to step the trading environment: {str(e)}",
                "extra": {"error": str(e)}
            }
            
    async def _wait_for_next_minute_boundary(self) -> None:
        """Wait until the next minute boundary for minute-level trading.
        
        This ensures we get complete minute kline data by waiting until the start
        of the next minute before fetching data.
        """
        now = datetime.now(timezone.utc)
        # Calculate seconds until next minute boundary
        if now.second > 0:
            # Calculate milliseconds to account for microsecond precision
            microseconds_until_next_minute = (60 - now.second) * 1000000 - now.microsecond
            wait_time = microseconds_until_next_minute / 1000000.0  # Convert to seconds
            if wait_time > 0:
                logger.info(f"| ⏳ Waiting {wait_time:.2f} seconds until next minute boundary (current: {now.strftime('%Y-%m-%d %H:%M:%S')})...")
                await asyncio.sleep(wait_time)
            else:
                logger.info(f"| ✅ Already at minute boundary (current: {now.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            logger.info(f"| ✅ Already at minute boundary (current: {now.strftime('%Y-%m-%d %H:%M:%S')})")
    
    def _safe_float_format(self, value: Any, default: float = 0.0) -> str:
        """Safely convert a value to float and format it to 2 decimal places."""
        try:
            if isinstance(value, (int, float)):
                return f"{float(value):.2f}"
            elif isinstance(value, str):
                return f"{float(value):.2f}"
            else:
                return f"{float(default):.2f}"
        except (ValueError, TypeError):
            return f"{float(default):.2f}"
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the Hyperliquid trading environment."""
        try:
            # Get account info
            account_result = await self.get_account()
            account_result = account_result.get("extra", {})
            account_info = account_result.get("account", {})
            metrics = account_result.get("metrics", {})
            account_string = ""
            account_string += dedent(f"""
                Timestamp: {account_info.get("time", "N/A")}
                Account Value: ${self._safe_float_format(account_info.get("margin_summary", {}).get("accountValue", 0))},
                Total Profit: ${self._safe_float_format(metrics.get("total_profit", 0))} ({self._safe_float_format(metrics.get("profit_percentage", 0))}%),
                Current Drawdown: ${self._safe_float_format(metrics.get("current_drawdown", 0))} ({self._safe_float_format(metrics.get("current_drawdown_percentage", 0))}%),
                Max Drawdown: ${self._safe_float_format(metrics.get("max_drawdown", 0))} ({self._safe_float_format(metrics.get("max_drawdown_percentage", 0))}%),
                Period Return: ${self._safe_float_format(metrics.get("period_return", 0))} ({self._safe_float_format(metrics.get("period_return_percentage", 0))}%),
            """)
            account_string = dedent(f"""
                <account>
                {account_string}
                </account>
            """)
            logger.info(f"| 📝 Account String: {account_string}")
            
            # Get positions
            positions_result = await self.get_positions()
            positions_result = positions_result.get("extra", {})
            positions_list = positions_result.get("positions", [])
            positions_string = ""
            for position in positions_list:
                return_on_equity = position.get('return_on_equity', 0)
                try:
                    return_on_equity_float = float(return_on_equity) * 100
                except (ValueError, TypeError):
                    return_on_equity_float = 0.0
                return_on_equity_pct = self._safe_float_format(return_on_equity_float)
                
                position_side = "LONG" if float(position['position_amt']) > 0 else "SHORT"
                
                positions_string += f"Symbol: {position['symbol']}, Position Side: {position_side}, Trade Type: {position['trade_type']}, Leverage: {position['leverage']}, Position Amount: {position['position_amt']}, Entry Price: {position['entry_price']}, Current Price: {position['mark_price']}, Return on Equity: {return_on_equity_pct}%\n"
            positions_string = dedent(f"""
                <positions>
                {positions_string}
                </positions>
            """)
            logger.info(f"| 📝 Positions String: {positions_string}")
            
            # Get orders
            orders_result = await self.get_orders()
            orders_result = orders_result.get("extra", {})
            orders_list = orders_result.get("orders", [])
            orders_string = ""
            for order in orders_list:
                quantity_str = self._safe_float_format(order.get('quantity', 0))
                price_str = self._safe_float_format(order.get('price', 0))
                orders_string += f"Order ID: {order['order_id']}, Symbol: {order['symbol']}, Trade Type: {order['trade_type']}, Order Type: {order['type']}, Order Side: {order['side']}, Quantity: {quantity_str}, Price: {price_str}, Status: {order['status']}\n"
            orders_string = dedent(f"""
                <orders>
                {orders_string}
                </orders>
            """)
            logger.info(f"| 📝 Orders String: {orders_string}")
            
            # Wait until the next minute boundary for minute-level trading
            await self._wait_for_next_minute_boundary()
            
            # Get data (placeholder for now)
            data_result = await self.get_data()
            data_result = data_result.get("extra", {})
            candles = {}
            for symbol, data in data_result.get("data", {}).items():
                candles[symbol] = data.get("candle", [])
            candles_string = ""
            for symbol, candles_list in candles.items():
                symbol_string = dedent(f"""Symbol: {symbol}""")
                
                for candle in candles_list:
                    
                    symbol_string += dedent(f"""
                            Timestamp: {candle["timestamp"]}
                            Open: {self._safe_float_format(candle.get("open", 0))}
                            High: {self._safe_float_format(candle.get("high", 0))}
                            Low: {self._safe_float_format(candle.get("low", 0))}
                            Close: {self._safe_float_format(candle.get("close", 0))}
                            Volume: {self._safe_float_format(candle.get("volume", 0))}
                    """)
                    
                    indicators = candle.get("indicators", {})
                    if len(indicators) > 0:
                        indicators_string = ", ".join([f"{indicator}: {self._safe_float_format(value)}" for indicator, value in indicators.items()])
                        symbol_string += dedent(f"""Indicators: {indicators_string}""")
                    else:
                        symbol_string += dedent(f"""Indicators: No indicators available now.""")
                        
                candles_string += symbol_string + "\n"
            
            data_string = dedent(f"""
                <data>
                {candles_string}
                </data>
            """)
            logger.info(f"| 📝 Data: {data_string}")
            
            state = dedent(f"""
                <state>
                {account_info}
                {orders_string}
                {data_string}
                </state>
            """)
            
            return {
                "state": state,
                "extra": {
                    "account": account_result,
                    "positions": positions_result,
                    "orders": orders_result,
                    "input": data_result,
                }
            }
        except AuthenticationError as e:
            return {
                "state": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            logger.error(f"Failed to get Hyperliquid state: {e}")
            return {
                "state": f"Failed to get Hyperliquid state: {str(e)}",
                "extra": {"error": str(e)}
            }

