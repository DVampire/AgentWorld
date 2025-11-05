"""Alpaca Trading Environment for AgentWorld - provides Alpaca trading operations as an environment."""

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
from src.environments.alpacaentry.service import AlpacaService
from src.environments.alpacaentry.exceptions import (
    AuthenticationError,
)
from src.environments.alpacaentry.types import (
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
from src.utils import dedent, assemble_project_path

@ecp.environment()
class AlpacaEnvironment(BaseEnvironment):
    """Alpaca Trading Environment that provides Alpaca trading operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="alpaca", description="The name of the Alpaca trading environment.")
    type: str = Field(default="Alpaca Trading", description="The type of the Alpaca trading environment.")
    description: str = Field(default="Alpaca trading environment for real-time data and trading operations", description="The description of the Alpaca trading environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the Alpaca trading environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the Alpaca trading environment including account information, positions, and market data.",
            "interaction": dedent(f"""
                Guidelines for interacting with the Alpaca trading environment:
                - Always check account status before placing orders
                - Verify sufficient buying power before buying
                - Check market hours before trading
                - Use paper trading for testing strategies
                - Monitor positions and orders regularly
            """),
        }
    }, description="The metadata of the Alpaca trading environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        account_name: str = None,
        symbol: Optional[Union[str, List[str]]] = None,
        data_type: Optional[Union[str, List[str]]] = None,
        alpaca_service: AlpacaService = None,
        **kwargs
    ):
        """
        Initialize the Alpaca paper trading environment.
        
        Args:
            base_dir (str): Base directory for Alpaca operations
            live (bool): Whether to use live trading
            auto_start_data_stream (bool): Whether to auto start data stream
            data_stream_symbols (List[str]): The symbols to stream data for
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.account_name = account_name
        self.symbol = symbol
        self.data_type = data_type
        self.alpaca_service = alpaca_service
        
    async def initialize(self) -> None:
        """Initialize the Alpaca trading environment."""
        logger.info(f"| 🚀 Alpaca Trading Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the Alpaca trading environment."""
        logger.info("| 🧹 Alpaca Trading Environment cleanup completed")

    async def get_account(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            A string containing detailed account information including buying power, cash, portfolio value, and account status.
        """
        try:
            
            request = GetAccountRequest(account_name=self.account_name)
            result = await self.alpaca_service.get_account(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            account = result.extra["account"]
            result_text = dedent(f"""
                Account Information:
                Account Number: {account["account_number"]}
                Status: {account["status"]}
                Currency: {account["currency"]}
                Buying Power: ${float(account["buying_power"]):,.2f}
                Cash: ${float(account["cash"]):,.2f}
                Portfolio Value: ${float(account["portfolio_value"]):,.2f}
                Equity: ${float(account["equity"]):,.2f}
                Pattern Day Trader: {account["pattern_day_trader"]}
                Trading Blocked: {account["trading_blocked"]}
                Shorting Enabled: {account["shorting_enabled"]}
                Day Trade Count: {account["daytrade_count"]}
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
                "message": f"Failed to get account information: {str(e)}",
                "extra": {"error": str(e)}
            }
        
    async def get_assets(self, status: Optional[str] = None, asset_class: Optional[str] = None) -> Dict[str, Any]:
        """Get all assets information.
        
        Returns:
            A string containing detailed assets information including symbols, names, types, and status.
        """
        try:
            request = GetAssetsRequest(status=status, asset_class=asset_class)
            result = await self.alpaca_service.get_assets(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
                
            assets = result.extra["assets"]
            result_text = dedent(f"""
                {len(assets)} assets found, list of assets:
                {", ".join([asset["symbol"] for asset in assets])}
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
                "message": f"Failed to get assets information: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get all open positions.
        
        Returns:
            A string containing detailed positions information including symbols, quantities, market values, and unrealized P&L.
        """
        try:
            request = GetPositionsRequest(account_name=self.account_name)
            result = await self.alpaca_service.get_positions(request)
            
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
                        qty = float(pos["qty"])
                        market_value = float(pos["market_value"])
                        unrealized_pl = float(pos["unrealized_pl"])
                        unrealized_plpc = float(pos["unrealized_plpc"])
                        current_price = float(pos["current_price"])
                        avg_entry_price = float(pos["avg_entry_price"])
                        
                        position_lines.append(
                            f"  {pos['symbol']}: {qty:+.2f} shares @ ${current_price:.2f} "
                            f"(Avg Entry: ${avg_entry_price:.2f}, Market Value: ${market_value:,.2f}, "
                            f"P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:.2%}))"
                        )
                    except (ValueError, TypeError, KeyError):
                        # Fallback to string representation if conversion fails
                        position_lines.append(
                            f"  {pos.get('symbol', 'N/A')}: {pos.get('qty', 'N/A')} shares "
                            f"(P&L: {pos.get('unrealized_pl', 'N/A')})"
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
        """Get historical data from database.
        
        Args:
            symbol: Symbol(s) to query (e.g., 'BTC/USD', 'AAPL', or ['BTC/USD', 'AAPL'])
            data_type: Type(s) of data to retrieve - 'quotes', 'trades', 'bars', 'orderbooks' (crypto only), or 'news'. Can be a single type or a list of types (e.g., ['bars', 'news'])
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-01 00:00:00'). If not provided, returns latest data.
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-31 23:59:59'). If not provided, returns latest data.
            limit: Optional maximum number of rows to return per symbol/data_type combination
            
        Returns:
            Dictionary with success, message, and extra containing the data organized by symbol:
            {
                "success": True,
                "message": "...",
                "extra": {
                    "data": {
                        "symbol1": {
                            "bars": [...],
                            "news": [...],
                            "quotes": [...]
                        },
                        "symbol2": {
                            "bars": [...],
                            "trades": [...]
                        }
                    },
                    "symbols": [...],
                    "data_types": [...],
                    "row_count": ...
                }
            }
        """
        try:
            
            request = GetDataRequest(
                symbol=self.symbol,
                data_type=self.data_type,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            result = await self.alpaca_service.get_data(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": result.extra
                }
            
            # Data is now organized by symbol: {symbol: {data_type: [...]}}
            data = result.extra.get("data", {})
            symbols = result.extra.get("symbols", [])
            data_types = result.extra.get("data_types", [])
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
                    "data_types": data_types,
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
        notional: Optional[float] = None,
        time_in_force: str = "ioc"
    ) -> Dict[str, Any]:
        """Create a market order.
        
        Args:
            symbol: Symbol to trade (e.g., 'AAPL', 'BTC/USD')
            side: Order side: 'buy' or 'sell'
            qty: Optional quantity to trade (for notional orders, use None)
            notional: Optional notional value to trade (for fractional shares, use this instead of qty)
            time_in_force: Time in force - 'day' (default), 'gtc', 'opg', 'cls', 'ioc', 'fok'
            
        Returns:
            Dictionary with success, message, and order information
        """
        try:
            
            request = CreateOrderRequest(
                account_name=self.account_name,
                symbol=symbol,
                side=side,
                qty=qty,
                notional=notional,
                time_in_force=time_in_force
            )
            result = await self.alpaca_service.create_order(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            order = result.extra["order"]
            result_text = dedent(f"""
                Order submitted successfully:
                Order ID: {order["id"]}
                Symbol: {order["symbol"]}
                Side: {order["side"]}
                Quantity: {order["qty"] or order["notional"]}
                Status: {order["status"]}
                Order Type: {order["order_type"]}
                Time in Force: {order["time_in_force"]}
                Submitted At: {order["submitted_at"]}
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
                "message": f"Failed to create order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_orders(
        self,
        status: str = "open",
        limit: Optional[int] = None,
        after: Optional[str] = None,
        until: Optional[str] = None,
        direction: str = "desc"
    ) -> Dict[str, Any]:
        """Get orders for an account.
        
        Args:
            account_name: Account name
            status: Filter by order status: 'open', 'closed', or 'all' (default: 'open')
            limit: Optional maximum number of orders to return
            after: Optional return orders after this date (ISO format)
            until: Optional return orders until this date (ISO format)
            direction: Sort direction: 'asc' or 'desc' (default: 'desc')
            
        Returns:
            Dictionary with success, message, and list of orders
        """
        try:
            request = GetOrdersRequest(
                account_name=self.account_name,
                status=status,
                limit=limit,
                after=after,
                until=until,
                direction=direction
            )
            result = await self.alpaca_service.get_orders(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            orders = result.extra["orders"]
            
            if len(orders) == 0:
                result_text = f"No {status} orders found."
            else:
                order_lines = []
                for order in orders:
                    qty_display = order.get("qty") or order.get("notional") or "N/A"
                    filled_qty = order.get("filled_qty", "0")
                    filled_price = order.get("filled_avg_price") or "N/A"
                    
                    order_lines.append(
                        f"  {order['symbol']}: {order['side']} {qty_display} "
                        f"(Status: {order['status']}, Filled: {filled_qty} @ {filled_price})"
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
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get a specific order by ID.
        
        Args:
            account_name: Account name
            order_id: Order ID
            
        Returns:
            Dictionary with success, message, and order information
        """
        try:
            request = GetOrderRequest(account_name=self.account_name, order_id=order_id)
            result = await self.alpaca_service.get_order(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            order = result.extra["order"]
            qty_display = order.get("qty") or order.get("notional") or "N/A"
            
            result_text = dedent(f"""
                Order Information:
                Order ID: {order["id"]}
                Client Order ID: {order["client_order_id"]}
                Symbol: {order["symbol"]}
                Side: {order["side"]}
                Quantity: {qty_display}
                Filled Quantity: {order.get("filled_qty", "0")}
                Filled Average Price: {order.get("filled_avg_price") or "N/A"}
                Status: {order["status"]}
                Order Type: {order["order_type"]}
                Time in Force: {order["time_in_force"]}
                Submitted At: {order.get("submitted_at") or "N/A"}
                Filled At: {order.get("filled_at") or "N/A"}
                Canceled At: {order.get("canceled_at") or "N/A"}
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
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order.
        
        Args:
            account_name: Account name
            order_id: Order ID to cancel
            
        Returns:
            Dictionary with success or failure message
        """
        try:
            request = CancelOrderRequest(account_name=self.account_name, order_id=order_id)
            result = await self.alpaca_service.cancel_order(request)
            
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
    
    async def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all orders for an account.
        """
        try:
            request = CancelAllOrdersRequest(account_name=self.account_name)
            result = await self.alpaca_service.cancel_all_orders(request)
            
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
                type="Alpaca Trading",
                description="Step the trading environment.")
    async def step(self, 
                   symbol: str = "BTC/USD", 
                   side: str = "HOLD", # BUY, SELL, HOLD
                   qty: float = 0.00, 
                   ) -> Dict[str, Any]:
        """Step the trading environment.
        
        Args:
            symbol (str): Symbol to trade (e.g., 'AAPL', 'BTC/USD')
            side (str): Order side: 'BUY', 'SELL', 'HOLD' (default: 'HOLD')
            qty (float): Quantity to trade (default: 0.01)
        Returns:
            Dictionary with success, message, and order information
        """
        side = side.lower()
        try:
            if side == "hold":
                return {
                    "success": True,
                    "message": "HOLD action performed successfully. No order submitted.",
                    "extra": {}
                }
            else:
                return await self.create_order(symbol, side, qty)
            
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
        
        This ensures we get complete minute bar data by waiting until the start
        of the next minute before fetching data.
        """
        now = datetime.now(timezone.utc)
        # Calculate seconds until next minute boundary
        # If we're at second 0, we're already at minute boundary, no need to wait
        # If we're at second 30, we need to wait 30 seconds to reach minute 1:00
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
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the Alpaca trading environment."""
        try:
            # Get account info
            account_request = GetAccountRequest(account_name=self.account_name)
            account_result = await self.alpaca_service.get_account(account_request)
            account_info = dedent(f"""
                <account_info>
                {account_result.message}
                </account_info>
            """)
            
            # Get Positions
            positions_request = GetPositionsRequest(account_name=self.account_name)
            positions_result = await self.alpaca_service.get_positions(positions_request)
            positions_string = dedent(f"""
                <positions>
                {positions_result.message}
                </positions>
            """)
            
            # Wait until the next minute boundary for minute-level trading
            await self._wait_for_next_minute_boundary()
            
            data_request = GetDataRequest(symbol=self.symbol, data_type=self.data_type)
            data_result = await self.alpaca_service.get_data(data_request)
            
            bars = {}
            for symbol, data in data_result.extra.get("data", {}).items():
                bars[symbol] = data.get("bars", [])
            
            bars_string = ""
            for symbol, bars in bars.items():
                bars_string += f"Symbol: {symbol}\n"
                bars_string += "Bars:\n"
                for bar in bars:
                    bars_string += json.dumps(bar, indent=4)
                bars_string += "\n"
            
            data_string = dedent(f"""
                <data>
                {bars_string}
                </data>
            """)
            logger.info(f"| 📝 Data: {data_string}")
            
            state = dedent(f"""
                <state>
                {account_info}
                {positions_string}
                {data_string}
                </state>
            """)
            
            return {
                "state": state,
                "extra": {
                    "account": account_result.extra,
                    "positions": positions_result.extra,
                    "data": data_result.extra,
                }
            }
        except AuthenticationError as e:
            return {
                "state": str(e),
                "extra": {"error": str(e)}
            }
        except Exception as e:
            logger.error(f"Failed to get Alpaca state: {e}")
            return {
                "state": f"Failed to get Alpaca state: {str(e)}",
                "extra": {"error": str(e)}
            }
