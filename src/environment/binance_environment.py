"""Binance Trading Environment for AgentWorld - provides Binance trading operations as an environment."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(verbose=True)
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Type, List, Union
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.environment.types import Environment
from src.environment.server import ecp
from src.environment.binanceentry.service import BinanceService
from src.environment.binanceentry.exceptions import (
    AuthenticationError,
)
from src.environment.binanceentry.types import (
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
)
from src.utils import dedent, assemble_project_path
from src.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class BinanceEnvironment(Environment):
    """Binance Trading Environment that provides Binance trading operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="binance", description="The name of the Binance trading environment.")
    description: str = Field(default="Binance trading environment for real-time data and trading operations", description="The description of the Binance trading environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the Binance trading environment including account information, positions, and market data.",
            "interaction": dedent(f"""
                Guidelines for interacting with the Binance trading environment:
                - Always check account status before placing orders
                - Verify sufficient balance before buying
                - Default to perpetual futures trading for orders
                - Default to futures WebSocket for klines data
                - Monitor positions and orders regularly
            """),
        }
    }, description="The metadata of the Binance trading environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        account_name: str = None,
        symbol: Optional[Union[str, List[str]]] = None,
        data_type: Optional[Union[str, List[str]]] = None,
        binance_service: BinanceService = None,
        **kwargs
    ):
        """
        Initialize the Binance trading environment.
        
        Args:
            base_dir (str): Base directory for Binance operations
            account_name (str): Account name to use
            symbol (str or List[str]): Symbol(s) to trade (e.g., 'BTCUSDT', ['BTCUSDT', 'ETHUSDT'])
            data_type (str or List[str]): Data type(s) to retrieve (default: 'klines')
            binance_service (BinanceService): Binance service instance
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.account_name = account_name
        self.symbol = symbol
        self.data_type = data_type
        self.binance_service = binance_service
    
    async def initialize(self) -> None:
        """Initialize the Binance trading environment."""
        logger.info(f"| 🚀 Binance Trading Environment initialized at: {self.base_dir}")
    
    async def cleanup(self) -> None:
        """Cleanup the Binance trading environment."""
        logger.info("| 🧹 Binance Trading Environment cleanup completed")
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information (defaults to futures account).
        
        Returns:
            A string containing detailed account information including balances, permissions, and futures account info.
        """
        try:
            request = GetAccountRequest(account_name=self.account_name)
            result = await self.binance_service.get_account(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            account = result.extra["account"]
            result_text = dedent(f"""
                Account Information:
                Account Type: {account.get("account_type", "N/A")}
                Permissions: {', '.join(account.get("permissions", []))}
                
                Spot Balances:
                {self._format_balances(account.get("balances", []))}
                
                Futures Account:
                Total Wallet Balance: {account.get("futures_total_wallet_balance", "N/A")}
                Available Balance: {account.get("futures_available_balance", "N/A")}
                Total Unrealized Profit: {account.get("futures_total_unrealized_profit", "N/A")}
                
                Futures Assets:
                {self._format_futures_assets(account.get("futures_assets", []))}
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
    
    def _format_balances(self, balances: List[Dict]) -> str:
        """Format spot balances for display."""
        if not balances:
            return "  No spot balances"
        
        lines = []
        for balance in balances:
            free = float(balance.get("free", 0))
            locked = float(balance.get("locked", 0))
            total = free + locked
            if total > 0:
                lines.append(f"  {balance.get('asset', 'N/A')}: Free={free}, Locked={locked}, Total={total}")
        
        return "\n".join(lines) if lines else "  No non-zero balances"
    
    def _format_futures_assets(self, assets: List[Dict]) -> str:
        """Format futures assets for display."""
        if not assets:
            return "  No futures assets"
        
        lines = []
        for asset in assets:
            asset_name = asset.get("asset", "N/A")
            wallet_balance = asset.get("walletBalance", "N/A")
            available_balance = asset.get("availableBalance", "N/A")
            lines.append(f"  {asset_name}: Wallet={wallet_balance}, Available={available_balance}")
        
        return "\n".join(lines) if lines else "  No futures assets"
    
    async def get_assets(self, status: Optional[str] = None, asset_class: Optional[str] = None) -> Dict[str, Any]:
        """Get all assets information.
        
        Returns:
            A string containing detailed assets information including symbols, names, types, and status.
        """
        try:
            request = GetAssetsRequest(status=status, asset_class=asset_class)
            result = await self.binance_service.get_assets(request)
            
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
    
    async def get_positions(self, trade_type: Optional[TradeType] = TradeType.PERPETUAL) -> Dict[str, Any]:
        """Get all open positions (defaults to perpetual futures positions).
        
        Args:
            trade_type: Trade type to filter by (default: PERPETUAL for futures positions)
        
        Returns:
            A string containing detailed positions information including symbols, quantities, entry prices, and unrealized P&L.
        """
        try:
            request = GetPositionsRequest(account_name=self.account_name, trade_type=trade_type)
            result = await self.binance_service.get_positions(request)
            
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
                        position_side = pos.get("position_side", "N/A")
                        trade_type_str = pos.get("trade_type", "N/A")
                        
                        position_lines.append(
                            f"  {symbol} ({trade_type_str}): {position_amt:+.6f} @ Entry: {entry_price}, "
                            f"Mark: {mark_price}, Leverage: {leverage}x, Side: {position_side}, "
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
        """Get historical klines data from database.
        
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
                data_type="klines",
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            result = await self.binance_service.get_data(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": result.extra
                }
            
            # Data is organized by symbol: {symbol: {data_type: [...]}}
            data = result.extra.get("data", {})
            symbols = result.extra.get("symbols", [])
            data_types = result.extra.get("data_type", "klines")
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
        order_type: str = "MARKET",
        leverage: Optional[int] = None,
        position_side: Optional[str] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """Create an order (defaults to perpetual futures market order).
        
        Args:
            symbol: Symbol to trade (e.g., 'BTCUSDT', 'ETHUSDT')
            side: Order side: 'buy' or 'sell'
            qty: Quantity to trade
            price: Price for limit orders (required for LIMIT order type)
            order_type: Order type - 'MARKET' (default) or 'LIMIT'
            leverage: Leverage for perpetual futures (optional)
            time_in_force: Time in force for limit orders - 'GTC', 'IOC', 'FOK' (default: 'GTC')
            
        Returns:
            Dictionary with success, message, and order information
        """
        try:
            from src.environment.binanceentry.types import OrderType
            
            order_type_enum = OrderType.MARKET if order_type.upper() == "MARKET" else OrderType.LIMIT
            
            request = CreateOrderRequest(
                account_name=self.account_name,
                symbol=symbol,
                side=side,
                trade_type=TradeType.PERPETUAL,  # Default to perpetual futures
                order_type=order_type_enum,
                qty=qty,
                price=price,
                leverage=leverage,
                position_side=position_side,
                time_in_force=time_in_force
            )
            result = await self.binance_service.create_order(request)
            
            if not result.success:
                return {
                    "success": False,
                    "message": result.message,
                    "extra": {"error": result.message}
                }
            
            order = result.extra["order"]
            result_text = dedent(f"""
                Order submitted successfully:
                Order ID: {order["order_id"]}
                Symbol: {order["symbol"]}
                Side: {order["side"]}
                Quantity: {order["quantity"]}
                Status: {order["status"]}
                Order Type: {order["type"]}
                Trade Type: {order["trade_type"]}
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
        trade_type: Optional[TradeType] = TradeType.PERPETUAL,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
        order_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get orders for an account (defaults to perpetual futures orders).
        
        Args:
            trade_type: Trade type to filter by (default: PERPETUAL for futures orders)
            symbol: Optional symbol to filter by
            limit: Optional maximum number of orders to return
            order_id: Optional order ID to filter by
            
        Returns:
            Dictionary with success, message, and list of orders
        """
        try:
            from src.environment.binanceentry.types import GetOrdersRequest
            
            request = GetOrdersRequest(
                account_name=self.account_name,
                trade_type=trade_type,
                symbol=symbol,
                limit=limit,
                order_id=order_id
            )
            result = await self.binance_service.get_orders(request)
            
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
                    filled_qty = order.get("executed_qty", "0")
                    
                    order_lines.append(
                        f"  {order['symbol']}: {order['side']} {qty_display} "
                        f"(Status: {order['status']}, Filled: {filled_qty}, Trade Type: {order.get('trade_type', 'N/A')})"
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
        """Get a specific order by ID (defaults to perpetual futures order).
        
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
            result = await self.binance_service.get_order(request)
            
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
                Client Order ID: {order.get("client_order_id", "N/A")}
                Symbol: {order["symbol"]}
                Side: {order["side"]}
                Quantity: {qty_display}
                Executed Quantity: {order.get("executed_qty", "0")}
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
        """Cancel an order (defaults to perpetual futures order).
        
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
            result = await self.binance_service.cancel_order(request)
            
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
        """Cancel all orders for an account (defaults to perpetual futures orders).
        
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
            result = await self.binance_service.cancel_all_orders(request)
            
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
                description="Step the trading environment for perpetual futures trading.")
    async def step(self, 
                   symbol: str = "BTCUSDT", 
                   action: str = "HOLD",  # LONG, SHORT, HOLD
                   qty: float = 0.00,
                   leverage: Optional[int] = 10,
                   ) -> Dict[str, Any]:
        """Step the trading environment for perpetual futures trading.
        
        Args:
            symbol (str): Symbol to trade (e.g., 'BTCUSDT', 'ETHUSDT')
            action (str): Trading action for perpetual futures:
                - 'LONG': Open long position (BUY with LONG positionSide)
                - 'SHORT': Open short position (SELL with SHORT positionSide)
                - 'HOLD': Do nothing (default)
            qty (float): Quantity to trade (default: 0.00)
            leverage (int): Leverage for perpetual futures (default: 10x)
        Returns:
            Dictionary with success, message, and order information
        """
        action = action.upper()
        try:
            if action == "HOLD":
                return {
                    "success": True,
                    "message": "HOLD action performed successfully. No order submitted.",
                    "extra": {}
                }
            elif action == "LONG":
                # Open long position: BUY with LONG positionSide
                return await self.create_order(
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    leverage=leverage,
                    position_side="LONG"
                )
            elif action == "SHORT":
                # Open short position: SELL with SHORT positionSide
                return await self.create_order(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    leverage=leverage,
                    position_side="SHORT"
                )
            else:
                return {
                    "success": False,
                    "message": f"Invalid action: {action}. Must be LONG, SHORT, or HOLD",
                    "extra": {"error": f"Invalid action: {action}"}
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
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the Binance trading environment."""
        try:
            # Get account info (futures account info is included)
            account_request = GetAccountRequest(account_name=self.account_name)
            account_result = await self.binance_service.get_account(account_request)
            account_info = dedent(f"""
                <account_info>
                {account_result.message}
                </account_info>
            """)
            
            # Get Positions (default to perpetual futures)
            positions_request = GetPositionsRequest(account_name=self.account_name, trade_type=TradeType.PERPETUAL)
            positions_result = await self.binance_service.get_positions(positions_request)
            positions_string = dedent(f"""
                <positions>
                {positions_result.message}
                </positions>
            """)
            
            # Wait until the next minute boundary for minute-level trading
            await self._wait_for_next_minute_boundary()
            
            data_request = GetDataRequest(symbol=self.symbol, data_type="klines")
            data_result = await self.binance_service.get_data(data_request)
            
            klines = {}
            for symbol, data in data_result.extra.get("data", {}).items():
                klines[symbol] = data.get("klines", [])
            
            klines_string = ""
            for symbol, klines_list in klines.items():
                klines_string += f"Symbol: {symbol}\n"
                klines_string += "Klines:\n"
                for kline in klines_list:
                    klines_string += json.dumps(kline, indent=4)
                klines_string += "\n"
            
            data_string = dedent(f"""
                <data>
                {klines_string}
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
            logger.error(f"Failed to get Binance state: {e}")
            return {
                "state": f"Failed to get Binance state: {str(e)}",
                "extra": {"error": str(e)}
            }

