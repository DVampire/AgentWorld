"""Alpaca Trading Environment for AgentWorld - provides Alpaca trading operations as an environment."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(verbose=True)

from typing import Any, Dict, Optional, Type, List
from pydantic import BaseModel, Field, SecretStr, ConfigDict
from decimal import Decimal
from datetime import datetime

from src.logger import logger
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol import ecp
from src.environments.alpacaentry.service import AlpacaService
from src.environments.alpacaentry.exceptions import (
    AuthenticationError,
    NotFoundError,
    OrderError,
    DataError,
    InsufficientFundsError,
    InvalidSymbolError,
    MarketClosedError
)
from src.environments.alpacaentry.types import (
    GetAccountRequest,
    GetPositionsRequest,
    GetPositionRequest,
    GetBarsRequest,
    GetLatestBarsRequest,
    GetQuotesRequest,
    GetLatestQuotesRequest,
    GetTradesRequest,
    GetLatestTradesRequest,
    SubmitOrderRequest,
    GetOrdersRequest,
    GetOrderRequest,
    CancelOrderRequest,
    CancelAllOrdersRequest,
    ClosePositionRequest,
    CloseAllPositionsRequest
)
from src.utils import dedent, get_env, assemble_project_path

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
        api_key: Optional[SecretStr] = None,
        secret_key: Optional[SecretStr] = None,
        data_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Alpaca paper trading environment.
        
        Args:
            base_dir (str): Base directory for Alpaca operations
            api_key (Optional[SecretStr]): Alpaca API key
            secret_key (Optional[SecretStr]): Alpaca secret key
            data_url (Optional[str]): Data API base URL
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.api_key = (api_key or get_env("ALPACA_PAPER_TRAING_API_KEY")).get_secret_value()
        self.secret_key = (secret_key or get_env("ALPACA_PAPER_TRAING_SECRET_KEY")).get_secret_value()
        
        # Initialize Alpaca paper trading service
        self.alpaca_service = AlpacaService(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )
        
    async def initialize(self) -> None:
        """Initialize the Alpaca trading environment."""
        await self.alpaca_service.initialize()
        logger.info(f"| 🚀 Alpaca Trading Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the Alpaca trading environment."""
        await self.alpaca_service.cleanup()
        logger.info("| 🧹 Alpaca Trading Environment cleanup completed")

    # --------------- Account Operations ---------------
    @ecp.action(name="get_account", 
                type="Alpaca Trading", 
                description="Get account information including buying power, cash, and portfolio value")
    async def get_account(self) -> str:
        """Get account information.
        
        Returns:
            A string containing detailed account information including buying power, cash, portfolio value, and account status.
        """
        try:
            request = GetAccountRequest()
            result = await self.alpaca_service.get_account(request)
            
            if not result.success or not result.account:
                return f"| ❌ {result.message}"
            
            account = result.account
            result_text = dedent(f"""
                Account Information:
                Account Number: {account.account_number}
                Status: {account.status}
                Currency: {account.currency}
                Buying Power: ${account.buying_power:,.2f}
                Cash: ${account.cash:,.2f}
                Portfolio Value: ${account.portfolio_value:,.2f}
                Equity: ${account.equity:,.2f}
                Pattern Day Trader: {account.pattern_day_trader}
                Trading Blocked: {account.trading_blocked}
                Shorting Enabled: {account.shorting_enabled}
                Day Trade Count: {account.daytrade_count}
                """)
            return result_text
        except AuthenticationError as e:
            return str(e)
        except Exception as e:
            return f"Failed to get account information: {str(e)}"

    # --------------- Position Operations ---------------
    @ecp.action(name="get_positions", 
                type="Alpaca Trading", 
                description="Get all current positions")
    async def get_positions(self) -> str:
        """Get all current positions.
        
        Returns:
            A string containing detailed information about all current positions including symbol, quantity, market value, and unrealized P&L.
        """
        try:
            request = GetPositionsRequest()
            result = await self.alpaca_service.get_positions(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            if not result.positions:
                return "No positions found."
            
            positions_text = "Current Positions:\n"
            for pos in result.positions:
                positions_text += dedent(f"""
                    Symbol: {pos.symbol}
                    Quantity: {pos.qty}
                    Side: {pos.side}
                    Market Value: ${pos.market_value}
                    Cost Basis: ${pos.cost_basis}
                    Unrealized P&L: ${pos.unrealized_pl} ({pos.unrealized_plpc}%)
                    Current Price: ${pos.current_price}
                    Change Today: {pos.change_today}%
                    ---
                """)
            
            return positions_text
        except Exception as e:
            return f"Failed to get positions: {str(e)}"

    @ecp.action(name="get_position", 
                type="Alpaca Trading", 
                description="Get position for a specific symbol")
    async def get_position(self, symbol: str) -> str:
        """Get position for a specific symbol.
        
        Args:
            symbol (str): The stock symbol to get position for.
        
        Returns:
            A string containing detailed position information for the specified symbol.
        """
        try:
            request = GetPositionRequest(symbol=symbol.upper())
            result = await self.alpaca_service.get_position(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            if not result.position:
                return f"No position found for {symbol.upper()}"
            
            pos = result.position
            result_text = dedent(f"""
                Position for {pos.symbol}:
                Quantity: {pos.qty}
                Side: {pos.side}
                Market Value: ${pos.market_value}
                Cost Basis: ${pos.cost_basis}
                Unrealized P&L: ${pos.unrealized_pl} ({pos.unrealized_plpc}%)
                Current Price: ${pos.current_price}
                Change Today: {pos.change_today}%
                """)
            return result_text
        except Exception as e:
            return f"Failed to get position for {symbol}: {str(e)}"

    # --------------- Market Data Operations ---------------
    @ecp.action(name="get_latest_bars", 
                type="Alpaca Trading", 
                description="Get latest bar data for symbols")
    async def get_latest_bars(self, symbols: List[str]) -> str:
        """Get latest bar data for symbols.
        
        Args:
            symbols (List[str]): List of stock symbols to get bar data for.
        
        Returns:
            A string containing latest bar data including OHLCV information.
        """
        try:
            symbols_upper = [s.upper() for s in symbols]
            request = GetLatestBarsRequest(symbols=symbols_upper)
            result = await self.alpaca_service.get_latest_bars(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            if not result.bars:
                return "No bar data found."
            
            bars_text = "Latest Bar Data:\n"
            for symbol, bar in result.bars.items():
                bars_text += dedent(f"""
                    Symbol: {symbol}
                    Timestamp: {bar.timestamp}
                    Open: ${bar.open}
                    High: ${bar.high}
                    Low: ${bar.low}
                    Close: ${bar.close}
                    Volume: {bar.volume:,}
                    ---
                """)
            
            return bars_text
        except Exception as e:
            return f"Failed to get latest bars: {str(e)}"

    @ecp.action(name="get_historical_bars", 
                type="Alpaca Trading", 
                description="Get historical bar data for symbols")
    async def get_historical_bars(
        self, 
        symbols: List[str], 
        timeframe: str = "1Min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """Get historical bar data for symbols.
        
        Args:
            symbols (List[str]): List of stock symbols to get bar data for.
            timeframe (str): Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day). Defaults to "1Min".
            start (Optional[str]): Start time in ISO format. Defaults to None.
            end (Optional[str]): End time in ISO format. Defaults to None.
            limit (Optional[int]): Maximum number of bars. Defaults to None.
        
        Returns:
            A string containing historical bar data.
        """
        try:
            symbols_upper = [s.upper() for s in symbols]
            
            start_dt = datetime.fromisoformat(start) if start else None
            end_dt = datetime.fromisoformat(end) if end else None
            
            request = GetBarsRequest(
                symbols=symbols_upper,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
                limit=limit
            )
            result = await self.alpaca_service.get_bars(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            if not result.bars:
                return "No historical bar data found."
            
            bars_text = f"Historical Bar Data ({timeframe}):\n"
            for symbol, bars in result.bars.items():
                bars_text += f"\n{symbol} ({len(bars)} bars):\n"
                for bar in bars[-5:]:  # Show last 5 bars
                    bars_text += dedent(f"""
                        Time: {bar.timestamp}
                        OHLC: ${bar.open} / ${bar.high} / ${bar.low} / ${bar.close}
                        Volume: {bar.volume:,}
                    """)
                if len(bars) > 5:
                    bars_text += f"... and {len(bars) - 5} more bars\n"
            
            return bars_text
        except Exception as e:
            return f"Failed to get historical bars: {str(e)}"

    @ecp.action(name="get_latest_quotes", 
                type="Alpaca Trading", 
                description="Get latest quote data for symbols")
    async def get_latest_quotes(self, symbols: List[str]) -> str:
        """Get latest quote data for symbols.
        
        Args:
            symbols (List[str]): List of stock symbols to get quote data for.
        
        Returns:
            A string containing latest quote data including bid/ask prices and sizes.
        """
        try:
            symbols_upper = [s.upper() for s in symbols]
            request = GetLatestQuotesRequest(symbols=symbols_upper)
            result = await self.alpaca_service.get_latest_quotes(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            if not result.quotes:
                return "No quote data found."
            
            quotes_text = "Latest Quote Data:\n"
            for symbol, quote in result.quotes.items():
                quotes_text += dedent(f"""
                    Symbol: {symbol}
                    Timestamp: {quote.timestamp}
                    Bid: ${quote.bid_price} (Size: {quote.bid_size:,})
                    Ask: ${quote.ask_price} (Size: {quote.ask_size:,})
                    Spread: ${quote.ask_price - quote.bid_price:.4f}
                    ---
                """)
            
            return quotes_text
        except Exception as e:
            return f"Failed to get latest quotes: {str(e)}"

    @ecp.action(name="get_latest_trades", 
                type="Alpaca Trading", 
                description="Get latest trade data for symbols")
    async def get_latest_trades(self, symbols: List[str]) -> str:
        """Get latest trade data for symbols.
        
        Args:
            symbols (List[str]): List of stock symbols to get trade data for.
        
        Returns:
            A string containing latest trade data including price and size.
        """
        try:
            symbols_upper = [s.upper() for s in symbols]
            request = GetLatestTradesRequest(symbols=symbols_upper)
            result = await self.alpaca_service.get_latest_trades(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            if not result.trades:
                return "No trade data found."
            
            trades_text = "Latest Trade Data:\n"
            for symbol, trade in result.trades.items():
                trades_text += dedent(f"""
                    Symbol: {symbol}
                    Timestamp: {trade.timestamp}
                    Price: ${trade.price}
                    Size: {trade.size:,}
                    Exchange: {trade.exchange or 'N/A'}
                    ---
                """)
            
            return trades_text
        except Exception as e:
            return f"Failed to get latest trades: {str(e)}"

    # --------------- Order Operations ---------------
    @ecp.action(name="submit_order", 
                type="Alpaca Trading", 
                description="Submit a trading order")
    async def submit_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[int] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        extended_hours: bool = False
    ) -> str:
        """Submit a trading order.
        
        Args:
            symbol (str): Stock symbol to trade.
            side (str): Order side ("buy" or "sell").
            qty (Optional[int]): Number of shares to trade.
            notional (Optional[float]): Dollar amount to trade (alternative to qty).
            order_type (str): Order type ("market", "limit", "stop", "stop_limit", "trailing_stop").
            time_in_force (str): Time in force ("day", "gtc", "ioc", "fok").
            limit_price (Optional[float]): Limit price for limit orders.
            stop_price (Optional[float]): Stop price for stop orders.
            extended_hours (bool): Whether to allow extended hours trading.
        
        Returns:
            A string indicating the success or failure of the order submission.
        """
        try:
            if not qty and not notional:
                return "Either qty or notional must be specified."
            
            if qty and notional:
                return "Cannot specify both qty and notional."
            
            request = SubmitOrderRequest(
                symbol=symbol.upper(),
                side=side.lower(),
                qty=qty,
                notional=Decimal(str(notional)) if notional else None,
                type=order_type.lower(),
                time_in_force=time_in_force.lower(),
                limit_price=Decimal(str(limit_price)) if limit_price else None,
                stop_price=Decimal(str(stop_price)) if stop_price else None,
                extended_hours=extended_hours
            )
            
            result = await self.alpaca_service.submit_order(request)
            
            if not result.success or not result.order:
                return f"| ❌ {result.message}"
            
            order = result.order
            order_text = dedent(f"""
                Order Submitted Successfully:
                Order ID: {order.id}
                Symbol: {order.symbol}
                Side: {order.side.upper()}
                Quantity: {order.qty or 'N/A'}
                Notional: ${order.notional or 'N/A'}
                Type: {order.type.upper()}
                Status: {order.status.upper()}
                Time in Force: {order.time_in_force.upper()}
                Submitted At: {order.submitted_at}
                """)
            
            return order_text
        except InsufficientFundsError as e:
            return f"| ❌ Insufficient funds: {str(e)}"
        except InvalidSymbolError as e:
            return f"| ❌ Invalid symbol: {str(e)}"
        except MarketClosedError as e:
            return f"| ❌ Market closed: {str(e)}"
        except OrderError as e:
            return f"| ❌ Order error: {str(e)}"
        except Exception as e:
            return f"Failed to submit order: {str(e)}"

    @ecp.action(name="get_orders", 
                type="Alpaca Trading", 
                description="Get order history")
    async def get_orders(
        self,
        status: str = "open",
        limit: Optional[int] = None
    ) -> str:
        """Get order history.
        
        Args:
            status (str): Order status filter ("open", "closed", "all"). Defaults to "open".
            limit (Optional[int]): Maximum number of orders to return. Defaults to None.
        
        Returns:
            A string containing order history information.
        """
        try:
            request = GetOrdersRequest(status=status, limit=limit)
            result = await self.alpaca_service.get_orders(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            if not result.orders:
                return f"No {status} orders found."
            
            orders_text = f"{status.upper()} Orders:\n"
            for order in result.orders:
                orders_text += dedent(f"""
                    Order ID: {order.id}
                    Symbol: {order.symbol}
                    Side: {order.side.upper()}
                    Quantity: {order.qty or 'N/A'}
                    Filled: {order.filled_qty}
                    Type: {order.type.upper()}
                    Status: {order.status.upper()}
                    Submitted: {order.submitted_at}
                    ---
                """)
            
            return orders_text
        except Exception as e:
            return f"Failed to get orders: {str(e)}"

    @ecp.action(name="get_order", 
                type="Alpaca Trading", 
                description="Get specific order details")
    async def get_order(self, order_id: str) -> str:
        """Get specific order details.
        
        Args:
            order_id (str): The order ID to get details for.
        
        Returns:
            A string containing detailed order information.
        """
        try:
            request = GetOrderRequest(order_id=order_id)
            result = await self.alpaca_service.get_order(request)
            
            if not result.success or not result.order:
                return f"| ❌ {result.message}"
            
            order = result.order
            order_text = dedent(f"""
                Order Details:
                Order ID: {order.id}
                Client Order ID: {order.client_order_id}
                Symbol: {order.symbol}
                Side: {order.side.upper()}
                Quantity: {order.qty or 'N/A'}
                Filled Quantity: {order.filled_qty}
                Filled Avg Price: ${order.filled_avg_price or 'N/A'}
                Type: {order.type.upper()}
                Status: {order.status.upper()}
                Time in Force: {order.time_in_force.upper()}
                Limit Price: ${order.limit_price or 'N/A'}
                Stop Price: ${order.stop_price or 'N/A'}
                Submitted At: {order.submitted_at}
                Updated At: {order.updated_at}
                """)
            
            return order_text
        except NotFoundError as e:
            return f"| ❌ Order not found: {str(e)}"
        except Exception as e:
            return f"Failed to get order: {str(e)}"

    @ecp.action(name="cancel_order", 
                type="Alpaca Trading", 
                description="Cancel a specific order")
    async def cancel_order(self, order_id: str) -> str:
        """Cancel a specific order.
        
        Args:
            order_id (str): The order ID to cancel.
        
        Returns:
            A string indicating the success or failure of the order cancellation.
        """
        try:
            request = CancelOrderRequest(order_id=order_id)
            result = await self.alpaca_service.cancel_order(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            return f"| ✅ Order {order_id} canceled successfully"
        except NotFoundError as e:
            return f"| ❌ Order not found: {str(e)}"
        except Exception as e:
            return f"Failed to cancel order: {str(e)}"

    @ecp.action(name="cancel_all_orders", 
                type="Alpaca Trading", 
                description="Cancel all open orders")
    async def cancel_all_orders(self) -> str:
        """Cancel all open orders.
        
        Returns:
            A string indicating the success or failure of canceling all orders.
        """
        try:
            request = CancelAllOrdersRequest()
            result = await self.alpaca_service.cancel_all_orders(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            return f"| ✅ Canceled {result.canceled_count} orders successfully"
        except Exception as e:
            return f"Failed to cancel all orders: {str(e)}"

    # --------------- Position Management Operations ---------------
    @ecp.action(name="close_position", 
                type="Alpaca Trading", 
                description="Close a position for a specific symbol")
    async def close_position(
        self,
        symbol: str,
        qty: Optional[int] = None,
        percentage: Optional[float] = None
    ) -> str:
        """Close a position for a specific symbol.
        
        Args:
            symbol (str): Stock symbol to close position for.
            qty (Optional[int]): Specific quantity to close.
            percentage (Optional[float]): Percentage of position to close.
        
        Returns:
            A string indicating the success or failure of closing the position.
        """
        try:
            if qty and percentage:
                return "Cannot specify both qty and percentage."
            
            request = ClosePositionRequest(
                symbol=symbol.upper(),
                qty=qty,
                percentage=Decimal(str(percentage)) if percentage else None
            )
            
            result = await self.alpaca_service.close_position(request)
            
            if not result.success or not result.order:
                return f"| ❌ {result.message}"
            
            order = result.order
            return dedent(f"""
                Position Closed Successfully:
                Symbol: {symbol.upper()}
                Order ID: {order.id}
                Status: {order.status.upper()}
                """)
        except NotFoundError as e:
            return f"| ❌ Position not found: {str(e)}"
        except Exception as e:
            return f"Failed to close position: {str(e)}"

    @ecp.action(name="close_all_positions", 
                type="Alpaca Trading", 
                description="Close all positions")
    async def close_all_positions(self, cancel_orders: bool = True) -> str:
        """Close all positions.
        
        Args:
            cancel_orders (bool): Whether to cancel open orders first. Defaults to True.
        
        Returns:
            A string indicating the success or failure of closing all positions.
        """
        try:
            request = CloseAllPositionsRequest(cancel_orders=cancel_orders)
            result = await self.alpaca_service.close_all_positions(request)
            
            if not result.success:
                return f"| ❌ {result.message}"
            
            return f"| ✅ Closed {len(result.orders)} positions successfully"
        except Exception as e:
            return f"Failed to close all positions: {str(e)}"

    # --------------- Environment Interface Methods ---------------
    async def get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "type": "alpaca_trading",
            "base_url": self.base_url,
            "is_paper_trading": "paper" in self.base_url,
            "authenticated": self.alpaca_service is not None,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if self.alpaca_service is None:
                return {"status": "unhealthy", "error": "Not initialized"}
            
            # Test service access by getting account info
            request = GetAccountRequest()
            result = await self.alpaca_service.get_account(request)
            
            if not result.success:
                return {"status": "unhealthy", "error": result.message}
            
            return {
                "status": "healthy",
                "account_number": result.account.account_number,
                "account_status": result.account.status,
                "is_paper_trading": "paper" in self.base_url,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the Alpaca trading environment."""
        try:
            # Get account info
            account_request = GetAccountRequest()
            account_result = await self.alpaca_service.get_account(account_request)
            
            # Get positions
            positions_request = GetPositionsRequest()
            positions_result = await self.alpaca_service.get_positions(positions_request)
            
            # Get open orders
            orders_request = GetOrdersRequest(status="open")
            orders_result = await self.alpaca_service.get_orders(orders_request)
            
            state_text = dedent(f"""
                Alpaca Paper Trading Environment:
                Account Status: {account_result.account.status if account_result.success else 'Unknown'}
                Buying Power: ${account_result.account.buying_power:,.2f} if account_result.success else 'Unknown'
                Cash: ${account_result.account.cash:,.2f} if account_result.success else 'Unknown'
                Portfolio Value: ${account_result.account.portfolio_value:,.2f} if account_result.success else 'Unknown'
                Positions: {len(positions_result.positions) if positions_result.success else 0}
                Open Orders: {len(orders_result.orders) if orders_result.success else 0}
                Mode: Paper Trading
            """)
            
            return {
                "state": state_text,
                "extra": {
                    "account": account_result.account if account_result.success else None,
                    "positions": positions_result.positions if positions_result.success else [],
                    "open_orders": orders_result.orders if orders_result.success else [],
                },
            }
        except Exception as e:
            logger.error(f"Failed to get Alpaca state: {e}")
            return {
                "state": str(e),
                "extra": {
                    "error": str(e),
                },
            }
