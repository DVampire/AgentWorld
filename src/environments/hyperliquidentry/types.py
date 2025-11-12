"""Hyperliquid trading service data types."""

from typing import Optional, List, Dict, Any, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field


class DataStreamType(str, Enum):
    """Data type."""
    CANDLE = "candle"  # OHLCV candlestick data (renamed from KLINES)
    TRADES = "trades"  # Trade execution data
    L2BOOK = "l2Book"  # Level 2 order book data


class OrderType(str, Enum):
    """Order type."""
    MARKET = "Market"
    LIMIT = "Limit"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "B"
    SELL = "A"


class TradeType(str, Enum):
    """Trade type - perpetual futures only for Hyperliquid."""
    PERPETUAL = "perpetual"


class AccountInfo(BaseModel):
    """Hyperliquid account model."""
    name: str
    address: str  # Wallet address (e.g., "0x...")
    private_key: Optional[str] = None  # Private key for signing (optional, required for trading)


class GetAccountRequest(BaseModel):
    """Request for getting account information."""
    account_name: str


class GetAssetsRequest(BaseModel):
    """Request for getting assets (trading symbols)."""
    status: Optional[str] = Field(None, description="Filter by asset status")
    asset_class: Optional[str] = Field(None, description="Filter by asset class")


class GetPositionsRequest(BaseModel):
    """Request for getting positions."""
    account_name: str
    trade_type: Optional[TradeType] = Field(None, description="Filter by trade type: 'perpetual'")


class GetDataRequest(BaseModel):
    """Request for getting historical data from database."""
    symbol: Union[str, List[str]] = Field(description="Symbol(s) to query (e.g., 'BTC', 'ETH')")
    data_type: Literal["candle", "trades", "l2Book"] = Field(default="candle", description="Type of data to retrieve: 'candle' for OHLCV data, 'trades' for trade data, 'l2Book' for order book data")
    start_date: Optional[str] = Field(None, description="Start date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-01 00:00:00'). If not provided, returns latest data.")
    end_date: Optional[str] = Field(None, description="End date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-31 23:59:59'). If not provided, returns latest data.")
    limit: Optional[int] = Field(None, description="Maximum number of rows to return (optional). If no date range provided, returns latest N records.")


class CreateOrderRequest(BaseModel):
    """Request for creating an order."""
    account_name: str = Field(description="Account name to use for the order")
    symbol: str = Field(description="Symbol to trade (e.g., 'BTC', 'ETH')")
    side: Literal["buy", "sell"] = Field(description="Order side: 'buy' or 'sell'")
    trade_type: TradeType = Field(default=TradeType.PERPETUAL, description="Trade type: 'perpetual' for perpetual futures order")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type: 'Market' for market order, 'Limit' for limit order")
    qty: Optional[float] = Field(None, description="Quantity to trade")
    price: Optional[float] = Field(None, description="Price for limit order (required for LIMIT order type)")
    leverage: Optional[int] = Field(None, description="Leverage for perpetual futures (optional)")
    time_in_force: Optional[Literal["Gtc", "Ioc", "Alo"]] = Field(default="Gtc", description="Time in force for limit orders: 'Gtc' (good till cancel), 'Ioc' (immediate or cancel), 'Alo' (add liquidity only)")
    stop_loss_price: Optional[float] = Field(None, description="Stop loss trigger price (optional). If provided, creates a stop loss order after main order.")
    take_profit_price: Optional[float] = Field(None, description="Take profit trigger price (optional). If provided, creates a take profit order after main order.")


class GetOrdersRequest(BaseModel):
    """Request for getting orders."""
    account_name: str = Field(description="Account name")
    trade_type: Optional[TradeType] = Field(None, description="Filter by trade type: 'perpetual'")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    limit: Optional[int] = Field(None, description="Maximum number of orders to return")
    order_id: Optional[str] = Field(None, description="Filter by order ID")


class GetOrderRequest(BaseModel):
    """Request for getting a specific order."""
    account_name: str = Field(description="Account name")
    order_id: str = Field(description="Order ID")
    symbol: str = Field(description="Symbol")
    trade_type: TradeType = Field(description="Trade type: 'perpetual'")


class CancelOrderRequest(BaseModel):
    """Request for canceling an order."""
    account_name: str = Field(description="Account name")
    order_id: str = Field(description="Order ID to cancel")
    symbol: str = Field(description="Symbol")
    trade_type: TradeType = Field(description="Trade type: 'perpetual'")


class CancelAllOrdersRequest(BaseModel):
    """Request for canceling all orders."""
    account_name: str = Field(description="Account name")
    symbol: Optional[str] = Field(None, description="Symbol to cancel orders for (if None, cancels all orders)")
    trade_type: Optional[TradeType] = Field(None, description="Trade type: 'perpetual'")


class CloseOrderRequest(BaseModel):
    """Request for closing a position."""
    account_name: str = Field(description="Account name to use for closing the position")
    symbol: str = Field(description="Symbol to close position for (e.g., 'BTC', 'ETH')")
    side: Literal["buy", "sell"] = Field(description="Order side to close position: 'buy' to close SHORT, 'sell' to close LONG")
    size: float = Field(description="Position size to close (in base units, e.g., 0.1 BTC)")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type: 'Market' for market order, 'Limit' for limit order")
    price: Optional[float] = Field(None, description="Price for limit order (required for LIMIT order type)")
    trade_type: TradeType = Field(default=TradeType.PERPETUAL, description="Trade type: 'perpetual' for perpetual futures")

