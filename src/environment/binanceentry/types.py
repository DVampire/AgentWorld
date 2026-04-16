"""Binance trading service data types."""

from typing import Optional, List, Dict, Any, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field


class DataStreamType(str, Enum):
    """Data type."""
    KLINES = "klines"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class TradeType(str, Enum):
    """Trade type - spot or perpetual futures."""
    SPOT = "spot"
    PERPETUAL = "perpetual"


class AccountInfo(BaseModel):
    """Binance account model."""
    api_key: str
    secret_key: str
    name: str


class GetAccountRequest(BaseModel):
    """Request for getting account information."""
    account_name: str


class GetAssetsRequest(BaseModel):
    """Request for getting assets (cryptocurrency symbols)."""
    status: Optional[str] = Field(None, description="Filter by asset status")
    asset_class: Optional[str] = Field(None, description="Filter by asset class (not used for crypto)")


class GetPositionsRequest(BaseModel):
    """Request for getting positions."""
    account_name: str
    trade_type: Optional[TradeType] = Field(None, description="Filter by trade type: 'spot' or 'perpetual'")


class GetDataRequest(BaseModel):
    """Request for getting historical data from database."""
    symbol: Union[str, List[str]] = Field(description="Symbol(s) to query (e.g., 'BTCUSDT', 'ETHUSDT')")
    data_type: Literal["klines"] = Field(default="klines", description="Type of data to retrieve: klines")
    start_date: Optional[str] = Field(None, description="Start date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-01 00:00:00'). If not provided, returns latest data.")
    end_date: Optional[str] = Field(None, description="End date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-31 23:59:59'). If not provided, returns latest data.")
    limit: Optional[int] = Field(None, description="Maximum number of rows to return (optional). If no date range provided, returns latest N records.")


class CreateOrderRequest(BaseModel):
    """Request for creating an order."""
    account_name: str = Field(description="Account name to use for the order")
    symbol: str = Field(description="Symbol to trade (e.g., 'BTCUSDT', 'ETHUSDT')")
    side: Literal["buy", "sell"] = Field(description="Order side: 'buy' or 'sell'")
    trade_type: TradeType = Field(description="Trade type: 'spot' for spot market order, 'perpetual' for perpetual futures order")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type: 'MARKET' for market order, 'LIMIT' for limit order")
    qty: Optional[float] = Field(None, description="Quantity to trade")
    price: Optional[float] = Field(None, description="Price for limit order (required for LIMIT order type)")
    leverage: Optional[int] = Field(None, description="Leverage for perpetual futures (optional, default is 1x)")
    position_side: Optional[Literal["LONG", "SHORT", "BOTH"]] = Field(None, description="Position side for perpetual futures: 'LONG' (open long/close short), 'SHORT' (open short/close long), 'BOTH' (hedge mode). Default: 'BOTH'")
    time_in_force: Optional[Literal["GTC", "IOC", "FOK"]] = Field(default="GTC", description="Time in force for limit orders: 'GTC' (good till cancel), 'IOC' (immediate or cancel), 'FOK' (fill or kill)")


class GetOrdersRequest(BaseModel):
    """Request for getting orders."""
    account_name: str = Field(description="Account name")
    trade_type: Optional[TradeType] = Field(None, description="Filter by trade type: 'spot' or 'perpetual'")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    limit: Optional[int] = Field(None, description="Maximum number of orders to return")
    order_id: Optional[int] = Field(None, description="Filter by order ID")


class GetOrderRequest(BaseModel):
    """Request for getting a specific order."""
    account_name: str = Field(description="Account name")
    order_id: str = Field(description="Order ID")
    symbol: str = Field(description="Symbol")
    trade_type: TradeType = Field(description="Trade type: 'spot' or 'perpetual'")


class CancelOrderRequest(BaseModel):
    """Request for canceling an order."""
    account_name: str = Field(description="Account name")
    order_id: str = Field(description="Order ID to cancel")
    symbol: str = Field(description="Symbol")
    trade_type: TradeType = Field(description="Trade type: 'spot' or 'perpetual'")


class CancelAllOrdersRequest(BaseModel):
    """Request for canceling all orders."""
    account_name: str = Field(description="Account name")
    symbol: Optional[str] = Field(None, description="Symbol to cancel orders for (if None, cancels all orders)")
    trade_type: Optional[TradeType] = Field(None, description="Trade type: 'spot' or 'perpetual'")

