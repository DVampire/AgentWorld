"""Alpaca trading service data types."""

from typing import Optional, List, Dict, Any, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field


class DataStreamType(str, Enum):
    """Data type."""
    QUOTES = "quotes"
    TRADES = "trades"
    BARS = "bars"
    ORDERBOOKS = "orderbooks"
    NEWS = "news"


class AccountInfo(BaseModel):
    """Alpaca account model."""
    api_key: str
    secret_key: str
    name: str

class GetAccountRequest(BaseModel):
    """Request for getting account information."""
    account_name: str

class GetAssetsRequest(BaseModel):
    """Request for getting assets."""
    status: Optional[str] = Field(None, description="Filter by asset status")
    asset_class: Optional[str] = Field(None, description="Filter by asset class")

class GetPositionsRequest(BaseModel):
    """Request for getting positions."""
    account_name: str

class GetDataRequest(BaseModel):
    """Request for getting historical data from database."""
    symbol: Union[str, List[str]] = Field(description="Symbol(s) to query (e.g., 'BTC/USD', 'AAPL', or ['BTC/USD', 'AAPL'])")
    data_type: Union[Literal["quotes", "trades", "bars", "orderbooks", "news"], List[Literal["quotes", "trades", "bars", "orderbooks", "news"]]] = Field(
        description="Type(s) of data to retrieve: quotes, trades, bars, orderbooks (crypto only), or news. Can be a single type or a list of types."
    )
    start_date: Optional[str] = Field(None, description="Start date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-01 00:00:00'). If not provided, returns latest data.")
    end_date: Optional[str] = Field(None, description="End date in format 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-31 23:59:59'). If not provided, returns latest data.")
    limit: Optional[int] = Field(None, description="Maximum number of rows to return (optional). If no date range provided, returns latest N records.")

class CreateOrderRequest(BaseModel):
    """Request for creating a market order."""
    account_name: str = Field(description="Account name to use for the order")
    symbol: str = Field(description="Symbol to trade (e.g., 'AAPL', 'BTC/USD')")
    qty: Optional[float] = Field(None, description="Quantity to trade (for notional orders, use None)")
    notional: Optional[float] = Field(None, description="Notional value to trade (for fractional shares, use this instead of qty)")
    side: Literal["buy", "sell"] = Field(description="Order side: 'buy' or 'sell'")
    time_in_force: Literal["day", "gtc", "opg", "cls", "ioc", "fok"] = Field(
        default="day", 
        description="Time in force: 'day' (default), 'gtc' (good till canceled), 'opg' (opening), 'cls' (closing), 'ioc' (immediate or cancel), 'fok' (fill or kill)"
    )

class GetOrdersRequest(BaseModel):
    """Request for getting orders."""
    account_name: str = Field(description="Account name")
    status: Optional[Literal["open", "closed", "all"]] = Field(
        default="open",
        description="Filter by order status: 'open', 'closed', or 'all'"
    )
    limit: Optional[int] = Field(None, description="Maximum number of orders to return")
    after: Optional[str] = Field(None, description="Return orders after this date (ISO format)")
    until: Optional[str] = Field(None, description="Return orders until this date (ISO format)")
    direction: Optional[Literal["asc", "desc"]] = Field(default="desc", description="Sort direction: 'asc' or 'desc'")

class GetOrderRequest(BaseModel):
    """Request for getting a specific order."""
    account_name: str = Field(description="Account name")
    order_id: str = Field(description="Order ID")

class CancelOrderRequest(BaseModel):
    """Request for canceling an order."""
    account_name: str = Field(description="Account name")
    order_id: str = Field(description="Order ID to cancel")

class CancelAllOrdersRequest(BaseModel):
    """Request for canceling all orders."""
    account_name: str = Field(description="Account name")