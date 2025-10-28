"""Alpaca trading service data types."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field


@dataclass
class AlpacaAccount:
    """Alpaca account information."""
    id: str
    account_number: str
    status: str
    currency: str
    buying_power: Decimal
    cash: Decimal
    portfolio_value: Decimal
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    trade_suspended_by_user: bool
    multiplier: int
    shorting_enabled: bool
    equity: Decimal
    last_equity: Decimal
    long_market_value: Decimal
    short_market_value: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    last_maintenance_margin: Decimal
    sma: Decimal
    daytrade_count: int


@dataclass
class AlpacaPosition:
    """Alpaca position information."""
    asset_id: str
    symbol: str
    exchange: str
    asset_class: str
    qty: str
    side: str
    market_value: str
    cost_basis: str
    unrealized_pl: str
    unrealized_plpc: str
    unrealized_intraday_pl: str
    unrealized_intraday_plpc: str
    current_price: str
    lastday_price: str
    change_today: str


@dataclass
class AlpacaBar:
    """Alpaca bar data."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    trade_count: Optional[int] = None
    vwap: Optional[Decimal] = None


@dataclass
class AlpacaQuote:
    """Alpaca quote data."""
    symbol: str
    timestamp: datetime
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    exchange: Optional[str] = None


@dataclass
class AlpacaTrade:
    """Alpaca trade data."""
    symbol: str
    timestamp: datetime
    price: Decimal
    size: int
    exchange: Optional[str] = None
    id: Optional[str] = None


@dataclass
class AlpacaOrder:
    """Alpaca order information."""
    id: str
    client_order_id: str
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime
    filled_at: Optional[datetime]
    expired_at: Optional[datetime]
    canceled_at: Optional[datetime]
    failed_at: Optional[datetime]
    replaced_at: Optional[datetime]
    replaced_by: Optional[str]
    replaces: Optional[str]
    asset_id: str
    symbol: str
    asset_class: str
    notional: Optional[str]
    qty: Optional[str]
    filled_qty: str
    filled_avg_price: Optional[str]
    order_class: str
    order_type: str
    type: str
    side: str
    time_in_force: str
    limit_price: Optional[str]
    stop_price: Optional[str]
    price: Optional[str]
    trail_percent: Optional[str]
    trail_price: Optional[str]
    hwm: Optional[str]
    status: str
    extended_hours: bool
    legs: Optional[List[Dict[str, Any]]] = None
    commission: Optional[str] = None


# Request/Result types for service layer

class GetAccountRequest(BaseModel):
    """Request for getting account information."""
    pass


class GetAccountResult(BaseModel):
    """Result of getting account information."""
    account: Optional[AlpacaAccount] = None
    success: bool
    message: str


class GetPositionsRequest(BaseModel):
    """Request for getting positions."""
    pass


class GetPositionsResult(BaseModel):
    """Result of getting positions."""
    positions: List[AlpacaPosition] = []
    success: bool
    message: str


class GetPositionRequest(BaseModel):
    """Request for getting a specific position."""
    symbol: str = Field(..., description="Stock symbol")


class GetPositionResult(BaseModel):
    """Result of getting a specific position."""
    position: Optional[AlpacaPosition] = None
    success: bool
    message: str


class GetBarsRequest(BaseModel):
    """Request for getting historical bars."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    timeframe: str = Field("1Min", description="Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)")
    start: Optional[datetime] = Field(None, description="Start time")
    end: Optional[datetime] = Field(None, description="End time")
    limit: Optional[int] = Field(None, description="Maximum number of bars")
    asof: Optional[datetime] = Field(None, description="As-of time")
    feed: Optional[str] = Field(None, description="Data feed")
    currency: Optional[str] = Field(None, description="Currency")


class GetBarsResult(BaseModel):
    """Result of getting historical bars."""
    bars: Dict[str, List[AlpacaBar]] = {}
    success: bool
    message: str


class GetLatestBarsRequest(BaseModel):
    """Request for getting latest bars."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    feed: Optional[str] = Field(None, description="Data feed")
    currency: Optional[str] = Field(None, description="Currency")


class GetLatestBarsResult(BaseModel):
    """Result of getting latest bars."""
    bars: Dict[str, AlpacaBar] = {}
    success: bool
    message: str


class GetQuotesRequest(BaseModel):
    """Request for getting quotes."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    start: Optional[datetime] = Field(None, description="Start time")
    end: Optional[datetime] = Field(None, description="End time")
    limit: Optional[int] = Field(None, description="Maximum number of quotes")
    feed: Optional[str] = Field(None, description="Data feed")
    currency: Optional[str] = Field(None, description="Currency")


class GetQuotesResult(BaseModel):
    """Result of getting quotes."""
    quotes: Dict[str, List[AlpacaQuote]] = {}
    success: bool
    message: str


class GetLatestQuotesRequest(BaseModel):
    """Request for getting latest quotes."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    feed: Optional[str] = Field(None, description="Data feed")
    currency: Optional[str] = Field(None, description="Currency")


class GetLatestQuotesResult(BaseModel):
    """Result of getting latest quotes."""
    quotes: Dict[str, AlpacaQuote] = {}
    success: bool
    message: str


class GetTradesRequest(BaseModel):
    """Request for getting trades."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    start: Optional[datetime] = Field(None, description="Start time")
    end: Optional[datetime] = Field(None, description="End time")
    limit: Optional[int] = Field(None, description="Maximum number of trades")
    feed: Optional[str] = Field(None, description="Data feed")
    currency: Optional[str] = Field(None, description="Currency")


class GetTradesResult(BaseModel):
    """Result of getting trades."""
    trades: Dict[str, List[AlpacaTrade]] = {}
    success: bool
    message: str


class GetLatestTradesRequest(BaseModel):
    """Request for getting latest trades."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    feed: Optional[str] = Field(None, description="Data feed")
    currency: Optional[str] = Field(None, description="Currency")


class GetLatestTradesResult(BaseModel):
    """Result of getting latest trades."""
    trades: Dict[str, AlpacaTrade] = {}
    success: bool
    message: str


class SubmitOrderRequest(BaseModel):
    """Request for submitting an order."""
    symbol: str = Field(..., description="Stock symbol")
    qty: Optional[int] = Field(None, description="Quantity")
    notional: Optional[Decimal] = Field(None, description="Notional amount")
    side: Literal["buy", "sell"] = Field(..., description="Order side")
    type: Literal["market", "limit", "stop", "stop_limit", "trailing_stop"] = Field("market", description="Order type")
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = Field("day", description="Time in force")
    limit_price: Optional[Decimal] = Field(None, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, description="Stop price")
    trail_percent: Optional[Decimal] = Field(None, description="Trail percent")
    trail_price: Optional[Decimal] = Field(None, description="Trail price")
    extended_hours: bool = Field(False, description="Extended hours trading")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    order_class: Literal["simple", "bracket", "oco", "oto"] = Field("simple", description="Order class")
    take_profit_limit_price: Optional[Decimal] = Field(None, description="Take profit limit price")
    stop_loss_stop_price: Optional[Decimal] = Field(None, description="Stop loss stop price")
    stop_loss_limit_price: Optional[Decimal] = Field(None, description="Stop loss limit price")


class SubmitOrderResult(BaseModel):
    """Result of submitting an order."""
    order: Optional[AlpacaOrder] = None
    success: bool
    message: str


class GetOrdersRequest(BaseModel):
    """Request for getting orders."""
    status: Optional[Literal["open", "closed", "all"]] = Field("open", description="Order status")
    limit: Optional[int] = Field(None, description="Maximum number of orders")
    after: Optional[datetime] = Field(None, description="After timestamp")
    until: Optional[datetime] = Field(None, description="Until timestamp")
    direction: Optional[Literal["asc", "desc"]] = Field("desc", description="Sort direction")
    nested: bool = Field(True, description="Include nested orders")


class GetOrdersResult(BaseModel):
    """Result of getting orders."""
    orders: List[AlpacaOrder] = []
    success: bool
    message: str


class GetOrderRequest(BaseModel):
    """Request for getting a specific order."""
    order_id: str = Field(..., description="Order ID")


class GetOrderResult(BaseModel):
    """Result of getting a specific order."""
    order: Optional[AlpacaOrder] = None
    success: bool
    message: str


class CancelOrderRequest(BaseModel):
    """Request for canceling an order."""
    order_id: str = Field(..., description="Order ID")


class CancelOrderResult(BaseModel):
    """Result of canceling an order."""
    success: bool
    message: str


class CancelAllOrdersRequest(BaseModel):
    """Request for canceling all orders."""
    pass


class CancelAllOrdersResult(BaseModel):
    """Result of canceling all orders."""
    success: bool
    message: str
    canceled_count: int = 0


class ClosePositionRequest(BaseModel):
    """Request for closing a position."""
    symbol: str = Field(..., description="Stock symbol")
    qty: Optional[int] = Field(None, description="Quantity to close")
    percentage: Optional[Decimal] = Field(None, description="Percentage to close")


class ClosePositionResult(BaseModel):
    """Result of closing a position."""
    order: Optional[AlpacaOrder] = None
    success: bool
    message: str


class CloseAllPositionsRequest(BaseModel):
    """Request for closing all positions."""
    cancel_orders: bool = Field(True, description="Cancel open orders")


class CloseAllPositionsResult(BaseModel):
    """Result of closing all positions."""
    orders: List[AlpacaOrder] = []
    success: bool
    message: str
