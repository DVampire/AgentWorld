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