"""Alpaca trading service data types."""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field

class GetAccountRequest(BaseModel):
    """Request for getting account information."""
    pass

class GetAssetsRequest(BaseModel):
    """Request for getting assets."""
    status: Optional[str] = Field(None, description="Filter by asset status")
    asset_class: Optional[str] = Field(None, description="Filter by asset class")