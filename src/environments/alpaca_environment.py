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
)
from src.environments.alpacaentry.types import (
    GetAccountRequest,
    GetPositionsRequest,
    GetAssetsRequest,
    GetAssetsResult,
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
        
    @ecp.action(name="get_assets", 
                type="Alpaca Trading", 
                description="Get all assets information including symbols, names, types, and status")
    async def get_assets(self, status: Optional[str] = None, asset_class: Optional[str] = None) -> str:
        """Get all assets information.
        
        Returns:
            A string containing detailed assets information including symbols, names, types, and status.
        """
        try:
            request = GetAssetsRequest(status=status, asset_class=asset_class)
            result = await self.alpaca_service.get_assets(request)
            
            if not result.success or not result.assets:
                return f"| ❌ {result.message}"
                
            assets = result.assets
            result_text = dedent(f"""
                Assets Information:
                {len(assets)} assets found
                """)
            return result_text
        except AuthenticationError as e:
            return str(e)
        except Exception as e:
            return f"Failed to get assets information: {str(e)}"

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
