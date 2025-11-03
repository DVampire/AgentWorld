"""Alpaca trading service implementation using alpaca-py."""

from typing import Optional
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv(verbose=True)


from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest as AlpacaGetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import (StockHistoricalDataClient,
                                    CryptoHistoricalDataClient,
                                    NewsClient,
                                    OptionHistoricalDataClient)
from alpaca.data.live import (CryptoDataStream,
                              StockDataStream,
                              NewsDataStream, 
                              OptionDataStream)
from alpaca.common.exceptions import APIError

from src.environments.alpacaentry.types import (
    AlpacaAccount,
    AlpacaAsset,
    GetAccountRequest,
    GetAccountResult,
    GetAssetsRequest,
    GetAssetsResult,
)
from src.environments.alpacaentry.exceptions import (
    AlpacaError,
    AuthenticationError,
)
from src.logger import logger


class AlpacaService:
    """Alpaca paper trading service using alpaca-py."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        live: bool = False,
    ):
        """Initialize Alpaca paper trading service.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.data_url = "https://data.alpaca.markets"
        self.live = live
        
        self._trading_client: Optional[TradingClient] = None
        
        self._stock_data_client: Optional[StockHistoricalDataClient] = None
        self._crypto_data_client: Optional[CryptoHistoricalDataClient] = None
        self._news_client: Optional[NewsClient] = None
        self._option_data_client: Optional[OptionHistoricalDataClient] = None
        
        self._crypto_data_stream: Optional[CryptoDataStream] = None
        self._stock_data_stream: Optional[StockDataStream] = None
        self._news_data_stream: Optional[NewsDataStream] = None
        self._option_data_stream: Optional[OptionDataStream] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the Alpaca paper trading service."""
        try:
            # Initialize trading client (paper trading only)
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=not self.live
            )
            
            # Initialize data client
            self._stock_data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            self._crypto_data_client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            self._news_client = NewsClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            self._option_data_client = OptionHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            self._crypto_data_stream = CryptoDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            self._stock_data_stream = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            self._news_data_stream = NewsDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            self._option_data_stream = OptionDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test connection by getting account info
            account = self._trading_client.get_account()
            logger.info(f"Connected to Alpaca paper trading account: {account.account_number}")
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Invalid Alpaca credentials: {e}")
            raise AlpacaError(f"Failed to initialize Alpaca service: {e}")
        except Exception as e:
            raise AlpacaError(f"Failed to initialize Alpaca service: {e}")

    async def cleanup(self) -> None:
        """Cleanup the Alpaca service."""
        self._trading_client = None
        self._data_client = None

    # Account methods
    async def get_account(self, request: GetAccountRequest) -> GetAccountResult:
        """Get account information."""
        try:
            account = self._trading_client.get_account()
            
            alpaca_account = AlpacaAccount(
                id=account.id,
                account_number=account.account_number,
                status=account.status,
                currency=account.currency,
                buying_power=Decimal(str(account.buying_power)),
                cash=Decimal(str(account.cash)),
                portfolio_value=Decimal(str(account.portfolio_value)),
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked,
                transfers_blocked=account.transfers_blocked,
                account_blocked=account.account_blocked,
                created_at=account.created_at,
                trade_suspended_by_user=account.trade_suspended_by_user,
                multiplier=int(account.multiplier),
                shorting_enabled=account.shorting_enabled,
                equity=Decimal(str(account.equity)),
                last_equity=Decimal(str(account.last_equity)),
                long_market_value=Decimal(str(account.long_market_value)),
                short_market_value=Decimal(str(account.short_market_value)),
                initial_margin=Decimal(str(account.initial_margin)),
                maintenance_margin=Decimal(str(account.maintenance_margin)),
                last_maintenance_margin=Decimal(str(account.last_maintenance_margin)),
                sma=Decimal(str(account.sma)),
                daytrade_count=int(account.daytrade_count)
            )
            
            return GetAccountResult(
                account=alpaca_account,
                success=True,
                message="Account information retrieved successfully"
            )
            
        except APIError as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}")
            raise AlpacaError(f"Failed to get account: {e}")
        except Exception as e:
            raise AlpacaError(f"Failed to get account: {e}")
        
    async def get_assets(self, request: GetAssetsRequest) -> GetAssetsResult:
        """Get assets information."""
        try:
            # Convert string parameters to Alpaca enums if provided
            alpaca_status = None
            if request.status:
                try:
                    # Try to convert string to AssetStatus enum
                    if isinstance(request.status, str):
                        # Try direct conversion first
                        try:
                            alpaca_status = AssetStatus(request.status)
                        except ValueError:
                            # Try case-insensitive match
                            for status in AssetStatus:
                                if status.value.upper() == request.status.upper():
                                    alpaca_status = status
                                    break
                    else:
                        alpaca_status = request.status
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to convert status '{request.status}' to AssetStatus: {e}")
            
            alpaca_asset_class = None
            if request.asset_class:
                try:
                    # Try to convert string to AssetClass enum
                    if isinstance(request.asset_class, str):
                        # Try direct conversion first
                        try:
                            alpaca_asset_class = AssetClass(request.asset_class)
                        except ValueError:
                            # Try case-insensitive match
                            for asset_class in AssetClass:
                                if asset_class.value.upper() == request.asset_class.upper():
                                    alpaca_asset_class = asset_class
                                    break
                    else:
                        alpaca_asset_class = request.asset_class
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to convert asset_class '{request.asset_class}' to AssetClass: {e}")
            
            # Create Alpaca SDK request object
            alpaca_request = None
            if alpaca_status or alpaca_asset_class:
                alpaca_request = AlpacaGetAssetsRequest(
                    status=alpaca_status,
                    asset_class=alpaca_asset_class
                )
            
            assets = self._trading_client.get_all_assets(filter=alpaca_request)
            
            alpaca_assets = []
            for asset in assets:
                alpaca_assets.append(AlpacaAsset(
                    id=asset.id,
                    symbol=asset.symbol,
                    name=asset.name,
                    asset_class=asset.asset_class,
                    exchange=asset.exchange,
                    status=asset.status,
                    tradable=asset.tradable,
                    marginable=asset.marginable,
                    shortable=asset.shortable,
                    easy_to_borrow=asset.easy_to_borrow,
                    fractionable=asset.fractionable
                ))
            
            return GetAssetsResult(
                assets=alpaca_assets,
                success=True,
                message=f"Retrieved {len(alpaca_assets)} assets"
            )
            
        except APIError as e:
            raise AlpacaError(f"Failed to get assets: {e}")
        except Exception as e:
            raise AlpacaError(f"Failed to get assets: {e}")