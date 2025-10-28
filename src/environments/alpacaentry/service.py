"""Alpaca trading service implementation using alpaca-py."""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv(verbose=True)

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderClass, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError

from src.environments.alpacaentry.types import (
    AlpacaAccount,
    AlpacaPosition,
    AlpacaBar,
    AlpacaQuote,
    AlpacaTrade,
    AlpacaOrder,
    GetAccountRequest,
    GetAccountResult,
    GetPositionsRequest,
    GetPositionsResult,
    GetPositionRequest,
    GetPositionResult,
    GetBarsRequest,
    GetBarsResult,
    GetLatestBarsRequest,
    GetLatestBarsResult,
    GetQuotesRequest,
    GetQuotesResult,
    GetLatestQuotesRequest,
    GetLatestQuotesResult,
    GetTradesRequest,
    GetTradesResult,
    GetLatestTradesRequest,
    GetLatestTradesResult,
    SubmitOrderRequest,
    SubmitOrderResult,
    GetOrdersRequest,
    GetOrdersResult,
    GetOrderRequest,
    GetOrderResult,
    CancelOrderRequest,
    CancelOrderResult,
    CancelAllOrdersRequest,
    CancelAllOrdersResult,
    ClosePositionRequest,
    ClosePositionResult,
    CloseAllPositionsRequest,
    CloseAllPositionsResult,
)
from src.environments.alpacaentry.exceptions import (
    AlpacaError,
    AuthenticationError,
    NotFoundError,
    OrderError,
    DataError,
    InsufficientFundsError,
    InvalidSymbolError,
    MarketClosedError,
)
from src.logger import logger


class AlpacaService:
    """Alpaca paper trading service using alpaca-py."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
    ):
        """Initialize Alpaca paper trading service.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.data_url = "https://data.alpaca.markets"
        
        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None

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
                paper=True
            )
            
            # Initialize data client
            self._data_client = StockHistoricalDataClient(
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

    # Position methods
    async def get_positions(self, request: GetPositionsRequest) -> GetPositionsResult:
        """Get all positions."""
        try:
            positions = self._trading_client.get_all_positions()
            
            alpaca_positions = []
            for pos in positions:
                alpaca_positions.append(AlpacaPosition(
                    asset_id=pos.asset_id,
                    symbol=pos.symbol,
                    exchange=pos.exchange,
                    asset_class=pos.asset_class,
                    qty=str(pos.qty),
                    side=pos.side,
                    market_value=str(pos.market_value),
                    cost_basis=str(pos.cost_basis),
                    unrealized_pl=str(pos.unrealized_pl),
                    unrealized_plpc=str(pos.unrealized_plpc),
                    unrealized_intraday_pl=str(pos.unrealized_intraday_pl),
                    unrealized_intraday_plpc=str(pos.unrealized_intraday_plpc),
                    current_price=str(pos.current_price),
                    lastday_price=str(pos.lastday_price),
                    change_today=str(pos.change_today)
                ))
            
            return GetPositionsResult(
                positions=alpaca_positions,
                success=True,
                message=f"Retrieved {len(alpaca_positions)} positions"
            )
            
        except APIError as e:
            raise AlpacaError(f"Failed to get positions: {e}")
        except Exception as e:
            raise AlpacaError(f"Failed to get positions: {e}")

    async def get_position(self, request: GetPositionRequest) -> GetPositionResult:
        """Get a specific position."""
        try:
            position = self._trading_client.get_open_position(request.symbol)
            
            alpaca_position = AlpacaPosition(
                asset_id=position.asset_id,
                symbol=position.symbol,
                exchange=position.exchange,
                asset_class=position.asset_class,
                qty=str(position.qty),
                side=position.side,
                market_value=str(position.market_value),
                cost_basis=str(position.cost_basis),
                unrealized_pl=str(position.unrealized_pl),
                unrealized_plpc=str(position.unrealized_plpc),
                unrealized_intraday_pl=str(position.unrealized_intraday_pl),
                unrealized_intraday_plpc=str(position.unrealized_intraday_plpc),
                current_price=str(position.current_price),
                lastday_price=str(position.lastday_price),
                change_today=str(position.change_today)
            )
            
            return GetPositionResult(
                position=alpaca_position,
                success=True,
                message=f"Position for {request.symbol} retrieved successfully"
            )
            
        except APIError as e:
            if e.status_code == 404:
                return GetPositionResult(
                    position=None,
                    success=True,
                    message=f"No position found for {request.symbol}"
                )
            raise AlpacaError(f"Failed to get position: {e}")
        except Exception as e:
            raise AlpacaError(f"Failed to get position: {e}")

    # Data methods
    async def get_bars(self, request: GetBarsRequest) -> GetBarsResult:
        """Get historical bars."""
        try:
            # Convert timeframe string to TimeFrame object
            timeframe_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
                "1Day": TimeFrame(1, TimeFrameUnit.Day),
            }
            
            timeframe = timeframe_map.get(request.timeframe, TimeFrame(1, TimeFrameUnit.Minute))
            
            bars_request = StockBarsRequest(
                symbol_or_symbols=request.symbols,
                timeframe=timeframe,
                start=request.start,
                end=request.end,
                limit=request.limit,
                asof=request.asof,
                feed=request.feed,
                currency=request.currency
            )
            
            bars_response = self._data_client.get_stock_bars(bars_request)
            
            alpaca_bars = {}
            for symbol, bars in bars_response.items():
                alpaca_bars[symbol] = []
                for bar in bars:
                    alpaca_bars[symbol].append(AlpacaBar(
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        open=Decimal(str(bar.open)),
                        high=Decimal(str(bar.high)),
                        low=Decimal(str(bar.low)),
                        close=Decimal(str(bar.close)),
                        volume=int(bar.volume),
                        trade_count=bar.trade_count,
                        vwap=Decimal(str(bar.vwap)) if bar.vwap else None
                    ))
            
            return GetBarsResult(
                bars=alpaca_bars,
                success=True,
                message=f"Retrieved bars for {len(request.symbols)} symbols"
            )
            
        except APIError as e:
            raise DataError(f"Failed to get bars: {e}")
        except Exception as e:
            raise DataError(f"Failed to get bars: {e}")

    async def get_latest_bars(self, request: GetLatestBarsRequest) -> GetLatestBarsResult:
        """Get latest bars."""
        try:
            bars_request = StockBarsRequest(
                symbol_or_symbols=request.symbols,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                limit=1,
                feed=request.feed,
                currency=request.currency
            )
            
            bars_response = self._data_client.get_stock_bars(bars_request)
            
            alpaca_bars = {}
            for symbol, bars in bars_response.items():
                if bars:
                    bar = bars[0]  # Get the latest bar
                    alpaca_bars[symbol] = AlpacaBar(
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        open=Decimal(str(bar.open)),
                        high=Decimal(str(bar.high)),
                        low=Decimal(str(bar.low)),
                        close=Decimal(str(bar.close)),
                        volume=int(bar.volume),
                        trade_count=bar.trade_count,
                        vwap=Decimal(str(bar.vwap)) if bar.vwap else None
                    )
            
            return GetLatestBarsResult(
                bars=alpaca_bars,
                success=True,
                message=f"Retrieved latest bars for {len(alpaca_bars)} symbols"
            )
            
        except APIError as e:
            raise DataError(f"Failed to get latest bars: {e}")
        except Exception as e:
            raise DataError(f"Failed to get latest bars: {e}")

    async def get_quotes(self, request: GetQuotesRequest) -> GetQuotesResult:
        """Get historical quotes."""
        try:
            quotes_request = StockQuotesRequest(
                symbol_or_symbols=request.symbols,
                start=request.start,
                end=request.end,
                limit=request.limit,
                feed=request.feed,
                currency=request.currency
            )
            
            quotes_response = self._data_client.get_stock_quotes(quotes_request)
            
            alpaca_quotes = {}
            for symbol, quotes in quotes_response.items():
                alpaca_quotes[symbol] = []
                for quote in quotes:
                    alpaca_quotes[symbol].append(AlpacaQuote(
                        symbol=symbol,
                        timestamp=quote.timestamp,
                        bid_price=Decimal(str(quote.bid_price)),
                        bid_size=int(quote.bid_size),
                        ask_price=Decimal(str(quote.ask_price)),
                        ask_size=int(quote.ask_size),
                        exchange=quote.exchange
                    ))
            
            return GetQuotesResult(
                quotes=alpaca_quotes,
                success=True,
                message=f"Retrieved quotes for {len(request.symbols)} symbols"
            )
            
        except APIError as e:
            raise DataError(f"Failed to get quotes: {e}")
        except Exception as e:
            raise DataError(f"Failed to get quotes: {e}")

    async def get_latest_quotes(self, request: GetLatestQuotesRequest) -> GetLatestQuotesResult:
        """Get latest quotes."""
        try:
            quotes_request = StockQuotesRequest(
                symbol_or_symbols=request.symbols,
                limit=1,
                feed=request.feed,
                currency=request.currency
            )
            
            quotes_response = self._data_client.get_stock_quotes(quotes_request)
            
            alpaca_quotes = {}
            for symbol, quotes in quotes_response.items():
                if quotes:
                    quote = quotes[0]  # Get the latest quote
                    alpaca_quotes[symbol] = AlpacaQuote(
                        symbol=symbol,
                        timestamp=quote.timestamp,
                        bid_price=Decimal(str(quote.bid_price)),
                        bid_size=int(quote.bid_size),
                        ask_price=Decimal(str(quote.ask_price)),
                        ask_size=int(quote.ask_size),
                        exchange=quote.exchange
                    )
            
            return GetLatestQuotesResult(
                quotes=alpaca_quotes,
                success=True,
                message=f"Retrieved latest quotes for {len(alpaca_quotes)} symbols"
            )
            
        except APIError as e:
            raise DataError(f"Failed to get latest quotes: {e}")
        except Exception as e:
            raise DataError(f"Failed to get latest quotes: {e}")

    async def get_trades(self, request: GetTradesRequest) -> GetTradesResult:
        """Get historical trades."""
        try:
            trades_request = StockTradesRequest(
                symbol_or_symbols=request.symbols,
                start=request.start,
                end=request.end,
                limit=request.limit,
                feed=request.feed,
                currency=request.currency
            )
            
            trades_response = self._data_client.get_stock_trades(trades_request)
            
            alpaca_trades = {}
            for symbol, trades in trades_response.items():
                alpaca_trades[symbol] = []
                for trade in trades:
                    alpaca_trades[symbol].append(AlpacaTrade(
                        symbol=symbol,
                        timestamp=trade.timestamp,
                        price=Decimal(str(trade.price)),
                        size=int(trade.size),
                        exchange=trade.exchange,
                        id=trade.id
                    ))
            
            return GetTradesResult(
                trades=alpaca_trades,
                success=True,
                message=f"Retrieved trades for {len(request.symbols)} symbols"
            )
            
        except APIError as e:
            raise DataError(f"Failed to get trades: {e}")
        except Exception as e:
            raise DataError(f"Failed to get trades: {e}")

    async def get_latest_trades(self, request: GetLatestTradesRequest) -> GetLatestTradesResult:
        """Get latest trades."""
        try:
            trades_request = StockTradesRequest(
                symbol_or_symbols=request.symbols,
                limit=1,
                feed=request.feed,
                currency=request.currency
            )
            
            trades_response = self._data_client.get_stock_trades(trades_request)
            
            alpaca_trades = {}
            for symbol, trades in trades_response.items():
                if trades:
                    trade = trades[0]  # Get the latest trade
                    alpaca_trades[symbol] = AlpacaTrade(
                        symbol=symbol,
                        timestamp=trade.timestamp,
                        price=Decimal(str(trade.price)),
                        size=int(trade.size),
                        exchange=trade.exchange,
                        id=trade.id
                    )
            
            return GetLatestTradesResult(
                trades=alpaca_trades,
                success=True,
                message=f"Retrieved latest trades for {len(alpaca_trades)} symbols"
            )
            
        except APIError as e:
            raise DataError(f"Failed to get latest trades: {e}")
        except Exception as e:
            raise DataError(f"Failed to get latest trades: {e}")

    # Order methods
    async def submit_order(self, request: SubmitOrderRequest) -> SubmitOrderResult:
        """Submit an order."""
        try:
            # Create order request based on type
            if request.type == "market":
                order_request = MarketOrderRequest(
                    symbol=request.symbol,
                    qty=request.qty,
                    notional=float(request.notional) if request.notional else None,
                    side=OrderSide.BUY if request.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY if request.time_in_force == "day" else TimeInForce.GTC,
                    extended_hours=request.extended_hours,
                    client_order_id=request.client_order_id
                )
            elif request.type == "limit":
                order_request = LimitOrderRequest(
                    symbol=request.symbol,
                    qty=request.qty,
                    notional=float(request.notional) if request.notional else None,
                    side=OrderSide.BUY if request.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY if request.time_in_force == "day" else TimeInForce.GTC,
                    limit_price=float(request.limit_price) if request.limit_price else None,
                    extended_hours=request.extended_hours,
                    client_order_id=request.client_order_id
                )
            elif request.type == "stop":
                order_request = StopOrderRequest(
                    symbol=request.symbol,
                    qty=request.qty,
                    notional=float(request.notional) if request.notional else None,
                    side=OrderSide.BUY if request.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY if request.time_in_force == "day" else TimeInForce.GTC,
                    stop_price=float(request.stop_price) if request.stop_price else None,
                    extended_hours=request.extended_hours,
                    client_order_id=request.client_order_id
                )
            elif request.type == "stop_limit":
                order_request = StopLimitOrderRequest(
                    symbol=request.symbol,
                    qty=request.qty,
                    notional=float(request.notional) if request.notional else None,
                    side=OrderSide.BUY if request.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY if request.time_in_force == "day" else TimeInForce.GTC,
                    limit_price=float(request.limit_price) if request.limit_price else None,
                    stop_price=float(request.stop_price) if request.stop_price else None,
                    extended_hours=request.extended_hours,
                    client_order_id=request.client_order_id
                )
            elif request.type == "trailing_stop":
                order_request = TrailingStopOrderRequest(
                    symbol=request.symbol,
                    qty=request.qty,
                    notional=float(request.notional) if request.notional else None,
                    side=OrderSide.BUY if request.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY if request.time_in_force == "day" else TimeInForce.GTC,
                    trail_percent=float(request.trail_percent) if request.trail_percent else None,
                    trail_price=float(request.trail_price) if request.trail_price else None,
                    extended_hours=request.extended_hours,
                    client_order_id=request.client_order_id
                )
            else:
                raise OrderError(f"Unsupported order type: {request.type}")
            
            # Submit the order
            order = self._trading_client.submit_order(order_request)
            
            alpaca_order = AlpacaOrder(
                id=order.id,
                client_order_id=order.client_order_id,
                created_at=order.created_at,
                updated_at=order.updated_at,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                expired_at=order.expired_at,
                canceled_at=order.canceled_at,
                failed_at=order.failed_at,
                replaced_at=order.replaced_at,
                replaced_by=order.replaced_by,
                replaces=order.replaces,
                asset_id=order.asset_id,
                symbol=order.symbol,
                asset_class=order.asset_class,
                notional=order.notional,
                qty=order.qty,
                filled_qty=order.filled_qty,
                filled_avg_price=order.filled_avg_price,
                order_class=order.order_class,
                order_type=order.order_type,
                type=order.type,
                side=order.side,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                price=order.price,
                trail_percent=order.trail_percent,
                trail_price=order.trail_price,
                hwm=order.hwm,
                status=order.status,
                extended_hours=order.extended_hours,
                legs=order.legs,
                commission=order.commission
            )
            
            return SubmitOrderResult(
                order=alpaca_order,
                success=True,
                message=f"Order submitted successfully: {order.id}"
            )
            
        except APIError as e:
            if e.status_code == 422:
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            elif e.status_code == 404:
                raise InvalidSymbolError(f"Invalid symbol: {e}")
            raise OrderError(f"Failed to submit order: {e}")
        except Exception as e:
            raise OrderError(f"Failed to submit order: {e}")

    async def get_orders(self, request: GetOrdersRequest) -> GetOrdersResult:
        """Get orders."""
        try:
            orders = self._trading_client.get_orders(
                status=request.status,
                limit=request.limit,
                after=request.after,
                until=request.until,
                direction=request.direction,
                nested=request.nested
            )
            
            alpaca_orders = []
            for order in orders:
                alpaca_orders.append(AlpacaOrder(
                    id=order.id,
                    client_order_id=order.client_order_id,
                    created_at=order.created_at,
                    updated_at=order.updated_at,
                    submitted_at=order.submitted_at,
                    filled_at=order.filled_at,
                    expired_at=order.expired_at,
                    canceled_at=order.canceled_at,
                    failed_at=order.failed_at,
                    replaced_at=order.replaced_at,
                    replaced_by=order.replaced_by,
                    replaces=order.replaces,
                    asset_id=order.asset_id,
                    symbol=order.symbol,
                    asset_class=order.asset_class,
                    notional=order.notional,
                    qty=order.qty,
                    filled_qty=order.filled_qty,
                    filled_avg_price=order.filled_avg_price,
                    order_class=order.order_class,
                    order_type=order.order_type,
                    type=order.type,
                    side=order.side,
                    time_in_force=order.time_in_force,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    price=order.price,
                    trail_percent=order.trail_percent,
                    trail_price=order.trail_price,
                    hwm=order.hwm,
                    status=order.status,
                    extended_hours=order.extended_hours,
                    legs=order.legs,
                    commission=order.commission
                ))
            
            return GetOrdersResult(
                orders=alpaca_orders,
                success=True,
                message=f"Retrieved {len(alpaca_orders)} orders"
            )
            
        except APIError as e:
            raise OrderError(f"Failed to get orders: {e}")
        except Exception as e:
            raise OrderError(f"Failed to get orders: {e}")

    async def get_order(self, request: GetOrderRequest) -> GetOrderResult:
        """Get a specific order."""
        try:
            order = self._trading_client.get_order_by_id(request.order_id)
            
            alpaca_order = AlpacaOrder(
                id=order.id,
                client_order_id=order.client_order_id,
                created_at=order.created_at,
                updated_at=order.updated_at,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                expired_at=order.expired_at,
                canceled_at=order.canceled_at,
                failed_at=order.failed_at,
                replaced_at=order.replaced_at,
                replaced_by=order.replaced_by,
                replaces=order.replaces,
                asset_id=order.asset_id,
                symbol=order.symbol,
                asset_class=order.asset_class,
                notional=order.notional,
                qty=order.qty,
                filled_qty=order.filled_qty,
                filled_avg_price=order.filled_avg_price,
                order_class=order.order_class,
                order_type=order.order_type,
                type=order.type,
                side=order.side,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                price=order.price,
                trail_percent=order.trail_percent,
                trail_price=order.trail_price,
                hwm=order.hwm,
                status=order.status,
                extended_hours=order.extended_hours,
                legs=order.legs,
                commission=order.commission
            )
            
            return GetOrderResult(
                order=alpaca_order,
                success=True,
                message=f"Order {request.order_id} retrieved successfully"
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise NotFoundError(f"Order {request.order_id} not found")
            raise OrderError(f"Failed to get order: {e}")
        except Exception as e:
            raise OrderError(f"Failed to get order: {e}")

    async def cancel_order(self, request: CancelOrderRequest) -> CancelOrderResult:
        """Cancel an order."""
        try:
            self._trading_client.cancel_order_by_id(request.order_id)
            
            return CancelOrderResult(
                success=True,
                message=f"Order {request.order_id} canceled successfully"
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise NotFoundError(f"Order {request.order_id} not found")
            raise OrderError(f"Failed to cancel order: {e}")
        except Exception as e:
            raise OrderError(f"Failed to cancel order: {e}")

    async def cancel_all_orders(self, request: CancelAllOrdersRequest) -> CancelAllOrdersResult:
        """Cancel all orders."""
        try:
            canceled_orders = self._trading_client.cancel_orders()
            
            return CancelAllOrdersResult(
                success=True,
                message=f"Canceled {len(canceled_orders)} orders",
                canceled_count=len(canceled_orders)
            )
            
        except APIError as e:
            raise OrderError(f"Failed to cancel all orders: {e}")
        except Exception as e:
            raise OrderError(f"Failed to cancel all orders: {e}")

    async def close_position(self, request: ClosePositionRequest) -> ClosePositionResult:
        """Close a position."""
        try:
            order = self._trading_client.close_position(
                symbol=request.symbol,
                qty=request.qty,
                percentage=float(request.percentage) if request.percentage else None
            )
            
            alpaca_order = AlpacaOrder(
                id=order.id,
                client_order_id=order.client_order_id,
                created_at=order.created_at,
                updated_at=order.updated_at,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                expired_at=order.expired_at,
                canceled_at=order.canceled_at,
                failed_at=order.failed_at,
                replaced_at=order.replaced_at,
                replaced_by=order.replaced_by,
                replaces=order.replaces,
                asset_id=order.asset_id,
                symbol=order.symbol,
                asset_class=order.asset_class,
                notional=order.notional,
                qty=order.qty,
                filled_qty=order.filled_qty,
                filled_avg_price=order.filled_avg_price,
                order_class=order.order_class,
                order_type=order.order_type,
                type=order.type,
                side=order.side,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                price=order.price,
                trail_percent=order.trail_percent,
                trail_price=order.trail_price,
                hwm=order.hwm,
                status=order.status,
                extended_hours=order.extended_hours,
                legs=order.legs,
                commission=order.commission
            )
            
            return ClosePositionResult(
                order=alpaca_order,
                success=True,
                message=f"Position {request.symbol} closed successfully"
            )
            
        except APIError as e:
            if e.status_code == 404:
                raise NotFoundError(f"Position {request.symbol} not found")
            raise OrderError(f"Failed to close position: {e}")
        except Exception as e:
            raise OrderError(f"Failed to close position: {e}")

    async def close_all_positions(self, request: CloseAllPositionsRequest) -> CloseAllPositionsResult:
        """Close all positions."""
        try:
            orders = self._trading_client.close_all_positions(
                cancel_orders=request.cancel_orders
            )
            
            alpaca_orders = []
            for order in orders:
                alpaca_orders.append(AlpacaOrder(
                    id=order.id,
                    client_order_id=order.client_order_id,
                    created_at=order.created_at,
                    updated_at=order.updated_at,
                    submitted_at=order.submitted_at,
                    filled_at=order.filled_at,
                    expired_at=order.expired_at,
                    canceled_at=order.canceled_at,
                    failed_at=order.failed_at,
                    replaced_at=order.replaced_at,
                    replaced_by=order.replaced_by,
                    replaces=order.replaces,
                    asset_id=order.asset_id,
                    symbol=order.symbol,
                    asset_class=order.asset_class,
                    notional=order.notional,
                    qty=order.qty,
                    filled_qty=order.filled_qty,
                    filled_avg_price=order.filled_avg_price,
                    order_class=order.order_class,
                    order_type=order.order_type,
                    type=order.type,
                    side=order.side,
                    time_in_force=order.time_in_force,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    price=order.price,
                    trail_percent=order.trail_percent,
                    trail_price=order.trail_price,
                    hwm=order.hwm,
                    status=order.status,
                    extended_hours=order.extended_hours,
                    legs=order.legs,
                    commission=order.commission
                ))
            
            return CloseAllPositionsResult(
                orders=alpaca_orders,
                success=True,
                message=f"Closed {len(alpaca_orders)} positions"
            )
            
        except APIError as e:
            raise OrderError(f"Failed to close all positions: {e}")
        except Exception as e:
            raise OrderError(f"Failed to close all positions: {e}")
