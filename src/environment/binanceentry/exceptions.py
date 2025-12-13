"""Binance service exceptions."""


class BinanceError(Exception):
    """Base exception for Binance service errors."""
    pass


class AuthenticationError(BinanceError):
    """Authentication error."""
    pass


class NotFoundError(BinanceError):
    """Resource not found error."""
    pass


class OrderError(BinanceError):
    """Order-related error."""
    pass


class DataError(BinanceError):
    """Data-related error."""
    pass


class InsufficientFundsError(BinanceError):
    """Insufficient funds error."""
    pass


class InvalidSymbolError(BinanceError):
    """Invalid symbol error."""
    pass


class MarketClosedError(BinanceError):
    """Market closed error."""
    pass

