"""Alpaca service exceptions."""


class AlpacaError(Exception):
    """Base exception for Alpaca service errors."""
    pass


class AuthenticationError(AlpacaError):
    """Authentication error."""
    pass


class NotFoundError(AlpacaError):
    """Resource not found error."""
    pass


class OrderError(AlpacaError):
    """Order-related error."""
    pass


class DataError(AlpacaError):
    """Data-related error."""
    pass


class InsufficientFundsError(AlpacaError):
    """Insufficient funds error."""
    pass


class InvalidSymbolError(AlpacaError):
    """Invalid symbol error."""
    pass


class MarketClosedError(AlpacaError):
    """Market closed error."""
    pass
