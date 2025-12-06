"""Hyperliquid service exceptions."""


class HyperliquidError(Exception):
    """Base exception for Hyperliquid service errors."""
    pass


class AuthenticationError(HyperliquidError):
    """Authentication error."""
    pass


class NotFoundError(HyperliquidError):
    """Resource not found error."""
    pass


class OrderError(HyperliquidError):
    """Order-related error."""
    pass


class DataError(HyperliquidError):
    """Data-related error."""
    pass


class InsufficientFundsError(HyperliquidError):
    """Insufficient funds error."""
    pass


class InvalidSymbolError(HyperliquidError):
    """Invalid symbol error."""
    pass


class MarketClosedError(HyperliquidError):
    """Market closed error."""
    pass

