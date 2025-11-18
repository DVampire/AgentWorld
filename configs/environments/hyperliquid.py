"""Configuration for Hyperliquid Environment."""

indicators = [
    "ATR",
    "BB",
    "CCI",
    "EMA",
    "KDJ",
    "MACD",
    "MFI",
    "OBV",
    "RSI",
    "SMA",
]

# Hyperliquid Environment Configuration
environment = dict(
    base_dir="workdir/hyperliquid",
    account_name="account1",
    symbol=["BTC", "ETH"],
    data_type=["candle"],
    hyperliquid_service=None,
)