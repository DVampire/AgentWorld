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
online_hyperliquid_environment = dict(
    base_dir="workdir/online_hyperliquid",
    account_name="account1",
    symbol=["BTC", "ETH"],
    data_type=["candle"],
    hyperliquid_service=None,
    require_grad=False,
)

offline_hyperliquid_environment = dict(
    base_dir="workdir/offline_hyperliquid",
    account_name="account1",
    symbol=["BTC", "ETH"],
    data_type=["candle"],
    hyperliquid_service=None,
    require_grad=False,
)