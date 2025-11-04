"""Configuration for Alpaca Environment."""

# Alpaca Environment Configuration
environment = dict(
    auto_start_data_stream=True,
    data_stream_symbols=["BTC/USD"],
    base_dir="workdir/alpaca"
)