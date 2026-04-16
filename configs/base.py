#---------------GENERAL CONFIG-------------
tag = "base"
workdir = f"workdir/{tag}"
log_path = "base.log"
use_local_proxy = False
model_name = "oenrouter/gemini-3-flash-preview"

#---------------MEMORY CONFIG---------------
memory_config = dict(
    type = "general_memory_system",
    model_name = "gpt-4.1",
    max_summaries = 20,
    max_insights = 100
)

#---------------MAX TOKENS CONFIG---------------
max_tokens = 16384

#---------------Window Size Config---------------
window_size = (1024, 768)

#---------------ALPACA SERVICE CONFIG---------------
alpaca_service = dict(
    base_dir=workdir,
    accounts=None,
    live=False,
    auto_start_data_stream=True,
    symbol=["BTC/USD"],
    data_type=["bars"],
)

#---------------BINANCE SERVICE CONFIG---------------
binance_service = dict(
    base_dir=workdir,
    accounts=None,
    live=False,
    auto_start_data_stream=True,
    symbol=["BTCUSDT"],
    data_type=["klines"],
)