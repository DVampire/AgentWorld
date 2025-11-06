from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .environments.binance import environment as binance_environment
    from .agents.online_trading import online_trading_agent

tag = "online_trading_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
model_name = "gpt-4.1"
symbols = ["BTCUSDT", "ETHUSDT"]
data_type = ["klines"]

env_names = [
    "binance",
]
agent_names = ["online_trading"]
tool_names = [
    'done', 
]

#-----------------BINANCE ENVIRONMENT CONFIG-----------------
binance_service = dict(
    base_dir=workdir,
    accounts=None,
    live=False,
    auto_start_data_stream=True,
    symbol=symbols,
    data_type=data_type,
)
binance_environment.update(dict(
    base_dir=workdir,
    symbol=symbols,
    data_type=data_type,
))

#-----------------ONLINE TRADING AGENT CONFIG-----------------
online_trading_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_config=memory_config,
)
