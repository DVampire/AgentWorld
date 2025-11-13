from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .environments.hyperliquid import environment as hyperliquid_environment
    from .agents.online_trading import online_trading_agent

tag = "online_trading_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
model_name = "gpt-5"
symbols = ["BTC", "ETH"]
data_type = ["candle"]

env_names = [
    "hyperliquid",
]
agent_names = ["online_trading"]
tool_names = [
    'done', 
]

#-----------------HYPERLIQUID ENVIRONMENT CONFIG-----------------
hyperliquid_service = dict(
    base_dir=workdir,
    accounts=None,
    live=True,
    auto_start_data_stream=True,
    symbol=symbols,
    data_type=data_type,
)
hyperliquid_environment.update(dict(
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
