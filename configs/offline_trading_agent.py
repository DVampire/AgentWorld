from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .environments.hyperliquid import offline_hyperliquid_environment as offline_hyperliquid_environment, indicators as hyperliquid_indicators
    from .agents.offline_trading import offline_trading_agent

tag = "offline_trading_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
model_name = "gpt-5"
symbols = ["BTC"]
data_type = ["candle"]

env_names = [
    "offline_hyperliquid",
]
agent_names = ["offline_trading"]
tool_names = [
    'done', 
]

#---------------ONLINE TRADING MEMORY CONFIG---------------
memory_config.update(
    type="offline_trading_memory_system",
    model_name=model_name,
    max_summaries=10,
    max_insights=10,
)

#-----------------HYPERLIQUID ENVIRONMENT CONFIG-----------------
hyperliquid_service = dict(
    base_dir=workdir,
    accounts=None,
    live=False,
    symbol=symbols,
    data_type=data_type,
)
offline_hyperliquid_environment.update(dict(
    base_dir=workdir,
    symbol=symbols,
    data_type=data_type,
))

#-----------------ONLINE TRADING AGENT CONFIG-----------------
offline_trading_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_config=memory_config,
)
