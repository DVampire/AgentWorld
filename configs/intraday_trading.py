from mmengine.config import read_base
with read_base():
    from .base import memory_config
    from .environments.intraday_trading import environment as intraday_trading_environment, dataset as intraday_trading_dataset
    from .agents.intraday_trading import intraday_trading_agent

tag = "intraday_trading"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"

#---------------INTRADAY TRADING ENVIRONMENT CONFIG--------
symbol = "AAPL"
start_timestamp = "2015-05-01"
split_timestamp = "2025-01-01"
end_timestamp = "2025-05-01"
level = "1day"
intraday_trading_dataset.update(
    symbol=symbol,
    start_timestamp=start_timestamp,    
    end_timestamp=end_timestamp,
    level=level
)
intraday_trading_environment.update(
    base_dir=workdir,
    mode="test",
    dataset_cfg=intraday_trading_dataset,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp
)

env_names = [
    "intraday_trading"
]
agent_names = ["intraday_trading"]
tool_names = [
    "todo"
]

#-----------------INTERDAY TRADING AGENT CONFIG-----------------
memory_config.update(
    type="trading_memory_system",
)
intraday_trading_agent.update(
    workdir=workdir,
    memory_config=memory_config,
)