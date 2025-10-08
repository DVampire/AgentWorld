from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .environments.interday_trading import environment as interday_trading_environment, dataset as interday_trading_dataset
    from .agents.interday_trading import interday_trading_agent

tag = "interday_trading"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"

#---------------INTERDAY TRADING ENVIRONMENT CONFIG--------
symbol = "AAPL"
start_timestamp = "2015-05-01"
split_timestamp = "2025-01-01"
end_timestamp = "2025-05-01"
level = "1day"
interday_trading_dataset.update(
    symbol=symbol,
    start_timestamp=start_timestamp,    
    end_timestamp=end_timestamp,
    level=level
)
interday_trading_environment.update(
    base_dir=workdir,
    mode="test",
    dataset_cfg=interday_trading_dataset,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp
)

env_names = [
    "interday_trading"
]
agent_names = ["interday_trading"]
tool_names = [
    "todo"
]

#-----------------INTERDAY TRADING AGENT CONFIG-----------------
memory_config.update(
    type="trading_memory_system",
)
interday_trading_agent.update(
    workdir=workdir,
    memory_config=memory_config,
)