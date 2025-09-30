from mmengine.config import read_base
with read_base():
    from .environments.trading_offline import environment as trading_offline_environment, dataset as trading_offline_dataset
    from .agents.trading_offline import trading_offline_agent

tag = "trading_offline"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"

#---------------TRADING OFFLINE ENVIRONMENT CONFIG--------
symbol = "AAPL"
start_timestamp = "2015-05-01"
split_timestamp = "2025-01-01"
end_timestamp = "2025-05-01"
level = "1day"
trading_offline_dataset.update(
    symbol=symbol,
    start_timestamp=start_timestamp,    
    end_timestamp=end_timestamp,
    level=level
)
trading_offline_environment.update(
    base_dir=workdir,
    mode="test",
    dataset_cfg=trading_offline_dataset,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp
)

env_names = [
    "trading_offline"
]
agent_names = ["trading_offline"]
tool_names = [
    "todo"
]

#-----------------TRADING OFFLINE AGENT CONFIG-----------------
trading_offline_agent.update(
    workdir=workdir,
)