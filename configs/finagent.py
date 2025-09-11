from mmengine.config import read_base
with read_base():
    from .environments.trading_offline import environment as trading_offline_environment, dataset as trading_offline_dataset

tag = "finagent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"

#---------------TRADING OFFLINE ENVIRONMENT CONFIG--------
symbol = "AAPL"
start_timestamp = "2015-05-01"
split_timestamp = "2023-05-01"
end_timestamp = "2025-05-01"
level = "1day"
trading_offline_dataset.update(
    symbol=symbol,
    start_timestamp=start_timestamp,    
    end_timestamp=end_timestamp,
    level=level
)
trading_offline_environment.update(
    mode="test",
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp
)

env_names = [
    "trading_offline"
]

agent = dict(
    type = "FinAgent",
    name = "finagent",
    model_name = "gpt-4.1",
    prompt_name = "finagent",  # Use explicit tool usage template
    tools = [
        # trading offline tools
        "step",
    ],
    max_steps = 10,
    env_names = env_names
)