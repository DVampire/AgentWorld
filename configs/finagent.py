from mmengine.config import read_base
with read_base():
    from .environments.trading_offline import environment, dataset, metric, controller

tag = "finagent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"

#---------------TRADING OFFLINE ENVIRONMENT CONFIG--------
symbol = "AAPL"
start_timestamp = "2015-05-01"
split_timestamp = "2023-05-01"
end_timestamp = "2025-05-01"
level = "1day"
dataset.update(
    symbol=symbol,
    start_timestamp=start_timestamp,    
    end_timestamp=end_timestamp,
    level=level
)
environment.update(
    mode="test",
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp
)

agent = dict(
    type = "FinAgent",
    name = "finagent",
    model_name = "gpt-4.1",
    prompt_name = "finagent",  # Use explicit tool usage template
    tools = [
        "trading_offline_action",
    ],
    max_iterations = 10,
)