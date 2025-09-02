from mmengine.config import read_base
with read_base():
    from .base import trading_offline_tool, browser_tool

tag = "finagent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"

#---------------BROWSER TOOL CONFIG-----------------------#
browser_tool.update(
    model_name="bs-gpt-4.1"
)

#---------------TRADING OFFLINE TOOL CONFIG---------------#
symbol = "AAPL"
start_timestamp = "2015-05-01"
split_timestamp = "2023-05-01"
end_timestamp = "2025-05-01"
level = "1day"
trading_offline_tool.dataset.update(
    symbol=symbol,
    start_timestamp=start_timestamp,    
    end_timestamp=end_timestamp,
    level=level
)
trading_offline_tool.environment.update(
    mode="test",
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp
)