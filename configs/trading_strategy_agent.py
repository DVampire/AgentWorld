from mmengine.config import read_base
with read_base():
    from .agents.trading_strategy import trading_strategy_agent
    from .environments.quickbacktest import environment as quickbacktest_environment
    from .environments.file_system import environment as file_system_environment
    from .memory.general_memory_system import memory_system as general_memory_system

tag = "trading_strategy_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
# model_name = "openrouter/gemini-3-flash-preview"
model_name = "openrouter/gemini-3-flash-preview"
env_names = [
    "quickbacktest",
]
memory_names = [
    "general_memory_system",
]
agent_names = [
    "trading_strategy",
]
tool_names = [
    'done',
    'todo',
]

#-----------------MEMORY SYSTEM CONFIG-----------------
general_memory_system.update(
    base_dir=f"{workdir}/memory/general_memory_system",
    model_name=model_name,
    max_summaries=10,
    max_insights=10,
    require_grad=False,
)

#-----------------QUICK BACKTEST ENVIRONMENT CONFIG-----------------
quickbacktest_environment.update(
    base_dir=f"{workdir}/environment/quickbacktest",
    require_grad=False,
)

#-----------------TRADING STRATEGY AGENT CONFIG-----------------
trading_strategy_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
)
