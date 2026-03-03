from mmengine.config import read_base
with read_base():
    from .agents.trading_strategy import trading_strategy_agent
    from .environments.quickbacktest import environment as quick_backtest_environment
    from .environments.signal_research import environment as signal_research_environment
    from .memory.general_memory_system import memory_system as general_memory_system
    from .tools.deep_researcher import deep_researcher_tool
    # from .memory.optimizer_memory_system import memory_system as optimizer_memory_system


tag = "trading_strategy_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
# model_name = "openrouter/gemini-3-flash-preview"
model_name = "openrouter/gemini-3-flash-preview"
# model_name = "openrouter/claude-opus-4.5"
env_names = [
    "quickbacktest",
    "signal_research"
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system",
]
agent_names = [
    "trading_strategy",
]
tool_names = [
    'done',
    'todo',
    "deep_researcher",
]

#-----------------MEMORY SYSTEM CONFIG-----------------
general_memory_system.update(
    base_dir=f"{workdir}/memory/general_memory_system",
    model_name=model_name,
    max_summaries=20,
    max_insights=20,
    require_grad=False,
)

#-----------------SIGNAL RESEARCH ENVIRONMENT CONFIG-----------------
signal_research_environment.update(
    base_dir="environment/signal_research",
    require_grad=False,
)

quick_backtest_environment.update(
    base_dir="environment/quick_backtest",
    require_grad=False,
)


deep_researcher_tool.update(
    model_name=model_name,
    base_dir="tool/deep_researcher",
)

# optimizer_memory_system.update(
#     base_dir="memory/optimizer_memory_system",
#     model_name=model_name,
#     max_records_per_session=10,
#     require_grad=False,
# )

#-----------------TRADING STRATEGY AGENT CONFIG-----------------
trading_strategy_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
    max_steps = 15
)
