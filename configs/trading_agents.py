from mmengine.config import read_base

with read_base():
    from .agents.trading_strategy import trading_strategy_agent
    from .agents.trading_signal_evaluation import trading_signal_evaluation_agent
    from .agents.trading_strategy_evaluation import trading_strategy_evaluation_agent
    from .agents.trading_signal import trading_signal_agent
    from .environments.quickbacktest import environment as quick_backtest_environment
    from .environments.signal_research import environment as signal_research_environment
    from .environments.signal_evaluate import environment as signal_evaluate_environment
    from .environments.strategy_evaluate import environment as strategy_evaluate_environment
    from .memory.general_memory_system import memory_system as general_memory_system
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.skill_generator import skill_generator_tool
    from .tools.mdify import mdify_tool
    # from .memory.optimizer_memory_system import memory_system as optimizer_memory_system


tag = "trading_agents"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
# model_name = "openrouter/gemini-3-flash-preview"
model_name = "openrouter/gemini-3-flash-preview"
# model_name = "openrouter/claude-opus-4.5"

env_names = [
    "signal_research",
    "signal_evaluate",
    "quickbacktest",
    "strategy_evaluate"

]

memory_names = [
    "general_memory_system",
]
agent_names = [
    "trading_strategy",
    "trading_signal",
    "trading_signal_evaluation",
    "trading_strategy_evaluation"
]
tool_names = [
    'done',
    'todo',
    "deep_researcher",
    "skill_generator",
    "mdify"
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

signal_evaluate_environment.update(
    base_dir="environment/signal_evaluate",
    require_grad=False,
)

strategy_evaluate_environment.update(
    base_dir="environment/strategy_evaluate",
    require_grad=False,
)

quick_backtest_environment.update(
    base_dir="environment/quick_backtest",
    require_grad=False,
)

skill_generator_tool.update(
    model_name="openrouter/gemini-3-flash-preview",
    base_dir="skill",
)

deep_researcher_tool.update(
    model_name=model_name,
    base_dir="tool/deep_researcher",
)

#-----------------TRADING STRATEGY AGENTS CONFIG-----------------
trading_strategy_agent.update(
    workdir = workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
    max_steps = 20
)

trading_signal_evaluation_agent.update(
    workdir = workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
    prompt_name = "trading_signal_evaluation",
    max_steps = 10
)

trading_signal_agent.update(
    workdir = workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
    max_steps = 20
)
trading_strategy_evaluation_agent.update(
    workdir = workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
    prompt_name = "trading_strategy_evaluation",
    max_steps = 10
)


