from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.tool_calling import tool_calling_agent
    # from .tools.browser import browser_tool
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.mdify import mdify_tool
    from .tools.plotter import plotter_tool
    from .tools.bash import bash_tool
    from .tools.todo import todo_tool
    from .memory.general_memory_system import memory_system as general_memory_system
    from .memory.optimizer_memory_system import memory_system as optimizer_memory_system

tag = "trade_optimization_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
# model_name = "openrouter/gemini-3-flash-preview"
model_name = "openrouter/claude-sonnet-4.5"

env_names = [
    
]
memory_names = [
    "optimizer_memory_system"
]
agent_names = [
    "tool_calling"
]
tool_names = [
    'done',
    'todo',
    "deep_researcher",

]


todo_tool.update(
    base_dir="tool/todo",
    require_grad=False,
)
#-----------------DEEP RESEARCHER TOOL CONFIG-----------------
deep_researcher_tool.update(
    model_name="openrouter/o3",
    base_dir="tool/deep_researcher",
)



#-----------------MEMORY SYSTEM CONFIG-----------------
general_memory_system.update(
    base_dir="memory/general_memory_system",
    model_name=model_name,
    max_summaries=10,
    max_insights=10,
    require_grad=False,
)

#-----------------TOOL CALLING AGENT CONFIG-----------------
tool_calling_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
)