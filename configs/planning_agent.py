from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.planning import planning_agent
    from .tools.todo import todo_tool

tag = "planning_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
model_name = "openrouter/gemini-3-flash-preview"

tool_names = [
    "bash_tool",
    "todo_tool",
    "done_tool"
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system"
]
agent_names = [
    "planning"
]

#-----------------TODO TOOL CONFIG-----------------
todo_tool.update(
    base_dir="tool/todo",
    require_grad=False,
)

#-----------------PLANNING AGENT CONFIG-----------------
planning_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
)

