from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.deep_analyzer import deep_analyzer_agent
    from .tools.todo import todo_tool

tag = "deep_analyzer_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
model_name = "openrouter/gemini-3-flash-preview"

tool_names = [
    "bash_tool",
    "todo_tool",
    "done_tool",
    "python_interpreter_tool",
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system"
]
agent_names = [
    "deep_analyzer"
]

#-----------------TODO TOOL CONFIG-----------------
todo_tool.update(
    base_dir="tool/todo",
    require_grad=False,
)

#-----------------DEEP ANALYZER AGENT CONFIG-----------------
deep_analyzer_agent.update(
    workdir=f"{workdir}/agent/deep_analyzer_agent",
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
)