from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.opencode import opencode_agent

tag = "opencode_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
model_name = "openrouter/gemini-3.1-pro-preview"

tool_names = [
    "bash_tool",
    "done_tool",
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system"
]
agent_names = [
    "opencode_agent"
]
skill_names = [
    "hello_world_skill",
]

#-----------------OPENCODE AGENT CONFIG-----------------
opencode_agent.update(
    workdir=f"{workdir}/agent/opencode_agent",
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
)