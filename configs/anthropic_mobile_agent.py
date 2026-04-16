from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .environments.anthropic_mobile import environment as anthropic_mobile_environment
    from .agents.anthropic_mobile import anthropic_mobile_agent

tag = "anthropic_mobile_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
model_name = "computer-use-claude-4.5-sonnet"

env_names = [
    "anthropic_mobile", 
]
agent_names = ["anthropic_mobile"]
tool_names = [
    'done', 
    'todo', 
]

#-----------------MOBILE ENVIRONMENT CONFIG-----------------
anthropic_mobile_environment.update(dict(
    base_dir=workdir,
))

#-----------------TOOL CALLING AGENT CONFIG-----------------
anthropic_mobile_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_config=memory_config,
)
