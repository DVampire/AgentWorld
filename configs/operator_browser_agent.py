from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .environments.operator_browser import environment as operator_browser_environment
    from .agents.operator_browser import operator_browser_agent

tag = "operator_browser_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
model_name = "computer-browser-use"
# model_name = "gpt-4.1"

env_names = [
    "operator_browser",
]
agent_names = ["operator_browser"]
tool_names = [  
    'done', 
    'todo',
]

#-----------------OPERATOR BROWSER ENVIRONMENT CONFIG-----------------
operator_browser_environment.update(dict(
    base_dir=workdir,
))

#-----------------OPERATOR BROWSER AGENT CONFIG-----------------
operator_browser_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_config=memory_config,
)
