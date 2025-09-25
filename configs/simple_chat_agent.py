from mmengine.config import read_base
with read_base():
    from .base import memory
    from .agents.simple_chat import simple_chat_agent

tag = "simple_chat_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"

env_names = []
agent_names = ["simple_chat"]
tool_names = []

#-----------------TOOL CALLING AGENT CONFIG-----------------
simple_chat_agent.update(
    workdir=workdir,
)
