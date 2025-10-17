from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .environments.file_system import environment as file_system_environment
    from .environments.github import environment as github_environment
    from .agents.tool_calling import tool_calling_agent

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
model_name = "gpt-4.1"

env_names = [
    "file_system", 
    "github",
]
agent_names = ["tool_calling"]
tool_names = [
    'bash',        
    'python_interpreter', 
    'done', 
    'todo', 
    'web_fetcher', 
    'web_searcher', 
    'mdify', 
    "browser",
]

#-----------------FILE SYSTEM ENVIRONMENT CONFIG-----------------
file_system_environment.update(dict(
    base_dir=workdir,
))

#-----------------GITHUB ENVIRONMENT CONFIG-----------------
github_environment.update(dict(
    base_dir=workdir,
))

#-----------------TOOL CALLING AGENT CONFIG-----------------
tool_calling_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_config=memory_config,
)
