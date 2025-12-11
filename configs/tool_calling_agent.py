from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.tool_calling import tool_calling_agent
    from .tools.browser import browser_tool
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.deep_analyzer import deep_analyzer_tool
    from .environments.file_system import environment as file_system_environment

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
model_name = "openrouter/gpt-4.1"

env_names = [
    "file_system",
]
agent_names = ["tool_calling"]
tool_names = [
    'bash',        
    'python_interpreter', 
    'done', 
    'todo',  
    'mdify', 
    "deep_analyzer",
    "deep_researcher",
    "browser",
]

#-----------------BROWSER TOOL CONFIG-----------------
browser_tool.update(
    base_dir=f"{workdir}/browser",
)

#-----------------DEEP RESEARCHER TOOL CONFIG-----------------
deep_researcher_tool.update(
    model_name="openrouter/o3",
    base_dir=f"{workdir}/deep_researcher",
)

#-----------------DEEP ANALYZER TOOL CONFIG-----------------
deep_analyzer_tool.update(
    model_name="openrouter/o3",
    base_dir=f"{workdir}/deep_analyzer",
)

#-----------------TOOL CALLING AGENT CONFIG-----------------
tool_calling_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_config=memory_config,
)

#-----------------FILE SYSTEM ENVIRONMENT CONFIG-----------------
file_system_environment.update(
    base_dir=f"{workdir}/file_system",
)