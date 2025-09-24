from mmengine.config import read_base
with read_base():
    from .base import memory
    from .environments.file_system import environment as file_system_environment
    from .environments.operator_browser import environment as operator_browser_environment
    from .tools.browser import browser_tool
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.deep_analyzer import deep_analyzer_tool
    from .agents.tool_calling import tool_calling_agent

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"

env_names = [
    "file_system", 
    # "operator_browser"
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
    'browser',
    'deep_researcher',
    'deep_analyzer',
]

#-----------------FILE SYSTEM ENVIRONMENT CONFIG-----------------
file_system_environment.update(dict(
    base_dir=workdir,
))

#-----------------OPERATOR BROWSER ENVIRONMENT CONFIG-----------------
# operator_browser_environment.update(
#     base_dir=workdir,
# )

#-----------------TOOL CALLING AGENT CONFIG-----------------
tool_calling_agent.update(
    workdir=workdir,
)
