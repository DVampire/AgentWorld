from mmengine.config import read_base
with read_base():
    from .base import deep_researcher_tool
    from .environments.file_system import environment as file_system_environment, controller as file_system_controller
    from .environments.browser import environment as browser_environment, controller as browser_controller

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"

#-----------------FILE SYSTEM ENVIRONMENT CONFIG-----------------
file_system_environment.update(dict(
    base_dir=workdir,
))
file_system_controller.update(dict(
    environment=file_system_environment,
))

#-----------------BROWSER ENVIRONMENT CONFIG-----------------
browser_environment.update(dict(
    downloads_dir=f"{workdir}/browser/downloads",
))
browser_controller.update(dict(
    environment=browser_environment,
))

#-----------------AGENT CONFIG-----------------------------------
agent = dict(
    type = "ToolCallingAgent",
    name = "tool_calling_agent",
    model_name = "gpt-4.1",
    prompt_name = "tool_calling",  # Use explicit tool usage template
    tools = [
        "bash",
        "python_interpreter",
        "done",
        # "weather",
        # "web_fetcher",
        # "web_searcher",
        "deep_researcher",
        
        # file system tools
        "file_operations",
        "directory_operations",
        "search_operations",
        "permission_operations",
        
        # browser tools (extracted from browser-use)
        "navigation_operations",
        "interaction_operations",
        "data_extraction_operations",
        "tab_management_operations",
    ],
    max_steps = 10,
    controllers = None,
)
