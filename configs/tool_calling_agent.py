from mmengine.config import read_base
with read_base():
    from .base import browser_tool, deep_researcher_tool, deep_analyzer_tool, memory
    from .environments.file_system import environment as file_system_environment
    from .environments.trading_offline import environment as trading_offline_environment, dataset as trading_offline_dataset
    from .environments.github import environment as github_environment

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"

env_names = [
    "file_system",
    "github",
]

#-----------------FILE SYSTEM ENVIRONMENT CONFIG-----------------
file_system_environment.update(dict(
    base_dir=workdir,
))

#-----------------GITHUB ENVIRONMENT CONFIG-----------------
github_environment.update(dict(
    base_dir=workdir
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
        "browser",
        "done",
        "weather",
        "web_fetcher",
        "web_searcher",
        "deep_researcher",
        "deep_analyzer",
        "todo",
        
        # --file system tools--
        "read",
        "write",
        "replace",
        "delete",
        "copy",
        "rename",
        "get_info",
        "create_dir",
        "delete_dir",
        "tree",
        "describe",
        "search",
        "change_permissions",
        
        # --github tools--
        # repository tools
        "create_repository",
        "get_repository",
        "fork_repository",
        "delete_repository",
        # git tools
        "git_init",
        "git_clone",
        "git_commit",
        "git_push",
        "git_pull",
        "git_fetch",
        "git_status"
        # branch tools
        "git_create_branch",
        "git_checkout_branch",
        "git_list_branches",
        "git_delete_branch",
    ],
    max_steps = 10,
    env_names = env_names
)
