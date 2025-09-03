from mmengine.config import read_base
with read_base():
    from .base import browser_tool, deep_researcher_tool

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"

agent = dict(
    type = "ToolCallingAgent",
    name = "tool_calling_agent",
    model_name = "gpt-4.1",
    prompt_name = "tool_calling",  # Use explicit tool usage template
    tools = [
        "bash",
        "file",
        "project",
        "python_interpreter",
        "browser",
        "done",
        "weather",
        "web_fetcher",
        "web_searcher",
        "deep_researcher",
    ],
    max_iterations = 10
)
