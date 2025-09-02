_base_ = './base.py'

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
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
    ],
    max_iterations = 10
)
