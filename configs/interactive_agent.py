# Interactive agent configuration
_base_ = './base.py'

# System settings
tag = "interactive_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

# Model settings
use_local_proxy = False
version = "0.1.0"

# Agent configuration
interactive_mode = True
auto_continue = False
agent = dict(
    name="interactive_agent",
    type="InteractiveAgent",
    model_name="gpt-4.1",
    prompt_name="interactive",
    tools=[
        "bash",
        "file", 
        "project",
        "python_interpreter",
        "browser"
    ],
    max_iterations=50,
    interactive_mode=interactive_mode,
    auto_continue=auto_continue,
    max_steps=100,
    review_steps=10
)
