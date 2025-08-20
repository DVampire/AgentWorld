workdir = "workdir"
tag = "agent"
exp_path = f"{workdir}/{tag}"
log_path = "agent.log"

model_id = "gpt-4.1"
version = "0.1.0"

agent = dict(
    type = "ToolCallingAgent",
    name = "tool_calling_agent",
    model_name = "gpt-4o",
    prompt_name = "tool_calling_explicit",  # Use explicit tool usage template
    tools = [
        "bash",
        "file",
        "project",
        "python_interpreter",
        "browser",
    ],
    max_iterations = 10,
    verbose = True
)
