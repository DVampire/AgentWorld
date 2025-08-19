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
        "get_current_time",
        "search_web",
        "calculate",
        "weather_lookup",
        "file_operations",
        "async_web_request",
        "batch_calculation"
    ],
    max_iterations = 10,
    verbose = True
)
