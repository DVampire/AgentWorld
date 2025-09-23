tool_calling_agent = dict(
    type = "Agent",
    name = "tool_calling",
    model_name = "gpt-4.1",
    prompt_name = "tool_calling",  # Use explicit tool usage template
    tools = [],
    max_steps = 10,
    env_names = []
)