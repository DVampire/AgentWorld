tool_calling_agent = dict(
    workdir = "workdir/tool_calling",
    name = "tool_calling",
    type = "Agent",
    description = "A tool calling agent that can call tools to complete tasks.",
    model_name = "gpt-5",
    prompt_name = "tool_calling",
    memory_config = None,
    max_tools = 10,
    max_steps = 50,
    review_steps = 5,
    log_max_length = 1000,
)