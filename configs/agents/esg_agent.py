esg_agent = dict(
    workdir = "workdir/esg_agent",
    name = "esg_agent",
    type = "Agent",
    description = "An ESG agent specialized in retrieving, analyzing, and generating reports from ESG data.",
    model_name = "openrouter/gpt-4.1",
    prompt_name = "esg_agent",
    memory_name = "general_memory_system",
    max_tools = 10,
    max_steps = 30,
    review_steps = 5,
    log_max_length = 1000,
    require_grad = False,
)

