planning_agent = dict(
    workdir = "workdir/planning",
    name = "planning_agent",
    type = "Agent",
    description = "A planning agent that decomposes complex tasks and coordinates sub-agents.",
    model_name = "openrouter/o3",
    prompt_name = "planning_agent",
    memory_name = "general_memory_system",
    max_tools = 10,
    max_rounds = 20,
    max_steps = 50,
    review_steps = 5,
    log_max_length = 1000,
    require_grad = False,
)

