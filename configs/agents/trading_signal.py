trading_signal_agent = dict(
    workdir = "workdir/trading_signal",
    name = "trading_signal",
    type = "Agent",
    description = "A trading signal agent that can develop trading signals.",
    model_name = "gpt-5",
    prompt_name = "trading_signal",
    memory_config = None,
    max_tools = 100,
    max_steps = 8,
    review_steps = 5,
    log_max_length = 1000,
    require_grad = False,
)