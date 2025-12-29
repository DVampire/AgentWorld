offline_trading_agent = dict(
    workdir = "workdir/offline_trading",
    name = "offline_trading",
    type = "Agent",
    description = "A offline trading agent that can trade offline.",
    model_name = "gpt-5",
    prompt_name = "offline_trading",
    memory_config = None,
    max_tools = 10,
    max_steps = -1,
    review_steps = 5,
    log_max_length = 1000,
    require_grad = False,
)