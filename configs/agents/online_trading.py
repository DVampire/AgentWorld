online_trading_agent = dict(
    workdir = "workdir/online_trading",
    name = "online_trading",
    type = "Agent",
    description = "A online trading agent that can trade online.",
    model_name = "gpt-5",
    prompt_name = "online_trading",
    memory_config = None,
    max_tools = 10,
    max_steps = 50,
    review_steps = 5,
    log_max_length = 1000,
)