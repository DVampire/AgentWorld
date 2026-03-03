trading_strategy_agent = dict(
    workdir = "workdir/trading_strategy",
    name = "trading_strategy",
    type = "Agent",
    description = "A trading strategy agent that can develop and optimize trading strategies.",
    model_name = "gpt-5",
    prompt_name = "trading_strategy",
    memory_config = None,
    max_tools = 100,
    max_steps = 8,
    review_steps = 5,
    log_max_length = 1000,
    require_grad = False,
)