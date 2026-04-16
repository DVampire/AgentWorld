trading_strategy_evaluation_agent = dict(
    workdir = "workdir/trading_strategy_evaluation",
    name = "trading_strategy_evaluation",
    type = "Agent",
    description = "A trading benchmark agent that can evaluate trading strategies.",
    model_name = "gpt-5",
    prompt_name = "trading_strategy_evaluation",
    memory_config = None,
    max_tools = 100,
    max_steps = 8,
    review_steps = 5,
    log_max_length = 1000,
    require_grad = False,
)