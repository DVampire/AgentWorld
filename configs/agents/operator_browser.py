"""Configuration for Operator Browser Agent."""
operator_browser_agent = dict(
    workdir = "workdir/operator_browser",
    name = "operator_browser",
    type = "Agent",
    model_name = "gpt-4.1",
    prompt_name = "operator_browser",
    memory_config = None,
    max_steps = 50
)

