"""Configuration for Anthropic Mobile Agent."""

anthropic_mobile_agent = dict(
    workdir="./workdir/anthropic_mobile_agent",
    model_name="computer-use-claude-4.5-sonnet",
    prompt_name="anthropic_mobile",
    memory_config=None,
    max_steps=30,
    review_steps=5,
    log_max_length=500
)