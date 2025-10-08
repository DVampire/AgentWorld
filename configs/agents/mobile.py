"""Configuration for Mobile Agent."""

mobile_agent = dict(
    workdir="./workdir/mobile_agent",
    model_name="gpt-4.1",
    prompt_name="mobile",
    memory_config=None,
    max_steps=30,
    review_steps=5,
    log_max_length=500
)