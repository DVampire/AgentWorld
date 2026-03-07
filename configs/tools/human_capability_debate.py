human_capability_debate_tool = dict(
    name="human_capability_debate",
    model_name="openrouter/gpt-5.4-pro",
    agent_models=[
        "openrouter/gemini-3.1-pro-preview",
        "openrouter/gpt-5.4-pro",
        "openrouter/claude-opus-4.6",
        "openrouter/grok-4.1-fast",
    ],
    base_dir="workdir/human_capability_debate",
    require_grad=False,
)
