deep_analyzer_light_agent = dict(
    workdir = "workdir/deep_analyzer_light_agent",
    name = "deep_analyzer_light_agent",
    type = "Agent",
    description = "A lightweight deep analysis agent that performs single-round analysis of tasks with optional image attachments.",
    model_name = "openrouter/gemini-3.1-flash-lite-preview",
    memory_name = "general_memory_system",
    analyzer_llm_models = [
        "openrouter/gemini-3.1-pro-preview",
        "openrouter/claude-opus-4.6",
        "openrouter/gpt-5.4-pro",
    ],
    require_grad = False,
)
