deep_researcher_light_agent = dict(
    workdir = "workdir/deep_researcher_light",
    name = "deep_researcher_light_agent",
    type = "Agent",
    description = "A lightweight deep research agent that performs single-round web search and content analysis, producing a concise summary.",
    model_name = "openrouter/gemini-3.1-flash-lite-preview",
    memory_name = "general_memory_system",
    require_grad = False,
    use_llm_search = False,
    search_llm_models = [
        "openrouter/gemini-3.1-flash-lite-preview-plugins",
    ],
)
