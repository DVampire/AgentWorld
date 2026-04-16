deep_researcher_agent = dict(
    workdir = "workdir/deep_researcher",
    name = "deep_researcher_agent",
    type = "Agent",
    description = "A deep research agent that performs multi-round web search and content analysis, producing a structured Markdown report.",
    model_name = "openrouter/gemini-3-flash-preview",
    memory_name = "general_memory_system",
    require_grad = False,
    # Research-specific config
    max_rounds = 3,
    num_results = 5,
    use_llm_search = False,
    search_llm_models = [
        "openrouter/o3-deep-research",
        "openrouter/sonar-deep-research",
    ],
)
