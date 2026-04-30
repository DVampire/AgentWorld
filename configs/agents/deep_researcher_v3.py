deep_researcher_v3_agent = dict(
    workdir="workdir/deep_researcher_v3_agent",
    name="deep_researcher_v3_agent",
    description=(
        "Deep research agent (v3) that performs multi-round web search with "
        "conflict detection, supporting both text and multimodal image queries."
    ),
    model_name="newapi/gemini-3.1-pro-preview",
    prompt_name="deep_researcher_v3",
    memory_name="general_memory_system",
    max_rounds=3,
    max_steps=10,
    num_results=10,
    llm_search_models=[
        "openrouter/gemini-3.1-pro-preview-plugins",
    ],
    enable_search_log=True,
    require_grad=False,
)
