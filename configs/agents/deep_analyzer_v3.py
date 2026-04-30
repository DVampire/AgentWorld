deep_analyzer_v3_agent = dict(
    workdir="workdir/deep_analyzer_v3_agent",
    name="deep_analyzer_v3_agent",
    description=(
        "Deep analysis agent (v3) that processes text, PDF, image, audio, and video files "
        "using parallel multi-model analysis with a think-action loop."
    ),
    model_name="newapi/gemini-3.1-pro-preview",
    prompt_name="deep_analyzer_v3",
    memory_name="general_memory_system",
    max_rounds=3,
    max_steps=10,
    general_analyze_models=[
        "newapi/gemini-3.1-pro-preview",
    ],
    llm_analyze_models=[
        "newapi/gemini-3.1-pro-preview",
    ],
    require_grad=False,
)
