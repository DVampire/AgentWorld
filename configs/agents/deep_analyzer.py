deep_analyzer_agent = dict(
    workdir = "workdir/deep_analyzer",
    name = "deep_analyzer_agent",
    type = "Agent",
    description = "A deep analysis agent that performs multi-step analysis of tasks with files (text, PDF, image, audio, video).",
    model_name = "openrouter/gemini-3-flash-preview",
    file_model_name = "openrouter/gemini-3-flash-preview-plugins",
    memory_name = "general_memory_system",
    require_grad = False,
    max_rounds = 3,
    max_steps = 3,
    chunk_size = 500,
)
