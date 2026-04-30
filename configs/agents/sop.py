sop_agent = dict(
    workdir="workdir/sop_agent",
    name="sop_agent",
    description=(
        "SOP agent that executes domain-specific Standard Operating Procedures "
        "phase-by-phase using registered skills and tools."
    ),
    model_name="newapi/gemini-3.1-pro-preview",
    prompt_name="sop",
    memory_name="general_memory_system",
    max_tools=10,
    max_steps=20,
    review_steps=5,
    require_grad=False,
)
