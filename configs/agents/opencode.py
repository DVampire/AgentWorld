opencode_agent = dict(
    workdir="workdir/opencode_agent",
    name="opencode_agent",
    type="Agent",
    description=(
        "Coding agent powered by the OpenCode CLI (opencode). "
        "Has access to Read, Write, Edit, Bash, Glob, Grep, WebSearch and WebFetch tools. "
        "Best for multi-file coding tasks, debugging, and navigating real codebases. "
    ),
    model_name="openrouter/claude-opus-4.6",
    require_grad=False,
    max_iterations=30,
)
