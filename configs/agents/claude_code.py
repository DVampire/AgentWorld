claude_code_agent = dict(
    workdir="workdir/bus/agent/claude_code_agent",
    name="claude_code_agent",
    type="Agent",
    description=(
        "Coding agent powered by the Claude Code CLI (claude-agent-sdk). "
        "Has access to Read, Write, Edit, Bash, Glob, Grep, WebSearch and WebFetch tools. "
        "Best for multi-file coding tasks, debugging, and navigating real codebases. "
        "Requires ANTHROPIC_API_KEY and the claude CLI binary."
    ),
    model_name="claude-opus-4-6",
    require_grad=False,
    max_iterations=30,
)
