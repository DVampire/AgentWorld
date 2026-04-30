"""Configuration for BrowserUseAgent."""

browser_use_agent = dict(
    workdir="workdir/browser_use_agent",
    name="browser_use_agent",
    description=(
        "A standalone browser-use execution agent for real webpage interaction tasks."
    ),
    model_name="newapi/gemini-3.1-pro-preview",
    prompt_name="browser_use",
    memory_name=None,
    base_dir="workdir/browser_use_agent/browser",
    max_browser_steps=50,
    browser_start_timeout_sec=120,
    require_grad=False,
)
