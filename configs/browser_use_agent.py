from mmengine.config import read_base

with read_base():
    from .base import max_tokens, window_size
    from .agents.browser_use import browser_use_agent


tag = "browser_use_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"
model_name = "newapi/gemini-3.1-pro-preview"

tool_names = []
memory_names = []
skill_names = []
agent_names = ["browser_use_agent"]


browser_use_agent.update(
    workdir=f"{workdir}/agent/browser_use_agent",
    base_dir=f"{workdir}/browser",
    model_name=model_name,
    prompt_name="browser_use",
    require_grad=False,
)
