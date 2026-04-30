from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.planning import planning_agent
    from .agents.deep_researcher_v3 import deep_researcher_v3_agent
    from .agents.deep_analyzer_v3 import deep_analyzer_v3_agent
    from .agents.opencode import opencode_agent

    from .tools.bash import bash_tool
    from .tools.todo import todo_tool
    from .memory.general_memory_system import memory_system as general_memory_system
    from .memory.optimizer_memory_system import memory_system as optimizer_memory_system
    from .benchmarks.gaia import gaia_benchmark

tag = "v3_bus"
workdir = f"workdir/{tag}"
log_path = "v3_bus.log"
max_tokens = 32768

use_local_proxy = True
version = "0.1.0"
model_name = "int_openrouter/gemini-3.1-pro-preview"

tool_names = [
    "bash_tool",
    "python_interpreter_tool",
    "done_tool",
    "todo_tool",
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system",
]
agent_names = [
    "planning_agent",
    "deep_researcher_v3_agent",
    "deep_analyzer_v3_agent",
    "opencode_agent"
]
skill_names = [
    "hello_world_skill",
]

# -----------------TOOL CONFIG-----------------
bash_tool.update(require_grad=False)
todo_tool.update(
    base_dir="tool/todo_tool",
    require_grad=False,
)

# -----------------MEMORY CONFIG-----------------
general_memory_system.update(
    base_dir="memory/general_memory_system",
    model_name="int_openrouter/gemini-3-flash-preview",
    max_summaries=10,
    max_insights=10,
    require_grad=False,
)
optimizer_memory_system.update(
    base_dir="memory/optimizer_memory_system",
    model_name="int_openrouter/gemini-3-flash-preview",
    max_records_per_session=10,
    require_grad=False,
)

# -----------------AGENT CONFIG-----------------
planning_agent.update(
    workdir=f"{workdir}/agent/planning_agent",
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    max_rounds=20,
)
deep_researcher_v3_agent.update(
    workdir=f"{workdir}/agent/deep_researcher_v3_agent",
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    max_rounds = 3,
    max_steps = 20,
    num_results=10,
    summary_model_name="int_openrouter/grok-4.1-fast",
    llm_search_models=[
        "int_openrouter/gemini-3.1-pro-preview-plugins", # official search model with plugins support
    ],
    fetch_timeout=20.0
)
deep_analyzer_v3_agent.update(
    workdir=f"{workdir}/agent/deep_analyzer_v3_agent",
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    max_rounds = 3,
    max_steps = 20,
    summary_model_name="int_openrouter/grok-4.1-fast",
    general_analyze_models = [
        "int_openrouter/gemini-3.1-pro-preview-plugins", # can analyze audio, video, image, pdf, etc.
    ],
    llm_analyze_models = [
        "int_openrouter/gpt-5.4", # can analyze complex text with image
    ],
    advanced_analyze_models = [
        "int_openrouter/gpt-5.4-pro", # expensive but powerful model for deep analysis
    ],
    use_memory=True
)
opencode_agent.update(
    workdir=f"{workdir}/agent/opencode_agent",
    model_name="int_openrouter/claude-opus-4.6",
    summary_model_name="int_openrouter/grok-4.1-fast",
    memory_name=memory_names[0],
    require_grad=False,
)
