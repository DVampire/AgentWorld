from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.tool_calling import tool_calling_agent
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.deep_analyzer import deep_analyzer_tool
    from .tools.mdify import mdify_tool
    from .tools.plotter import plotter_tool
    from .tools.bash import bash_tool
    from .tools.todo import todo_tool
    from .tools.skill_generator import skill_generator_tool
    from .tools.future_capability_debate import future_capability_debate_tool
    from .environments.file_system import environment as file_system_environment
    from .memory.general_memory_system import memory_system as general_memory_system
    from .memory.optimizer_memory_system import memory_system as optimizer_memory_system

tag = "future_capability_debate"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
model_name = "openrouter/gemini-3-flash-preview"

env_names = [
    "file_system"
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system"
]
agent_names = [
    "tool_calling"
]
tool_names = [
    'done',
    'todo',
    "future_capability_debate",
]
skill_names = [
    "hello-world",
]

#-----------------BASH TOOL CONFIG-----------------
bash_tool.update(
    require_grad=False,
)
#-----------------MDIFY TOOL CONFIG-----------------
mdify_tool.update(
    base_dir="tool/mdify",
)
todo_tool.update(
    base_dir="tool/todo",
    require_grad=False,
)
#-----------------DEEP RESEARCHER TOOL CONFIG-----------------
deep_researcher_tool.update(
    model_name="openrouter/o3",
    base_dir="tool/deep_researcher",
)

#-----------------DEEP ANALYZER TOOL CONFIG-----------------
deep_analyzer_tool.update(
    model_name="openrouter/o3",
    base_dir="tool/deep_analyzer",
    require_grad=False,
)

#-----------------PLOTTER TOOL CONFIG-----------------
plotter_tool.update(
    model_name="openrouter/o3",
    base_dir="tool/plotter",
)
#-----------------SKILL GENERATOR TOOL CONFIG-----------------
skill_generator_tool.update(
    model_name="openrouter/gemini-3-flash-preview",
    base_dir="skill",
)
#-----------------FUTURE CAPABILITY DEBATE TOOL CONFIG-----------------
future_capability_debate_tool.update(
    model_name="openrouter/gemini-3-flash-preview",
    agent_models=[
        "openrouter/gemini-3-flash-preview",
        "openrouter/gpt-5.2",
        "openrouter/claude-sonnet-4.5",
        "openrouter/grok-4.1-fast",
    ],
    base_dir="tool/future_capability_debate",
)
#-----------------MEMORY SYSTEM CONFIG-----------------
general_memory_system.update(
    base_dir="memory/general_memory_system",
    model_name=model_name,
    max_summaries=10,
    max_insights=10,
    require_grad=False,
)
optimizer_memory_system.update(
    base_dir="memory/optimizer_memory_system",
    model_name=model_name,
    max_records_per_session=10,
    require_grad=False,
)

#-----------------FILE SYSTEM ENVIRONMENT CONFIG-----------------
file_system_environment.update(
    base_dir="environment/file_system",
    require_grad=False,
)

#-----------------TOOL CALLING AGENT CONFIG-----------------
tool_calling_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
)
