from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.esg_agent import esg_agent
    from .tools.retriever import retriever_tool
    from .tools.plotter import plotter_tool
    from .tools.reporter import reporter_tool
    from .tools.browser import browser_tool
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.deep_analyzer import deep_analyzer_tool
    from .tools.mdify import mdify_tool
    from .memory.general_memory_system import memory_system as general_memory_system

tag = "esg_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "1.0.0"
model_name = "openrouter/gemini-3-flash-preview"
concurrency = 1

env_names = []
memory_names = [
    "general_memory_system"
]
agent_names = [
    "esg_agent"
]
tool_names = [
    'bash',        
    'python_interpreter', 
    'done', 
    'todo',  
    'retriever',
    'plotter',
    'reporter',
    'browser',
    'deep_researcher',
    'deep_analyzer',
    "reformulator",
]

#-----------------RETRIEVER TOOL CONFIG-----------------
retriever_tool.update(
    base_dir=f"{workdir}/tool/retriever",
    model_name=model_name,
    top_k=5,
    query_mode="naive",
    extract_metadata=True,
)
#-----------------PLOTTER TOOL CONFIG-----------------
plotter_tool.update(
    model_name=model_name,
    base_dir=f"{workdir}/tool/plotter",
)
#-----------------BROWSER TOOL CONFIG-----------------
browser_tool.update(
    model_name="openrouter/gpt-4.1",
    base_dir=f"{workdir}/tool/browser",
)
#-----------------DEEP RESEARCHER TOOL CONFIG-----------------
deep_researcher_tool.update(
    model_name="openrouter/gemini-3-flash-preview",
    base_dir=f"{workdir}/tool/deep_researcher",
    use_llm_search = True, # Only use LLM search if search_llm_models is provided
    search_llm_models = ["openrouter/gemini-3-flash-preview-plugins"]
)

#-----------------DEEP ANALYZER TOOL CONFIG-----------------
deep_analyzer_tool.update(
    model_name="openrouter/gemini-3-flash-preview",
    base_dir=f"{workdir}/tool/deep_analyzer",
)
#-----------------REPORT TOOL CONFIG-----------------
reporter_tool.update(
    model_name=model_name,
    base_dir=f"{workdir}/tool/reporter",
)
#-----------------MDIFY TOOL CONFIG-----------------
mdify_tool.update(
    base_dir=f"{workdir}/tool/mdify",
)
#-----------------GENERAL MEMORY SYSTEM CONFIG-----------------
general_memory_system.update(
    base_dir=f"{workdir}/memory/general_memory_system",
    model_name=model_name,
    max_summaries=10,
    max_insights=10,
)
#-----------------ESG AGENT CONFIG-----------------
esg_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0]
)

