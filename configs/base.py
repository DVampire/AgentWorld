#---------------GENERAL CONFIG-------------
tag = "base"
workdir = f"workdir/{tag}"
log_path = "base.log"
use_local_proxy = False

#---------------TOOLS CONFIG---------------
browser_tool = dict(
    model_name = "bs-gpt-4.1",
)

deep_researcher_tool = dict(
    model_name = "o3",
)

deep_analyzer_tool = dict(
    model_name = "o3",
)

#---------------MEMORY CONFIG---------------
memory = dict(
    model_name = "gpt-4.1",
    max_summaries = 20,
    max_insights = 100
)