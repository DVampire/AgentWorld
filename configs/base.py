#---------------GENERAL CONFIG-------------
tag = "base"
workdir = f"workdir/{tag}"
log_path = "base.log"
use_local_proxy = False

#---------------MEMORY CONFIG---------------
memory_config = dict(
    type = "general_memory_system",
    model_name = "gpt-4.1",
    max_summaries = 20,
    max_insights = 100
)

#---------------MAX TOKENS CONFIG---------------
max_tokens = 16384

#---------------Window Size Config---------------
window_size = (1024, 768)