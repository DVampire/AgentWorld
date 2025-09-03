from mmengine.config import read_base
with read_base():
    from .environments.trading_offline import dataset, environment, metric

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

trading_offline_tool = dict(
    dataset = dataset,
    environment = environment,
    metric = metric
)