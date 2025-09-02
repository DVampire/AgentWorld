from mmengine.registry import Registry

AGENTS = Registry("agents", locations=["src.agents"])

COLLATE_FN = Registry("collate_fn", locations=["src.datasets"])
DATALOADER = Registry("dataloader", locations=["src.datasets"])
SCALER = Registry("scaler", locations=["src.datasets"])
DATASETS = Registry("datasets", locations=["src.datasets"])

METRIC = Registry("metric", locations=["src.metric"])

ENVIRONMENTS = Registry("environments", locations=["src.environments"])