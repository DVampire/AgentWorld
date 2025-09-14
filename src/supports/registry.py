from mmengine.registry import Registry
COLLATE_FN = Registry("collate_fn", locations=["src.supports.datasets"])
DATALOADER = Registry("dataloader", locations=["src.supports.datasets"])
SCALER = Registry("scaler", locations=["src.supports.datasets"])
DATASETS = Registry("datasets", locations=["src.supports.datasets"])
METRIC = Registry("metric", locations=["src.supports.metric"])