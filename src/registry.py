from mmengine.registry import Registry

COLLATE_FN = Registry("collate_fn", locations=["src.data"])
DATALOADER = Registry("dataloader", locations=["src.data"])
SCALER = Registry("scaler", locations=["src.data"])
DATASETS = Registry("dataset", locations=["src.data"])
METRIC = Registry("metric", locations=["src.metrics"])
INDICATOR = Registry("indicator", locations=["src.indicators"])