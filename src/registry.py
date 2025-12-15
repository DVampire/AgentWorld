from mmengine.registry import Registry

COLLATE_FN = Registry("collate_fn", locations=["src.data"])
DATALOADER = Registry("dataloader", locations=["src.data"])
SCALER = Registry("scaler", locations=["src.data"])
DATASET = Registry("dataset", locations=["src.data"])
METRIC = Registry("metric", locations=["src.metric"])
INDICATOR = Registry("indicator", locations=["src.indicator"])

MEMORY_SYSTEM = Registry("memory_system", locations=["src.memory"])
TOOL = Registry("tool", locations=["src.tool"])