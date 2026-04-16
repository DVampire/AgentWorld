from mmengine.registry import Registry

COLLATE_FN = Registry("collate_fn", locations=["src.data"])
DATALOADER = Registry("dataloader", locations=["src.data"])
SCALER = Registry("scaler", locations=["src.data"])
DATASET = Registry("dataset", locations=["src.data"])
METRIC = Registry("metric", locations=["src.metric"])
INDICATOR = Registry("indicator", locations=["src.indicator"])

MEMORY_SYSTEM = Registry("memory_system", locations=["src.memory"])
TOOL = Registry("tool", locations=["src.tool"])
ENVIRONMENT = Registry("environment", locations=["src.environment"])
AGENT = Registry("agent", locations=["src.agent"])
PROMPT = Registry("prompt", locations=["src.prompt"])
DOWNLOADER = Registry("downloader", locations=["src.download"])
PROCESSOR = Registry("processor", locations=["src.process"])
BENCHMARK = Registry("benchmark", locations=["src.benchmark"])
SKILL = Registry("skill", locations=["src.skill"])