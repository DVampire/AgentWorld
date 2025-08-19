from mmengine.registry import Registry

AGENTS = Registry("agents", locations=["src.agents"])
ENVIRONMENTS = Registry("environments", locations=["src.environments"])