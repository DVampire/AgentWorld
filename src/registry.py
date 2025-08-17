from mmengine.registry import Registry

AGNTES = Registry("agents", locations=["src.agents"])
ENVIRONMENTS = Registry("environments", locations=["src.environments"])