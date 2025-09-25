from mmengine.config import read_base
with read_base():
    from .base import memory
    from .agents.simple_chat import simple_chat_agent
    from .agents.debate_manager import debate_manager_agent

tag = "multi_agent_debate"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = False
version = "0.1.0"

env_names = []
agent_names = ["alice", "bob", "debate_manager", "simple_chat"]
tool_names = []

#-----------------ALICE AGENT CONFIG-----------------
alice_agent = simple_chat_agent.copy()
alice_agent.update(
    workdir=workdir,
    name="alice",
    description="Alice is a helpful assistant. She is an expert with professional knowledge in the field of finance and stocks.",
)

#-----------------BOB AGENT CONFIG-----------------
bob_agent = simple_chat_agent.copy()
bob_agent.update(
    workdir=workdir,
    name="bob",
    description="Bob is a helpful assistant. He is an expert with professional knowledge in the field of mathematics.",
)

#-----------------DEBATE MANAGER CONFIG-----------------
debate_manager_agent = debate_manager_agent.copy()
debate_manager_agent.update(
    workdir=workdir,
    name="debate_manager",
    description="A debate manager that coordinates multiple agents in a debate.",
)
