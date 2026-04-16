"""AgentBus — session-isolated message bus that drives the planner loop.

Startup::

    from src.agent.server import agent_manager
    from src.interaction import bus

    await agent_manager.initialize()    # register agents
    await bus.initialize()    # sync agent names into bus

Submit a task::

    from src.task import Task
    from src.session import SessionContext

    ctx = SessionContext()
    task = Task(content="Analyse the Q3 earnings report.", session_id=ctx.id)
    response = await bus.submit(task, session_ctx=ctx)
    print(response.payload)

Flow::

    submit(task) → session queue → session worker
      └→ planner round 1: call planner → PlanDecision
           └→ dispatch agents concurrently (asyncio.gather)
           └→ collect results
      └→ planner round 2: call planner with results → PlanDecision
           └→ dispatch / done
      └→ ...
      └→ planner returns is_done=True → resolve caller Future
"""

from .bus import AgentBus, bus
from .types import BusEvent, BusMessage, BusMessageType, DeliveryMode

__all__ = [
    "bus",
    "AgentBus",
    "BusMessage",
    "BusMessageType",
    "DeliveryMode",
    "BusEvent",
]
