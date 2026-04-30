"""Run a task through the v3 AgentBus.

Validates the five new v3 agents end-to-end:
  - PlanningAgent      (planning.py)
  - DeepResearcherV3   (deep_researcher_v3.py)
  - DeepAnalyzerV3     (deep_analyzer_v3.py)
  - OpencodeAgent v2   (opencode_agent_v2.py)
  - SopAgent           (sop.py)

Usage:
    python examples/run_v3_bus.py
    python examples/run_v3_bus.py --config configs/v3_bus.py
    python examples/run_v3_bus.py --task "your task here"
    python examples/run_v3_bus.py --max-rounds 5
"""

import os
import sys
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.prompt import prompt_manager
from src.memory import memory_manager
from src.tool import tool_manager
from src.skill import skill_manager
from src.agent import agent_manager
from src.interaction import bus
from src.task import Task
from src.session import SessionContext


def parse_args():
    parser = argparse.ArgumentParser(description="Run a task through the v3 AgentBus")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "v3_bus.py"),
        help="config file path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Write a Python function that computes the nth Fibonacci number using memoization, then test it for n=10.",
        help="task to run",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="maximum planner rounds before giving up",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config settings in xxx=yyy format",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")

    logger.info("| Initializing model manager...")
    await model_manager.initialize()
    logger.info(f"| Model manager ready: {await model_manager.list()}")

    logger.info("| Initializing prompt manager...")
    await prompt_manager.initialize()
    logger.info(f"| Prompt manager ready: {await prompt_manager.list()}")

    logger.info("| Initializing memory manager...")
    await memory_manager.initialize(memory_names=config.memory_names)
    logger.info(f"| Memory manager ready: {await memory_manager.list()}")

    logger.info("| Initializing tools...")
    await tool_manager.initialize(tool_names=config.tool_names)
    logger.info(f"| Tools ready: {await tool_manager.list()}")

    logger.info("| Initializing skills...")
    skill_names = getattr(config, "skill_names", None)
    await skill_manager.initialize(skill_names=skill_names)
    logger.info(f"| Skills ready: {await skill_manager.list()}")

    logger.info("| Initializing v3 agents...")
    await agent_manager.initialize(agent_names=config.agent_names)
    logger.info(f"| Agents ready: {await agent_manager.list()}")

    logger.info("| Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| Version manager ready: {json.dumps(await version_manager.list(), indent=4)}")

    logger.info("| Initializing AgentBus...")
    await bus.initialize()
    logger.info(f"| Bus agents: {await bus.list()}")

    ctx = SessionContext()
    task = Task(content=args.task, session_id=ctx.id)

    logger.info(f"| Task: {task.content}")
    logger.info(f"| Session: {ctx.id}")
    logger.info(f"| Max rounds: {args.max_rounds}")

    logger.info("| Submitting task to v3 AgentBus...")
    response = await bus.submit(task, ctx=ctx, max_rounds=args.max_rounds)

    success = response.payload.get("success", False)
    result = response.payload.get("result", "")
    error = response.payload.get("error")
    
    

    logger.info("=" * 60)
    if success:
        logger.info("| Task completed successfully")
        logger.info(f"| Result: {result}")
    else:
        logger.info("| Task failed")
        logger.info(f"| Error: {error or result}")
    logger.info("=" * 60)

    events = await bus.get_event_log(session_id=ctx.id)
    logger.info(f"| Bus events for this session: {len(events)}")
    for evt in events:
        agent = evt.agent_name or "-"
        detail = evt.detail or ""
        logger.info(f"|   \\[{evt.event_type}] {agent} {('| ' + detail) if detail else ''}")

    await bus.shutdown()
    logger.info("| Done.")


if __name__ == "__main__":
    asyncio.run(main())
