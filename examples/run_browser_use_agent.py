"""Run BrowserUseAgent as a standalone example.

Usage:
    python examples/run_browser_use_agent.py
    python examples/run_browser_use_agent.py --task "Open example.com and return page title."
    python examples/run_browser_use_agent.py --files path/to/file1 path/to/file2
"""

import os
import sys
import json
from pathlib import Path
import argparse
import asyncio

from dotenv import load_dotenv
from mmengine import DictAction

load_dotenv(verbose=True)

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
from src.session import SessionContext


def parse_args():
    parser = argparse.ArgumentParser(description="Run BrowserUseAgent example")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "browser_use_agent.py"),
        help="config file path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Open https://github.com/DVampire/Autogenesis and tell me the page title and main heading.",
        help="browser task to execute",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="optional attached files",
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

    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize()
    logger.info(f"| ✅ Model manager initialized: {await model_manager.list()}")

    logger.info("| 📁 Initializing prompt manager...")
    await prompt_manager.initialize()
    logger.info(f"| ✅ Prompt manager initialized: {await prompt_manager.list()}")

    logger.info("| 📁 Initializing memory manager...")
    memory_names = getattr(config, "memory_names", [])
    await memory_manager.initialize(memory_names=memory_names)
    logger.info(f"| ✅ Memory manager initialized: {await memory_manager.list()}")

    logger.info("| 🛠️ Initializing tools...")
    tool_names = getattr(config, "tool_names", [])
    await tool_manager.initialize(tool_names=tool_names)
    logger.info(f"| ✅ Tools initialized: {await tool_manager.list()}")

    logger.info("| 🎯 Initializing skills...")
    skill_names = getattr(config, "skill_names", None)
    await skill_manager.initialize(skill_names=skill_names)
    logger.info(f"| ✅ Skills initialized: {await skill_manager.list()}")

    logger.info("| 🤖 Initializing agents...")
    await agent_manager.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await agent_manager.list()}")

    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")

    task = args.task
    files = args.files
    ctx = SessionContext()

    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    logger.info(f"| 🆔 Session: {ctx.id}")

    response = await agent_manager(
        name="browser_use_agent",
        input={
            "task": task,
            "files": files,
        },
        ctx=ctx,
    )

    logger.info("=" * 60)
    logger.info(f"| Success: {response.success}")
    logger.info(f"| Message: {response.message}")
    if response.extra and response.extra.data:
        logger.info(f"| Extra: {json.dumps(response.extra.data, indent=2, ensure_ascii=False)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
