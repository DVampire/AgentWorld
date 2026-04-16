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
from src.environment import environment_manager
from src.agent import agent_manager
from src.transformation import transformation
from src.session.types import SessionContext
from src.utils import generate_unique_id

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "future_capability_debate.py"), help="config file path")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

async def main():
    args = parse_args()
    
    config.initialize(config_path = args.config, args = args)
    logger.initialize(config = config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize()
    logger.info(f"| ✅ Model manager initialized: {await model_manager.list()}")
    
    # Initialize prompt manager
    logger.info("| 📁 Initializing prompt manager...")
    await prompt_manager.initialize()
    logger.info(f"| ✅ Prompt manager initialized: {await prompt_manager.list()}")
    
    # Initialize memory manager
    logger.info("| 📁 Initializing memory manager...")
    await memory_manager.initialize(memory_names=config.memory_names)
    logger.info(f"| ✅ Memory manager initialized: {await memory_manager.list()}")
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tool_manager.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tool_manager.list()}")
    
    # Initialize skills
    logger.info("| 🎯 Initializing skills...")
    skill_names = getattr(config, 'skill_names', None)
    await skill_manager.initialize(skill_names=skill_names)
    logger.info(f"| ✅ Skills initialized: {await skill_manager.list()}")

    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await environment_manager.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {environment_manager.list()}")
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await agent_manager.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await agent_manager.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    
    sector_file = os.path.join(os.path.dirname(__file__), "files", "sector.txt")
    with open(sector_file, "r", encoding="utf-8") as f:
        sectors = [line.strip() for line in f if line.strip()]
    logger.info(f"| 📋 Loaded {len(sectors)} sectors from {sector_file}")

    concurrency = getattr(config, "concurrency", 4)
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    total_count = len(sectors)

    async def run_with_semaphore(sector: str):
        nonlocal completed_count
        async with semaphore:
            task = f"发掘{sector}行业中因 AI 出现而被根本性重塑的未来能力，包括 AI 时代催生的全新能力和长期被低估但因稀缺性被放大的能力。"
            logger.info(f"| 🚀 Starting task for sector: {sector}")
            ctx = SessionContext()
            input = {
                "name": "tool_calling",
                "input": {
                    "task": task,
                    "files": []
                },
                "ctx": ctx
            }
            try:
                await agent_manager(**input)
            except Exception as e:
                logger.error(f"| ❌ Failed task for sector: {sector}, error: {e}")
            finally:
                completed_count += 1
                if completed_count % concurrency == 0 or completed_count == total_count:
                    logger.info(f"| 📊 Progress: {completed_count}/{total_count} tasks completed")

    tasks = [run_with_semaphore(sector) for sector in sectors]
    await asyncio.gather(*tasks)
    logger.info(f"| ✅ All {total_count} tasks completed.")
    
if __name__ == "__main__":
    asyncio.run(main())
