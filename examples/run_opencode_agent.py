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
from src.session.types import SessionContext
from src.utils import assemble_project_path

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "opencode_agent.py"), help="config file path")

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
    
    # Initialize version manager
    logger.info("| Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {await version_manager.list()}")

    # Initialize model manager
    logger.info("| Initializing model manager...")
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
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await agent_manager.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await agent_manager.list()}")
    
    # Example task
    task = "Generate a Python function to calculate the factorial of a number and return the result of factorial(5)."  # You can change this to any coding task you want
    files = []
    
    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    
    # Session context
    ctx = SessionContext()
    
    input = {
        "name": "opencode_agent",
        "input": {
            "task": task,
            "files": files
        },
        "ctx": ctx
    }
    response = await agent_manager(**input)
    logger.info(f"| 📤 Response: {response.message}")
    
if __name__ == "__main__":
    asyncio.run(main())