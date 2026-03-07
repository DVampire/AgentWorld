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
from src.tool import tcp
from src.skill import scp
from src.environment import ecp
from src.agent import acp
from src.transformation import transformation
from src.session.types import SessionContext
from src.utils import generate_unique_id

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")

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
    await tcp.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tcp.list()}")
    
    # Initialize skills
    logger.info("| 🎯 Initializing skills...")
    skill_names = getattr(config, 'skill_names', None)
    await scp.initialize(skill_names=skill_names)
    logger.info(f"| ✅ Skills initialized: {await scp.list()}")

    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await acp.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    
    # Example task
    # task = """If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."""
    # task = """Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations."""
    # task = "Write a mini game about a cat that can fly and fight enemies, and then push it to github."
    task = "Generate an add two numbers skill to add 1 and 2 and return the result."
    # task = "开启一个AI能力自博弈讨论，讨论主题和行业是：AI在生物医药行业的应用。"
    files = []
    
    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    
    # Session context
    ctx = SessionContext()
    
    input = {
        "name": "tool_calling",
        "input": {
            "task": task,
            "files": files
        },
        "ctx": ctx
    }
    await acp(**input)
    
if __name__ == "__main__":
    asyncio.run(main())