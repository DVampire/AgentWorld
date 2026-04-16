"""Example of running the InteractiveAgent with Cursor-style interaction."""
import os
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import json
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.memory import memory_manager
from src.prompt import prompt_manager
from src.environment import environment_manager
from src.tool import tool_manager
from src.version import version_manager
from src.utils import assemble_project_path


def parse_args():
        parser = argparse.ArgumentParser(description='Online Trading Agent Example')
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

async def test_file_system():
    
    input = {
        "name": "file_system",
        "action": "read",
        "input": {
            "file_path": assemble_project_path("tests/files/test.txt")
        }
    }
    
    res = await environment_manager(**input)
    logger.info(f"| 📝 Action result: {res}")
    
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
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await environment_manager.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {await environment_manager.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    
    await test_file_system()
    

if __name__ == "__main__":
    asyncio.run(main())
