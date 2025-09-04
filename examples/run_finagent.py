"""Example of running the InteractiveAgent with Cursor-style interaction."""

import os
import sys
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
from src.registry import AGENTS
from src.registry import CONTROLLERS
from src.registry import ENVIRONMENTS
from src.registry import DATASETS
from src.models import model_manager
from src.tools import tool_manager

def parse_args():
    parser = argparse.ArgumentParser(description='FinAgent Example')
    parser.add_argument("--config", default=os.path.join(root, "configs", "finagent.py"), help="config file path")
    
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
    
    # Initialize configuration
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list_models()}")
    
    # Initialize controllers
    logger.info("| 🎮 Initializing controllers...")
    controllers = []
    dataset_config = config.dataset
    dataset = DATASETS.build(dataset_config)
    environment_config = config.environment
    environment_config.update(dict(
        dataset=dataset
    ))
    environment = ENVIRONMENTS.build(environment_config)
    controller_config = config.controller
    controller_config.update(dict(
        environment=environment
    ))
    controller = CONTROLLERS.build(controller_config)
    controllers.append(controller)
    logger.info(f"| ✅ Controllers initialized: {controllers}")
    
    # Initialize tool manager
    logger.info("| 🛠️ Initializing tool manager...")
    await tool_manager.init_tools(controllers)
    logger.info(f"| ✅ Tool manager initialized: {tool_manager.list_tools()}")

if __name__ == "__main__":
    asyncio.run(main())
