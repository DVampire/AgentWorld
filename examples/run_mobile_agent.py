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
from src.infrastructures.models import model_manager
from src.tools import tcp
from src.environments import ecp
from src.agents import acp
from src.transformation import transformation

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "mobile_agent.py"), help="config file path")

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
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize(config.tool_names)
    logger.info(f"| ✅ Tools initialized: {tcp.list()}")

    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(config.agent_names)
    logger.info(f"| ✅ Agents initialized: {acp.list()}")
    
    # Transformation ECP to TCP
    logger.info("| 🔄 Transformation start...")
    await transformation.transform(type="e2t", env_names=config.env_names)
    logger.info(f"| ✅ Transformation completed: {tcp.list()}")
    
    # Example task
    task = "Open the '备忘录' app and write 'Hello, World!'."
    files = []
    
    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    
    input = {
        "name": "mobile",
        "input": {
            "task": task,
            "files": files
        }
    }
    await acp.ainvoke(**input)
    
if __name__ == "__main__":
    asyncio.run(main())