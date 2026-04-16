import os
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio
import json

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.tools import tool_manager
from src.environments import environment_manager
from src.agents import agent_manager
from src.transformation import transformation
from src.models import model_manager
from src.environments.hyperliquidentry.service import OnlineHyperliquidService
from src.utils import get_env

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "online_trading_agent.py"), help="config file path")

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
    
    # Initialize Hyperliquid service
    logger.info("| 🔧 Initializing Hyperliquid service...")
    hyperliquid_service = None
    try:
        accounts = get_env("HYPERLIQUID_ACCOUNTS").get_secret_value()
        if accounts:
            accounts = json.loads(accounts)
            config.hyperliquid_service.update(dict(accounts=accounts))
            hyperliquid_service = OnlineHyperliquidService(**config.hyperliquid_service)
        else:
            raise ImportError("HYPERLIQUID_ACCOUNTS is not set in .env file.")
    except Exception as e:
        raise ImportError(f"Error initializing OnlineHyperliquidService: {e}") from e
    await hyperliquid_service.initialize()
    for env_name in config.env_names:
        env_config = config.get(f"{env_name}_environment", None)
        env_config.update(dict(hyperliquid_service=hyperliquid_service))
    logger.info(f"| ✅ Hyperliquid service initialized.")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await environment_manager.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {environment_manager.list()}")
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tool_manager.initialize(config.tool_names)
    logger.info(f"| ✅ Tools initialized: {tool_manager.list()}")

    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await agent_manager.initialize(config.agent_names)
    logger.info(f"| ✅ Agents initialized: {agent_manager.list()}")
    
    # Transformation ECP to TCP
    logger.info("| 🔄 Transformation start...")
    await transformation.transform(type="e2t", env_names=config.env_names)
    logger.info(f"| ✅ Transformation completed: {tool_manager.list()}")
    
    # Example task
    task = "Start trading on BTC and maximize the profit until the environment is done."
    files = []
    
    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    
    input = {
        "name": "online_trading",
        "input": {
            "task": task,
            "files": files
        }
    }
    await agent_manager.ainvoke(**input)
    
if __name__ == "__main__":
    asyncio.run(main())