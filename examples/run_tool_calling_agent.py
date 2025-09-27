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
from src.models import model_manager
from src.tools import tool_manager

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
    
    # File system controller
    file_system_controller_config = config.file_system_controller
    file_system_controller = CONTROLLERS.build(file_system_controller_config)
    controllers.append(file_system_controller)
    
    # Browser controller
    browser_controller_config = config.browser_controller
    browser_controller = CONTROLLERS.build(browser_controller_config)
    controllers.append(browser_controller)
    
    logger.info(f"| ✅ Controllers initialized: {controllers}")
    
    # Initialize tool manager
    logger.info("| 🛠️ Initializing tool manager...")
    await tool_manager.init_tools(controllers)
    logger.info(f"| ✅ Tool manager initialized: {tool_manager.list_tools()}")
    
    # Build agent
    logger.info("| 🎮 Building agent...")
    agent_config = config.agent
    agent_config.update(dict(
        controllers=controllers
    ))
    agent = AGENTS.build(agent_config)
    logger.info(f"| ✅ Agent built: {agent}")
    
    """Test browser tool integration."""
    logger.info("| 🚀 Testing browser tool integration")
    
    task = "使用browser中的工具，访问百度网站，搜索北京明天天气预报，给出答案"
    logger.info(f"| 📋 Task: {task}")
    
    await agent.run(task)

if __name__ == "__main__":
    asyncio.run(main())