"""Example of running the Browser-Use Agent for web automation tasks."""

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
    parser = argparse.ArgumentParser(description='Browser-Use Agent Example')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")
    parser.add_argument("--task", default="访问百度搜索北京明天天气，获取明天的天气预报信息", help="browser task to execute")

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
    file_system_controller_config = config.file_system_controller
    file_system_controller = CONTROLLERS.build(file_system_controller_config)
    controllers.append(file_system_controller)
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
    
    """Browser-use agent functionality test."""
    logger.info("| 🌐 Testing browser-use agent functionality")
    
    # Browser task
    task = args.task
    logger.info(f"| 📋 Browser Task: {task}")
    
    try:
        # Run the browser agent
        logger.info("| 🚀 Starting browser agent execution...")
        result = await agent.run(task)
        
        logger.info("| ✅ Browser task completed!")
        logger.info("| 📄 Browser Result:")
        logger.info("| " + "-"*50)
        if isinstance(result, dict):
            logger.info(f"| Final Response: {result.get('final_response', result)}")
            logger.info(f"| Success: {result.get('success', 'Unknown')}")
            logger.info(f"| Iterations: {result.get('iterations', 'Unknown')}")
        else:
            logger.info(f"| Result: {result}")
        logger.info("| " + "-"*50)
        
    except KeyboardInterrupt:
        logger.info("| 👋 User interrupted the execution")
    except Exception as e:
        logger.error(f"| ❌ Browser task failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
