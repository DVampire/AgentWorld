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
from src.models import model_manager
from src.tools import tool_manager

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Agent Example')
    parser.add_argument("--config", default=os.path.join(root, "configs", "interactive_agent.py"), help="config file path")
    
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
    logger.info("| 🔧 Initializing model manager...")
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list_models()}")
    
    # Initialize tool manager
    logger.info("| 🔧 Initializing tool manager...")
    await tool_manager.init_tools()
    logger.info(f"| ✅ Tool manager initialized: {tool_manager.list_tools()}")
    
    # Build interactive agent
    logger.info("| 🔧 Building interactive agent...")
    agent = AGENTS.build(config.agent)
    logger.info(f"| ✅ Interactive agent built: {agent}")
    logger.info(f"| 🎮 Interactive mode: {'ON' if agent.interactive_mode else 'OFF'}")
    logger.info(f"| ⚡ Auto continue: {'ON' if agent.auto_continue else 'OFF'}")
    
    # Get task from user or command line
    task = "Write a Python script that calculates the Fibonacci sequence."
    logger.info(f"| 📋 Task: {task}")
    
    try:
        # Run the interactive agent
        logger.info("| 🚀 Starting interactive execution...")
        result = await agent.run(task)
        
        logger.info("| ✅ Task completed!")
        logger.info(f"| 📊 Final result: {result}")
        
    except KeyboardInterrupt:
        logger.info("| 👋 User interrupted the execution")
    except Exception as e:
        logger.error(f"| ❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
