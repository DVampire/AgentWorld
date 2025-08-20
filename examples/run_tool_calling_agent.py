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
from src.registry import AGENTS
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
    logger.info(f"| Config: {config}")
    
    await model_manager.init_models()
    logger.info(f"| Model: {model_manager.list_models()}")
    
    await tool_manager.init_tools()
    logger.info(f"| Tool: {tool_manager.list_tools()}")
    
    agent = AGENTS.build(config.agent)
    logger.info(f"| Agent: {agent}")
    
    """Test streaming execution mode."""
    logger.info("\n" + "="*60)
    logger.info("Testing streaming execution mode")
    logger.info("="*60)
    
    task = "使用python code计算 10 + 5 的值"
    logger.info(f"| Task: {task}")
    
    logger.info("\n🔄 Streaming execution process:")
    logger.info("-" * 50)
    
    async for update in agent.run_streaming(task):
        # Print the update in a user-friendly format
        if update["type"] == "task_start":
            logger.info(f"🚀 Task started: {update['task']}")
            logger.info(f"   Agent: {update['agent_name']}")
            
        elif update["type"] == "tool_calling":
            logger.info(f"   🔧 Agent calling tool: {update['tool_name']}")
            logger.info(f"      Input: {update['tool_input']}")
            
        elif update["type"] == "tool_result":
            logger.info(f"   ✅ Tool {update['tool_name']} succeeded")
            logger.info(f"      Result: {update['result'][:100]}...")
            
        elif update["type"] == "final_response":
            logger.info(f"\n🏁 Final Response:")
            logger.info(f"   Final Response: {update['final_response']}")
            
        # Simulate some delay for demonstration
        await asyncio.sleep(0.5)
    
    logger.info("-" * 50)
    logger.info("| Streaming execution completed")


if __name__ == "__main__":
    asyncio.run(main())