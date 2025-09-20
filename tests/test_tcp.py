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
from src.infrastructures import model_manager

def parse_args():
    parser = argparse.ArgumentParser(description='Tool Calling Agent Example')
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

async def test_bash():
    from src.tools import tcp
    try:
        print("🔌 Registering MCP tools with TCP...")
        
        tools = tcp.list_tools()
        for index, tool in enumerate(tools):
            print(f"Tool {index}: {tool.name}, Type: {tool.type}, Description: {tool.description}")
            
        tool = tcp.get_tool("bash")
        input = {
            "command": "echo 'Hello, World!'"
        }
        result = await tool.ainvoke(input=input)
        print(result)
            
    except Exception as e:
        print(f"❌ Error registering MCP tools: {e}")
    
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
    
    await test_bash()

if __name__ == "__main__":
    asyncio.run(main())
