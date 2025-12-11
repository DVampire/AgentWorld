"""Test script for browser tool functionality."""

import asyncio
import sys
import os
from pathlib import Path
import argparse
from mmengine import DictAction
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.tool import tcp

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

async def test_browser_tool():
    """Test the browser tool directly."""
    
    # Test parameters
    task = "Go to google.com and search for 'python programming' get the first result."
    base_dir = "workdir/test_browser_tool"
    
    print("🧪 Testing browser tool...")
    print(f"Task: {task}")
    print(f"Data directory: {base_dir}")
    
    try:
        # Invoke the browser tool
        input = {
            "name": "browser",
            "input": {
                "task": task,
                "base_dir": base_dir
            }
        }
        
        result = await tcp(**input)
        
        print("\n📋 Browser tool result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        if result and "Error" not in str(result):
            print("✅ Browser tool test successful!")
        else:
            print("❌ Browser tool test failed!")
            
    except Exception as e:
        print(f"❌ Error testing browser tool: {e}")
        import traceback
        traceback.print_exc()
        
        
async def test_deep_researcher_tool():
    """Test the deep researcher tool directly."""
    
    # Test parameters
    task = "Search for the latest news about Apple on Google."
    
    print("🧪 Testing deep researcher tool...")
    print(f"Task: {task}")
    
    try:
        # Invoke the deep researcher tool
        input = {
            "name": "deep_researcher",
            "input": {
                "task": task,
            }
        }
        
        result = await tcp(**input)
        
        print("\n📋 Deep researcher tool result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        if result and "Error" not in str(result):
            print("✅ Deep researcher tool test successful!")
        else:
            print("❌ Deep researcher tool test failed!")
            
    except Exception as e:
        print(f"❌ Error testing deep researcher tool: {e}")
        import traceback
        traceback.print_exc()
        
async def main():
    args = parse_args()
    
    config.initialize(config_path = args.config, args = args)
    logger.initialize(config = config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize()
    logger.info(f"| ✅ Model manager initialized: {model_manager.list()}")
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tcp.list()}")
    
    # Initialize version manager
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {await version_manager.list()}")
    
    # await test_browser_tool()
    await test_deep_researcher_tool()
    logger.info("| 🚪 Test completed")
    
if __name__ == "__main__":
    asyncio.run(main())
