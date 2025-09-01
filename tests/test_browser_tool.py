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
from src.models import model_manager
from src.tools.mcp_tools.browser import browser

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
    
    # Mock context for testing
    class MockContext:
        async def info(self, msg):
            print(f"INFO: {msg}")
        
        async def warning(self, msg):
            print(f"WARNING: {msg}")
        
        async def error(self, msg):
            print(f"ERROR: {msg}")
    
    ctx = MockContext()
    
    # Test parameters
    task = "Go to google.com and search for 'python programming' get the first result."
    data_dir = "workdir/test_browser_tool"
    
    print("🧪 Testing browser tool...")
    print(f"Task: {task}")
    print(f"Data directory: {data_dir}")
    
    try:
        # Call the browser tool
        result = await browser(task, data_dir, ctx)
        
        print("\n📋 Browser tool result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        if result and "Error" not in result:
            print("✅ Browser tool test successful!")
        else:
            print("❌ Browser tool test failed!")
            
    except Exception as e:
        print(f"❌ Error testing browser tool: {e}")
        import traceback
        traceback.print_exc()
        
async def main():
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await model_manager.init_models(use_local_proxy=False)
    logger.info(f"| Models: {model_manager.list_models()}")
    
    await test_browser_tool()
    
    
if __name__ == "__main__":
    asyncio.run(main())
