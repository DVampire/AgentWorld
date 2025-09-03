"""Test script for browser tool functionality."""

import asyncio
import sys
import os
from pathlib import Path
import argparse
from mmengine import DictAction
from dotenv import load_dotenv
from inspect import cleandoc

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.models import model_manager
from src.tools import tool_manager
from src.utils import assemble_project_path

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
        # Call the browser tool
        browser = tool_manager.get_tool("browser")
        print(f"Browser tool: {browser}")
        print(f"Browser tool args schema: {browser.args_schema}")
        
        # Use the correct parameter name
        result = await browser.ainvoke(input={"task": task, "base_dir": base_dir})
        
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
        
async def test_web_fetcher_tool():
    """Test the web fetcher tool directly."""
    
    # Test parameters
    url = "https://www.google.com"
    
    print("🧪 Testing web fetcher tool...")
    print(f"URL: {url}")
    
    try:
        # Call the web fetcher tool
        web_fetcher = tool_manager.get_tool("web_fetcher")
        print(f"Web fetcher tool: {web_fetcher}")
        print(f"Web fetcher tool args schema: {web_fetcher.args_schema}")
        
        # Use the correct parameter name
        result = await web_fetcher.ainvoke(input={"url": url})
        
        print("\n📋 Web fetcher tool result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        if result and "Error" not in str(result):
            print("✅ Web fetcher tool test successful!")
        else:
            print("❌ Web fetcher tool test failed!")
            
    except Exception as e:
        print(f"❌ Error testing web fetcher tool: {e}")
        import traceback
        traceback.print_exc()
        
async def test_web_searcher_tool():
    """Test the web searcher tool directly."""
    
    # Test parameters
    query = "python programming"
    
    print("🧪 Testing web searcher tool...")
    print(f"Query: {query}")
    
    try:
        # Call the web searcher tool
        web_searcher = tool_manager.get_tool("web_searcher")
        print(f"Web searcher tool: {web_searcher}")
        print(f"Web searcher tool args schema: {web_searcher.args_schema}")
        
        # Use the correct parameter name
        result = await web_searcher.ainvoke(input={"query": query})
        
        print("\n📋 Web searcher tool result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        if result and "Error" not in str(result):
            print("✅ Web searcher tool test successful!")
        else:
            print("❌ Web searcher tool test failed!")
            
    except Exception as e:
        print(f"❌ Error testing web searcher tool: {e}")
        import traceback
        traceback.print_exc()
        
async def test_deep_researcher_tool():
    """Test the deep researcher tool directly."""
    
    # Test parameters
    task = "Find out the Pokémon IDs in the image."
    image = assemble_project_path("tests/pokemon.jpg")
    
    print("🧪 Testing deep researcher tool...")
    print(f"Task: {task}")
    print(f"Image: {image}")
    
    try:
        # Call the deep researcher tool
        deep_researcher = tool_manager.get_tool("deep_researcher")
        print(f"Deep researcher tool: {deep_researcher}")
        print(f"Deep researcher tool args schema: {deep_researcher.args_schema}")
        
        # Use the correct parameter name
        result = await deep_researcher.ainvoke(input={"task": task, "image": image})
        
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
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await model_manager.init_models(use_local_proxy=False)
    logger.info(f"| Models: {model_manager.list_models()}")
    
    await tool_manager.init_tools()
    logger.info(f"| Tools: {tool_manager.list_tools()}")
    
    # await test_browser_tool()
    # await test_web_fetcher_tool()
    # await test_web_searcher_tool()
    await test_deep_researcher_tool()
    
if __name__ == "__main__":
    asyncio.run(main())
