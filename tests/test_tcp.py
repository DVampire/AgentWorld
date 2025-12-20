"""Test script for browser tool functionality."""

import asyncio
import sys
import os
import json
from pathlib import Path
import argparse
from mmengine import DictAction
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.memory import memory_manager
from src.prompt import prompt_manager
from src.model import model_manager
from src.version import version_manager
from src.environment import ecp
from src.tool import tcp

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "esg_agent.py"), help="config file path")

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
        
        tool_info = await tcp.get_info("deep_researcher")
        
        print("=" * 50)
        print("Function_calling:")
        print(tool_info.function_calling)
        print("=" * 50)
        print("Args schema:")
        print(tool_info.args_schema)
        print("=" * 50)
        print("Text:")
        print(tool_info.text)
        print("=" * 50)
        
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
        
        
async def test_bash_tool():
    """Test the bash tool directly."""
    
    # Test parameters
    command = "ls -l"
    
    print("🧪 Testing bash tool...")
    print(f"Command: {command}")
    
    try:
        # Invoke the bash tool
        input = {
            "name": "bash",
            "input": {
                "command": command,
            }
        }
        
        result = await tcp(**input)
        
        print("\n📋 Bash tool result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        if result and "Error" not in str(result):
            print("✅ Bash tool test successful!")
        else:
            print("❌ Bash tool test failed!")
            
    except Exception as e:
        print(f"❌ Error testing bash tool: {e}")
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
    logger.info(f"| ✅ Model manager initialized: {await model_manager.list()}")
    
    # Initialize prompt manager
    logger.info("| 📁 Initializing prompt manager...")
    await prompt_manager.initialize()
    logger.info(f"| ✅ Prompt manager initialized: {await prompt_manager.list()}")
    
    # Initialize memory manager
    logger.info("| 📁 Initializing memory manager...")
    await memory_manager.initialize(memory_names=config.memory_names)
    logger.info(f"| ✅ Memory manager initialized: {await memory_manager.list()}")
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tcp.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {await ecp.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    
    # await test_browser_tool()
    await test_deep_researcher_tool()
    # await test_bash_tool()
    logger.info("| 🚪 Test completed")
    
if __name__ == "__main__":
    asyncio.run(main())
