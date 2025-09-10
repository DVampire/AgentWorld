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
from src.environments import ecp

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
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    ecp.build_environment("file_system", env_config=config.file_system_environment)
    logger.info(f"| ✅ Environments initialized: {ecp.get_registered_environments()}")
    
    # Initialize tool manager
    logger.info("| 🛠️ Initializing tool manager...")
    await tool_manager.init_tools(env_names=["file_system"])
    logger.info(f"| ✅ Tool manager initialized: {tool_manager.list_tools()}")
    
    # Build agent
    logger.info("| 🎮 Building agent...")
    agent_config = config.agent
    agent_config.update(dict(
        env_names=["file_system"]
    ))
    agent = AGENTS.build(agent_config)
    logger.info(f"| ✅ Agent built: {agent}")
    
    """Test streaming execution mode."""
    logger.info("| 🚀 Testing streaming execution mode")
    
    # task = "请找到图片中所有Pokemon的编号，并返回一个列表。"
    # files = [assemble_project_path("tests/files/pokemon.jpg")]
    
    task = "帮我生成一个简单的python脚本并保存为prime.py，计算100以内的质数，并返回一个列表。"
    files = []
    
    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    
    await agent.run(task, files)

if __name__ == "__main__":
    asyncio.run(main())