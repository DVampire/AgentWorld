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
from src.registry import DATASETS
from src.models import model_manager
from src.tools import tool_manager
from src.environments import ecp
from src.utils import assemble_project_path

async def test_file_system():
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
    
    args = parse_args()
    
    # Initialize configuration
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list_models()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    env_names = ["file_system"]
    for env_name in env_names:
        await ecp.build_environment(env_name, env_config=config.get(f"{env_name}_environment"))
        logger.info(f"| ✅ Environments initialized: {ecp.get_environment_info(env_name)}")
    
    input = {
        "env_name": "file_system",
        "action_name": "read",
        "file_path": assemble_project_path("tests/files/test.txt")
    }
    
    res = await ecp.call_action(**input)
    logger.info(f"| ✅ Action result: {res}")
    
async def test_github():
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
    
    args = parse_args()
    
    # Initialize configuration
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list_models()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    env_names = ["github"]
    for env_name in env_names:
        await ecp.build_environment(env_name, env_config=config.get(f"{env_name}_environment"))
        logger.info(f"| ✅ Environments initialized: {ecp.get_environment_info(env_name)}")
    
    input = {
        "env_name": "github",
        "action_name": "create_repository",
        "name": "test_repo",
        "description": "Test repository",
        "private": False,
        "auto_init": True
    }
    
    res = await ecp.call_action(**input)
    logger.info(f"| ✅ Action result: {res}")
    
async def test_trading_offline():
    def parse_args():
        parser = argparse.ArgumentParser(description='FinAgent Example')
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
    
    args = parse_args()
    
    # Initialize configuration
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list_models()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    env_names = ["trading_offline"]
    trading_offline_dataset = config.trading_offline_dataset
    trading_offline_dataset = DATASETS.build(trading_offline_dataset)
    trading_offline_environment = config.trading_offline_environment
    trading_offline_environment.update(dict(
        dataset=trading_offline_dataset
    ))
    for env_name in env_names:
        await ecp.build_environment(env_name, env_config=config.get(f"{env_name}_environment"))
        logger.info(f"| ✅ Environments initialized: {ecp.get_environment_info(env_name)}")
        
    input = {
        "env_name": "trading_offline",
        "action_name": "step",
        "action": "BUY"
    }
    
    res = await ecp.call_action(**input)
    logger.info(f"| ✅ Action result: {res}")
    
async def main():
    
    # Test file system
    # await test_file_system()
    
    # Test github
    await test_github()
    
    # Test trading offline
    # await test_trading_offline()
    

if __name__ == "__main__":
    asyncio.run(main())
