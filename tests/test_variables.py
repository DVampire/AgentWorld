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
from src.environment import environment_manager
from src.tool import tool_manager
from src.agent import agent_manager

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
    await tool_manager.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tool_manager.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await environment_manager.initialize(env_names=config.env_names)
    logger.info(f"| ✅ Environments initialized: {await environment_manager.list()}")
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await agent_manager.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await agent_manager.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    
    # # Step 1: Get variables from prompt manager
    # variables = await prompt_manager.get_variables(prompt_name="tool_calling")
    # print("="*100)
    # print(variables)
    # print(type(variables), len(variables))
    # sub_variables = variables[0].variables
    # print(sub_variables)
    # print(type(sub_variables), len(sub_variables))
    # print("="*100)
    
    # print("="*100)
    # trainable_variables = await prompt_manager.get_trainable_variables(prompt_name="tool_calling")
    # print(trainable_variables)
    # print(type(trainable_variables), len(trainable_variables))
    
    # sub_variables = trainable_variables[0].variables
    # print(sub_variables)
    # print(type(sub_variables), len(sub_variables))
    # print("="*100)
    
    # # Step 2: Get variables from tools
    # variables = await tool_manager.get_variables()
    # print("="*100)
    # print(variables)
    # print(type(variables), len(variables))
    # print("="*100)
    
    # # Step 3: Get trainable variables from tools
    # trainable_variables = await tool_manager.get_trainable_variables()
    # print("="*100)
    # print(trainable_variables)
    # print(type(trainable_variables), len(trainable_variables))
    # print("="*100)
    
    # # Step 4: Get variables from memory manager
    # variables = await memory_manager.get_variables()
    # print("="*100)
    # print(variables)
    # print(type(variables), len(variables))
    # print("="*100)
    
    # # Step 5: Get trainable variables from memory manager
    # trainable_variables = await memory_manager.get_trainable_variables()
    # print("="*100)
    # print(trainable_variables)
    # print(type(trainable_variables), len(trainable_variables))
    # print("="*100)
    
    # # Step 6: Get variables from environments
    # variables = await environment_manager.get_variables()
    # print("="*100)
    # print(variables)
    # print(type(variables), len(variables))
    # print("="*100)
    
    # # Step 7: Get trainable variables from environments
    # trainable_variables = await environment_manager.get_trainable_variables()
    # print("="*100)
    # print(trainable_variables)
    # print(type(trainable_variables), len(trainable_variables))
    # print("="*100)
    
    # Step 8: Get variables from agents
    variables = await agent_manager.get_variables()
    print("="*100)
    print(variables)
    print(type(variables), len(variables))
    print("="*100)
    
    # Step 9: Get trainable variables from agents
    trainable_variables = await agent_manager.get_trainable_variables()
    print("="*100)
    print(trainable_variables)
    print(type(trainable_variables), len(trainable_variables))
    print("="*100)
    
if __name__ == "__main__":
    asyncio.run(main())
