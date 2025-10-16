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
from src.agents.prompts import PromptManager

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
    
    prompt_manager = PromptManager(prompt_name="tool_calling")
    
    system_message = prompt_manager.get_system_message(reload=False)
    print(system_message.content)
    
    agent_message_modules = {
        "agent_context": "<agent_context>You are a helpful assistant.</agent_context>",
        "environment_context": "<environment_context>You are in a room with a table and a chair.</environment_context>",
        "tool_context": "<tool_context>You have a tool called 'tool1'.</tool_context>",
        "examples": "<examples>You can use the tool to help you.</examples>",
    }
    
    agent_message = prompt_manager.get_agent_message(modules=agent_message_modules, reload=True)
    print(agent_message.content)
    
if __name__ == "__main__":
    asyncio.run(main())
