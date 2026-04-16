"""
Comprehensive test suite for the memory system.
Tests all components: MemorySystem, SessionMemory, CombinedMemory, and MemoryManager.
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
import os
from pathlib import Path
import argparse
from mmengine import DictAction
import sys

from dotenv import load_dotenv
load_dotenv(verbose=True)


root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.memory import MemoryManager, EventType
from src.models import model_manager
from src.logger import logger
from src.config import config


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
    
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info(f"| Models: {model_manager.list_models()}")
    
    memory_manager = MemoryManager(**config.memory)
    logger.info(f"| Memory Manager: {memory_manager}")
    
    session_id = "test_session"
    await memory_manager.start_session(session_id)
    
    for i in range(10):
        await memory_manager.add_event(step_number=i,
                                       event_type=EventType.ACTION_STEP,
                                       data = dict(
                                           test_data = f"test_data_{i}"
                                       ),
                                       agent_name="test_agent",
                                       task_id="test_task",
                                       session_id=session_id)
        
    state = await memory_manager.get_state(session_id, n=5)
    logger.info(f"| State: {state}")
    
if __name__ == "__main__":
    asyncio.run(main())