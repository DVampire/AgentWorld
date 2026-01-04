"""Test script for benchmark manager functionality."""

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
from src.benchmark import benchmark_manager
from src.version import version_manager

def parse_args():
    parser = argparse.ArgumentParser(description='Test Benchmark Manager')
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

async def test_benchmark_manager(benchmark_name: str):
    """Test the benchmark manager directly."""
    
    print(f"🧪 Testing benchmark manager with benchmark: {benchmark_name}")
    
    try:

        # 1. Reset benchmark progress
        print(f"🔄 Resetting progress for {benchmark_name}...")
        task = await benchmark_manager.reset(benchmark_name)
        logger.info(f"| 📋 Task: {task}")
        prediction = "The answer is 204."
        task_id = task["task_id"]
        print(f"📋 Evaluating task for {benchmark_name}...")
        result = await benchmark_manager.eval_task(benchmark_name, prediction, task_id=task_id)
        print(f"📋 Result: {result}")
        
        # 2. Get next task
        next_task = await benchmark_manager.step(benchmark_name)
        print(f"📋 Next task: {next_task}")
        prediction = "The answer is 204."
        task_id = next_task["task_id"]
        print(f"📋 Evaluating next task for {benchmark_name}...")
        result = await benchmark_manager.eval_task(benchmark_name, prediction, task_id=task_id)
        print(f"📋 Result: {result}")
        
        # 3. Get stats
        # stats = await benchmark_manager.get_stats(benchmark_name)
        # print(f"📋 Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
            
            
async def main():
    args = parse_args()
    
    config.initialize(config_path = args.config, args = args)
    logger.initialize(config = config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize benchmark manager
    logger.info("| 🛠️ Initializing benchmark manager...")
    await benchmark_manager.initialize(benchmark_names=[
        "aime24",
    ])
    logger.info(f"| ✅ Benchmark manager initialized: {await benchmark_manager.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    
    await test_benchmark_manager("aime24")
    
    print("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    print("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())

