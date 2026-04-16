"""
Example script that optimizes a tool calling agent with the Reflection optimizer.

The Reflection optimizer updates the agent prompt via the following loop:
1. Execute the task and collect the result.
2. Reflect on the execution to identify issues and improvements.
3. Refine the prompt based on the reflection.
4. Repeat until the configured number of iterations is reached.
"""

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
from src.model import model_manager
from src.version import version_manager
from src.prompt import prompt_manager
from src.memory import memory_manager
from src.tool import tool_manager
from src.environment import environment_manager
from src.agent import agent_manager
from src.optimizer import ReflectionOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Reflection optimization for tool calling agent')
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
    
    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
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
    logger.info(f"| ✅ Environments initialized: {environment_manager.list()}")
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await agent_manager.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await agent_manager.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized")
    
    # Get the agent instance; use the synchronous get_info() helper to access AgentInfo and then use .instance.
    agent = await agent_manager.get("tool_calling")
    
    # Example task; replace with the task you want to optimize.
    task = "Solve the equation x^2 + 2x - 3 = 0 with detailed steps and optimize the solution."
    files = []
    
    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    logger.info(f"| 🤖 Using Reflection optimization method")
    logger.info(f"| 💡 Reflection optimizer uses the agent's own model for reflection and improvement")
    
    # Run the agent with Reflection optimization.
    # Note: the Reflection optimizer relies on the agent's own model for reflection, so no extra optimizer_model is required.
    optimizer = ReflectionOptimizer(
        workdir=config.workdir,
        prompt_name="reflection_optimizer",
        model_name="openrouter/gemini-3-flash-preview",
        memory_name="optimizer_memory_system",
        optimize_trainable_variables=True,
        optimize_solution=True
    )
    await optimizer.optimize(agent=agent, 
                             task=task, 
                             files=files)


if __name__ == "__main__":
    asyncio.run(main())

