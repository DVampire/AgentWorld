"""
使用TextGrad优化Tool Calling Agent的示例
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
from src.models import model_manager
from src.tools import tcp
from src.environments import ecp
from src.agents import acp
from src.transformation import transformation
from src.optimizers.textgrad_optimizer import optimize_agent_with_textgrad


def parse_args():
    parser = argparse.ArgumentParser(description='optimized tool calling agent with textgrad')
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
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize(config.tool_names)
    logger.info(f"| ✅ Tools initialized: {tcp.list()}")
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(config.agent_names)
    logger.info(f"| ✅ Agents initialized: {acp.list()}")
    
    # Transformation ECP to TCP
    logger.info("| 🔄 Transformation start...")
    await transformation.transform(type="e2t", env_names=config.env_names)
    logger.info(f"| ✅ Transformation completed: {tcp.list()}")
    
    # Get agent instance - 使用get_info()同步方法获取AgentInfo，然后访问instance
    agent_info = acp.get_info("tool_calling")
    if agent_info is None:
        raise ValueError(f"Agent 'tool_calling' not found. Available agents: {acp.list()}")
    agent = agent_info.instance
    
    # Example task
    task = "How are you!"
    files = []
    
    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")
    
    # 运行带优化的Agent
    await optimize_agent_with_textgrad(
        agent=agent,
        task=task,
        files=files,
        optimization_steps=3,  # 优化迭代次数
        optimizer_model="gpt-4o",  # 用于优化的模型
        log_dir=config.workdir  # 日志目录
    )
    
    # 最后用优化后的提示词再运行一次
    logger.info(f"\n| {'='*60}")
    logger.info(f"| 🎯 Final run with optimized prompts")
    logger.info(f"| {'='*60}\n")
    
    final_result = await agent.ainvoke(task=task, files=files)
    logger.info(f"| ✅ Final result: {final_result}")


if __name__ == "__main__":
    asyncio.run(main())
