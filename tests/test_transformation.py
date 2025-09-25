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
from src.infrastructures.models import model_manager
from src.tools import tcp
from src.environments import ecp
from src.agents import acp
from src.transformation import transformation

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
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize()
    logger.info(f"| ✅ Tools initialized: {tcp.list()}")

    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(config.agent_names)
    logger.info(f"| ✅ Agents initialized: {acp.list()}")
    
    # # Transformation T2E
    # logger.info("| 🔄 Transformation start...")
    # await transformation.transform(type="t2e", tool_names=[
    #     "bash",
    #     "python_interpreter",
    # ])
    # logger.info(f"| ✅ Transformation completed: {ecp.list()}")
    
    # # Transformation T2A
    # logger.info("| 🔄 Transformation start...")
    # await transformation.transform(type="t2a", tool_names=[
    #     "bash",
    #     "python_interpreter",
    # ])
    # logger.info(f"| ✅ Transformation completed: {acp.list()}")
    
    # # Transformation E2T
    # logger.info("| 🔄 Transformation start...")
    # await transformation.transform(type="e2t", env_names=[
    #     "file_system",
    # ])
    # logger.info(f"| ✅ Transformation completed: {tcp.list()}")

    # # Transformation E2A
    # logger.info("| 🔄 Transformation start...")
    # await transformation.transform(type="e2a", env_names=[
    #     "file_system",
    # ])
    # logger.info(f"| ✅ Transformation completed: {acp.list()}")
    
    # # Transformation A2T
    # logger.info("| 🔄 Transformation start...")
    # await transformation.transform(type="a2t", agent_names=[
    #     "tool_calling",
    # ])
    # logger.info(f"| ✅ Transformation completed: {tcp.list()}")
    
    # Transformation A2E
    logger.info("| 🔄 Transformation start...")
    await transformation.transform(type="a2e", agent_names=[
        "tool_calling",
    ])
    logger.info(f"| ✅ Transformation completed: {ecp.list()}")
    
    
if __name__ == "__main__":
    asyncio.run(main())