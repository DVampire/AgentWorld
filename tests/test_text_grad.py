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
from src.optimizers.protocol.type import Variable

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
    
    prompt = Variable(
        name="system_prompt",
        type="system_prompt",
        description="系统提示词",
        template="你是一个{{role}}，请{{task}}",
        variables=[
            Variable(name="role", 
                     type="system_prompt_module",
                     description="角色",
                     variables="AI助手",
                     require_grad=True),
            Variable(name="task", 
                     type="system_prompt_module", 
                     description="任务", 
                     variables="回答问题",
                     require_grad=True)
        ],
        require_grad=True
    )
    
    other_variable = Variable(
        name="other_variable",
        type="other_variable",
        description="其他变量",
        template="{{other_variable_module}}",
        variables=[
            Variable(name="other_variable_module",
                     type="other_variable_module",
                     description="其他变量模块", 
                     variables="其他变量模块",
                     require_grad=False)
        ],
        require_grad= False
    )
    
    prompt = prompt + other_variable

    all_vars = prompt.get_all_variables()
    print(all_vars)
    
    graph = prompt.generate_graph()
    graph.render("graph", directory=".", format="png")
    print("Graph saved to graph.png")


if __name__ == "__main__":
    asyncio.run(main())
