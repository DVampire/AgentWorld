from langchain_openai import ChatOpenAI
import os
import sys
import base64
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.utils import assemble_project_path
from src.infrastructures.models import model_manager
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

screenshot_1_path = assemble_project_path("tests/files/google.png")
screenshot_1_base64 = base64.b64encode(open(screenshot_1_path, "rb").read()).decode("utf-8")

model = model_manager.get("computer-browser-use")

# Construct input message
input_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": (
                "Search for 'OpenAI Computer Use API' on Google"
            ),
        },
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{screenshot_1_base64}",
        },
    ],
}


async def main():
    
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| ✅ Model manager initialized: {model_manager.list()}")
    
    response = await model.ainvoke(
        [input_message],
        reasoning={
            "generate_summary": "concise",
        },
    )
    print(response.content)
    
import asyncio
asyncio.run(main())

