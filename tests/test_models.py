import os
import sys
import json
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
from src.message import message_manager
from src.message import HumanMessage, SystemMessage

from src.tool import tcp

from src.utils import assemble_project_path, encode_file_base64, make_file_url


async def test_acompletion():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing acompletion with different models")
    models = [
        "openrouter/gpt-4o",
        "openrouter/gpt-4.1",
        "openrouter/gpt-5",
        "openrouter/gpt-5.1",
        "openrouter/o3",
        "openrouter/gemini-2.5-flash",
        "openrouter/gemini-2.5-pro",
        "openrouter/claude-4.5-sonnet",
    ]
    
    image_url = make_file_url(file_path=assemble_project_path("tests/files/pokemon.jpg"))
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": "What are the names of the Pokémon in the image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager.acompletion(
            model=model,
            messages=messages
        )
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
    logger.info(f"| --------------------------------------------------")

async def test_aembedding():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing aembedding with different models")
    
    models = [
        "openrouter/text-embedding-3-large",
    ]
    
    messages = [
        HumanMessage(content="What is the capital of France?"),
        HumanMessage(content="What is the capital of Germany?")
    ]
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager.aembedding(
            model=model,
            messages=messages
        )
        
        logger.info(f"| {model} Response: length {len(response.extra['embeddings'])}, shape {response.extra['embeddings'][0].shape}")
    logger.info(f"| --------------------------------------------------")

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
    await model_manager.initialize()
    logger.info(f"| Model manager initialized")
    
    # Initialize message manager
    await message_manager.initialize()
    logger.info(f"| Message manager initialized")
    
    # Initialize tools
    await tcp.initialize()
    logger.info(f"| Tools initialized: {await tcp.list()}")
    exit()
    
    # await test_acompletion()
    await test_aembedding()


if __name__ == "__main__":
    asyncio.run(main())