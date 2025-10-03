import imp
import os
import sys
import numpy as np
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.infrastructures.models import model_manager, HumanMessage
from src.utils import assemble_project_path
from src.utils import make_image_url, encode_image_base64

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

async def test_general_models():
    messages = [
        HumanMessage(content=[
            {
                "type": "text",
                "text": "What is animal in the image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": make_image_url(encode_image_base64(Image.open(assemble_project_path("tests/files/cat.png"))))
                }
            }
        ]),
    ]
    
    for model_name in [
        # "gpt-4o", 
        "gpt-4.1", 
        # "gpt-5", 
        # "o1", 
        # "o3",
        # "claude-3.7-sonnet",
        # "claude-4-sonnet",
        # "gemini-2.5-pro",  
    ]:
        model = model_manager.get(model_name)
        response = await model.ainvoke(messages)
        logger.info(f"| {model_name} Response: {response}")
        

async def test_deep_research_models():
    
    messages = [
        HumanMessage(content="What is the capital of France?")
    ]
    
    for model_name in [
        # "o3-deep-research",
        # "o4-mini-deep-research",
        # "gpt-4o-search-preview",
    ]:
        model = model_manager.get(model_name)
        response = await model.ainvoke(messages)
        logger.info(f"| {model_name} Response: {response}")
        
async def test_transcribe_models():
    for model_name in [
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
    ]:
        model = model_manager.get(model_name)
        
        # Test with file path
        audio_path = assemble_project_path("tests/files/audio.mp3")
        
        # Test 1: File path (original way)
        messages = [
            HumanMessage(content=audio_path),
        ]
        response = await model.ainvoke(messages)
        logger.info(f"| {model_name} (file path) Response: {response}")
        
async def test_embedding_models():
    for model_name in [
        "text-embedding-3-large",
    ]:
        model = model_manager.get(model_name)
        response = await model.aembed_query("Hello, world!")
        response = np.array(response)
        logger.info(f"| {model_name} Response: {response.shape}")
        

async def test_computer_browser_use_models():
    """Test computer-use-preview model with responses API.
    
    Note: computer-use-preview uses OpenAI's responses API, not chat completions.
    The model is already bound with computer_use_preview tool in model_manager.
    Do not use response_format with structured output, as it conflicts with the responses API.
    """
    
    for model_name in [
        "computer-browser-use",
    ]:
        # Create message with proper format for responses API
        messages = [
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "What is the next action to search 'python programming' on google?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": make_image_url(encode_image_base64(Image.open(assemble_project_path("tests/files/google.png"))))
                    }
                }
            ])
        ]
        
        # Get the model (already configured with computer_use_preview tool)
        model = model_manager.get(model_name)
        
        # Invoke with reasoning parameter for computer-use-preview
        # Changed from "generate_summary" to "summary" as per OpenAI API
        response = await model.ainvoke(
            messages,
            reasoning={"summary": "concise"},
        )
        
        logger.info(f"| {model_name} Response: {response}")
        
async def main():
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| Models: {model_manager.list()}")
    
    # await test_general_models()
    # await test_deep_research_models()
    # await test_transcribe_models()
    # await test_embedding_models()
    await test_computer_browser_use_models()
    
    
if __name__ == "__main__":
    asyncio.run(main())