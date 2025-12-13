import os
import sys
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio
from pydantic import BaseModel, Field
from typing import List

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.message import (
    HumanMessage,
    SystemMessage,
    ContentPartText,
    ContentPartImage,
    ImageURL,
    ContentPartAudio,
    AudioURL,
    ContentPartVideo,
    VideoURL,
    ContentPartPdf,
    PdfURL,
)

from src.tool import tcp

from src.utils import assemble_project_path, make_file_url


async def test_chat():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing chat with different models")
    models = [
        # OpenAI models
        "openrouter/gpt-4o",
        "openrouter/gpt-4.1",
        "openrouter/gpt-5",
        "openrouter/gpt-5.1",
        "openrouter/o3",
        # "openai/gpt-4o",
        # "openai/gpt-4.1",
        # "openai/gpt-5",
        # "openai/gpt-5.1",
        # "openai/o3",
        
        # Anthropic models
        # "openrouter/claude-sonnet-3.7",
        # "openrouter/claude-sonnet-4",
        # "openrouter/claude-opus-4",
        # "openrouter/claude-sonnet-4.5",
        # "openrouter/claude-opus-4.5",
        # "anthropic/claude-sonnet-3.7",
        # "anthropic/claude-sonnet-4",
        # "anthropic/claude-sonnet-4.5",
        
        # Gemini models
        # "openrouter/gemini-2.5-flash",
        # "openrouter/gemini-2.5-pro",
        # "openrouter/gemini-3-pro-preview",
        # "google/gemini-2.5-flash",
        # "google/gemini-2.5-pro",
        # "google/gemini-3-pro-preview",
    ]
    
    image_url = make_file_url(file_path=assemble_project_path("tests/files/pokemon.jpg"))
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="What are the names of the Pokémon in the image?"),
            ContentPartImage(image_url=ImageURL(url=image_url, detail="high")),
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager(
            model=model,
            messages=messages
        )
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
    logger.info(f"| --------------------------------------------------")

async def test_transcription():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing transcription with different models")
    models = [
        # "openai/gpt-4o-transcribe",
        "openrouter/gemini-2.5-flash",
        # "google/gemini-2.5-flash",
    ]
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="Please transcribe the audio file and provide the transcription. Only return the transcription, no other text or formatting."),
            ContentPartAudio(audio_url=AudioURL(url=make_file_url(file_path="tests/files/audio.mp3"))),
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager(model=model, messages=messages)
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
    logger.info(f"| --------------------------------------------------")


async def test_embedding():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing embedding with different models")
    models = [
        # "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
        # "openai/text-embedding-ada-002",
    ]
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="Please embed the text and provide the embedding."),
            ContentPartText(text="The text is: The quick brown fox jumps over the lazy dog."),
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager(model=model, messages=messages)
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
    logger.info(f"| --------------------------------------------------")

async def test_video():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing video with different models")
    models = [
        "openrouter/gemini-2.5-flash",
        # "google/gemini-2.5-flash",
    ]
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="Please analyze the video and provide the analysis. Only return the analysis, no other text or formatting."),
            ContentPartVideo(video_url=VideoURL(url=make_file_url(file_path="tests/files/video.MOV"))),
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager(model=model, messages=messages)
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
    logger.info(f"| --------------------------------------------------")


async def test_pdf():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing PDF with different models")
    models = [
        "openrouter/gemini-2.5-flash",
        # "google/gemini-2.5-flash",
    ]
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="Please analyze the PDF and provide the analysis. Only return the analysis, no other text or formatting."),
            ContentPartPdf(pdf_url=PdfURL(url=make_file_url(file_path="tests/files/pdf.pdf"))),
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager(model=model, messages=messages)
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
    logger.info(f"| --------------------------------------------------")


async def test_response_format():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing response format with different models")
    models = [
        # OpenAI models
        "openrouter/gpt-4o",
        "openrouter/gpt-4.1",
        "openrouter/gpt-5",
        "openrouter/gpt-5.1",
        "openrouter/o3",
        # "openai/gpt-4o",
        # "openai/gpt-4.1",
        # "openai/gpt-5",
        # "openai/gpt-5.1",
        # "openai/o3",
        
        # Anthropic models
        # "openrouter/claude-sonnet-3.7",
        # "openrouter/claude-sonnet-4",
        # "openrouter/claude-opus-4",
        # "openrouter/claude-sonnet-4.5",
        # "openrouter/claude-opus-4.5",
        # "anthropic/claude-sonnet-4.5",
        
        # Gemini models
        # "openrouter/gemini-2.5-flash",
        # "openrouter/gemini-2.5-pro",
        # "openrouter/gemini-3-pro-preview",
        # "google/gemini-2.5-flash",
        # "google/gemini-2.5-pro",
        # "google/gemini-3-pro-preview",
    ]
    
    class Add(BaseModel):
        a: int = Field(description="The first number to add")
        b: int = Field(description="The second number to add")
        result: int = Field(description="The result of the addition")

    class AddList(BaseModel):
        adds: List[Add] = Field(description="The list of additions to perform")
        reasoning: str = Field(description="The reasoning process of the addition")
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="Please add the numbers 1 and 2, 3 and 4, 5 and 6, 7 and 8, 9 and 10."),
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager(model=model, messages=messages, response_format=AddList)
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
    logger.info(f"| --------------------------------------------------")

async def test_tool_calling():
    logger.info(f"| --------------------------------------------------")
    logger.info(f"| Testing tool calling with different models")
    models = [
        # OpenAI models
        "openrouter/gpt-4o",
        "openrouter/gpt-4.1",
        "openrouter/gpt-5",
        "openrouter/gpt-5.1",
        "openrouter/o3",
        # "openai/gpt-4o",
        # "openai/gpt-4.1",
        # "openai/gpt-5",
        # "openai/gpt-5.1",
        # "openai/o3",
        
        # Anthropic models
        # "openrouter/claude-sonnet-3.7",
        # "openrouter/claude-sonnet-4",
        # "openrouter/claude-opus-4",
        # "openrouter/claude-sonnet-4.5",
        # "openrouter/claude-opus-4.5",
        # "anthropic/claude-sonnet-3.7",
        # "anthropic/claude-sonnet-4",
        # "anthropic/claude-sonnet-4.5",
        
        # Gemini models
        # "openrouter/gemini-2.5-flash",
        # "openrouter/gemini-2.5-pro",
        # "openrouter/gemini-3-pro-preview",
        # "google/gemini-2.5-flash",
        # "google/gemini-2.5-pro",
        # "google/gemini-3-pro-preview",
    ]
    
    tools = [
        await tcp.get('bash'),
    ]
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="Please run the command 'ls -l' and return the output."),
        ]),
    ]
    
    for model in models:
        logger.info(f"| Testing {model}")
        response = await model_manager(
            model=model, 
            messages=messages,
            tools=tools,
        )
        logger.info(f"| {model} Response: {json.dumps(response.model_dump(), indent=4)}")
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
    logger.info(f"| Model manager initialized: {model_manager.list()}")
    
    # Initialize tools
    await tcp.initialize()
    logger.info(f"| Tools initialized: {await tcp.list()}")

    await test_chat()
    # await test_response_format()
    # await test_tool_calling()
    # await test_transcription()
    # await test_embedding()
    # await test_video()
    # await test_pdf()


if __name__ == "__main__":
    asyncio.run(main())