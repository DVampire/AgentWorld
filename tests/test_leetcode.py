"""Test script for AIME benchmark with REAL Model Inference (Full Loop)."""

import asyncio
import sys
import os
import argparse
import re
import time
from pathlib import Path
from mmengine import DictAction
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Union

# Load environment variables
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.benchmark import benchmark_manager
from src.benchmark.types import Task, Stats
from src.model.manager import model_manager
from src.message.types import HumanMessage, SystemMessage, ContentPartText, ContentPartImage, ImageURL

# ==========================================
# Configuration Section
# ==========================================
TARGET_MODEL = "openrouter/gemini-3-flash-preview" 

class Response(BaseModel):
    reasoning: str = Field(description="The reasoning process")
    code: str = Field(description="The final code")

def sanitize_filename(name: str) -> str:
    """Clean filename, remove illegal characters"""
    name = str(name).replace('\n', ' ').replace('\r', '')
    return re.sub(r'[\\/*?:"<>|]', '', name).strip()

def parse_markdown_with_images(markdown_text: str) -> Union[str, list]:
    """
    Parse markdown text and convert it to a message content format that supports images.
    
    Supports standard markdown image syntax:
    - ![](url) - image without alt text
    - ![alt text](url) - image with alt text
    
    Example:
        Input: "**Example:**\n\n![](https://example.com/image.jpg)\n\nText after"
        Output: [
            ContentPartText(text="**Example:**\n\n"),
            ContentPartImage(image_url=ImageURL(url="https://example.com/image.jpg", ...)),
            ContentPartText(text="\n\nText after")
        ]
    
    Args:
        markdown_text: The markdown text that may contain image references like ![](url)
    
    Returns:
        If no images found: returns the original string
        If images found: returns a list of ContentPartText and ContentPartImage objects
    """
    # Pattern to match markdown image syntax: ![](url) or ![alt](url)
    # Matches: ![optional alt text](image_url)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    # Find all image matches with their positions
    matches = list(re.finditer(image_pattern, markdown_text))
    
    if not matches:
        # No images found, return as plain string
        return markdown_text
    
    # Build content list with text and image parts
    content_parts = []
    last_end = 0
    
    for match in matches:
        # Add text before the image (including whitespace)
        text_before = markdown_text[last_end:match.start()]
        # Only add non-empty text parts (preserve whitespace if it's meaningful)
        if text_before:
            content_parts.append(ContentPartText(text=text_before))
        
        # Extract image URL (group 2 is the URL in parentheses)
        image_url = match.group(2)
        
        # Determine media type from URL extension (handle URLs with query parameters)
        media_type = 'image/png'  # default fallback
        # Extract path before query parameters (e.g., "image.jpg?v=1" -> "image.jpg")
        url_path = image_url.split('?')[0].lower()
        if url_path.endswith('.jpg') or url_path.endswith('.jpeg'):
            media_type = 'image/jpeg'
        elif url_path.endswith('.png'):
            media_type = 'image/png'
        elif url_path.endswith('.gif'):
            media_type = 'image/gif'
        elif url_path.endswith('.webp'):
            media_type = 'image/webp'
        
        # Create image content part
        image_url_obj = ImageURL(url=image_url, media_type=media_type)
        content_parts.append(ContentPartImage(image_url=image_url_obj))
        
        last_end = match.end()
    
    # Add remaining text after the last image
    text_after = markdown_text[last_end:]
    if text_after:
        content_parts.append(ContentPartText(text=text_after))
    
    return content_parts

async def test_leetcode_benchmark(benchmark_name: str = "leetcode"):
    """
    Test the benchmark manager specifically for LeetCode using a REAL model.
    Uses response_format for structured output.
    """
    print(f"🧪 Testing benchmark manager with benchmark: {benchmark_name}")
    print(f"🤖 Using Model: {TARGET_MODEL}")
    
    # Define save directory
    save_dir = os.path.join(config.workdir, "benchmark", benchmark_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Created output directory: {save_dir}")
    
    # 1. Reset and get first task
    print(f"🔄 Resetting progress for LeetCode...")
    task = await benchmark_manager.reset(benchmark_name)
    
    if not task:
        logger.warning("⚠️ No tasks available to run (Dataset empty or all finished).")
        return

    # ==========================================
    # Loop Logic
    # ==========================================
    while task is not None:
        task_id = task.task_id
        start_time = time.time()
        
        try:
            print(f"\n" + "="*50)
            print(f"🚀 Processing Task ID: {task_id}")
            print("="*50)

            # --- 1. Prepare Prompt ---
            question_text = task.input
            
            # Parse markdown to extract images and convert to message content format
            question_content = parse_markdown_with_images(question_text)
            
            # Get system_prompt directly from task
            system_prompt_text = task.system_prompt
            
            logger.info(f"| 📋 [Task {task_id}] Input length: {len(question_text)}")
            if isinstance(question_content, list):
                image_count = sum(1 for part in question_content if isinstance(part, ContentPartImage))
                logger.info(f"| 🖼️ [Task {task_id}] Found {image_count} image(s) in question")

            messages = [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=question_content)
            ]

            # --- 2. Model Inference (Structured Output) ---
            print(f"⏳ [Task {task_id}] Model inferencing (Structured)...")
            
            try:
                # Call model_manager and pass response_format
                response = await model_manager(
                    model=TARGET_MODEL,
                    messages=messages,
                    response_format=Response,
                )
                
                if response.success:
                    # Get parsed object
                    response_model = response.extra.parsed_model
                    task.reasoning = response_model.reasoning
                    task.answer = response_model.code
                    
                    # --- Save Response to Markdown file ---
                    try:
                        safe_id = sanitize_filename(task_id)
                        filename = f"{safe_id}.md"
                        file_path = os.path.join(save_dir, filename)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(task.reasoning + "\n\n" + task.answer)
                        
                        print(f"💾 [Saved] Output saved to: {file_path}")
                        
                    except Exception as save_err:
                        logger.error(f"⚠️ Failed to save markdown file: {save_err}")

                else:
                    logger.error(f"| ⚠️ [Task {task_id}] Model API Error: {response.message}")
                    task.reasoning = "" 
                    task.answer = "" 
                    
            except Exception as e:
                logger.error(f"| ❌ [Task {task_id}] Critical Inference Error: {e}")
                task.reasoning = ""
                task.answer = ""
                
        except Exception as e:
            logger.error(f"❌ Error processing task {task_id}: {e}")
            import traceback
            traceback.print_exc()
        
        # ==========================================
        # Get Next Task
        # ==========================================
        print(f"⏭️ Fetching next task...")
        task = await benchmark_manager.step(benchmark_name)
        
    print("\n🎉 All tasks in the benchmark have been processed.")


async def main():
    parser = argparse.ArgumentParser(description='Test Benchmark Loop')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")
    parser.add_argument("--benchmark", default="leetcode", help="benchmark name to test")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings')
    args = parser.parse_args()
    
    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    
    logger.info("| 🧠 Initializing model manager...")
    if hasattr(model_manager, 'initialize'):
        await model_manager.initialize()
    
    benchmark_name = args.benchmark
    logger.info(f"| 🛠️ Initializing benchmark manager for {benchmark_name}...")
    await benchmark_manager.initialize(benchmark_names=[benchmark_name])
    
    await test_leetcode_benchmark(benchmark_name)
    
    print("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    print("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())
