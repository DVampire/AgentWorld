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
from src.message.types import HumanMessage, SystemMessage

# ==========================================
# Configuration Section
# ==========================================
TARGET_MODEL = "openrouter/gemini-3-flash-preview" 

class Response(BaseModel):
    reasoning: str = Field(description="The reasoning process")
    result: str = Field(description="The final result")

def sanitize_filename(name: str) -> str:
    """Clean filename, remove illegal characters"""
    name = str(name).replace('\n', ' ').replace('\r', '')
    return re.sub(r'[\\/*?:"<>|]', '', name).strip()

async def test_math_benchmark(benchmark_name: str = "aime25"):
    """
    Test the benchmark manager specifically for Math/AIME using a REAL model.
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
    print(f"🔄 Resetting progress for {benchmark_name}...")
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
            
            # Get system_prompt directly from task
            system_prompt_text = task.system_prompt
            
            logger.info(f"| 📋 [Task {task_id}] Input length: {len(question_text)}")

            messages = [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=question_text)
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
                    task.result = response_model.result
                    
                    # --- Save Response to Markdown file ---
                    try:
                        safe_id = sanitize_filename(task_id)
                        filename = f"{safe_id}.md"
                        file_path = os.path.join(save_dir, filename)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(task.reasoning + "\n\n" + task.result)
                        
                        print(f"💾 [Saved] Output saved to: {file_path}")
                        
                    except Exception as save_err:
                        logger.error(f"⚠️ Failed to save markdown file: {save_err}")

                else:
                    logger.error(f"| ⚠️ [Task {task_id}] Model API Error: {response.message}")
                    task.reasoning = "" 
                    task.result = "" 
                    
            except Exception as e:
                logger.error(f"| ❌ [Task {task_id}] Critical Inference Error: {e}")
                task.reasoning = ""
                task.result = ""

            # --- 3. Evaluation ---
            task.time = time.time() - start_time
            print(f"🤖 [Task {task_id}] Evaluating...")
            task = await benchmark_manager.eval(benchmark_name, task)
            
            print(f"🤖 [Task {task_id}] Result: {task.result}, Ground Truth: {task.ground_truth}")
            
            if task.score and task.score >= 1.0:
                print(f"✅ [Task {task_id}] Result: Correct (Score: {task.score}) | Time: {task.time:.2f}s")
            else:
                print(f"⚠️ [Task {task_id}] Result: Incorrect (Score: {task.score}) | Time: {task.time:.2f}s")

            # --- 4. Real-time Statistics ---
            stats = await benchmark_manager.stats(benchmark_name)
            if stats:
                attempted = stats.correct + stats.wrong
                print(f"📊 Overall Progress: {attempted}/{stats.total} | Accuracy: {stats.accuracy:.2%}")

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
    parser.add_argument("--benchmark", default="aime25", help="benchmark name to test")
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
    
    await test_math_benchmark(benchmark_name)
    
    print("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    print("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())
