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

# 加载环境变量
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
# 配置区域
# ==========================================
TARGET_MODEL = "openrouter/gemini-3-flash-preview" 

class Response(BaseModel):
    reasoning: str = Field(description="The reasoning process")
    answer: str = Field(description="The final answer")

def sanitize_filename(name: str) -> str:
    """清洗文件名，移移除非法字符"""
    name = str(name).replace('\n', ' ').replace('\r', '')
    return re.sub(r'[\\/*?:"<>|]', '', name).strip()

async def test_math_benchmark(benchmark_name: str = "aime25"):
    """
    Test the benchmark manager specifically for Math/AIME using a REAL model.
    Uses response_format for structured output.
    """
    print(f"🧪 Testing benchmark manager with benchmark: {benchmark_name}")
    print(f"🤖 Using Model: {TARGET_MODEL}")
    
    # 定义保存目录
    save_dir = os.path.join(config.workdir, "benchmark", benchmark_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Created output directory: {save_dir}")
    
    # 1. 重置并获取第一个任务
    print(f"🔄 Resetting progress for {benchmark_name}...")
    task = await benchmark_manager.reset(benchmark_name)
    
    if not task:
        logger.warning("⚠️ No tasks available to run (Dataset empty or all finished).")
        return

    # ==========================================
    # 循环逻辑
    # ==========================================
    while task is not None:
        task_id = task.task_id
        start_time = time.time()
        
        try:
            print(f"\n" + "="*50)
            print(f"🚀 Processing Task ID: {task_id}")
            print("="*50)

            # --- 1. 准备 Prompt ---
            question_text = task.input
            
            # 直接从 task 获取 system_prompt
            system_prompt_text = task.system_prompt
            
            logger.info(f"| 📋 [Task {task_id}] Input length: {len(question_text)}")

            messages = [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=question_text)
            ]

            # --- 2. 模型推理 (Structure Output) ---
            print(f"⏳ [Task {task_id}] Model inferencing (Structured)...")
            
            try:
                # 调用 model_manager 并传入 response_format
                response = await model_manager(
                    model=TARGET_MODEL,
                    messages=messages,
                    response_format=Response,
                )
                
                if response.success:
                    # 获取解析后的对象
                    response_model = response.extra.parsed_model
                    task.reasoning = response_model.reasoning
                    task.answer = response_model.answer
                    
                    # --- 保存 Response 到 Markdown 文件 ---
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

            # --- 3. 评测 ---
            task.time = time.time() - start_time
            print(f"🤖 [Task {task_id}] Evaluating...")
            task = await benchmark_manager.eval(benchmark_name, task)
            
            print(f"🤖 [Task {task_id}] Answer: {task.answer}, Ground Truth: {task.ground_truth}")
            
            if task.score and task.score >= 1.0:
                print(f"✅ [Task {task_id}] Result: Correct (Score: {task.score}) | Time: {task.time:.2f}s")
            else:
                print(f"⚠️ [Task {task_id}] Result: Incorrect (Score: {task.score}) | Time: {task.time:.2f}s")

            # --- 4. 实时统计 ---
            stats = await benchmark_manager.stats(benchmark_name)
            if stats:
                attempted = stats.correct + stats.wrong
                print(f"📊 Overall Progress: {attempted}/{stats.total} | Accuracy: {stats.accuracy:.2%}")

        except Exception as e:
            logger.error(f"❌ Error processing task {task_id}: {e}")
            import traceback
            traceback.print_exc()
        
        # ==========================================
        # 获取下一个任务
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
