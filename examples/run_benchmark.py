"""Test script for AIME benchmark with REAL Model Inference (Full Loop)."""

import asyncio
import sys
import os
import argparse
import re
import time
import json
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

async def test_math_benchmark(benchmark_name: str = "aime25", max_concurrency: int = 5):
    """
    Test the benchmark manager specifically for Math/AIME using a REAL model.
    Uses response_format for structured output with concurrent processing.
    
    Args:
        benchmark_name: Name of the benchmark to test
        max_concurrency: Maximum number of concurrent tasks to process
    """
    print(f"🧪 Testing benchmark manager with benchmark: {benchmark_name}")
    print(f"🤖 Using Model: {TARGET_MODEL}")
    print(f"⚡ Max Concurrency: {max_concurrency}")
    
    # 定义保存目录
    save_dir = os.path.join(config.workdir, "benchmark", benchmark_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Created output directory: {save_dir}")
    
    # 1. 重置并收集所有任务
    print(f"🔄 Resetting progress for {benchmark_name}...")
    task = await benchmark_manager.reset(benchmark_name)
    
    if not task:
        logger.warning("⚠️ No tasks available to run (Dataset empty or all finished).")
        return

    # 收集所有任务
    print(f"📦 Collecting all tasks...")
    all_tasks = []
    while task is not None:
        all_tasks.append(task)
        task = await benchmark_manager.step(benchmark_name)
    
    total_tasks = len(all_tasks)
    print(f"✅ Collected {total_tasks} tasks. Starting concurrent processing...")
    
    # 创建 Semaphore 限制并发数
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # 用于跟踪进度
    completed_count = 0
    completed_lock = asyncio.Lock()
    
    async def process_single_task(task: Task) -> Task:
        """处理单个任务的协程函数"""
        nonlocal completed_count  # 必须在函数开始处声明
        task_id = task.task_id
        start_time = time.time()
        
        async with semaphore:  # 使用 Semaphore 限制并发
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
                    # 调用 model_manager 并传入 response_format，添加超时
                    try:
                        response = await asyncio.wait_for(
                            model_manager(
                                model=TARGET_MODEL,
                                messages=messages,
                                response_format=Response,
                            ),
                            timeout=600.0  # 10分钟超时
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"| ⏱️ [Task {task_id}] Model API Timeout (600s)")
                        task.reasoning = ""
                        task.result = ""
                        response = None
                    
                    if response and response.success:
                        # 获取解析后的对象
                        response_model = response.extra.parsed_model
                        task.reasoning = response_model.reasoning
                        task.result = response_model.answer
                        
                        # --- 保存 Response 到 Markdown 文件 ---
                        try:
                            safe_id = sanitize_filename(task_id)
                            filename = f"{safe_id}.md"
                            file_path = os.path.join(save_dir, filename)
                            
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(task.reasoning + "\n\n" + task.result)
                            
                            print(f"💾 [Saved] Output saved to: {file_path}")
                            
                        except Exception as save_err:
                            logger.error(f"⚠️ Failed to save markdown file: {save_err}")

                    elif response:
                        logger.error(f"| ⚠️ [Task {task_id}] Model API Error: {response.message}")
                        task.reasoning = "" 
                        task.result = "" 
                    else:
                        task.reasoning = ""
                        task.result = ""
                        
                except Exception as e:
                    logger.error(f"| ❌ [Task {task_id}] Critical Inference Error: {e}")
                    import traceback
                    traceback.print_exc()
                    task.reasoning = ""
                    task.result = ""

                # --- 3. 评测 ---
                task.time = time.time() - start_time
                print(f"🤖 [Task {task_id}] Evaluating...")
                
                try:
                    evaluated_task = await asyncio.wait_for(
                        benchmark_manager.eval(benchmark_name, task),
                        timeout=30.0  # 30秒超时
                    )
                except asyncio.TimeoutError:
                    logger.error(f"| ⏱️ [Task {task_id}] Evaluation Timeout (30s)")
                    evaluated_task = task
                except Exception as e:
                    logger.error(f"| ❌ [Task {task_id}] Evaluation Error: {e}")
                    evaluated_task = task
                
                if evaluated_task:
                    print(f"🤖 [Task {task_id}] Answer: {evaluated_task.result}, Ground Truth: {evaluated_task.ground_truth}")
                    
                    if evaluated_task.score and evaluated_task.score >= 1.0:
                        print(f"✅ [Task {task_id}] Result: Correct (Score: {evaluated_task.score}) | Time: {evaluated_task.time:.2f}s")
                    else:
                        print(f"⚠️ [Task {task_id}] Result: Incorrect (Score: {evaluated_task.score}) | Time: {evaluated_task.time:.2f}s")
                    
                    # 更新进度
                    async with completed_lock:
                        completed_count += 1
                        print(f"📊 Progress: {completed_count}/{total_tasks} tasks completed ({completed_count/total_tasks*100:.1f}%)")
                    
                    return evaluated_task
                else:
                    async with completed_lock:
                        completed_count += 1
                        print(f"📊 Progress: {completed_count}/{total_tasks} tasks completed ({completed_count/total_tasks*100:.1f}%)")
                    return task

            except Exception as e:
                logger.error(f"❌ Error processing task {task_id}: {e}")
                import traceback
                traceback.print_exc()
                async with completed_lock:
                    completed_count += 1
                    print(f"📊 Progress: {completed_count}/{total_tasks} tasks completed ({completed_count/total_tasks*100:.1f}%)")
                return task
    
    # 并发处理所有任务，使用 return_exceptions=True 避免单个任务失败导致整体卡住
    print(f"🚀 Starting concurrent processing of {total_tasks} tasks...")
    try:
        processed_tasks = await asyncio.gather(
            *[process_single_task(t) for t in all_tasks],
            return_exceptions=True
        )
        
        # 检查是否有异常
        for i, result in enumerate(processed_tasks):
            if isinstance(result, Exception):
                logger.error(f"❌ Task {all_tasks[i].task_id} failed with exception: {result}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        logger.error(f"❌ Fatal error in gather: {e}")
        import traceback
        traceback.print_exc()
    
    # 最终统计
    print("\n" + "="*50)
    print("📊 Final Statistics")
    print("="*50)
    stats = await benchmark_manager.stats(benchmark_name)
    if stats:
        attempted = stats.correct + stats.wrong
        print(f"📊 Overall Progress: {attempted}/{stats.total} | Accuracy: {stats.accuracy:.2%}")
        print(f"✅ Correct: {stats.correct} | ❌ Wrong: {stats.wrong}")
        print(f"⏱️  Average Time: {stats.average_time:.2f}s")
        
        # 保存统计信息
        try:
            with open(os.path.join(save_dir, "stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats.model_dump(), f, indent=4, ensure_ascii=False)
            print(f"💾 Statistics saved to: {os.path.join(save_dir, 'stats.json')}")
        except Exception as e:
            logger.error(f"⚠️ Failed to save statistics: {e}")
    else:
        logger.warning("⚠️ No statistics available")
    
    print("\n🎉 All tasks in the benchmark have been processed.")


async def main():
    parser = argparse.ArgumentParser(description='Test Benchmark Loop')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")
    parser.add_argument("--benchmark", default="gpqa", help="benchmark name to test")
    parser.add_argument("--max-concurrency", type=int, default=16, help="Maximum number of concurrent tasks (default: 5)")
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
    
    await test_math_benchmark(benchmark_name, max_concurrency=args.max_concurrency)
    
    print("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    print("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())
