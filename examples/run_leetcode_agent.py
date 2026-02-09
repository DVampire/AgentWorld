"""Test script for AIME benchmark with REAL Model Inference (Full Loop).

支持并发推理模式：
- 多个任务可以同时进行模型推理
- 提交评测时自动串行化（使用锁）
- 共享单个浏览器实例
"""

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
from typing import Union, List, Optional

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
from src.benchmark.leetcode import CodeSubmitter

# ==========================================
# Configuration Section (可通过命令行参数覆盖)
# ==========================================
TARGET_MODEL = "openrouter/gemini-3-flash-preview"  # 默认模型，可通过 --model 覆盖
TARGET_LANGUAGE = "python3"  # 默认语言，可通过 --language 覆盖
MAX_CONCURRENT_INFERENCE = 1  # 最大并发推理数量
BATCH_SIZE = 1  # 每批次评测的任务数量

# 🚫 定义不使用图片解析的模型列表
NON_VISION_MODELS = [
    "openrouter/deepseek-v3.2",
    "openrouter/qwen3-max",
    # 你可以在这里添加任何你想强制纯文本输入的模型
]

# 支持的编程语言列表
SUPPORTED_LANGUAGES = [
    "python3", "python", "cpp", "c++", "java", "javascript", "typescript",
    "c", "csharp", "c#", "go", "ruby", "swift", "rust", "scala", "kotlin", "php"
]
class Response(BaseModel):
    reasoning: str = Field(description="The reasoning process")
    result: str = Field(description="The generated code")

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

async def inference_single_task(
    task: Task,
    save_dir: str,
    semaphore: asyncio.Semaphore
) -> Task:
    """
    仅执行单个任务的推理阶段（不包括评测）。
    推理使用 semaphore 限制并发数。
    
    时间记录：
    - inference_start_time: 推理开始时间（获取信号量后）
    - inference_time: 推理耗时
    """
    task_id = task.task_id
    
    # 使用信号量限制并发推理数量
    async with semaphore:
        # ✅ 在获取信号量后立即记录推理开始时间
        inference_start_time = time.time()
        task.extra["inference_start_time"] = inference_start_time
        
        try:
            logger.info(f"| 🚀 [Task {task_id}] Starting inference...")
            
            # --- 1. Prepare Prompt ---
            question_text = task.input
            # 检查模型是否支持视觉/是否被禁用视觉
            if TARGET_MODEL in NON_VISION_MODELS:
                logger.info(f"| 🙈 [Task {task_id}] Vision disabled for model {TARGET_MODEL}, using text only.")
                question_content = question_text
            else:
                question_content = parse_markdown_with_images(question_text)
            
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
            logger.info(f"| ⏳ [Task {task_id}] Model inferencing...")
            
            try:
                response = await model_manager(
                    model=TARGET_MODEL,
                    messages=messages,
                    response_format=Response,
                    max_completion_tokens=65536
                )
                
                if response.success:
                    response_model = response.extra.parsed_model
                    task.reasoning = response_model.reasoning
                    task.result = response_model.result
                    
                    # Save Response to Markdown file
                    try:
                        file_name = f"{task.extra['file_name']}.md"
                        file_path = os.path.join(save_dir, file_name)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(task.reasoning + "\n\n" + task.result)
                        
                        logger.info(f"| 💾 [Task {task_id}] Output saved to: {file_path}")
                        
                    except Exception as save_err:
                        logger.error(f"| ⚠️ [Task {task_id}] Failed to save markdown: {save_err}")

                else:
                    logger.error(f"| ⚠️ [Task {task_id}] Model API Error: {response.message}")
                    
                    raw_output = (
                        getattr(response, "raw", None)
                        or getattr(response, "text", None)
                        or getattr(response.extra, "raw_response", None)
                        or getattr(response.extra, "text", None)
                    )

                    if raw_output:
                        logger.error(f"| 🧨 [Task {task_id}] RAW MODEL OUTPUT:\n{raw_output}")
                    else:
                        logger.error(f"| 🧨 [Task {task_id}] No raw model output found.")

                    task.reasoning = ""
                    task.result = ""
                    
            except Exception as e:
                logger.error(f"| ❌ [Task {task_id}] Critical Inference Error: {e}")
                task.reasoning = ""
                task.result = ""
            
            # ✅ 记录推理结束时间和耗时
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            task.extra["inference_time"] = inference_time
            
            logger.info(f"| ✅ [Task {task_id}] Inference complete in {inference_time:.2f}s")
                
        except Exception as e:
            logger.error(f"| ❌ [Task {task_id}] Error in inference phase: {e}")
            import traceback
            traceback.print_exc()
            task.reasoning = ""
            task.result = ""
            task.extra["inference_time"] = time.time() - inference_start_time
    
    return task


async def test_leetcode_benchmark(benchmark_name: str = "leetcode"):
    """
    Test the benchmark manager specifically for LeetCode using a REAL model.
    Uses PIPELINE mode: inference and evaluation run in parallel.
    
    流水线模式说明：
    - 推理完成 batch_size 个任务后，立即 push + 评测
    - 同时继续推理下一批任务
    - 推理失败的任务直接保存结果，不入队列
    - 共享一个浏览器实例
    """
    print(f"\n{'='*60}")
    print(f"🧪 LeetCode Benchmark Test (Pipeline Mode)")
    print(f"{'='*60}")
    print(f"🤖 Model: {TARGET_MODEL}")
    print(f"💻 Language: {TARGET_LANGUAGE}")
    print(f"⚡ Max concurrent inference: {MAX_CONCURRENT_INFERENCE}")
    print(f"📦 Batch size for evaluation: {BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    # Define save directory
    save_dir = os.path.join(config.workdir, "benchmark", benchmark_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Created output directory: {save_dir}")
    
    # 1. Reset and collect all tasks
    print(f"🔄 Resetting progress for LeetCode...")
    task = await benchmark_manager.reset(benchmark_name)
    
    if not task:
        logger.warning("⚠️ No tasks available to run (Dataset empty or all finished).")
        summarize_benchmark_results(benchmark_name)
        return

    # ==========================================
    # 收集所有待处理任务
    # ==========================================
    all_tasks: List[Task] = [task]
    while True:
        next_task = await benchmark_manager.step(benchmark_name)
        if next_task is None:
            break
        all_tasks.append(next_task)
    
    total_tasks = len(all_tasks)
    print(f"📋 Collected {total_tasks} tasks for processing\n")
    
    # ==========================================
    # 获取 benchmark 实例
    # ==========================================
    benchmark = await benchmark_manager.get(benchmark_name)
    
    # ==========================================
    # 流水线模式：推理完成直接入队，满5个自动评测
    # ==========================================
    inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
    
    # 统计变量（使用锁保护）
    stats = {
        "inference_done": 0,
        "inference_errors": 0,
        "queued": 0,
        "direct_saved": 0,
        "batches_evaluated": 0
    }
    stats_lock = asyncio.Lock()
    
    start_time = time.time()
    
    def print_status():
        """实时打印状态"""
        queue_size = benchmark.get_queue_size() if benchmark else 0
        print(f"\r🔄 Inferred: {stats['inference_done']}/{total_tasks} | "
              f"Queue: {queue_size}/{BATCH_SIZE} | "
              f"Batches: {stats['batches_evaluated']} | "
              f"Errors: {stats['direct_saved']}", end="", flush=True)
    
    async def inference_and_queue_worker(task: Task):
        """
        推理工作器：完成推理后直接入队。
        队列满 batch_size 时自动触发 push + 评测。
        使用 benchmark 内部的队列锁确保同一时间只有一个 task 入队。
        """
        # 1. 执行推理
        inferred_task = await inference_single_task(task, save_dir, inference_semaphore)
        
        # 2. 更新统计并入队
        async with stats_lock:
            stats["inference_done"] += 1
            
            if not inferred_task.result:
                # 推理失败，直接保存结果，不入队列
                stats["direct_saved"] += 1
                if hasattr(benchmark, 'save_error_result_directly'):
                    await benchmark.save_error_result_directly(inferred_task, prediction="response_error")
                logger.info(f"| ⚠️ [Task {inferred_task.task_id}] No code, saved directly")
            else:
                # 推理成功，入队列（benchmark 内部有锁）
                stats["queued"] += 1
                
        # 入队操作放在 stats_lock 外面，避免死锁
        # benchmark.queue_for_eval 内部有自己的锁
        if inferred_task.result:
            triggered_eval = await benchmark.queue_for_eval(inferred_task)
            if triggered_eval:
                async with stats_lock:
                    stats["batches_evaluated"] += 1
        
        print_status()
        return inferred_task
    
    # ==========================================
    # 启动流水线：所有推理任务并发执行
    # ==========================================
    print(f"🚀 Starting pipeline: inference → queue → auto-evaluate when batch full\n")
    
    # 创建并执行所有推理任务
    inference_tasks = [inference_and_queue_worker(t) for t in all_tasks]
    await asyncio.gather(*inference_tasks, return_exceptions=True)
    
    # 处理队列中剩余的任务（不满 batch_size 的最后一批）
    remaining = benchmark.get_queue_size()
    if remaining > 0:
        print(f"\n\n📤 Flushing remaining {remaining} tasks...")
        await benchmark.flush_eval_queue()
        async with stats_lock:
            stats["batches_evaluated"] += 1
    
    total_time = time.time() - start_time
    
    # ==========================================
    # 最终统计
    # ==========================================
    print(f"\n\n{'='*60}")
    print(f"🎉 All {total_tasks} tasks processed! (Pipeline Mode)")
    print(f"{'='*60}")
    print(f"🤖 Model: {TARGET_MODEL}")
    print(f"💻 Language: {TARGET_LANGUAGE}")
    print(f"{'='*60}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"{'='*60}")
    print(f"📊 Total tasks: {total_tasks}")
    print(f"📊 Inference completed: {stats['inference_done']}")
    print(f"📊 Queued for eval: {stats['queued']}")
    print(f"📊 Batches evaluated: {stats['batches_evaluated']}")
    print(f"📊 Direct saved (no code): {stats['direct_saved']}")
    print(f"{'='*60}")
    
    summarize_benchmark_results(benchmark_name)

def summarize_benchmark_results(benchmark_name: str):
    """
    Summarize benchmark results from results.jsonl.
    """

    import os
    import json
    from collections import Counter

    results_path = os.path.join(
        config.workdir, "benchmark", benchmark_name, "results.jsonl"
    )

    if not os.path.exists(results_path):
        logger.warning(f"⚠️ Results file not found: {results_path}")
        return

    total = 0
    score_1_cnt = 0
    pred_counter = Counter()

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("⚠️ Skipping invalid json line")
                continue

            total += 1

            if data.get("score") == 1.0:
                score_1_cnt += 1

            pred = data.get("prediction", "").strip()
            if pred:
                pred_counter[pred] += 1

    # =========================
    # 📊 Print Summary
    # =========================
    print("\n" + "=" * 60)
    print(f"📊 Benchmark Summary: {benchmark_name}")
    print(f"📊 Use model: {TARGET_MODEL}")
    print(f"📊 Use language: {TARGET_LANGUAGE}")
    print("=" * 60)
    print(f"Total tasks: {total}")
    print(f"score == 1.0 (Accepted): {score_1_cnt}")

    # 常见失败类型
    for key in [
        "Time Limit Exceeded",
        "Timeout",
        "Memory Limit Exceeded",
        "Compile Error",
        "Runtime Error",
        "Wrong Answer",
        "response_error",
    ]:
        print(f"{key}: {pred_counter.get(key, 0)}")

    print("=" * 60)

async def main():
    global TARGET_MODEL, TARGET_LANGUAGE, BATCH_SIZE
    
    parser = argparse.ArgumentParser(description='Test Benchmark Loop')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")
    parser.add_argument("--benchmark", default="leetcode", help="benchmark name to test")
    parser.add_argument("--model", default=TARGET_MODEL, help=f"Model to use for inference (default: {TARGET_MODEL})")
    parser.add_argument("--language", default=TARGET_LANGUAGE, choices=SUPPORTED_LANGUAGES, 
                        help=f"Programming language for LeetCode solutions (default: {TARGET_LANGUAGE})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for evaluation (default: {BATCH_SIZE})")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings')
    args = parser.parse_args()
    
    # 更新全局配置
    TARGET_MODEL = args.model
    TARGET_LANGUAGE = args.language
    BATCH_SIZE = args.batch_size
    
    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    
    logger.info("| 🧠 Initializing model manager...")
    if hasattr(model_manager, 'initialize'):
        await model_manager.initialize()
    
    benchmark_name = args.benchmark
    logger.info(f"| 🛠️ Initializing benchmark manager for {benchmark_name}...")
    
    await benchmark_manager.initialize(benchmark_names=[benchmark_name])
    
    # 设置 benchmark 的语言和批次大小
    benchmark = await benchmark_manager.get(benchmark_name)
    if benchmark:
        benchmark.language = TARGET_LANGUAGE
        benchmark.batch_size = BATCH_SIZE
        logger.info(f"| 🔧 Configured benchmark: language={TARGET_LANGUAGE}, batch_size={BATCH_SIZE}")
    
    await test_leetcode_benchmark(benchmark_name)
    
#     code = """
# #
# # @lc app=leetcode id=1 lang=python3
# #
# # [1] Two Sum
# #
# # @lc code=start
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         hashmap = {}
#         for i, num in enumerate(nums):
#             complement = target - num
#             if complement in hashmap:
#                 return [hashmap[complement], i]
#             hashmap[num] = i
#         return []
# # @lc code=end
#     """
    
#     file_name = "1.two-sum.py"
    
#     await submitter.submit_code(code, file_name)
    
    
    print("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    print("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())

