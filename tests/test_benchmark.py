"""Test script for LeetCode benchmark with REAL Model Inference (Full Loop)."""

import asyncio
import sys
import os
import argparse
import re
from pathlib import Path
from mmengine import DictAction
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.benchmark import benchmark_manager
from src.model.manager import model_manager
from src.message.types import HumanMessage, SystemMessage

# ==========================================
# 配置区域
# ==========================================
TARGET_MODEL = "openrouter/gemini-3-flash-preview" 

# 定义结构化输出的 Schema
class LeetCodeSolution(BaseModel):
    """
    LeetCode solution format.
    """
    reasoning: str = Field(
        description="Detailed step-by-step reasoning, algorithm analysis, and complexity analysis."
    )
    code: str = Field(
        description=(
            "The complete Python solution code. "
            "IMPORTANT: The code MUST be strictly enclosed between the standard LeetCode markers:\n"
            "# @lc code=start\n"
            "...\n"
            "# @lc code=end\n"
            "Do NOT wrap this block in markdown code ticks (```)."
        )
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Test LeetCode Benchmark Loop')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings')
    args = parser.parse_args()
    return args

def sanitize_filename(name: str) -> str:
    """清洗文件名，移除非法字符"""
    name = str(name).replace('\n', ' ').replace('\r', '')
    return re.sub(r'[\\/*?:"<>|]', '', name).strip()

async def test_leetcode_benchmark(benchmark_name: str = "leetcode"):
    """
    Test the benchmark manager specifically for LeetCode using a REAL model.
    Uses response_format for structured output.
    """
    print(f"🧪 Testing benchmark manager with benchmark: {benchmark_name}")
    print(f"🤖 Using Model: {TARGET_MODEL}")
    
    # 定义保存目录
    save_dir = Path(root) / "tmp" / "model_output" / "gemini3_flash_struct"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
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
        task_id = task.get("task_id", "unknown")
        
        try:
            print(f"\n" + "="*50)
            print(f"🚀 Processing Task ID: {task_id}")
            print("="*50)

            # --- 1. 准备 Prompt ---
            question_text = task.get("input") or task.get("question")
            
            # 直接从 task 获取 system_prompt，不进行硬编码修改
            system_prompt_text = task.get("system_prompt", "You are a helpful coding assistant.")
            
            logger.info(f"| 📋 [Task {task_id}] Input length: {len(question_text)}")

            messages = [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=question_text)
            ]

            # --- 2. 模型推理 (Structure Output) ---
            print(f"⏳ [Task {task_id}] Model inferencing (Structured)...")
            prediction_content = ""
            
            try:
                # 调用 model_manager 并传入 response_format
                response = await model_manager(
                    model=TARGET_MODEL,
                    messages=messages,
                    response_format=LeetCodeSolution, # <--- 关键修改
                    stream=False
                )
                
                if response.success:
                    # 获取解析后的对象
                    solution_obj: LeetCodeSolution = response.extra.parsed_model
                    
                    # 提取代码用于提交
                    prediction_content = solution_obj.code
                    reasoning_content = solution_obj.reasoning
                    
                    # --- 保存 Response 到 Markdown 文件 ---
                    try:
                        raw_title = task.get("title") or task.get("question_title") or task.get("name") or "Unknown"
                        safe_id = sanitize_filename(task_id)
                        safe_title = sanitize_filename(raw_title)
                        filename = f"{safe_id}_{safe_title}.md"
                        file_path = save_dir / filename
                        
                        # 将 reasoning 和 code 组合写入文件，方便人工查看
                        file_content = (
                            f"# Reasoning\n{reasoning_content}\n\n"
                            f"# Code\n{prediction_content}"
                        )
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(file_content)
                        
                        print(f"💾 [Saved] Output saved to: {file_path}")
                        
                    except Exception as save_err:
                        logger.error(f"⚠️ Failed to save markdown file: {save_err}")

                else:
                    logger.error(f"| ⚠️ [Task {task_id}] Model API Error: {response.message}")
                    prediction_content = "" 
                    
            except Exception as e:
                logger.error(f"| ❌ [Task {task_id}] Critical Inference Error: {e}")
                prediction_content = ""

            # --- 3. 提交评测 ---
            # 注意：如果 extraction 失败或者模型出错，prediction_content 为空
            print(f"🤖 [Task {task_id}] Submitting code to LeetCode...")
            
            if prediction_content:
                score = await benchmark_manager.eval_task(
                    benchmark_name, 
                    prediction=prediction_content, 
                    task_id=task_id
                )
                
                if score == 1.0:
                    print(f"✅ [Task {task_id}] Result: Accepted (Score: {score})")
                else:
                    print(f"⚠️ [Task {task_id}] Result: Failed/Partial (Score: {score})")
            else:
                print(f"⏭️ [Task {task_id}] Skipping eval due to empty model output.")
                # 即使没有输出也提交一次空值，以便 benchmark manager 推进到下一题(视具体逻辑而定)
                await benchmark_manager.eval_task(benchmark_name, prediction="", task_id=task_id)

            # --- 4. 实时统计 ---
            stats = await benchmark_manager.get_stats(benchmark_name)
            print(f"📊 Overall Progress: {stats.get('progress_percent')} | Accuracy: {stats.get('accuracy', 0):.2%}")

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
    args = parse_args()
    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    
    logger.info("| 🧠 Initializing model manager...")
    if hasattr(model_manager, 'initialize'):
        await model_manager.initialize()
    
    logger.info("| 🛠️ Initializing benchmark manager...")
    await benchmark_manager.initialize(benchmark_names=["leetcode"])
    
    await test_leetcode_benchmark("leetcode")
    
    print("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    print("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())