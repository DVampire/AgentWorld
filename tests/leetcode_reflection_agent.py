"""Simple Self-Reflection LeetCode Agent

简化版self-reflection解题agent：
- 去掉可训练参数的优化，只保留solution的优化
- 优化solution时，将问题、生成的代码以及评测结果都当作输入
- 最多迭代3轮，如果还是无法解题就停止
- 保持多协程模式
"""

import asyncio
import sys
import os
import argparse
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mmengine import DictAction

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
from src.benchmark.leetcode import CodeSubmitter, submit_lock

# ==========================================
# 批量Push队列管理器
# ==========================================
class BatchPushManager:
    """
    批量Push管理器：收集多个文件后一次性push，减少git操作频率
    
    触发push的条件：
    1. 队列达到batch_size
    2. 队列不满但超过timeout_seconds（默认2分钟）- 通过后台定时器主动触发
    """
    def __init__(self, batch_size: int = 5, timeout_seconds: float = 120.0, idle_timeout: float = 15.0):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds  # 硬超时时间（秒）
        self.idle_timeout = idle_timeout  # 空闲超时（队列有文件但没有新文件加入的时间）
        self._pending_files: List[Dict[str, Any]] = []  # [{"filename": str, "code": str, "future": asyncio.Future}]
        self._lock = asyncio.Lock()
        self._submitter = None
        self._first_file_time: Optional[float] = None  # 队列中第一个文件加入的时间
        self._last_add_time: Optional[float] = None  # 最后一个文件加入的时间
        self._timeout_task: Optional[asyncio.Task] = None  # 超时定时器任务
        self._running = False  # 是否正在运行
        self._expected_total: Optional[int] = None  # 预期的总文件数
        self._added_count: int = 0  # 已加入队列的文件总数（包括已push的）
    
    def set_submitter(self, submitter: CodeSubmitter):
        """设置submitter实例"""
        self._submitter = submitter
    
    def set_expected_total(self, total: int):
        """
        设置预期的总文件数。当队列达到这个数时会立即push，不需要等待batch_size或超时。
        这用于处理任务数少于batch_size的情况。
        """
        self._expected_total = total
        self._added_count = 0
        logger.info(f"| 📊 Expected total files: {total}")
    
    def start(self):
        """启动管理器"""
        self._running = True
    
    async def stop(self):
        """停止管理器，取消定时器"""
        self._running = False
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None
    
    def _check_timeout(self) -> tuple[bool, str]:
        """
        检查是否超时（调用前需要持有锁）
        
        Returns:
            (is_timeout, reason): 是否超时及原因
        """
        if not self._pending_files:
            return False, ""
        
        now = time.time()
        
        # 检查空闲超时（最后一个文件加入后一段时间没有新文件）
        if self._last_add_time is not None:
            idle_elapsed = now - self._last_add_time
            if idle_elapsed >= self.idle_timeout:
                return True, f"idle timeout ({self.idle_timeout}s)"
        
        # 检查硬超时（第一个文件加入后的总时间）
        if self._first_file_time is not None:
            total_elapsed = now - self._first_file_time
            if total_elapsed >= self.timeout_seconds:
                return True, f"hard timeout ({self.timeout_seconds}s)"
        
        return False, ""
    
    def _cancel_timeout_task(self):
        """取消当前的超时定时器任务"""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None
    
    def _schedule_timeout_task(self):
        """调度超时定时器任务（使用空闲超时，更快响应）"""
        if not self._running:
            return
        
        # 取消已有的定时器
        self._cancel_timeout_task()
        
        if not self._pending_files:
            return
        
        # 使用空闲超时作为等待时间（更快响应）
        wait_time = self.idle_timeout
        
        async def timeout_callback():
            """超时回调：检查并触发push"""
            try:
                await asyncio.sleep(wait_time)
                async with self._lock:
                    # 再次检查是否超时（可能在等待期间已经被push了）
                    is_timeout, reason = self._check_timeout()
                    if self._pending_files and is_timeout:
                        logger.info(f"| ⏰ Timeout timer triggered: {reason}, pushing {len(self._pending_files)} files...")
                        await self._flush_queue()
                    elif self._pending_files:
                        # 还没超时，重新调度
                        self._schedule_timeout_task()
            except asyncio.CancelledError:
                pass  # 任务被取消，正常退出
            except Exception as e:
                logger.error(f"| ❌ Timeout task error: {e}")
        
        self._timeout_task = asyncio.create_task(timeout_callback())
    
    async def add_file_and_wait(self, filename: str, code: str) -> bool:
        """
        添加文件到队列并等待push完成
        
        Args:
            filename: 文件名
            code: 代码内容
            
        Returns:
            push是否成功
        """
        future = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            # 写入本地文件
            if self._submitter and self._submitter.repo_path:
                file_path = os.path.join(self._submitter.repo_path, filename)
                try:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                except Exception as e:
                    logger.error(f"| ❌ Failed to write file {filename}: {e}")
                    future.set_result(False)
                    return False
            
            # 记录时间
            now = time.time()
            is_first_file = not self._pending_files
            if is_first_file:
                self._first_file_time = now
            self._last_add_time = now  # 记录最后加入时间（用于空闲超时）
            
            self._pending_files.append({
                "filename": filename,
                "code": code,
                "future": future
            })
            
            # 更新已加入的文件计数
            self._added_count += 1
            
            queue_size = len(self._pending_files)
            elapsed = now - self._first_file_time if self._first_file_time else 0
            
            # 显示进度信息
            if self._expected_total:
                logger.info(f"| 📥 Added {filename} to push queue ({queue_size}/{self.batch_size}, "
                           f"total {self._added_count}/{self._expected_total}, waiting {elapsed:.1f}s)")
            else:
                logger.info(f"| 📥 Added {filename} to push queue ({queue_size}/{self.batch_size}, waiting {elapsed:.1f}s)")
            
            # 触发push的条件：队列满 或 超时 或 所有任务都已加入队列
            should_push = False
            push_reason = ""
            
            if queue_size >= self.batch_size:
                should_push = True
                push_reason = "batch full"
                self._cancel_timeout_task()  # 队列满了，取消定时器
            elif self._expected_total and self._added_count >= self._expected_total:
                # 所有预期的任务都已加入队列，立即push
                should_push = True
                push_reason = f"all {self._expected_total} tasks queued"
                self._cancel_timeout_task()
            else:
                # 检查超时
                is_timeout, timeout_reason = self._check_timeout()
                if is_timeout:
                    should_push = True
                    push_reason = timeout_reason
                    self._cancel_timeout_task()
            
            # 如果不需要立即push，启动/重新调度超时定时器
            if not should_push:
                self._schedule_timeout_task()
            
            if should_push:
                logger.info(f"| ⏰ Triggering push: {push_reason}")
                await self._flush_queue()
        
        # 等待push完成
        return await future
    
    async def flush(self) -> None:
        """强制刷新队列中的所有文件"""
        async with self._lock:
            self._cancel_timeout_task()  # 取消定时器
            if self._pending_files:
                await self._flush_queue()
    
    async def _flush_queue(self) -> None:
        """内部方法：执行批量push（调用前需要持有锁）"""
        if not self._pending_files or not self._submitter:
            return
        
        files_to_push = self._pending_files.copy()
        self._pending_files.clear()
        self._first_file_time = None  # 重置计时器
        self._last_add_time = None  # 重置最后加入时间
        self._cancel_timeout_task()  # 清理定时器
        
        filenames = [f["filename"] for f in files_to_push]
        logger.info(f"| 🚀 Batch pushing {len(filenames)} files...")
        
        try:
            # 批量push（不在这里执行 Codespace sync，因为可能有其他任务正在使用浏览器）
            # sync_codespace=False 会设置 _needs_sync 标志，在下次 eval_single_file 时执行 sync
            push_success = await self._submitter.batch_push(filenames, max_retries=3, sync_codespace=False)
            
            # 通知所有等待的任务
            for file_info in files_to_push:
                if not file_info["future"].done():
                    file_info["future"].set_result(push_success)
            
            if push_success:
                logger.info(f"| ✅ Batch push successful: {len(filenames)} files")
            else:
                logger.error(f"| ❌ Batch push failed for {len(filenames)} files")
                
        except Exception as e:
            logger.error(f"| ❌ Batch push error: {e}")
            for file_info in files_to_push:
                if not file_info["future"].done():
                    file_info["future"].set_result(False)
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return len(self._pending_files)
    
    def get_wait_time(self) -> float:
        """获取当前队列已等待的时间（秒）"""
        if self._first_file_time is None:
            return 0.0
        return time.time() - self._first_file_time


# 全局批量push管理器实例
# 全局批量push管理器实例（2分钟硬超时，15秒空闲超时）
batch_push_manager = BatchPushManager(batch_size=5, timeout_seconds=120.0, idle_timeout=15.0)


# ==========================================
# Configuration Section
# ==========================================
TARGET_MODEL = "openrouter/gemini-3-flash-preview"
TARGET_LANGUAGE = "kotlin"
MAX_CONCURRENT_INFERENCE = 1
BATCH_SIZE = 1
MAX_REFLECTION_ROUNDS = 3  # 最大反思迭代轮数
OPTIMIZATION_THRESHOLD = 65.0  # 性能优化阈值（beats百分比），超过此值则停止优化

# 判断模式：threshold（基于阈值）或 llm（LLM as a Judge）
JUDGE_MODE = "threshold"  # "threshold" or "llm"

# 批量push超时时间（秒），队列不满时超过此时间也会自动push
PUSH_TIMEOUT = 120.0  # 2分钟

# 不使用图片解析的模型列表
NON_VISION_MODELS = [
    "openrouter/deepseek-v3.2",
    "openrouter/qwen3-max",
]

# 支持的编程语言列表
SUPPORTED_LANGUAGES = [
    "python3", "python", "cpp", "c++", "java", "javascript", "typescript",
    "c", "csharp", "c#", "go", "ruby", "swift", "rust", "scala", "kotlin", "php"
]


class CodeResponse(BaseModel):
    """模型生成代码的响应格式"""
    reasoning: str = Field(description="The reasoning process")
    result: str = Field(description="The generated code")


class ReflectionResponse(BaseModel):
    """反思并改进代码的响应格式"""
    analysis: str = Field(description="Analysis of why the previous solution failed")
    improved_reasoning: str = Field(description="The improved reasoning process")
    improved_code: str = Field(description="The improved solution code")


class LLMJudgeResult(BaseModel):
    """LLM as a Judge 的评估结果"""
    should_stop: bool = Field(description="Whether optimization should stop")
    reasoning: str = Field(description="The reasoning for the decision")
    suggestion: str = Field(default="", description="Suggestion for improvement if should continue")


def parse_markdown_with_images(markdown_text: str) -> Union[str, list]:
    """
    Parse markdown text and convert it to a message content format that supports images.
    """
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = list(re.finditer(image_pattern, markdown_text))
    
    if not matches:
        return markdown_text
    
    content_parts = []
    last_end = 0
    
    for match in matches:
        text_before = markdown_text[last_end:match.start()]
        if text_before:
            content_parts.append(ContentPartText(text=text_before))
        
        image_url = match.group(2)
        media_type = 'image/png'
        url_path = image_url.split('?')[0].lower()
        if url_path.endswith('.jpg') or url_path.endswith('.jpeg'):
            media_type = 'image/jpeg'
        elif url_path.endswith('.png'):
            media_type = 'image/png'
        elif url_path.endswith('.gif'):
            media_type = 'image/gif'
        elif url_path.endswith('.webp'):
            media_type = 'image/webp'
        
        image_url_obj = ImageURL(url=image_url, media_type=media_type)
        content_parts.append(ContentPartImage(image_url=image_url_obj))
        
        last_end = match.end()
    
    text_after = markdown_text[last_end:]
    if text_after:
        content_parts.append(ContentPartText(text=text_after))
    
    return content_parts


def format_evaluation_result(eval_result: Dict[str, Any]) -> str:
    """格式化评测结果为易于模型理解的字符串"""
    status = eval_result.get("status", "Unknown")
    passed = eval_result.get("passed_cases", 0)
    total = eval_result.get("total_cases", 0)
    
    result_text = f"Status: {status}\n"
    result_text += f"Passed Cases: {passed}/{total}\n"
    
    if status == "Accepted":
        runtime = eval_result.get("runtime", 0)
        memory = eval_result.get("memory_usage", 0)
        runtime_beats = eval_result.get("runtime_beats", 0)
        memory_beats = eval_result.get("memory_beats", 0)
        result_text += f"Runtime: {runtime}ms (beats {runtime_beats}%)\n"
        result_text += f"Memory: {memory}MB (beats {memory_beats}%)\n"
    elif status == "Wrong Answer":
        result_text += "The solution produced incorrect output for some test cases.\n"
    elif status == "Time Limit Exceeded":
        result_text += "The solution is too slow. Consider optimizing the algorithm.\n"
    elif status == "Runtime Error":
        result_text += "The solution crashed during execution. Check for index errors, null pointers, etc.\n"
    elif status == "Memory Limit Exceeded":
        result_text += "The solution uses too much memory. Consider using more memory-efficient data structures.\n"
    elif status == "Compile Error":
        result_text += "The code failed to compile. Check for syntax errors.\n"
    
    if "details_text" in eval_result:
        # 只取前200字符的details
        details = eval_result["details_text"][:200]
        result_text += f"Details: {details}...\n"
    
    return result_text


async def generate_initial_solution(
    task: Task,
    model: str,
) -> CodeResponse:
    """生成初始解决方案"""
    question_text = task.input
    system_prompt_text = task.system_prompt
    
    # 检查模型是否支持视觉
    if model in NON_VISION_MODELS:
        question_content = question_text
    else:
        question_content = parse_markdown_with_images(question_text)
    
    messages = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=question_content)
    ]
    
    response = await model_manager(
        model=model,
        messages=messages,
        response_format=CodeResponse,
        max_completion_tokens=65536
    )
    
    if response.success:
        return response.extra.parsed_model
    else:
        raise Exception(f"Model API Error: {response.message}")


def should_stop_optimization(eval_result: Dict[str, Any], optimization_threshold: float = 50.0) -> tuple[bool, str]:
    """
    根据评测结果判断是否应该停止优化
    
    Args:
        eval_result: 评测结果
        optimization_threshold: 性能优化阈值（beats百分比），超过此值则停止优化
    
    Returns:
        (should_stop, reason): 是否停止及原因
    """
    status = eval_result.get("status", "")
    
    if status != "Accepted":
        # 非Accepted，不应该停止
        return False, f"Solution not accepted: {status}"
    
    # Accepted情况，检查性能指标
    runtime_beats = eval_result.get("runtime_beats", 0.0)
    memory_beats = eval_result.get("memory_beats", 0.0)
    
    # 如果运行时和内存都超过阈值，可以停止
    if runtime_beats >= optimization_threshold and memory_beats >= optimization_threshold:
        return True, f"Performance is good enough (runtime beats {runtime_beats}%, memory beats {memory_beats}%)"
    
    # # 如果只有一个指标超过阈值，看综合情况
    # if runtime_beats >= optimization_threshold:
    #     return True, f"Runtime performance is good (beats {runtime_beats}%), memory beats {memory_beats}%"
    
    # if memory_beats >= optimization_threshold:
    #     return True, f"Memory performance is good (beats {memory_beats}%), runtime beats {runtime_beats}%"
    
    # 性能不够好，继续优化
    return False, f"Performance can be improved (runtime beats {runtime_beats}%, memory beats {memory_beats}%)"


async def llm_judge_should_stop(
    task: Task,
    current_code: str,
    current_reasoning: str,
    eval_result: Dict[str, Any],
    model: str,
    round_num: int,
) -> tuple[bool, str]:
    """
    使用 LLM as a Judge 判断是否应该停止优化
    
    Args:
        task: 任务
        current_code: 当前代码
        current_reasoning: 当前推理
        eval_result: 评测结果
        model: 模型名称
        round_num: 当前轮数
    
    Returns:
        (should_stop, reason): 是否停止及原因
    """
    question_text = task.input
    eval_result_text = format_evaluation_result(eval_result)
    status = eval_result.get("status", "Unknown")
    
    # 构建 LLM Judge 的 prompt
    judge_system_prompt = f"""You are an expert code reviewer and judge for LeetCode solutions.
Your task is to evaluate whether the current solution is satisfactory and whether optimization should stop.

Current Round: {round_num}/{MAX_REFLECTION_ROUNDS}

Evaluation Criteria:
1. **Correctness**: Is the solution accepted by LeetCode? (This is the most important criterion)
2. **Efficiency**: If accepted, is the runtime and memory performance acceptable?
3. **Code Quality**: Is the code clean, readable, and well-structured?
4. **Improvement Potential**: Is there significant room for improvement?

Decision Guidelines:
- If the solution is NOT accepted (Wrong Answer, TLE, Runtime Error, etc.), you should NOT stop - we need to fix the bugs.
- If the solution is Accepted with good performance (beats > 50%), you can consider stopping.
- If the solution is Accepted but with poor performance, consider whether optimization is worth the effort.
- Consider the current round number - if we're at the last round, we might want to stop.

Output format:
The output should be a JSON object with the following fields:
{{
    "should_stop": true/false,
    "reasoning": "Your detailed reasoning for the decision",
    "suggestion": "If should_stop is false, provide a brief suggestion for improvement"
}}
"""

    judge_user_prompt = f"""## Problem
{question_text}

## Current Solution (Round {round_num})
### Reasoning:
{current_reasoning}

### Code:
```
{current_code}
```

## LeetCode Evaluation Result
{eval_result_text}

## Task
Based on the evaluation result and the current solution, decide whether we should:
1. STOP optimization (the solution is good enough)
2. CONTINUE optimization (there's room for improvement)

Please provide your judgment.
"""

    messages = [
        SystemMessage(content=judge_system_prompt),
        HumanMessage(content=judge_user_prompt)
    ]
    
    try:
        response = await model_manager(
            model=model,
            messages=messages,
            response_format=LLMJudgeResult,
            max_completion_tokens=4096
        )
        
        if response.success:
            judge_result: LLMJudgeResult = response.extra.parsed_model
            reason = f"LLM Judge: {judge_result.reasoning}"
            if not judge_result.should_stop and judge_result.suggestion:
                reason += f" | Suggestion: {judge_result.suggestion}"
            return judge_result.should_stop, reason
        else:
            # API 调用失败，默认继续优化
            return False, f"LLM Judge API failed: {response.message}, continuing optimization"
            
    except Exception as e:
        # 发生异常，默认继续优化
        logger.warning(f"| ⚠️ LLM Judge failed: {e}, continuing optimization")
        return False, f"LLM Judge exception: {str(e)}, continuing optimization"


async def should_stop_with_judge_mode(
    task: Task,
    current_code: str,
    current_reasoning: str,
    eval_result: Dict[str, Any],
    model: str,
    round_num: int,
    judge_mode: str,
    optimization_threshold: float,
) -> tuple[bool, str]:
    """
    根据 judge_mode 选择判断方式
    
    Args:
        judge_mode: "threshold" 或 "llm"
        其他参数同上
    
    Returns:
        (should_stop, reason): 是否停止及原因
    """
    if judge_mode == "llm":
        return await llm_judge_should_stop(
            task=task,
            current_code=current_code,
            current_reasoning=current_reasoning,
            eval_result=eval_result,
            model=model,
            round_num=round_num,
        )
    else:
        # 默认使用阈值模式
        return should_stop_optimization(eval_result, optimization_threshold)


async def generate_bugfix_reflection(
    task: Task,
    previous_code: str,
    previous_reasoning: str,
    eval_result: Dict[str, Any],
    model: str,
    round_num: int,
) -> ReflectionResponse:
    """
    生成Bug修复的反思解决方案（用于非Accepted情况）
    """
    question_text = task.input
    eval_result_text = format_evaluation_result(eval_result)
    status = eval_result.get("status", "Unknown")
    
    # 根据不同的错误类型，给出针对性的提示
    error_hints = '''
- Check boundary conditions and edge cases
- Verify your algorithm logic step by step
- Consider special inputs like empty arrays, single elements, negative numbers, etc.
- Make sure you understand the problem requirements correctly
'''
#     if status == "Wrong Answer":
#         error_hints = """
# - Check boundary conditions and edge cases
# - Verify your algorithm logic step by step
# - Consider special inputs like empty arrays, single elements, negative numbers, etc.
# - Make sure you understand the problem requirements correctly
# """
#     elif status == "Time Limit Exceeded":
#         error_hints = """
# - Your current algorithm is too slow, analyze the time complexity
# - Consider using more efficient data structures (hash map, heap, etc.)
# - Look for opportunities to reduce nested loops
# - Consider dynamic programming or greedy approaches if applicable
# """
#     elif status == "Runtime Error":
#         error_hints = """
# - Check for array index out of bounds
# - Check for null/None pointer access
# - Check for division by zero
# - Check for stack overflow (infinite recursion)
# - Verify data type ranges and overflow issues
# """
#     elif status == "Memory Limit Exceeded":
#         error_hints = """
# - Your solution uses too much memory
# - Consider using in-place algorithms
# - Avoid storing unnecessary data
# - Consider iterative instead of recursive approaches
# """
#     elif status == "Compile Error":
#         error_hints = """
# - Check for syntax errors
# - Verify all imports are correct
# - Check for type mismatches
# - Ensure all variables are properly defined
# """

#     reflection_system_prompt = f"""You are an expert programmer debugging a LeetCode solution.
# Your previous solution got "{status}". You need to carefully analyze why it failed and provide a fixed solution.
    reflection_system_prompt = f"""You are an expert programmer debugging a LeetCode solution.
You need to carefully analyze why it failed and provide a fixed solution.

IMPORTANT DEBUGGING GUIDELINES:
{error_hints}

You MUST:
1. Carefully analyze the error type and evaluation result
2. Identify the root cause of the failure
3. Provide a completely corrected solution,Do not assume the original approach is correct—validate the algorithm itself and fix or replace it if necessary
4. Make sure your code is syntactically correct and handles all edge cases

Output format:
The output should be a JSON object with the following fields:
{{
    "analysis": "Your detailed analysis of why the previous solution failed and what caused the {status}",
    "improved_reasoning": "Your new step-by-step reasoning process for the fixed solution",
    "improved_code": "Your complete fixed solution code"
}}
"""

    reflection_user_prompt = f"""## Problem
{question_text}

## Your Previous Solution (Round {round_num - 1})
### Reasoning:
{previous_reasoning}

### Code:
```
{previous_code}
```

## Evaluation Result
{eval_result_text}

## Task
Your solution got "{status}". Analyze why it failed and provide a fixed solution.
Make sure to:
1. Identify and fix the bug that caused the {status}
2. Handle all edge cases properly
3. Ensure the solution is efficient enough
4. Keep the same code template format (with @lc code=start and @lc code=end markers)
"""

    messages = [
        SystemMessage(content=reflection_system_prompt),
        HumanMessage(content=reflection_user_prompt)
    ]
    
    response = await model_manager(
        model=model,
        messages=messages,
        response_format=ReflectionResponse,
        max_completion_tokens=65536
    )
    
    if response.success:
        return response.extra.parsed_model
    else:
        raise Exception(f"Model API Error: {response.message}")


async def generate_optimization_reflection(
    task: Task,
    previous_code: str,
    previous_reasoning: str,
    eval_result: Dict[str, Any],
    model: str,
    round_num: int,
) -> ReflectionResponse:
    """
    生成性能优化的反思解决方案（用于Accepted但性能不够好的情况）
    """
    question_text = task.input
    eval_result_text = format_evaluation_result(eval_result)
    
    runtime_beats = eval_result.get("runtime_beats", 0.0)
    memory_beats = eval_result.get("memory_beats", 0.0)
    runtime = eval_result.get("runtime", 0)
    memory = eval_result.get("memory_usage", 0)
    
    reflection_system_prompt = f"""You are an expert programmer optimizing a LeetCode solution.
Your previous solution was ACCEPTED but the performance can be improved:
- Runtime: {runtime}ms (beats {runtime_beats}% of submissions)
- Memory: {memory}MB (beats {memory_beats}% of submissions)

Your goal is to optimize the solution for better time and/or space complexity.

OPTIMIZATION STRATEGIES TO CONSIDER:
1. **Time Optimization:**
   - Use more efficient algorithms (e.g., binary search instead of linear search)
   - Use better data structures (e.g., hash map for O(1) lookup)
   - Reduce unnecessary computations
   - Consider dynamic programming or memoization
   
2. **Space Optimization:**
   - Use in-place algorithms when possible
   - Avoid creating unnecessary copies of data
   - Consider iterative vs recursive approaches
   - Reuse variables when possible

IMPORTANT:
- The optimized solution MUST still be correct (pass all test cases)
- Focus on reducing time complexity first, then space complexity
- Explain the complexity improvement in your analysis

Output format:
The output should be a JSON object with the following fields:
{{
    "analysis": "Your analysis of the current solution's complexity and how you plan to optimize it",
    "improved_reasoning": "Your step-by-step reasoning for the optimized solution with complexity analysis",
    "improved_code": "Your complete optimized solution code"
}}
"""

    reflection_user_prompt = f"""## Problem
{question_text}

## Your Previous Solution (Round {round_num - 1}) - ACCEPTED
### Reasoning:
{previous_reasoning}

### Code:
```
{previous_code}
```

## Current Performance
{eval_result_text}

## Task
Your solution is correct but can be optimized. Current performance:
- Runtime beats only {runtime_beats}% of submissions
- Memory beats only {memory_beats}% of submissions

Please provide an optimized solution that:
1. Maintains correctness (passes all test cases)
2. Has better time complexity if possible
3. Has better space complexity if possible
4. Keep the same code template format (with @lc code=start and @lc code=end markers)
"""

    messages = [
        SystemMessage(content=reflection_system_prompt),
        HumanMessage(content=reflection_user_prompt)
    ]
    
    response = await model_manager(
        model=model,
        messages=messages,
        response_format=ReflectionResponse,
        max_completion_tokens=65536
    )
    
    if response.success:
        return response.extra.parsed_model
    else:
        raise Exception(f"Model API Error: {response.message}")


async def inference_with_reflection(
    task: Task,
    save_dir: str,
    semaphore: asyncio.Semaphore,
    benchmark: Any,
) -> Task:
    """
    带反思的推理：生成代码 -> 评测 -> 反思改进 -> 再评测，最多MAX_REFLECTION_ROUNDS轮
    """
    task_id = task.task_id
    
    async with semaphore:
        inference_start_time = time.time()
        task.extra["inference_start_time"] = inference_start_time
        
        current_code = ""
        current_reasoning = ""
        best_score = 0.0
        best_code = ""
        best_reasoning = ""
        best_eval_result = None
        
        try:
            logger.info(f"| 🚀 [Task {task_id}] Starting inference with reflection...")
            
            # ============ Round 1: 初始生成 ============
            logger.info(f"| 🔄 [Task {task_id}] Round 1: Generating initial solution...")
            
            try:
                initial_response = await generate_initial_solution(task, TARGET_MODEL)
                current_reasoning = initial_response.reasoning
                current_code = initial_response.result
                
                logger.info(f"| ✅ [Task {task_id}] Round 1: Initial solution generated")
                
            except Exception as e:
                logger.error(f"| ❌ [Task {task_id}] Round 1: Failed to generate initial solution: {e}")
                task.reasoning = ""
                task.result = ""
                task.extra["inference_time"] = time.time() - inference_start_time
                task.extra["final_round"] = 0
                task.extra["error"] = str(e)
                return task
            
            # 保存初始代码
            try:
                file_name = f"{task.extra['file_name']}_round1.md"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Round 1 - Initial Solution\n\n")
                    f.write(f"## Reasoning\n{current_reasoning}\n\n")
                    f.write(f"## Code\n```\n{current_code}\n```\n")
            except Exception as save_err:
                logger.warning(f"| ⚠️ [Task {task_id}] Failed to save round 1 markdown: {save_err}")
            
            # ============ 迭代评测和反思 ============
            for round_num in range(1, MAX_REFLECTION_ROUNDS + 1):
                logger.info(f"| 📊 [Task {task_id}] Round {round_num}: Evaluating solution...")
                
                if not current_code:
                    logger.warning(f"| ⚠️ [Task {task_id}] Round {round_num}: No code to evaluate")
                    break
                
                # 准备评测任务
                eval_task = Task(
                    task_id=f"{task_id}_round{round_num}",
                    input=task.input,
                    system_prompt=task.system_prompt,
                    ground_truth=task.ground_truth,
                    result=current_code,
                    reasoning=current_reasoning,
                    extra={
                        "file_name": f"{task.extra['file_name']}_r{round_num}",
                        "file_ext": task.extra.get("file_ext", "py"),
                        "task_start_time": time.time(),
                        "inference_time": 0,
                    }
                )
                
                # 写入文件到队列，批量push后再评测
                file_name = f"{eval_task.extra['file_name']}.{eval_task.extra['file_ext']}"
                
                try:
                    # 添加文件到批量队列并等待push完成
                    logger.info(f"| 📝 [Task {task_id}] Round {round_num}: Adding file to batch queue...")
                    push_success = await batch_push_manager.add_file_and_wait(file_name, current_code)
                    
                    if not push_success:
                        logger.error(f"| ❌ [Task {task_id}] Round {round_num}: Push failed")
                        continue
                    
                    # 使用锁确保浏览器评测操作串行化
                    eval_result = None
                    async with submit_lock:
                        logger.info(f"| 🔒 [Task {task_id}] Round {round_num}: Browser lock acquired, evaluating...")
                        
                        # 评测
                        eval_result = await benchmark._submitter.eval_single_file(file_name)
                    
                    # 锁已自动释放
                    if eval_result is None:
                        # 评测失败，继续下一轮
                        continue
                    
                    # 计算分数
                    passed = eval_result.get("passed_cases", 0)
                    total = eval_result.get("total_cases", 0)
                    status = eval_result.get("status", "Unknown")
                    score = float(passed) / float(total) if total > 0 else (1.0 if status == "Accepted" else 0.0)
                    is_accepted = (status == "Accepted")
                    
                    logger.info(f"| 📈 [Task {task_id}] Round {round_num}: Status = {status}, Score = {score:.2%} ({passed}/{total})")
                    
                    # 更新最佳结果（只有Accepted的结果才更新，或者当前没有Accepted的结果时更新）
                    if is_accepted:
                        if not best_eval_result or best_eval_result.get("status") != "Accepted":
                            # 第一次Accepted，直接更新
                            best_score = score
                            best_code = current_code
                            best_reasoning = current_reasoning
                            best_eval_result = eval_result
                        else:
                            # 已有Accepted结果，比较性能
                            prev_runtime_beats = best_eval_result.get("runtime_beats", 0)
                            curr_runtime_beats = eval_result.get("runtime_beats", 0)
                            if curr_runtime_beats > prev_runtime_beats:
                                best_score = score
                                best_code = current_code
                                best_reasoning = current_reasoning
                                best_eval_result = eval_result
                    elif score > best_score:
                        # 非Accepted情况，只有分数更高才更新
                        best_score = score
                        best_code = current_code
                        best_reasoning = current_reasoning
                        best_eval_result = eval_result
                    
                    # ============ 根据eval结果判断是否提前结束 ============
                    should_stop, stop_reason = await should_stop_with_judge_mode(
                        task=task,
                        current_code=current_code,
                        current_reasoning=current_reasoning,
                        eval_result=eval_result,
                        model=TARGET_MODEL,
                        round_num=round_num,
                        judge_mode=JUDGE_MODE,
                        optimization_threshold=OPTIMIZATION_THRESHOLD,
                    )
                    
                    if should_stop:
                        logger.info(f"| 🎉 [Task {task_id}] Round {round_num}: Stopping optimization - {stop_reason}")
                        task.extra["final_round"] = round_num
                        task.extra["stop_reason"] = stop_reason
                        break
                    
                    # 如果还有迭代次数，根据状态选择不同的反思策略
                    if round_num < MAX_REFLECTION_ROUNDS:
                        if is_accepted:
                            # Accepted但性能不够好，进行性能优化
                            logger.info(f"| ⚡ [Task {task_id}] Round {round_num + 1}: Generating OPTIMIZATION reflection...")
                            reflection_type = "optimization"
                            
                            try:
                                reflection_response = await generate_optimization_reflection(
                                    task=task,
                                    previous_code=current_code,
                                    previous_reasoning=current_reasoning,
                                    eval_result=eval_result,
                                    model=TARGET_MODEL,
                                    round_num=round_num + 1,
                                )
                            except Exception as e:
                                logger.error(f"| ❌ [Task {task_id}] Round {round_num + 1}: Optimization reflection failed: {e}")
                                task.extra["final_round"] = round_num
                                task.extra["stop_reason"] = "Optimization reflection failed"
                                break
                        else:
                            # 非Accepted，进行bug修复
                            logger.info(f"| 🔧 [Task {task_id}] Round {round_num + 1}: Generating BUGFIX reflection...")
                            reflection_type = "bugfix"
                            
                            try:
                                reflection_response = await generate_bugfix_reflection(
                                    task=task,
                                    previous_code=current_code,
                                    previous_reasoning=current_reasoning,
                                    eval_result=eval_result,
                                    model=TARGET_MODEL,
                                    round_num=round_num + 1,
                                )
                            except Exception as e:
                                logger.error(f"| ❌ [Task {task_id}] Round {round_num + 1}: Bugfix reflection failed: {e}")
                                task.extra["final_round"] = round_num
                                task.extra["stop_reason"] = "Bugfix reflection failed"
                                break
                        
                        # 更新当前代码
                        current_reasoning = reflection_response.improved_reasoning
                        current_code = reflection_response.improved_code
                        
                        # 保存反思结果
                        try:
                            file_name_md = f"{task.extra['file_name']}_round{round_num + 1}.md"
                            file_path_md = os.path.join(save_dir, file_name_md)
                            with open(file_path_md, "w", encoding="utf-8") as f:
                                f.write(f"# Round {round_num + 1} - {reflection_type.upper()} Reflection\n\n")
                                f.write(f"## Previous Result\n{format_evaluation_result(eval_result)}\n\n")
                                f.write(f"## Analysis\n{reflection_response.analysis}\n\n")
                                f.write(f"## Improved Reasoning\n{current_reasoning}\n\n")
                                f.write(f"## Improved Code\n```\n{current_code}\n```\n")
                        except Exception as save_err:
                            logger.warning(f"| ⚠️ [Task {task_id}] Failed to save round {round_num + 1} markdown: {save_err}")
                        
                        logger.info(f"| ✅ [Task {task_id}] Round {round_num + 1}: {reflection_type.capitalize()} reflection generated")
                    else:
                        task.extra["final_round"] = round_num
                        task.extra["stop_reason"] = "Max rounds reached"
                        
                except Exception as e:
                    logger.error(f"| ❌ [Task {task_id}] Round {round_num}: Evaluation error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 使用最佳结果
            task.reasoning = best_reasoning
            task.result = best_code
            task.score = best_score
            task.extra["best_score"] = best_score
            task.extra["best_eval_result"] = best_eval_result
            
            inference_end_time = time.time()
            task.extra["inference_time"] = inference_end_time - inference_start_time
            
            logger.info(f"| ✅ [Task {task_id}] Completed: Best score = {best_score:.2%} after {task.extra.get('final_round', MAX_REFLECTION_ROUNDS)} rounds")
            
        except Exception as e:
            logger.error(f"| ❌ [Task {task_id}] Critical error: {e}")
            import traceback
            traceback.print_exc()
            task.reasoning = ""
            task.result = ""
            task.extra["inference_time"] = time.time() - inference_start_time
            task.extra["error"] = str(e)
    
    return task


async def test_leetcode_with_reflection(benchmark_name: str = "leetcode"):
    """
    使用Self-Reflection测试LeetCode
    """
    print(f"\n{'='*60}")
    print(f"🧪 LeetCode Benchmark Test (Self-Reflection Mode)")
    print(f"{'='*60}")
    print(f"🤖 Model: {TARGET_MODEL}")
    print(f"💻 Language: {TARGET_LANGUAGE}")
    print(f"🔄 Max Reflection Rounds: {MAX_REFLECTION_ROUNDS}")
    print(f"⚖️ Judge Mode: {JUDGE_MODE}")
    if JUDGE_MODE == "threshold":
        print(f"📊 Optimization Threshold: {OPTIMIZATION_THRESHOLD}%")
    else:
        print(f"🤖 Using LLM as a Judge for optimization decisions")
    print(f"⚡ Max concurrent inference: {MAX_CONCURRENT_INFERENCE}")
    print(f"{'='*60}\n")
    
    # Define save directory
    save_dir = os.path.join(config.workdir, "benchmark", benchmark_name + "_reflection")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Created output directory: {save_dir}")
    
    # Reset and collect all tasks
    print(f"🔄 Resetting progress for LeetCode...")
    task = await benchmark_manager.reset(benchmark_name)
    
    if not task:
        logger.warning("⚠️ No tasks available to run.")
        return

    # 收集所有待处理任务
    all_tasks: List[Task] = [task]
    while True:
        next_task = await benchmark_manager.step(benchmark_name)
        if next_task is None:
            break
        all_tasks.append(next_task)
    
    total_tasks = len(all_tasks)
    print(f"📋 Collected {total_tasks} tasks for processing\n")
    
    # 获取 benchmark 实例
    benchmark = await benchmark_manager.get(benchmark_name)
    
    # 确保浏览器已初始化
    if not benchmark._submitter_started:
        logger.info("| 🚀 Initializing browser...")
        await benchmark._submitter.initialize()
        benchmark._submitter_started = True
    
    # 初始化批量push管理器
    batch_push_manager.set_submitter(benchmark._submitter)
    batch_push_manager.batch_size = BATCH_SIZE
    batch_push_manager.timeout_seconds = PUSH_TIMEOUT
    batch_push_manager.idle_timeout = 600.0  # 空闲15秒后自动push
    batch_push_manager.set_expected_total(total_tasks)  # 设置预期任务总数（用于首轮）
    batch_push_manager.start()  # 启动超时定时器功能
    logger.info(f"| 📦 Batch push manager initialized (batch_size={BATCH_SIZE}, "
                f"hard_timeout={PUSH_TIMEOUT}s, idle_timeout=15s, expected_total={total_tasks})")
    
    # 创建信号量限制并发
    inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
    
    # 统计变量
    stats = {
        "completed": 0,
        "accepted": 0,
        "partial": 0,
        "failed": 0,
        "total_rounds": 0,
    }
    stats_lock = asyncio.Lock()
    
    start_time = time.time()
    
    async def process_task(task: Task):
        """处理单个任务"""
        result_task = await inference_with_reflection(
            task=task,
            save_dir=save_dir,
            semaphore=inference_semaphore,
            benchmark=benchmark,
        )
        
        async with stats_lock:
            stats["completed"] += 1
            score = result_task.score or 0.0
            
            if score >= 1.0:
                stats["accepted"] += 1
            elif score > 0:
                stats["partial"] += 1
            else:
                stats["failed"] += 1
            
            final_round = result_task.extra.get("final_round", MAX_REFLECTION_ROUNDS)
            stats["total_rounds"] += final_round
            
            # 打印进度
            print(f"\r🔄 Progress: {stats['completed']}/{total_tasks} | "
                  f"✅ Accepted: {stats['accepted']} | "
                  f"🔶 Partial: {stats['partial']} | "
                  f"❌ Failed: {stats['failed']}", end="", flush=True)
        
        # 保存最终结果
        await save_final_result(result_task, benchmark, save_dir)
        
        return result_task
    
    # 启动所有任务
    print(f"🚀 Starting self-reflection inference...\n")
    
    tasks = [process_task(t) for t in all_tasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 刷新批量push队列中剩余的文件并停止定时器
    remaining = batch_push_manager.get_queue_size()
    if remaining > 0:
        logger.info(f"| 📤 Flushing remaining {remaining} files in push queue...")
        await batch_push_manager.flush()
    
    # 停止批量push管理器（取消超时定时器）
    await batch_push_manager.stop()
    
    total_time = time.time() - start_time
    
    # 最终统计
    print(f"\n\n{'='*60}")
    print(f"🎉 All {total_tasks} tasks processed! (Self-Reflection Mode)")
    print(f"{'='*60}")
    print(f"🤖 Model: {TARGET_MODEL}")
    print(f"💻 Language: {TARGET_LANGUAGE}")
    print(f"🔄 Max Reflection Rounds: {MAX_REFLECTION_ROUNDS}")
    print(f"{'='*60}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"📊 Avg rounds per task: {stats['total_rounds'] / total_tasks:.2f}")
    print(f"{'='*60}")
    print(f"✅ Accepted (100%): {stats['accepted']}")
    print(f"🔶 Partial: {stats['partial']}")
    print(f"❌ Failed (0%): {stats['failed']}")
    print(f"📊 Accuracy: {stats['accepted'] / total_tasks * 100:.2f}%")
    print(f"{'='*60}")


async def save_final_result(task: Task, benchmark: Any, save_dir: str):
    """保存最终结果到jsonl"""
    from src.benchmark.types import Result
    
    best_eval = task.extra.get("best_eval_result", {})
    
    result = Result(
        task_id=task.task_id,
        prompt=task.input,
        prediction=best_eval.get("status", "Unknown"),
        answer="None",
        score=task.score or 0.0,
        metrics={
            "final_round": task.extra.get("final_round", MAX_REFLECTION_ROUNDS),
            "stop_reason": task.extra.get("stop_reason", "Unknown"),
            "best_score": task.extra.get("best_score", 0.0),
            "passed_cases": best_eval.get("passed_cases", 0),
            "total_cases": best_eval.get("total_cases", 0),
            "runtime": best_eval.get("runtime", 0),
            "runtime_beats": best_eval.get("runtime_beats", 0.0),
            "memory_usage": best_eval.get("memory_usage", 0),
            "memory_beats": best_eval.get("memory_beats", 0.0),
            "inference_time": task.extra.get("inference_time", 0.0),
        },
        extra=None,
        start_time=task.extra.get("inference_start_time", time.time()),
        end_time=time.time(),
        spend_time=task.extra.get("inference_time", 0.0)
    )
    
    await benchmark._submitter.save_result(result)


def summarize_results(benchmark_name: str):
    """总结结果"""
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
    round_counter = Counter()
    
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            total += 1
            if data.get("score") == 1.0:
                score_1_cnt += 1
            
            pred = data.get("prediction", "").strip()
            if pred:
                pred_counter[pred] += 1
            
            final_round = data.get("metrics", {}).get("final_round", 0)
            round_counter[final_round] += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Benchmark Summary: {benchmark_name}")
    print(f"📊 Model: {TARGET_MODEL}")
    print(f"📊 Language: {TARGET_LANGUAGE}")
    print("=" * 60)
    print(f"Total tasks: {total}")
    print(f"Accepted (score=1.0): {score_1_cnt}")
    print(f"Accuracy: {score_1_cnt / total * 100:.2f}%")
    print("=" * 60)
    print("Result Distribution:")
    for key in ["Accepted", "Wrong Answer", "Time Limit Exceeded", "Runtime Error", 
                "Memory Limit Exceeded", "Compile Error", "Timeout"]:
        print(f"  {key}: {pred_counter.get(key, 0)}")
    print("=" * 60)
    print("Rounds Distribution:")
    for r in sorted(round_counter.keys()):
        print(f"  Round {r}: {round_counter[r]} tasks")
    print("=" * 60)


async def main():
    global TARGET_MODEL, TARGET_LANGUAGE, BATCH_SIZE, MAX_REFLECTION_ROUNDS, OPTIMIZATION_THRESHOLD, JUDGE_MODE, PUSH_TIMEOUT
    
    parser = argparse.ArgumentParser(description='LeetCode Self-Reflection Agent')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), 
                        help="config file path")
    parser.add_argument("--benchmark", default="leetcode", help="benchmark name")
    parser.add_argument("--model", default=TARGET_MODEL, 
                        help=f"Model to use (default: {TARGET_MODEL})")
    parser.add_argument("--language", default=TARGET_LANGUAGE, choices=SUPPORTED_LANGUAGES,
                        help=f"Programming language (default: {TARGET_LANGUAGE})")
    parser.add_argument("--max-rounds", type=int, default=MAX_REFLECTION_ROUNDS,
                        help=f"Max reflection rounds (default: {MAX_REFLECTION_ROUNDS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for evaluation (default: {BATCH_SIZE})")
    parser.add_argument("--opt-threshold", type=float, default=OPTIMIZATION_THRESHOLD,
                        help=f"Performance optimization threshold in %% (default: {OPTIMIZATION_THRESHOLD}). "
                             f"If runtime_beats or memory_beats exceeds this, stop optimizing. "
                             f"Only used when --judge-mode=threshold.")
    parser.add_argument("--judge-mode", default=JUDGE_MODE, choices=["threshold", "llm"],
                        help=f"Judge mode for stopping optimization (default: {JUDGE_MODE}). "
                             f"'threshold': stop based on beats percentage threshold. "
                             f"'llm': use LLM as a Judge to decide when to stop.")
    parser.add_argument("--push-timeout", type=float, default=PUSH_TIMEOUT,
                        help=f"Push timeout in seconds (default: {PUSH_TIMEOUT}). "
                             f"If queue is not full after this time, auto push.")
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
    MAX_REFLECTION_ROUNDS = args.max_rounds
    OPTIMIZATION_THRESHOLD = args.opt_threshold
    JUDGE_MODE = args.judge_mode
    PUSH_TIMEOUT = args.push_timeout
    
    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    
    logger.info("| 🧠 Initializing model manager...")
    if hasattr(model_manager, 'initialize'):
        await model_manager.initialize()
    
    benchmark_name = args.benchmark
    logger.info(f"| 🛠️ Initializing benchmark manager for {benchmark_name}...")
    
    await benchmark_manager.initialize(benchmark_names=[benchmark_name])
    
    # 设置 benchmark 的语言
    benchmark = await benchmark_manager.get(benchmark_name)
    if benchmark:
        benchmark.language = TARGET_LANGUAGE
        benchmark.batch_size = BATCH_SIZE
        logger.info(f"| 🔧 Configured benchmark: language={TARGET_LANGUAGE}, batch_size={BATCH_SIZE}")
    
    await test_leetcode_with_reflection(benchmark_name)
    
    summarize_results(benchmark_name)
    
    print("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    print("| 🚪 Test completed")


if __name__ == "__main__":
    asyncio.run(main())
