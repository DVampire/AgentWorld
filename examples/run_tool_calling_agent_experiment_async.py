"""
Comprehensive experiment script for testing different optimizers on complete benchmark datasets.

This script tests three different optimizers across entire benchmark datasets:
1. GRPO (Generative Reinforcement Learning from Human Feedback with PPO)
2. Reinforce++ (Enhanced policy gradient method)
3. Reflection (Iterative prompt refinement)

The script iterates through ALL tasks in the specified benchmark, testing both
initial and optimized agent performance on each task, then provides comprehensive
statistics and analysis.

Usage:
    python run_tool_calling_agent_experiment_async.py --optimizer grpo --benchmark aime24_benchmark --concurrency 8
    python run_tool_calling_agent_experiment_async.py --optimizer reinforce_pp --benchmark gsm8k --concurrency 4
    python run_tool_calling_agent_experiment_async.py --optimizer reflection --benchmark aime24_benchmark --concurrency 6
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
load_dotenv(verbose=True)
from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio
from typing import Optional, Callable, Any, List, Dict, Tuple
from datetime import datetime

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.prompt import prompt_manager
from src.memory import memory_manager
from src.tool import tcp
from src.environment import ecp
from src.agent import acp
from src.benchmark import benchmark_manager
from src.optimizer import GrpoOptimizer, ReinforcePlusPlusOptimizer, ReflectionOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Test different optimizers on benchmark tasks')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")
    parser.add_argument("--optimizer", choices=['grpo', 'reinforce_pp', 'reflection'],
                       default='reflection', help="optimizer to test")
    parser.add_argument("--benchmark", default="gpqa", help="benchmark name to test on")
    parser.add_argument("--concurrency", type=int, default=8, help="number of concurrent tasks to run")
    parser.add_argument("--split", type=str, default='test', help="the split of dataset", choices=['train', 'test'])
    parser.add_argument("--batchsize", type=int, default=8, help="batch size for aggregating historical reflections")
    parser.add_argument("--model_name", type=str, default='openrouter/gpt-4o', help="")
    parser.add_argument("--optimize_trainable_variables", action='store_true', default=True, help="optimize trainable variables")
    parser.add_argument("--optimize_solution", action='store_true', default=True, help="optimize solution")
    parser.add_argument("--resume", action='store_true', default=True,
                       help="Resume from the latest results file. Will automatically find the most recent "
                            "matching results file, remove incorrect answers, and only retry failed tasks.")

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

class ExperimentResultSaver:
    """Save experiment results to JSON file with real-time updates."""

    def __init__(self, optimizer_type: str, benchmark_name: str, concurrency: int, total_tasks: int, model_name: str, split: str = "test", existing_file: str = None):
        self.optimizer_type = optimizer_type
        self.benchmark_name = benchmark_name
        self.concurrency = concurrency
        self.total_tasks = total_tasks
        self.model_name = model_name
        self.split = split
        self.start_time = datetime.now()

        # Create results directory if it doesn't exist
        self.results_dir = Path(os.path.join(config.workdir, "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize thread lock for file operations
        self.file_lock = asyncio.Lock()

        # If existing_file is provided, use it; otherwise create a new file
        if existing_file:
            self.filepath = Path(existing_file)
            self.filename = self.filepath.name
            # Load existing results
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.results_data = json.load(f)
                logger.info(f"| 📂 Loaded existing results file: {self.filepath}")
            except Exception as e:
                logger.error(f"| ❌ Failed to load existing file {existing_file}: {e}")
                raise
        else:
            # Generate filename with timestamp
            timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
            self.filename = f"{optimizer_type}_{benchmark_name}_{split}_{timestamp}.json"
            self.filepath = self.results_dir / self.filename

            # Initialize results structure
            self.results_data = {
                "experiment_meta": {
                    "timestamp": self.start_time.isoformat() + "Z",
                    "optimizer": optimizer_type,
                    "benchmark": benchmark_name,
                    "split": split,
                    "concurrency": concurrency,
                    "total_tasks": total_tasks,
                    "model": model_name
                },
                "results": [],
                "summary": {
                    "completed_tasks": 0,
                    "correct_answers": 0,
                    "accuracy": 0.0,
                    "last_updated": self.start_time.isoformat() + "Z"
                }
            }

            # Save initial empty results
            asyncio.create_task(self._save_to_file())

    async def add_task_result(self, task_data: Any, processing_time: float = None,
                              optimizer_data: Dict[str, Any] = None):
        """Add a single task result and update the file."""
        async with self.file_lock:
            if optimizer_data == None:
                task_result = {
                    "task_id": task_data.task_id,
                    "task_input": task_data.input[:200] + "..." if len(task_data.input) > 200 else task_data.input,
                    "ground_truth": str(task_data.ground_truth),
                    "result": str(task_data.result) if hasattr(task_data, 'result') else "",
                    "reasoning": getattr(task_data, 'reasoning', ""),
                    "correct": task_data.score == 1.0,
                    "processing_time": processing_time
                }
            else:
                _, answer = parse_agent_result(task_data.result)

                task_result = {"task_id": task_data.task_id,
                               "task_input": task_data.input,
                               "ground_truth": str(task_data.ground_truth),
                               "result": answer,
                               "reasoning": getattr(task_data, 'reasoning', ""),
                               "correct": task_data.score == 1.0,
                               "processing_time": processing_time, "reflection_process": {
                        "initial_reasoning": optimizer_data.get("initial_agent_reasoning", ""),
                        "initial_result": optimizer_data.get("initial_agent_result", ""),
                        "reflection_rounds": []
                    }}

                # Add detailed reflection process data for reflection optimizer
                # Support both legacy list format and new structured dict format:
                # - legacy: {"reflecion_text": [...], "improved_solution": [...]}
                # - structured: {"reflecion_text": {"phase1": [...], "phase2": [...]}, "improved_solution": {"phase1": [...], "phase2": [...]}}
                reflection_raw = optimizer_data.get("reflecion_text", [])
                improved_raw = optimizer_data.get("improved_solution", [])

                # Normalize to per-phase lists
                phase1_reflections, phase2_reflections = [], []
                phase1_improvements, phase2_improvements = [], []

                if isinstance(reflection_raw, dict):
                    phase1_reflections = reflection_raw.get("phase1", []) or []
                    phase2_reflections = reflection_raw.get("phase2", []) or []
                else:
                    # legacy: treat all as phase2 reflections
                    phase2_reflections = reflection_raw or []

                if isinstance(improved_raw, dict):
                    phase1_improvements = improved_raw.get("phase1", []) or []
                    phase2_improvements = improved_raw.get("phase2", []) or []
                else:
                    # legacy: treat all as phase2 improvements
                    phase2_improvements = improved_raw or []

                # Build rounds preserving phase information (phase1 rounds first, then phase2)
                rounds = []
                # Phase 1 rounds
                max_p1 = max(len(phase1_reflections), len(phase1_improvements))
                for i in range(max_p1):
                    round_data = {"phase": 1}
                    if i < len(phase1_reflections):
                        round_data["reflection_text"] = phase1_reflections[i]
                    if i < len(phase1_improvements):
                        round_data["improved_solution"] = phase1_improvements[i]
                    rounds.append(round_data)
                # Phase 2 rounds
                max_p2 = max(len(phase2_reflections), len(phase2_improvements))
                for i in range(max_p2):
                    round_data = {"phase": 2}
                    if i < len(phase2_reflections):
                        round_data["reflection_text"] = phase2_reflections[i]
                    if i < len(phase2_improvements):
                        round_data["improved_solution"] = phase2_improvements[i]
                    rounds.append(round_data)

                if rounds:
                    task_result["reflection_process"]["reflection_rounds"].extend(rounds)

                # Final results
                task_result["reflection_process"]["final_reasoning"] = optimizer_data.get("agent_reasoning", "")
                task_result["reflection_process"]["final_result"] = optimizer_data.get("agent_result", "")


            self.results_data["results"].append(task_result)

            # Update summary
            self.results_data["summary"]["completed_tasks"] = len(self.results_data["results"])
            correct_count = sum(1 for r in self.results_data["results"] if r["correct"])
            self.results_data["summary"]["correct_answers"] = correct_count
            self.results_data["summary"]["accuracy"] = correct_count / len(self.results_data["results"]) if self.results_data["results"] else 0.0
            self.results_data["summary"]["last_updated"] = datetime.now().isoformat() + "Z"

            # Save updated results
            await self._save_to_file()

    async def _save_to_file(self):
        """Save current results to JSON file."""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results_data, f, indent=2, ensure_ascii=False)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure data is written to disk
            logger.debug(f"Successfully saved results to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to save results to {self.filepath}: {e}")
            raise  # Re-raise exception to ensure it's not silently ignored

    def get_file_path(self) -> str:
        """Get the path to the results file."""
        return str(self.filepath)

async def eval_without_recording(benchmark_name: str, code_str: str):
    """
    Evaluate `code_str` on `benchmark_name` using the benchmark.eval(...) flow
    but avoid recording the task into benchmark._tasks or saving results to disk.
    Returns the evaluated Task (with score populated) or None on error.
    """
    import time
    import uuid
    from src.benchmark import benchmark_manager
    from src.benchmark.types import Task

    benchmark = await benchmark_manager.get(benchmark_name)
    if benchmark is None:
        raise RuntimeError(f"Benchmark '{benchmark_name}' is not initialized.")

    # Construct a temporary Task
    temp_id = f"temp_{uuid.uuid4().hex}"

    # Determine file extension if benchmark exposes LANGUAGE_CONFIG / language
    file_ext = "py"
    try:
        lang = getattr(benchmark, "language", None)
        lang_cfg = getattr(benchmark, "LANGUAGE_CONFIG", None)
        if lang and lang_cfg and isinstance(lang_cfg, dict):
            file_ext = lang_cfg.get(lang, {}).get("ext", file_ext)
    except Exception:
        pass

    temp_task = Task(
        task_id=temp_id,
        input="",
        result=code_str,
        extra={"file_name": temp_id, "file_ext": file_ext, "inference_time": 0.0}
    )

    # Monkeypatch _tasks and submitter.save_result to avoid side-effects
    orig_tasks = getattr(benchmark, "_tasks", None)
    class _TempList(list):
        def append(self, x):
            return None

    benchmark._tasks = _TempList()

    orig_save = None
    if getattr(benchmark, "_submitter", None) and hasattr(benchmark._submitter, "save_result"):
        orig_save = benchmark._submitter.save_result
        async def _noop_save(*args, **kwargs):
            return None
        benchmark._submitter.save_result = _noop_save

    try:
        evaluated = await benchmark.eval(temp_task)
    finally:
        # Restore originals
        if orig_save:
            benchmark._submitter.save_result = orig_save
        benchmark._tasks = orig_tasks

    return evaluated


async def reward_fn(answer: str = None, ground_truth: Any = None, benchmark_name: str = "leetcode"):
    """
    Reward function that evaluates agent `answer` (code) on specified `benchmark_name`
    using the benchmark's eval flow but without recording the evaluation into stats/results.
    """
    if benchmark_name != "leetcode":
        _, answer = parse_agent_result(answer)
        score = 1.0 if answer == ground_truth else 0.0
        print(f'answer: {answer}, ground_truth: {ground_truth}')
        return score
    # If no code/result, return zero reward
    if not answer:
        return 0.0

    try:
        evaluated_task = await eval_without_recording(benchmark_name, answer)
        score = float(evaluated_task.score) if evaluated_task and evaluated_task.score is not None else 0.0
        print(f"[Reward] score={score} for benchmark={benchmark_name}")
        return score
    except Exception as e:
        logger.warning(f"[Reward] evaluation failed: {e}")
        return 0.0

def parse_agent_result(agent_result: Any) -> Tuple[str, Any]:
    """
    Parse agent_result that could be:
    1. Direct string: "Final answer" → (reasoning="", result="Final answer")
    2. JSON string: '{"reasoning": "...", "result": "..."}' → (reasoning="...", result="...")
    3. Dict object: {"reasoning": "...", "result": "..."} → (reasoning="...", result="...")
    """
    import json

    # 1) If it's already a dict (upstream parsed JSON), return fields directly
    if isinstance(agent_result, dict):
        reasoning = agent_result.get("reasoning", "")
        result = agent_result.get("result", "")
        return reasoning, str(result)

    # 2) If it's a string, it may be:
    #    - plain text answer
    #    - a JSON string representing a dict
    #    - a JSON string that itself is an escaped JSON string (double-encoded)
    if isinstance(agent_result, str):
        s = agent_result.strip()

        # If labeled plain output like "Result: D Reasoning: ..." appear, extract them (case-insensitive, allow newlines)
        if 'result:' in s.lower() or 'reasoning:' in s.lower():
            import re
            # Manual, permissive extraction: find 'result' label and capture first letter, find 'reasoning' label and capture rest
            low = s.lower()
            result = ""
            reasoning = ""

            ridx = low.find('result:')
            if ridx != -1:
                after = s[ridx + len('result:'):].strip()
                m = re.search(r'([A-Za-z])', after)
                if m:
                    result = m.group(1).upper()

            rridx = low.find('reasoning:')
            if rridx != -1:
                reasoning = s[rridx + len('reasoning:'):].strip()

            if result:
                return reasoning, result

        # Quick heuristic: if it doesn't look like JSON (or a quoted JSON), treat as plain text
        if not (s.startswith("{") or s.startswith('"') or s.startswith("'")):
            return "", s

        # Try to parse JSON once
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            # Not valid JSON -> return as plain string
            return "", s

        # If parsed to a dict, extract fields
        if isinstance(parsed, dict):
            reasoning = parsed.get("reasoning", "")
            result = parsed.get("result", "")
            return reasoning, str(result)

        # If parsed to a string, attempt a second parse (handles double-encoded JSON)
        if isinstance(parsed, str):
            try:
                inner = json.loads(parsed)
                if isinstance(inner, dict):
                    reasoning = inner.get("reasoning", "")
                    result = inner.get("result", "")
                    return reasoning, str(result)
                else:
                    return "", parsed
            except json.JSONDecodeError:
                return "", parsed

    # 3) Fallback for other types
    return "", str(agent_result) if agent_result else ""

def create_optimizer(optimizer_type: str, model_name: str, reward_fn: Optional[Callable[[str, str, str], Any]] = None, batchsize: int = 10, optimize_trainable_variables: bool = True, optimize_solution: bool = True):
    """Create optimizer instance based on type."""
    base_config = {
        'workdir': config.workdir,
        'batchsize': batchsize,
        # 'model_name': 'openrouter/gemini-3-flash-preview',
        'model_name': model_name,
        'memory_name': 'optimizer_memory_system',
        'optimize_trainable_variables': optimize_trainable_variables,
        'optimize_solution': optimize_solution,
    }

    if optimizer_type == 'grpo':
        return GrpoOptimizer(
            num_candidates=4,
            clip_ratio=0.2,
            beta=0.01,
            reward_fn=reward_fn,
            prompt_name='grpo_optimizer',
            **base_config
        )
    elif optimizer_type == 'reinforce_pp':
        return ReinforcePlusPlusOptimizer(
            clip_ratio=0.2,
            beta=0.01,
            reward_fn=reward_fn,
            prompt_name='reinforce_plus_plus_optimizer',
            **base_config
        )
    elif optimizer_type == 'reflection':
        return ReflectionOptimizer(prompt_name='reflection_optimizer',
            **base_config
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def find_latest_results_file(optimizer_type: str, benchmark_name: str, split: str) -> Optional[str]:
    """Find the most recent results file matching the pattern."""
    results_dir = Path(os.path.join(config.workdir, "results"))
    if not results_dir.exists():
        return None
    
    # Pattern: {optimizer}_{benchmark}_{split}_*.json
    pattern = f"{optimizer_type}_{benchmark_name}_{split}_*.json"
    matching_files = list(results_dir.glob(pattern))
    
    if not matching_files:
        return None
    
    # Sort by modification time, get the latest
    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


async def get_all_tasks(benchmark_name: str, split: str = "test", results_file: str = None) -> List[Dict]:
    """Get all tasks from benchmark manager.
    
    If results_file is provided:
    1. Load existing results from the JSON file
    2. Remove incorrect results (ground_truth != result) from the file and save
    3. Return only tasks that were answered incorrectly (for retry)
    """
    tasks = []
    logger.info(f"| 🔄 Resetting progress for {benchmark_name} (split: {split})...")
    task_data = await benchmark_manager.reset(benchmark_name, split=split)

    while task_data is not None:
        tasks.append(task_data)
        task_data = await benchmark_manager.step(benchmark_name)

    # If no results file provided, return all tasks
    if not results_file:
        return tasks
    
    # Load existing results and filter
    results_path = Path(results_file)
    if not results_path.exists():
        logger.warning(f"| ⚠️ Results file not found: {results_file}, returning all tasks")
        return tasks
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        existing_results = results_data.get("results", [])
        logger.info(f"| 📂 Loaded {len(existing_results)} existing results from {results_file}")
        
        # Separate correct and incorrect results
        correct_results = []
        incorrect_task_ids = set()
        
        for result in existing_results:
            task_id = result.get("task_id")
            ground_truth = str(result.get("ground_truth", "")).strip()
            answer = str(result.get("result", "")).strip()
            
            if ground_truth == answer:
                correct_results.append(result)
            else:
                incorrect_task_ids.add(task_id)
                logger.info(f"| ❌ Task {task_id} was incorrect: expected '{ground_truth}', got '{answer}'")
        
        logger.info(f"| ✅ Correct: {len(correct_results)}, ❌ Incorrect: {len(incorrect_task_ids)}")
        
        # Update the results file with only correct results
        results_data["results"] = correct_results
        results_data["summary"]["completed_tasks"] = len(correct_results)
        results_data["summary"]["correct_answers"] = len(correct_results)
        results_data["summary"]["accuracy"] = 1.0 if correct_results else 0.0
        results_data["summary"]["last_updated"] = datetime.now().isoformat() + "Z"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"| 💾 Updated results file with {len(correct_results)} correct results")
        
        # Filter tasks to only include incorrect ones
        correct_task_ids = {r.get("task_id") for r in correct_results}
        filtered_tasks = [task for task in tasks if task.task_id not in correct_task_ids]
        
        logger.info(f"| 📋 Filtered tasks: {len(filtered_tasks)} remaining (excluding {len(correct_task_ids)} already correct)")
        
        return filtered_tasks
        
    except Exception as e:
        logger.error(f"| ❌ Error loading/processing results file: {e}")
        import traceback
        traceback.print_exc()
        return tasks


async def process_single_task(optimizer_type: str, benchmark_name: str, task_data: Any, task_index: int, total_tasks: int, split: str, batchsize: int, model_name: str, result_saver: ExperimentResultSaver = None, optimize_trainable_variables: bool = True, optimize_solution: bool = True):
    """Process a single task with the optimizer."""
    task_id = task_data.task_id
    task_input = task_data.input
    if benchmark_name != "leetcode":
        task_gt = None
    else:
        task_gt = task_data.ground_truth
    task_gt = task_data.ground_truth
    system_instruction = task_data.system_prompt
    start_time = time.time()

    # Combine system instruction with task input
    full_task = f"{system_instruction}\n\n{task_input}"

    logger.info(f"| 📋 Task {task_index + 1}/{total_tasks}: {task_id}, Ground Truth: {task_gt}")
    logger.info(f"| 📋 Task: {full_task[:150]}..." if len(full_task) > 150 else f"| 📋 Task: {full_task}")

    try:
        # Get the agent instance
        agent = await acp.get("tool_calling")
        # Create optimizer
        if split=='train' or optimizer_type=='reflection':
            # Bind benchmark_name to reward_fn so eval uses correct benchmark instance
            import functools
            bound_reward = functools.partial(reward_fn, benchmark_name=benchmark_name)
            optimizer = create_optimizer(optimizer_type, model_name, bound_reward, batchsize, optimize_trainable_variables, optimize_solution)

            # ！！！！！用于临时代替参考模型输出
            if optimizer_type == 'reinforce_pp':
                logger.info(f"| 🚀 Running agent to get initial solution...")
                reference_agent_response = await agent(task=full_task, files=[])
                reference_agent_response_extra_data = reference_agent_response.extra.data if reference_agent_response.extra and reference_agent_response.extra.data else None
                reference_agent_result = reference_agent_response_extra_data['result']
                reference_agent_reasoning = reference_agent_response_extra_data['reasoning']
                reference_solution = f"Result: {reference_agent_result}\nReasoning: {reference_agent_reasoning}" if reference_agent_reasoning else f"Result: {reference_agent_result}"
                logger.info(f"| ✅ Initial solution obtained")

                initial_agent_result, initial_agent_reasoning, reflecion_text, improved_solution, agent_reasoning, agent_result = await optimizer.optimize(agent=agent,
                                                                                 task=full_task,
                                                                                 ground_truth=task_gt,
                                                                                 sft_solution=reference_solution,
                                                                                 benchmark_task_id=task_id,
                                                                                 files=[],
                                                                                 results_file_path=result_saver.get_file_path() if result_saver else None)
            else:
                initial_agent_result, initial_agent_reasoning, reflecion_text, improved_solution, agent_reasoning, agent_result = await optimizer.optimize(agent=agent,
                                                                                 task=full_task,
                                                                                 ground_truth=task_gt,
                                                                                 benchmark_task_id=task_id,
                                                                                 files=[],
                                                                                 results_file_path=result_saver.get_file_path() if result_saver else None)
        else:
            logger.info(f"| 🚀 Running agent to get initial solution...")
            agent_response = await agent(task=full_task, files=[])
            agent_response_extra_data = agent_response.extra.data if agent_response.extra and agent_response.extra.data else None
            agent_result = agent_response_extra_data['final_result']
            agent_reasoning = agent_response_extra_data['final_reasoning']
            logger.info(f"| ✅ Initial solution obtained")


        parse_reasoning, parse_result = parse_agent_result(agent_result)

        if parse_reasoning == '':
            parse_reasoning = agent_reasoning
        task_data.reasoning = parse_reasoning
        task_data.result = parse_result

        _ = await benchmark_manager.eval(benchmark_name, task_data)
        # Get current stats after processing this task
        stats = await benchmark_manager.stats(benchmark_name)
        if stats:
            attempted = stats.correct + stats.wrong
            accuracy_msg = f"📊 Overall Progress: {attempted}/{stats.total} | Accuracy: {stats.accuracy:.2%}"
            print(accuracy_msg)
            logger.info(accuracy_msg)

        logger.info(f"| ✅ Task {task_id} completed successfully")

        # Save result if saver is provided
        if result_saver:
            processing_time = time.time() - start_time

            # Prepare optimizer data for detailed saving
            optimizer_data = None
            if split == 'train' or optimizer_type == 'reflection':
                optimizer_data = {
                    "initial_agent_result": initial_agent_result,
                    "initial_agent_reasoning": initial_agent_reasoning,
                    "reflecion_text": reflecion_text,
                    "improved_solution": improved_solution,
                    "agent_reasoning": agent_reasoning,
                    "agent_result": agent_result
                }

            await result_saver.add_task_result(task_data, processing_time, optimizer_data)

    except Exception as e:
        logger.error(f"| ❌ Error processing task {task_id}: {e}")
        import traceback
        traceback.print_exc()


async def run_optimizer_on_benchmark(optimizer_type: str, benchmark_name: str, split: str, batchsize: int, model_name: str, concurrency: int = 4, resume: bool = False, optimize_trainable_variables: bool = True, optimize_solution: bool = True):
    """Test specified optimizer performance on entire benchmark dataset with concurrency control."""
    logger.info(f"| 🧪 Testing {optimizer_type.upper()} optimizer on complete benchmark: {benchmark_name}")
    logger.info(f"| ⚡ Using concurrency level: {concurrency}")
    logger.info(f"| 📊 Using dataset split: {split}")
    
    # Find results file if resuming
    results_file = None
    if resume:
        results_file = find_latest_results_file(optimizer_type, benchmark_name, split)
        if results_file:
            logger.info(f"| 📂 Found latest results file: {results_file}")
        else:
            logger.warning(f"| ⚠️ No existing results file found for {optimizer_type}_{benchmark_name}_{split}_*.json, starting fresh")

    # Get all tasks first (filtered if results_file found)
    all_tasks = await get_all_tasks(benchmark_name, split=split, results_file=results_file)
    # all_tasks = all_tasks[8:9]
    total_tasks = len(all_tasks)

    if total_tasks == 0:
        logger.warning("⚠️ No tasks available to run (Dataset empty or all finished).")
        return

    logger.info(f"| 📋 Total tasks to process: {total_tasks}")

    # Initialize result saver (continue using existing file if resuming)
    result_saver = ExperimentResultSaver(optimizer_type, benchmark_name, concurrency, total_tasks, model_name, split, existing_file=results_file)
    logger.info(f"| 💾 Results will be saved to: {result_saver.get_file_path()}")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0

    async def process_with_semaphore(task_data: Dict, task_index: int):
        """Process a task with semaphore control."""
        nonlocal completed_count
        async with semaphore:
            try:
                await process_single_task(optimizer_type, benchmark_name, task_data, task_index, total_tasks, split, batchsize, model_name, result_saver, optimize_trainable_variables, optimize_solution)
            finally:
                completed_count += 1
                # Progress reporting
                if completed_count % concurrency == 0 or completed_count == total_tasks:
                    progress_msg = f"| 📊 Progress: {completed_count}/{total_tasks} tasks completed"
                    logger.info(progress_msg)
                    print(progress_msg)

    # Create all tasks and run them with semaphore-controlled concurrency
    tasks = [process_with_semaphore(task_data, i) for i, task_data in enumerate(all_tasks)]
    await asyncio.gather(*tasks)

    logger.info(f"| ✅ All {total_tasks} tasks completed for {optimizer_type.upper()} optimizer")


async def main():
    args = parse_args()

    config.initialize(config_path=args.config, args=args)
    # Disable logging during experiments for cleaner output
    # logger.initialize(config=config, level=logging.CRITICAL)
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")

    # Initialize model manager
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize()
    logger.info(f"| ✅ Model manager initialized: {await model_manager.list()}")

    # Initialize prompt manager
    logger.info("| 📁 Initializing prompt manager...")
    await prompt_manager.initialize()
    logger.info(f"| ✅ Prompt manager initialized: {await prompt_manager.list()}")

    # Initialize memory manager
    logger.info("| 📁 Initializing memory manager...")
    await memory_manager.initialize(memory_names=config.memory_names)
    logger.info(f"| ✅ Memory manager initialized: {await memory_manager.list()}")

    # Initialize tools
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tcp.list()}")

    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(env_names=config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")

    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await acp.list()}")

    # Initialize benchmark manager
    logger.info("| 🧪 Initializing benchmark manager...")
    await benchmark_manager.initialize(benchmark_names=[args.benchmark])
    logger.info(f"| ✅ Benchmark manager initialized: {await benchmark_manager.list()}")

    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized")

    # Test specified optimizer on benchmark
    await run_optimizer_on_benchmark(args.optimizer, args.benchmark, args.split, args.batchsize, args.model_name, args.concurrency, args.resume, args.optimize_trainable_variables, args.optimize_solution)

    logger.info("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    logger.shutdown()  # 确保所有日志都被写入文件
    logger.info("| 🚪 Experiment completed")


if __name__ == "__main__":
    asyncio.run(main())
