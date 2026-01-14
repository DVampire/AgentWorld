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
import logging
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


class ExperimentResultSaver:
    """Save experiment results to JSON file with real-time updates."""

    def __init__(self, optimizer_type: str, benchmark_name: str, concurrency: int, total_tasks: int, model_name: str):
        self.optimizer_type = optimizer_type
        self.benchmark_name = benchmark_name
        self.concurrency = concurrency
        self.total_tasks = total_tasks
        self.model_name = model_name
        self.start_time = datetime.now()

        # Create results directory if it doesn't exist
        self.results_dir = Path("examples/results")
        self.results_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"{optimizer_type}_{benchmark_name}_{timestamp}.json"
        self.filepath = self.results_dir / self.filename

        # Initialize results structure
        self.results_data = {
            "experiment_meta": {
                "timestamp": self.start_time.isoformat() + "Z",
                "optimizer": optimizer_type,
                "benchmark": benchmark_name,
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
        self._save_to_file()

    def add_task_result(self, task_data: Any, processing_time: float = None):
        """Add a single task result and update the file."""
        task_result = {
            "task_id": task_data.task_id,
            "task_input": task_data.input[:200] + "..." if len(task_data.input) > 200 else task_data.input,
            "ground_truth": str(task_data.ground_truth),
            "result": str(task_data.result) if hasattr(task_data, 'result') else "",
            "reasoning": getattr(task_data, 'reasoning', ""),
            "correct": getattr(task_data, 'result', "") == str(task_data.ground_truth),
            "processing_time": processing_time
        }

        self.results_data["results"].append(task_result)

        # Update summary
        self.results_data["summary"]["completed_tasks"] = len(self.results_data["results"])
        correct_count = sum(1 for r in self.results_data["results"] if r["correct"])
        self.results_data["summary"]["correct_answers"] = correct_count
        self.results_data["summary"]["accuracy"] = correct_count / len(self.results_data["results"]) if self.results_data["results"] else 0.0
        self.results_data["summary"]["last_updated"] = datetime.now().isoformat() + "Z"

        # Save updated results
        self._save_to_file()

    def _save_to_file(self):
        """Save current results to JSON file."""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save results to {self.filepath}: {e}")

    def get_file_path(self) -> str:
        """Get the path to the results file."""
        return str(self.filepath)


def parse_args():
    parser = argparse.ArgumentParser(description='Test different optimizers on benchmark tasks')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")
    parser.add_argument("--optimizer", choices=['grpo', 'reinforce_pp', 'reflection'],
                       default='grpo', help="optimizer to test")
    parser.add_argument("--benchmark", default="aime24", help="benchmark name to test on")
    parser.add_argument("--concurrency", type=int, default=4, help="number of concurrent tasks to run")

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

async def reward_fn(answer: str = None, ground_truth: Any = None):
    _, answer = parse_agent_result(answer)
    score = 1.0 if answer == ground_truth else 0.0
    print(f'answer: {answer}, ground_truth: {ground_truth}')
    return score

def parse_agent_result(agent_result: Any) -> Tuple[str, Any]:
    """
    Parse agent_result that could be:
    1. Direct string: "Final answer" → (reasoning="", result="Final answer")
    2. JSON string: '{"reasoning": "...", "result": "..."}' → (reasoning="...", result="...")
    """
    import json

    # Case 1: Direct string result
    if isinstance(agent_result, str) and not agent_result.strip().startswith('{'):
        return "", agent_result.strip()

    # Case 2: JSON string with reasoning and result
    if isinstance(agent_result, str):
        try:
            parsed = json.loads(agent_result.strip())
            if isinstance(parsed, dict):
                reasoning = parsed.get("reasoning", "")
                result = parsed.get("result", "")
                return reasoning, str(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as direct string
            return "", agent_result.strip()

    # Fallback for other types
    return "", str(agent_result) if agent_result else ""

def create_optimizer(optimizer_type: str, reward_fn: Optional[Callable[[str, str, str], Any]] = None):
    """Create optimizer instance based on type."""
    base_config = {
        'workdir': config.workdir,
        'model_name': 'openrouter/gemini-3-flash-preview',
        'memory_name': 'optimizer_memory_system',
        'optimize_trainable_variables': False,
        'optimize_solution': True
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


async def get_all_tasks(benchmark_name: str) -> List[Dict]:
    """Get all tasks from benchmark manager."""
    tasks = []
    logger.info(f"| 🔄 Resetting progress for {benchmark_name}...")
    task_data = await benchmark_manager.reset(benchmark_name)

    while task_data is not None:
        tasks.append(task_data)
        task_data = await benchmark_manager.step(benchmark_name)

    return tasks


async def process_single_task(optimizer_type: str, benchmark_name: str, task_data: Any, task_index: int, total_tasks: int, result_saver: ExperimentResultSaver = None):
    """Process a single task with the optimizer."""
    task_id = task_data.task_id
    task_input = task_data.input
    task_gt = task_data.ground_truth
    system_instruction = task_data.system_prompt
    start_time = time.time()

    # Combine system instruction with task input
    full_task = f"{system_instruction}\n\n{task_input}"

    logger.info(f"\n📋 Task {task_index + 1}/{total_tasks}: {task_id}")
    print(f"\n📋 Task {task_index + 1}/{total_tasks}: {task_id}")
    logger.info(f"📋 Task: {full_task[:150]}..." if len(full_task) > 150 else f"📋 Task: {full_task}")

    try:
        # Get the agent instance
        agent = await acp.get("tool_calling")
        # Create optimizer
        optimizer = create_optimizer(optimizer_type, reward_fn)

        # ！！！！！用于临时代替参考模型输出
        if optimizer_type == 'reinforce_pp':
            logger.info(f"| 🚀 Running agent to get initial solution...")
            reference_agent_response = await agent(task=full_task, files=[])
            reference_agent_response_extra_data = reference_agent_response.extra.data if reference_agent_response.extra and reference_agent_response.extra.data else None
            reference_agent_result = reference_agent_response_extra_data['final_result']
            reference_agent_reasoning = reference_agent_response_extra_data['final_reasoning']
            reference_solution = f"Result: {reference_agent_result}\nReasoning: {reference_agent_reasoning}" if reference_agent_reasoning else f"Result: {reference_agent_result}"
            logger.info(f"| ✅ Initial solution obtained")

            agent_reasoning, agent_result = await optimizer.optimize(agent=agent,
                                                                             task=full_task,
                                                                             ground_truth=task_gt,
                                                                             sft_solution=reference_solution,
                                                                             benchmark_task_id=task_id,
                                                                             files=[])
        else:
            agent_reasoning, agent_result = await optimizer.optimize(agent=agent,
                                                                             task=full_task,
                                                                             ground_truth=task_gt,
                                                                             benchmark_task_id=task_id,
                                                                             files=[])

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
            result_saver.add_task_result(task_data, processing_time)

    except Exception as e:
        logger.error(f"| ❌ Error processing task {task_id}: {e}")
        import traceback
        traceback.print_exc()


async def run_optimizer_on_benchmark(optimizer_type: str, benchmark_name: str, concurrency: int = 4):
    """Test specified optimizer performance on entire benchmark dataset with concurrency control."""
    logger.info(f"| 🧪 Testing {optimizer_type.upper()} optimizer on complete benchmark: {benchmark_name}")
    logger.info(f"| ⚡ Using concurrency level: {concurrency}")

    # Get all tasks first
    all_tasks = await get_all_tasks(benchmark_name)
    total_tasks = len(all_tasks)

    if total_tasks == 0:
        logger.warning("⚠️ No tasks available to run (Dataset empty or all finished).")
        return

    logger.info(f"| 📋 Total tasks to process: {total_tasks}")

    # Initialize result saver
    model_name = 'openrouter/gemini-3-flash-preview'
    result_saver = ExperimentResultSaver(optimizer_type, benchmark_name, concurrency, total_tasks, model_name)
    logger.info(f"| 💾 Results will be saved to: {result_saver.get_file_path()}")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0

    async def process_with_semaphore(task_data: Dict, task_index: int):
        """Process a task with semaphore control."""
        nonlocal completed_count
        async with semaphore:
            try:
                await process_single_task(optimizer_type, benchmark_name, task_data, task_index, total_tasks, result_saver)
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
    await run_optimizer_on_benchmark(args.optimizer, args.benchmark, args.concurrency)

    logger.info("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    logger.info("| 🚪 Experiment completed")


if __name__ == "__main__":
    asyncio.run(main())
