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
    python run_tool_calling_agent_experiment.py --optimizer grpo --benchmark aime24_benchmark
    python run_tool_calling_agent_experiment.py --optimizer reinforce_pp --benchmark gsm8k
    python run_tool_calling_agent_experiment.py --optimizer reflection --benchmark aime24_benchmark
"""

import os
import sys
import logging
from dotenv import load_dotenv
load_dotenv(verbose=True)
from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio
from typing import Optional, Callable, Any

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
    parser.add_argument("--benchmark", default="aime24", help="benchmark name to test on")

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

async def reward_fn(benchmark_name: str, prediction: Optional[str] = None, task_id: Optional[str] = None):
    score = await benchmark_manager.eval_task(name=benchmark_name, prediction=prediction, task_id=task_id)
    return score

def create_optimizer(optimizer_type: str, reward_fn: Optional[Callable[[str, str, str], Any]] = None, benchmark_name: str = None):
    """Create optimizer instance based on type."""
    base_config = {
        'workdir': config.workdir,
        'prompt_name': 'reflection_optimizer',
        'model_name': 'openrouter/gemini-3-flash-preview',
        'memory_name': 'optimizer_memory_system',
        'optimize_trainable_variables': 'True',
        'optimize_solution': 'True',
        'benchmark_name': benchmark_name,

    }

    if optimizer_type == 'grpo':
        return GrpoOptimizer(
            num_candidates=4,
            clip_ratio=0.2,
            beta=0.01,
            reward_fn=reward_fn,
            **base_config
        )
    elif optimizer_type == 'reinforce_pp':
        return ReinforcePlusPlusOptimizer(
            clip_ratio=0.2,
            beta=0.01,
            reward_fn=reward_fn,
            **base_config
        )
    elif optimizer_type == 'reflection':
        return ReflectionOptimizer(**base_config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


async def run_optimizer_on_benchmark(optimizer_type: str, benchmark_name: str):
    """Test specified optimizer performance on entire benchmark dataset."""
    logger.info(f"| 🧪 Testing {optimizer_type.upper()} optimizer on complete benchmark: {benchmark_name}")

    # Get the agent instance
    agent = await acp.get("tool_calling")

    # Create fresh optimizer for each task (to avoid state carryover)
    logger.info(f"| 🤖 Starting {optimizer_type.upper()} optimization...")
    # Set benchmark_name as attribute on reward_fn for sync calls
    optimizer = create_optimizer(optimizer_type, reward_fn, benchmark_name)

    # Statistics tracking
    total_tasks = 0
    initial_scores = []
    optimized_scores = []
    improvements = []
    task_results = []

    # Reset benchmark progress
    logger.info(f"| 🔄 Resetting progress for {benchmark_name}...")
    task_data = await benchmark_manager.reset(benchmark_name)

    while task_data is not None:
        total_tasks += 1
        task_id = task_data.get("task_id", f"task_{total_tasks}")
        task_input = task_data.get("input", "")
        system_instruction = task_data.get("system_prompt", "")
        ground_truth = task_data.get("ground_truth", "")

        # Combine system instruction with task input
        full_task = f"{system_instruction}\n\n{task_input}"

        logger.info(f"\n📋 Task {total_tasks}: {task_id}")
        print(f"\n📋 Task {total_tasks}: {task_id}")
        logger.info(f"📋 Task: {full_task[:150]}..." if len(full_task) > 150 else f"📋 Task: {full_task}")

        # # Test initial performance before optimization
        # logger.info("| 🚀 Testing initial performance...")
        # initial_response = await agent(task=full_task, files=[])
        # initial_result = initial_response.extra.data.get('final_result', '') if initial_response.extra and initial_response.extra.data else ''
        #
        # # Evaluate initial answer
        # initial_score = await benchmark_manager.eval_task(name=benchmark_name, prediction=initial_result, task_id=task_id)
        # initial_scores.append(initial_score)
        # logger.info(f"| 🎯 Initial Score: {initial_score}")
        # print(f"| 🎯 Initial Score: {initial_score}")

        await optimizer.optimize(agent=agent,
                                 task=full_task,
                                 benchmark_task_id=task_id,
                                 files=[])

        # Test optimized performance
        logger.info("| 🚀 Testing optimized performance...")
        optimized_response = await agent(task=full_task, files=[])
        optimized_result = optimized_response.extra.data.get('final_result', '') if optimized_response.extra and optimized_response.extra.data else ''

        # Evaluate optimized answer
        optimized_score = await benchmark_manager.eval_task(name=benchmark_name, prediction=optimized_result, task_id=task_id)
        optimized_scores.append(optimized_score)
        logger.info(f"| 🎯 Optimized Score: {optimized_score}")
        print(f"| 🎯 Optimized Score: {optimized_score}")

        # # Calculate improvement
        # improvement = optimized_score - initial_score
        # improvements.append(improvement)

        # Store detailed results
        task_results.append({
            'task_id': task_id,
            # 'initial_score': initial_score,
            'optimized_score': optimized_score,
            # 'improvement': improvement,
            'ground_truth': ground_truth,
            # 'initial_result': initial_result[:200] if len(initial_result) > 200 else initial_result,
            'optimized_result': optimized_result[:200] if len(optimized_result) > 200 else optimized_result
        })

        # Progress indicator
        if total_tasks % 5 == 0:
            logger.info(f"| 📊 Progress: {total_tasks} tasks completed")


        # Get next task
        task_data = await benchmark_manager.step(benchmark_name)

    # Calculate overall statistics
    avg_initial = sum(initial_scores) / len(initial_scores)
    avg_optimized = sum(optimized_scores) / len(optimized_scores)
    avg_improvement = sum(improvements) / len(improvements)

    improved_count = sum(1 for imp in improvements if imp > 0)
    neutral_count = sum(1 for imp in improvements if imp == 0)
    regressed_count = sum(1 for imp in improvements if imp < 0)

    logger.info("\n🎯 === FINAL RESULTS ===")
    logger.info(f"| 📊 Total Tasks Tested: {total_tasks}")
    logger.info(f"| 📊 Average Initial Score: {avg_initial:.3f}")
    logger.info(f"| 📊 Average Optimized Score: {avg_optimized:.3f}")
    logger.info(f"| 📊 Average Improvement: {avg_improvement:+.3f}")
    logger.info("\n📈 Performance Distribution:")
    logger.info(f"|  ✅ Improved: {improved_count} tasks ({improved_count/total_tasks*100:.1f}%)")
    logger.info(f"|  🤝 Neutral: {neutral_count} tasks ({neutral_count/total_tasks*100:.1f}%)")
    logger.info(f"|  📉 Regressed: {regressed_count} tasks ({regressed_count/total_tasks*100:.1f}%)")

    if avg_improvement > 0:
        logger.info(f"\n🎉 OVERALL SUCCESS: {optimizer_type.upper()} improved average performance by {avg_improvement:.3f}")
    elif avg_improvement == 0:
        logger.info(f"\n🤝 OVERALL NEUTRAL: {optimizer_type.upper()} maintained average performance")
    else:
        logger.info(f"\n📉 OVERALL REGRESSION: {optimizer_type.upper()} decreased average performance by {abs(avg_improvement):.3f}")

    # Print detailed results for first few tasks
    logger.info("\n📋 Sample Task Results:")
    for i, result in enumerate(task_results[:3]):  # Show first 3 tasks
        logger.info(f"  Task {result['task_id']}: {result['initial_score']} → {result['optimized_score']} (Δ{result['improvement']:+.3f})")


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
    await run_optimizer_on_benchmark(args.optimizer, args.benchmark)

    logger.info("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    logger.info("| 🚪 Experiment completed")


if __name__ == "__main__":
    asyncio.run(main())

