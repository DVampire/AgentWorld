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

async def reward_fn(answer: str = None, ground_truth: Any = None):
    score = 1.0 if answer == ground_truth else 0.0
    print(f'answer: {answer}, ground_truth: {ground_truth}')
    return score

def create_optimizer(optimizer_type: str, reward_fn: Optional[Callable[[str, str, str], Any]] = None):
    """Create optimizer instance based on type."""
    base_config = {
        'workdir': config.workdir,
        'model_name': 'openrouter/gemini-3-flash-preview',
        'memory_name': 'optimizer_memory_system',
        'optimize_trainable_variables': True,
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


async def run_optimizer_on_benchmark(optimizer_type: str, benchmark_name: str):
    """Test specified optimizer performance on entire benchmark dataset."""
    logger.info(f"| 🧪 Testing {optimizer_type.upper()} optimizer on complete benchmark: {benchmark_name}")

    # Get the agent instance
    agent = await acp.get("tool_calling")

    # Create fresh optimizer for each task (to avoid state carryover)
    logger.info(f"| 🤖 Starting {optimizer_type.upper()} optimization...")
    optimizer = create_optimizer(optimizer_type, reward_fn)

    # Statistics tracking
    total_tasks = 0

    # Reset benchmark progress
    logger.info(f"| 🔄 Resetting progress for {benchmark_name}...")
    task_data = await benchmark_manager.reset(benchmark_name)

    while task_data is not None:
        total_tasks += 1
        task_id = task_data.task_id
        task_input = task_data.input
        task_gt = task_data.ground_truth
        system_instruction = task_data.system_prompt

        # Combine system instruction with task input
        full_task = f"{system_instruction}\n\n{task_input}"

        logger.info(f"\n📋 Task {total_tasks}: {task_id}")
        print(f"\n📋 Task {total_tasks}: {task_id}")
        logger.info(f"📋 Task: {full_task[:150]}..." if len(full_task) > 150 else f"📋 Task: {full_task}")

        # ！！！！！用于临时代替参考模型输出
        if optimizer_type == 'reinforce_pp':
            logger.info(f"| 🚀 Running agent to get initial solution...")
            reference_agent_response = await agent(task=full_task, files=[])
            reference_agent_response_extra_data = reference_agent_response.extra.data if reference_agent_response.extra and reference_agent_response.extra.data else None
            reference_agent_result = reference_agent_response_extra_data['result']
            reference_agent_reasoning = reference_agent_response_extra_data['reasoning']
            reference_solution = f"Result: {reference_agent_result}\nReasoning: {reference_agent_reasoning}" if reference_agent_reasoning else f"Result: {reference_agent_result}"
            logger.info(f"| ✅ Initial solution obtained")

            task_data.reasoning, task_data.result = await optimizer.optimize(agent=agent,
                                                                             task=full_task,
                                                                             ground_truth=task_gt,
                                                                             sft_solution=reference_solution,
                                                                             benchmark_task_id=task_id,
                                                                             files=[])
        else:
            task_data.reasoning, task_data.result = await optimizer.optimize(agent=agent,
                                                                             task=full_task,
                                                                             ground_truth=task_gt,
                                                                             benchmark_task_id=task_id,
                                                                             files=[])
        _ = await benchmark_manager.eval(benchmark_name, task_data)
        stats = await benchmark_manager.stats(benchmark_name)
        if stats:
            attempted = stats.correct + stats.wrong
            print(f"📊 Overall Progress: {attempted}/{stats.total} | Accuracy: {stats.accuracy:.2%}")

        # Progress indicator
        if total_tasks % 5 == 0:
            logger.info(f"| 📊 Progress: {total_tasks} tasks completed")

        # Get next task
        task_data = await benchmark_manager.step(benchmark_name)



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

