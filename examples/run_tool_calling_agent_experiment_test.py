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
from typing import Optional, Callable, Any, List, Dict

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
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"),
                        help="config file path")
    parser.add_argument("--optimizer", choices=['grpo', 'reinforce_pp', 'reflection'],
                        default='reinforce_pp', help="optimizer to test")
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


def create_optimizer(optimizer_type: str, reward_fn: Optional[Callable[[str, str, str], Any]] = None,
                     benchmark_name: str = None):
    """Create optimizer instance based on type."""
    base_config = {
        'workdir': config.workdir,
        'model_name': 'openrouter/gemini-3-flash-preview',
        'memory_name': 'optimizer_memory_system',
        'benchmark_name': benchmark_name,
        'optimize_trainable_variables': False,
        'optimize_solution': True
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
            prompt_name='reinforce_plus_plus_optimizer',
            **base_config
        )
    elif optimizer_type == 'reflection':
        return ReflectionOptimizer(prompt_name='reflection_optimizer',
                                   **base_config
                                   )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


async def collect_reference_solutions(benchmark_name: str) -> List[Dict[str, Any]]:
    """Collect baseline solutions from original agent for all tasks in benchmark."""
    logger.info(f"| 📊 Collecting reference solutions for {benchmark_name}...")

    # Create baseline agent
    reference_agent = await acp.get("tool_calling")

    # Reset benchmark progress
    task_data = await benchmark_manager.reset(benchmark_name)

    reference_solutions = []
    task_count = 0

    while task_data is not None:
        task_count += 1
        task_id = task_data.get("task_id", f"task_{task_count}")
        task_input = task_data.get("input", "")
        system_instruction = task_data.get("system_prompt", "")

        # Combine system instruction with task input
        full_task = f"{system_instruction}\n\n{task_input}"

        logger.info(f"| 📋 Collecting reference for task {task_count}: {task_id}")

        try:
            # Run baseline agent to get solution
            response = await reference_agent(task=full_task, files=[])

            # Extract solution from response
            reference_result = response.extra.data.get('final_result',
                                                       '') if response.extra and response.extra.data else ''
            reference_reasoning = response.extra.data.get('final_reasoning',
                                                          '') if response.extra and response.extra.data else ''
            reference_solution = f"Result: {reference_result}\nReasoning: {reference_reasoning}" if reference_reasoning else f"Result: {reference_result}"

            reference_solutions.append({
                'task_id': task_id,
                'task_data': task_data,
                'reference_solution': reference_solution,
                'full_task': full_task
            })

        except Exception as e:
            logger.error(f"| ❌ Failed to collect reference for task {task_id}: {e}")
            # Still add to list with empty solution
            reference_solutions.append({
                'task_id': task_id,
                'task_data': task_data,
                'reference_solution': "",
                'full_task': full_task
            })

        # Get next task
        task_data = await benchmark_manager.step(benchmark_name)

    logger.info(f"| ✅ Collected {len(reference_solutions)} reference solutions")
    return reference_solutions


async def run_optimizer_on_benchmark(optimizer_type: str, benchmark_name: str,
                                     reference_solutions: List[Dict[str, Any]]):
    """Test specified optimizer performance on entire benchmark dataset using pre-collected baseline solutions."""
    logger.info(f"| 🧪 Testing {optimizer_type.upper()} optimizer on complete benchmark: {benchmark_name}")

    # Get the agent instance
    agent = await acp.get("tool_calling")

    # Create fresh optimizer for each task (to avoid state carryover)
    logger.info(f"| 🤖 Starting {optimizer_type.upper()} optimization...")
    optimizer = create_optimizer(optimizer_type, reward_fn, benchmark_name)

    # Statistics tracking
    total_tasks = 0

    # Process each baseline solution
    for reference_data in reference_solutions:
        total_tasks += 1
        task_id = reference_data['task_id']
        reference_solution = reference_data['reference_solution']
        full_task = reference_data['full_task']

        logger.info(f"\n📋 Task {total_tasks}: {task_id}")
        print(f"\n📋 Task {total_tasks}: {task_id}")
        logger.info(f"📋 Task: {full_task[:150]}..." if len(full_task) > 150 else f"📋 Task: {full_task}")

        await optimizer.optimize(agent=agent,
                                 task=full_task,
                                 benchmark_task_id=task_id,
                                 sft_solution=reference_solution,  # Use pre-collected reference solution
                                 files=[])

        # Progress indicator
        if total_tasks % 5 == 0:
            logger.info(f"| 📊 Progress: {total_tasks} tasks completed")


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

    # Step 1: Collect baseline solutions from original agent
    logger.info(f"| 📊 Phase 1: Collecting baseline solutions...")
    reference_solutions = await collect_reference_solutions(args.benchmark)

    # Step 2: Reset component states to clean state
    logger.info(f"| 🔄 Phase 2: Resetting component states...")
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

    # Step 3: Run optimizer on benchmark using baseline solutions
    logger.info(f"| 🤖 Phase 3: Running {args.optimizer.upper()} optimization...")
    await run_optimizer_on_benchmark(args.optimizer, args.benchmark, reference_solutions)

    logger.info("| 🧹 Cleaning up...")
    await benchmark_manager.cleanup()
    logger.info("| 🚪 Experiment completed")


if __name__ == "__main__":
    asyncio.run(main())

