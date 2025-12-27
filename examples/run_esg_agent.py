import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio
import pandas as pd
from typing import List
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
from src.transformation import transformation
from src.data import ESGDataset
from src.utils import assemble_project_path

def parse_args():
    parser = argparse.ArgumentParser(description='Run ESG Agent')
    parser.add_argument("--config", default=os.path.join(root, "configs", "esg_agent.py"), help="config file path")
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


async def append_answer(entry: dict, jsonl_file: str) -> None:
    """Append an answer entry to a JSONL file.
    
    File append operations are atomic at the OS level for single-line writes,
    so no explicit locking is needed. Using asyncio.to_thread to avoid blocking event loop.
    """
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use asyncio.to_thread to run file I/O in a thread pool to avoid blocking
    def _write_file():
        with open(jsonl_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    await asyncio.to_thread(_write_file)
    
    assert os.path.exists(jsonl_file), "File not found!"
    logger.info(f"Answer exported to file: {jsonl_file.resolve()}")


async def filter_answers(answers_file: str) -> None:
    """Filter answers to remove invalid entries."""
    if not os.path.exists(answers_file):
        return
    
    try:
        # Run pandas operations in thread pool to avoid blocking
        def _filter():
            answer_df = pd.read_json(answers_file, lines=True)
            filtered_df = []
            
            for _, row in answer_df.iterrows():
                prediction = row.get('prediction')
                truth = row.get('true_answer', '?')
                
                # If the prediction is "Unable to determine", we set it to None
                if str(prediction) == "Unable to determine" or prediction is None:
                    continue
                
                # Processing the test dataset that not contains the true answer
                if truth == "?":
                    if prediction is not None:
                        filtered_df.append(row)
                # Processing the validation dataset that contains the true answer
                else:
                    if prediction is not None:
                        prediction = str(prediction)
                        # Simple validation: check if prediction is not empty
                        if prediction.strip():
                            filtered_df.append(row)
            
            if filtered_df:
                filtered_df = pd.DataFrame(filtered_df)
                filtered_df.to_json(answers_file, lines=True, orient='records')
                return len(answer_df), len(filtered_df)
            return len(answer_df), len(answer_df)
        
        original_count, filtered_count = await asyncio.to_thread(_filter)
        if original_count != filtered_count:
            logger.info(f"Previous answers filtered! {original_count} -> {filtered_count}")
    except Exception as e:
        logger.warning(f"Error filtering answers: {e}")


async def get_tasks_to_run(answers_file: str, dataset) -> List[dict]:
    """Get tasks that haven't been completed yet."""
    # Convert dataset to list of dicts
    data = dataset.data
    tasks = data.to_dict(orient='records')
    
    logger.info(f"Loading answers from {answers_file}...")
    
    try:
        if os.path.exists(answers_file):
            logger.info("Filtering answers starting.")
            await filter_answers(answers_file)
            logger.info("Filtering answers ending.")
            
            # Run pandas read in thread pool
            def _read_json():
                return pd.read_json(answers_file, lines=True)
            
            df = await asyncio.to_thread(_read_json)
            if "task_id" not in df.columns:
                logger.warning(f"Answers file {answers_file} does not contain 'task_id' column. Please check the file format.")
                return tasks
            
            done_questions = set(df["task_id"].tolist())
            logger.info(f"Found {len(done_questions)} previous results!")
            
            # Filter out completed tasks
            tasks_to_run = [task for task in tasks if task.get("task_id") not in done_questions]
            return tasks_to_run
        else:
            logger.info("No previous results found. Starting fresh.")
            return tasks
    except Exception as e:
        logger.warning(f"Error when loading records: {e}")
        logger.warning("No usable records! ▶️ Starting new.")
        return tasks


async def answer_single_question(config, example, save_path):
    """Answer a single question using the agent."""
    # Initialize variables outside try block to ensure they're always available
    task_id = example.get('task_id', f"task_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    question = example.get('question', example.get('task', ''))
    file_name = example.get('file_name', None)
    true_answer = example.get('true_answer', '?')
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    output = None
    iteration_limit_exceeded = False
    raised_exception = False
    exception = None
    
    try:
        # Get agent instance from acp
        agent_name = "esg_agent"
        agent = await acp.get(agent_name)
        
        if agent is None:
            raise ValueError(f"Agent {agent_name} not found. Make sure it's initialized.")
        
        logger.info(f"| Agent {agent_name} initialized")
        logger.info(f"Task Id: {task_id}, Final Answer: {true_answer}")

        task = question
        files = []
        if file_name:
            files.append(file_name)
        
        # Run agent 🚀
        final_result = await agent(task=task, files=files if files else None)
        
        output = str(final_result) if final_result else None
        iteration_limit_exceeded = agent.step_number >= agent.max_steps if hasattr(agent, 'step_number') and hasattr(agent, 'max_steps') else False
        
    except Exception as e:
        logger.error(f"Error on question {question} (task_id: {task_id}): {e}", exc_info=True)
        exception = e
        raised_exception = True
    
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Always save the result, even if there was an error
    annotated_example = {
        "agent_name": config.agent_names[0] if config.agent_names else "esg_agent",
        "question": question,
        "prediction": output,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example.get('task', question),
        "task_id": task_id,
        "true_answer": true_answer,
    }
    
    try:
        await append_answer(annotated_example, save_path)
        logger.info(f"| ✅ Task {task_id} result saved to {save_path}")
    except Exception as e:
        logger.error(f"| ❌ Failed to save result for task {task_id}: {e}", exc_info=True)

async def main():
    args = parse_args()
    
    config.initialize(config_path = args.config, args = args)
    logger.initialize(config = config)
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
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await acp.list()}")
    
    # Transformation ECP to TCP
    logger.info("| 🔄 Transformation start...")
    await transformation.transform(type="e2t", env_names=config.env_names)
    logger.info(f"| ✅ Transformation completed: {await tcp.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    
    # Load dataset
    dataset = ESGDataset(
        path=os.path.join(root, "datasets", "ESG"),
        name="all",
        split="test",
    )
    logger.info(f"| Loaded dataset: {len(dataset)} examples.")
    
    # Get tasks to run (filter out completed ones)
    save_path = os.path.join(config.workdir, "answers.jsonl")
    tasks_to_run = await get_tasks_to_run(save_path, dataset)
    tasks_to_run = [task for task in tasks_to_run if int(task.get("task")) == 3][:1]
    
    # Sort tasks by task_id
    tasks_to_run = list(sorted(tasks_to_run, key=lambda x: x.get("task_id", "")))
    logger.info(f"| Loaded {len(tasks_to_run)} tasks to run.")
    
    if not tasks_to_run:
        logger.info("| No tasks to run. All tasks are already completed.")
        return
    
    # Run tasks with controlled concurrency using semaphore
    concurrency = getattr(config, "concurrency", 4)
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    total_count = len(tasks_to_run)
    
    async def run_with_semaphore(task):
        """Run a task with semaphore control."""
        nonlocal completed_count
        async with semaphore:
            try:
                await answer_single_question(config, task, save_path)
            finally:
                completed_count += 1
                if completed_count % concurrency == 0 or completed_count == total_count:
                    logger.info(f"| Progress: {completed_count}/{total_count} tasks completed.")
    
    # Create all tasks and run them with semaphore-controlled concurrency
    tasks = [run_with_semaphore(task) for task in tasks_to_run]
    await asyncio.gather(*tasks)
    logger.info(f"| ✅ All {total_count} tasks completed.")


if __name__ == "__main__":
    asyncio.run(main())

