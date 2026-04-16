import os
import sys
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.prompt import prompt_manager
from src.memory import memory_manager
from src.tool import tool_manager
from src.environment import environment_manager
from src.agent import agent_manager
from src.transformation import transformation
from src.session.types import SessionContext
from src.optimizer import ReflectionOptimizer
import shutil
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "trading_strategy_agent.py"), help="config file path")

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
    await tool_manager.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tool_manager.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await environment_manager.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {environment_manager.list()}")

    env_names = ["signal_research","quickbacktest"]  # 使用信号研究环境的名称
    await transformation.transform(type="e2t", env_names=env_names)
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await agent_manager.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await agent_manager.list()}")
    
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    

    
    # Example task
    # task = """If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."""
    # task = """Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations."""
    # task = "Write a mini game about a cat that can fly and fight enemies, and then push it to github."


    # HYPOTHESIS = "Price displacement over 60D interacted with volume intensity ratio; targets potential exhaustion and reversa"
    # task = rf"Implement signal and corresponding strategy using the hypothesis {HYPOTHESIS} and other technical indicators.Try to achieve high win rate. Keep the result one when finished. Clear workdir regularly to delete unnecessary files."

    task = r"""Design or a qualified signal and strategy to beat benchmark 
    in out-sample test with positive return, and robust again bear, bull and sideways. 
    Current commison fee is 0.04% per order. Use 2024- mid 2025 for insample test.
    Clean images regularly.
    Do version control"""

    evolved_round = 3
    ctx  = SessionContext()

    memory = []
    files = []

    for i in range(1,evolved_round+1):
        input = {
        "name": "trading_strategy",
        "input": {
            "task": task,
            "files": files
        },
        "ctx": ctx
        }
        result = await agent_manager(**input)

        signal_iteration_dir = Path(config.workdir) / "environment"/ "SignalIterations.md"
        strategy_iteration_dir = Path(config.workdir) / "environment"/ "StrategyIterations.md"
        backtest_log_dir = Path(config.workdir) / "environment"/ "backtest_log.md"
        signal_list_dir = Path(config.workdir) / "environment"/ "signals.csv"

        logger.info(f"Round {i} agent response: {result.message}")

        df_signal = pd.read_csv(signal_list_dir) if signal_list_dir.exists() else pd.DataFrame()
        if not df_signal.empty:
            df_signal = df_signal.sort_values(by="hit_rate", ascending=False)
            df_signal = df_signal.drop_duplicates(subset=["signal"], keep="first")
        
        signal_in_md = df_signal.to_markdown(index=False) if not df_signal.empty else "No signal generated"
        df_signal.sort_index().to_csv(signal_list_dir, index=False) if not df_signal.empty else None

        optimzation_task = f"""
        <input>
        <agent_memory> {str(memory)} </agent_memory>
        <task> {task} </task>
        <previous_response> {result.message} </previous_response>
        <SignalIterationFile> {str(signal_iteration_dir.read_text()) if signal_iteration_dir.exists() else ""} </SignalIterationFile>
        <StrategyIterationFile> {str(strategy_iteration_dir.read_text()) if strategy_iteration_dir.exists() else ""} </StrategyIterationFile>
        <SIGNAL_LIST> {signal_in_md if not df_signal.empty else "No signal generated"} </SIGNAL_LIST>
        <backtest_log> {str(backtest_log_dir.read_text()) if backtest_log_dir.exists() else ""} </backtest_log> (Just for evaluation, can not modify its format)
        </input>
        """

        system_modules = dict(workdir=config.workdir)
        agent_message_modules = dict(task = optimzation_task)
        messages = await prompt_manager.get_messages(
                prompt_name="trading_optimizer_reflection",
                system_modules=system_modules,
                agent_modules= agent_message_modules
            )


        signal_img_folder = Path(config.workdir) / "environment" / "signal_research" / "images"
        strategy_img_folder = Path(config.workdir) / "environment" / "quick_backtest" / "images"

        response = await model_manager(model="openrouter/claude-sonnet-4.5",messages=messages)

        shutil.rmtree(signal_img_folder, ignore_errors=True)
        shutil.rmtree(strategy_img_folder, ignore_errors=True)
        os.makedirs(signal_img_folder, exist_ok=True)
        os.makedirs(strategy_img_folder, exist_ok=True)
        os.remove(signal_iteration_dir) if signal_iteration_dir.exists() else None
        os.remove(strategy_iteration_dir) if strategy_iteration_dir.exists() else None
        os.remove(backtest_log_dir) if backtest_log_dir.exists() else None

        

        memory.append({
            "round": i,
            "task": task,
            "response": response.message,
        })

        task = response.message
        logger.info(f"Round {i} optimization result: {task}")

    else:
        input = {
        "name": "trading_strategy",
        "input": {
            "task": task,
            "files": files
        },
        "ctx": ctx
        }
        result = await agent_manager(**input)

    logger.info(f"Final agent response: {result.message}")

    evaluation_memory = pd.DataFrame(memory)
    evaluation_memory.to_csv(Path(config.workdir) / "trading_strategy_agent_evaluation.csv", index=False)

    
if __name__ == "__main__":
    asyncio.run(main())