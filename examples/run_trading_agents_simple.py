import csv
import os
import sys
import json
from boltons.iterutils import one
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from time import time
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.prompt import prompt_manager
from src.memory import memory_manager
from src.tool import tcp
from src.environment import ecp
from src.agent import acp
from src.skill import scp
from src.transformation import transformation
from src.session.types import SessionContext
from src.optimizer import ReflectionOptimizer
import shutil
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "trading_agents.py"), help="config file path")

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
    await tcp.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tcp.list()}")

    logger.info("| 🎯 Initializing skills...")
    skill_names = getattr(config, 'skill_names', None)
    await scp.initialize(skill_names=skill_names)
    logger.info(f"| ✅ Skills initialized: {await scp.list()}")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(env_names = config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Initialize agents
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await acp.list()}")
    await transformation.transform(type="e2t", env_names=["signal_research"])
    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    

    signal_agent = await acp.get("trading_signal")
    bechmark_agent = await acp.get("trading_benchmark")
    strategy_agent = await acp.get("trading_strategy")

    strategy_feedback = None

    signal_session = SessionContext()
    benchmark_session = SessionContext()
    strategy_session = SessionContext()
    universal_tools = ["done","todo"]
    signal_count = 2

    async def signal_generation(n,benchmark_feedback=None,signal_count=1):
        if signal_count > 5:
            signal_count = 5

        for _ in range(n):

            await tcp.cleanup()

            await tcp.initialize(universal_tools)

            await transformation.transform(type="e2t", env_names=["signal_research"])

            await tcp.tool_context_manager.save_contract()
            style = "aggressive"
            summary = "High-tempo breakout pursuit in expansion regimes with pyramiding and volatility-based trailing exits."
            query  = "Pursue large upside moves by prioritizing early participation in expansion regimes detected strictly from current-bar activity (range expansion, momentum thrust, and participation/volume pressure). Define entries using fast-reacting comparisons and threshold triggers that allow earlier signals, even at the cost of more false positives, while still requiring at least one confirmation filter to avoid the weakest noise. Use dynamic position sizing that scales up rapidly (pyramiding) when the current bar continues to validate the expansion regime and the signal strengthens, but enforce a firm maximum exposure limit. Risk controls should be wider than conservative styles, tied to current volatility conditions, and complemented by a trailing exit that tightens as momentum decays. Impose a turnover discipline: allow higher trade frequency but cap the number of entries within a short window and include a rule that forces reduced sizing or a temporary pause after a cluster of adverse bars."

            
            signal_main_task = f"""Design signal strictly based on style,summary and query. the data is day-based. Implement {signal_count} signal/signals only.
                Style: {style} 
                Summary: {summary}
                Query: {query}
                Signal Limit: {signal_count} signal/signals. Leave the rest with -1.

            Transfer for evaluation when finished
            """

            
            task = f"""Main task: {signal_main_task}
            Feedback from benchmark agent: {benchmark_feedback if  benchmark_feedback else "N/A"}
            Example: 
            Reason: ...
            Result: ...
            """

            files = []
            response = await signal_agent(task=task,files=files,ctx=signal_session)

            await tcp.cleanup()

            await tcp.initialize(universal_tools)

            await transformation.transform(type="e2t", env_names=["signal_evaluate"])

            await tcp.tool_context_manager.save_contract()

            main_task = f"""Evaluate the quality of the signal. And simulate signal-based strategy trading performance. The data is 1d frequency.
            """
            info = response.message

            task = f"""Main task: {main_task} \n
            Reasoning from signal agent: {info} \n
            Format requiremment: Ouput explanation followed by Decision: [OK] or [NOT OK]
            Reason: ... 
            Decision: [NOT OK]
            Feedback: ...
            Backtest result...

            Criteria for [OK]:
            1) One signal time series IC >0.03 in at least one horizon (1d, 3d, 5d or 10d) and resonable autocorrelatiion and decay.
            2) Signal is profitable in simple backtest with reasonable assumption. Provide backtest result as evidence.
            3) Run backtest with >20% excessive returns

            If necessary, you can generate signal inside benchmark to test some factors.

            Assume commision fee is 0.04% per side
            """

            files = []
            response = await bechmark_agent(task=task,files=files,ctx=benchmark_session)

            explanation = response.message

            benchmark_feedback = explanation

            if "[OK]" in explanation:
                return False
            
        return True

    await signal_generation(n = 5,signal_count=signal_count)
    
if __name__ == "__main__":
    start_timing = time()
    asyncio.run(main())
    end_timing = time()
    print(f"Total execution time: {(end_timing - start_timing)/60:.2f} minutes")