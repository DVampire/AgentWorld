from __future__ import annotations
import csv
from datetime import datetime
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
from time import time
from src.logger import logger
from src.environment.quickbacktest.run import run_backtest,get_rank_ic
from src.model import model_manager
from src.version import version_manager
from src.prompt import prompt_manager
from src.memory import memory_manager
from src.tool import tcp
from src.environment import ecp
from src.agent import acp
from src.skill import scp
from src.transformation import transformation
from src.utils.args_utils import parse_tool_args as parse_json_args
from src.session.types import SessionContext
from src.environment.quickbacktest.cst_utils import extract_first_json_object
import shutil
import os
import pandas as pd

from src.environment.quickbacktest.judge_config import SignalJudge, StrategyJudge
from typing import Any, Dict


signal_judge = SignalJudge()

strategy_judge = StrategyJudge()

tmp_dir = Path(r"E:\ureca\AgentWorld\workdir\trading_agents")
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)

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

    await transformation.transform(type="e2t", env_names=["signal_research","signal_evaluate","strategy_evaluate","quickbacktest"])

    # Initialize version manager, must after tool, agent, environment initialized
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")
    



    signal_agent = await acp.get("trading_signal")
    signal_evaluate_agent = await acp.get("trading_signal_evaluation")
    strategy_agent = await acp.get("trading_strategy")
    strategy_evaluate_agent = await acp.get("trading_strategy_evaluation")


    signal_evaluate_env = await ecp.get("signal_evaluate")
    strategy_evaluate_env = await ecp.get("strategy_evaluate")



    signal_session = SessionContext()
    signal_evaluate_session = SessionContext()
    strategy_evaluate_session = SessionContext()
    strategy_session = SessionContext()

    all_tools = list(tcp.tool_context_manager._tool_configs.keys())

    universal_tools = ["done","todo"]

    print(all_tools)

    signal_tools = universal_tools + [tool for tool in all_tools if tool.startswith("signal_research.")]
    signal_evaluation_tools = universal_tools+[tool for tool in all_tools if tool.startswith("signal_evaluate.")]
    strategy_evaluation_tools = universal_tools+[tool for tool in all_tools if tool.startswith("strategy_evaluate.")]
    strategy_tools =universal_tools+ [tool for tool in all_tools if tool.startswith("quickbacktest.")]
    style = "conservative"
    summary = "Capital-preserving, regime-filtered entries with scaled exposure and strict risk-off overrides."
    query  = "Design a capital-preservation-first strategy that only participates when the current bar indicates an orderly, low-stress market regime, and otherwise stays defensive. Define a current-bar regime classifier using a volatility proxy and a candle/return-stability proxy, and require multi-signal confirmation before any entry (trend alignment plus a separate momentum/participation confirmation). Position sizing must be conservative and tiered: use smaller baseline exposure and allow a partial scale-in only when the same-bar signal strength is exceptionally high. Exits must include (i) a volatility-adjusted hard stop, (ii) a profit-taking rule that avoids giving back gains during sudden regime deterioration, and (iii) a time-based exit if price fails to progress despite favorable regime conditions. Add a strict risk-off override: whenever the current bar reflects stress (volatility spike, unstable candle structure, or failed participation), the strategy must immediately prefer flat/near-flat exposure regardless of other signals."
    
    print(signal_tools)

    async def signal_generation(n,signal_count=1):
        if signal_count > 5:
            signal_count = 5

        judge_result = None

        for i in range(n):

            await tcp.tool_context_manager.save_contract(tool_names = signal_tools)

            image_dir = Path(r"workdir\trading_agents\environment\signal_evaluate\images")
            for item in image_dir.iterdir():
                item.unlink()  # overwrite

            style = "conservative"
            summary = "Capital-preserving, regime-filtered entries with scaled exposure and strict risk-off overrides."
            query  = "Design a capital-preservation-first strategy that only participates when the current bar indicates an orderly, low-stress market regime, and otherwise stays defensive. Define a current-bar regime classifier using a volatility proxy and a candle/return-stability proxy, and require multi-signal confirmation before any entry (trend alignment plus a separate momentum/participation confirmation). Position sizing must be conservative and tiered: use smaller baseline exposure and allow a partial scale-in only when the same-bar signal strength is exceptionally high. Exits must include (i) a volatility-adjusted hard stop, (ii) a profit-taking rule that avoids giving back gains during sudden regime deterioration, and (iii) a time-based exit if price fails to progress despite favorable regime conditions. Add a strict risk-off override: whenever the current bar reflects stress (volatility spike, unstable candle structure, or failed participation), the strategy must immediately prefer flat/near-flat exposure regardless of other signals."

            
            signal_main_task = f"""Design signal strictly based on style,summary and query.
                Style: {style} 
                Summary: {summary}
                Query: {query}

            Design signal according to the style, summary, and query.
            Generate {signal_count} signals (one factor one file). Do not update/fix signals that have been proved valid with true hypotheses.
            Do version control for each signal. Name the file with version. Use fix for only syntax error.
            Verify code is runnable using getSignalQuantile tool
            Transfer for evaluation when finished
            Update signal based on suggested focus and recommended hypotheses from judge feedback. Only update signals that hypothesis is not true.
            """ +"\n"+ """
            {
                signals: [],
                signal_combination_hypothesis: "(in the form of equation using s1,s2....s5 to represent the signals)",
            }

            """

            task = f"""Main task: {signal_main_task}
            Feedback: {judge_result if  judge_result else "N/A"}
            """

            files = []
            response = await signal_agent(task=task,files=files,ctx=signal_session)

            await tcp.tool_context_manager.save_contract(tool_names = signal_evaluation_tools)


            main_task = f"""Evaluate the quality of the signal. The data is 1d frequency. 
            Run evaluations every round even it is in your memeory.
            Your result should be evidence-based
            """
            info = response.message

            task = f"""Main task: {main_task} \n
            Signal Result: {info} \n
            
            Hypothesis are strictly the same as the ones in each signal docstring.
            Call saveSignalEvaluation to save your result.
            """


            files = []
            response = await signal_evaluate_agent(task=task,files=files,ctx=signal_evaluate_session)


                # ex = extract_first_json_object(response.message)
            evaluation_result = await signal_evaluate_env.get_last_evaluation_result()
            judge_result = (signal_judge.step(evaluation_result))
            logger.info(f"Judge result for signal iteration {i+1}: {judge_result}")


            if judge_result["decision"] == "end":
                logger.info(f"Signal iteration ended by judge: {judge_result}")
                return judge_result
            else:
                continue

        return judge_result


    async def strategy_design(n,signal_combination=None, combo_hypothesis=None, rank_ic=None):

        judge_result = None
        for i in range(n):
            image_dir = Path(r"workdir\trading_agents\environment\strategy_evaluate\images")
            for item in image_dir.iterdir():
                item.unlink()  # overwrite
            

            await tcp.tool_context_manager.save_contract(tool_names = strategy_tools)



            task = """Generation strategy based on the signal hypothesis. Name the file with versions.Check it is runnable using tool. Transfer for evaluation. Call down only when transferred successfully."""
                    
            enhance_task = f"""Main task: {task} \n

            Best signal combination: {signal_combination if signal_combination else "N/A"}
            Signal combination hypothesis: {combo_hypothesis if combo_hypothesis else "N/A"}
            Rank IC: {rank_ic if rank_ic else "N/A"}
            Judge feedback: {judge_result if judge_result else "N/A"}

            Output in json format:

            Strategy design instructions: {query}

            """ + "\n"+ """
            {
                "signal_combinations": [...]
                "strategy_name": "<string>",
                "edge_hypothesis": "<string>"
                "backtest_result": {...}
            }
            """

            response = await strategy_agent(task=enhance_task,files=[],ctx=strategy_session)

            await tcp.tool_context_manager.save_contract(tool_names = strategy_evaluation_tools)
            task = f"""
            Main task: Evaluate the strategy that designe based on

            {response.message}

            Re-run evaluation every round even if the strategy is in your memory, because the strategy may be updated every round.

            Strategy design instructions: {query}

            Upload result by tool.
            """


            files = []
            response = await strategy_evaluate_agent(task=task,files=files,ctx=strategy_evaluate_session)


            judge_result = (strategy_judge.step(await strategy_evaluate_env.get_last_strategy_evaluation_result()))
            logger.info(f"Strategy judge result: {judge_result}")

            if judge_result["decision"] == "end":
                logger.info(f"Strategy iteration ended by judge: {judge_result}")
                return judge_result
            else:
                continue
        return judge_result


    async def backtest(signal,strategy) -> Dict[str, Any]:
        result_1 = run_backtest(
            start = datetime(2025, 1, 2),
            end = datetime(2026, 1, 1),
            data_dir = "datasets/backtest/binance",
            watermark_dir = "datasets/backtest/binance_state.duckdb",
            venue = "binance_um",
            symbol = "BTCUSDT",
            strategy_module=strategy,
            signal_modules=signal,
            base_dir="workdir/trading_agents/environment/strategy_evaluate",
            plot=True
        )

        return result_1
    # async def rank_ic(signal) -> Dict[str, Any]:
    #     result_1 = get_rank_ic(
    #         start = datetime(2025, 1, 2),
    #         end = datetime(2026, 1, 1),
    #         data_dir = "datasets/backtest/binance",
    #         watermark_dir = "datasets/backtest/binance_state.duckdb",
    #         venue = "binance_um",
    #         symbol = "BTCUSDT",
    #         signal_modules=signal,
    #         base_dir="workdir/trading_agents/environment/signal_evaluate",
    #         horizon=1
    #     )

    #     return result_1

    for _ in range(1):
        files = []
        signal_count = 5


        signal_result = await signal_generation(20,signal_count=signal_count)


        signal_combination = signal_result.get("best_signal_combination", [])
        combo_rank_ic = signal_result.get("best_combo_rank_ic", None)
        combo_hypothesis = signal_result.get("best_combo_hypothesis", "")



        print(signal_result)

        src_dir = Path(r"workdir\trading_agents\environment\signal_evaluate\signals")
        dst_dir = Path(r"workdir\trading_agents\environment\quick_backtest\signals")
        EXCLUDE = {"__init__.py", "__pycache__"}

        for item in src_dir.iterdir():
            if item.name in EXCLUDE:
                continue

            if item.is_file():
                target = dst_dir / item.name

                if target.exists():
                    target.unlink()  # overwrite

                shutil.move(str(item), target)


        signal_combination = signal_result.get("best_signal_combination", [])
        rank_ic = signal_result.get("best_combo_rank_ic", None)
        combo_hypothesis = signal_result.get("best_combo_hypothesis", "")




        # print(signal_combination, rank_ic, combo_hypothesis)
        strategy_result = await strategy_design(20, signal_combination=signal_combination, combo_hypothesis=combo_hypothesis, rank_ic=rank_ic)

    iteration_result = strategy_result

    backtest_result = await backtest(signal=iteration_result["next_step_path"]["best_signal_combinations"], strategy=iteration_result["next_step_path"]["best_strategy_name"])

    logger.info(f"Final backtest result: {backtest_result}")

if __name__ == "__main__":
    start_timing = time()
    asyncio.run(main())
    end_timing = time()
    print(f"Total execution time: {(end_timing - start_timing)/60:.2f} minutes")