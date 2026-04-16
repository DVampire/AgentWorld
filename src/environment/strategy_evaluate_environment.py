import csv
import json
import os
from typing import Any, Dict, Any, Dict, List, Literal, Optional, Type, Union
from unittest import result
from ddgs.engines import module_path
from matplotlib.pylab import roll
from pydantic import  Field, ConfigDict
from src.logger import logger
from src.environment.server import environment_manager
from src.environment.types import Environment
from src.registry import ENVIRONMENT
from src.environment.quickbacktest.run import get_signal_quantile,ClassLoader,run_signal_evaluation,run_strategy_evaluation
from src.utils import dedent, assemble_project_path
from src.utils.utils import parse_code_blobs
from src.environment.quickbacktest.run import run_backtest,ClassLoader,get_signal_quantile,dict_to_markdown_table
from src.environment.quickbacktest.cst_utils import patch_file,PatchConfig,extract_first_json_object
import src.environment.quickbacktest.signal_research as s
from pathlib import Path
import shutil
from datetime import datetime
from src.prompt import prompt_manager
import inspect
from dateutil.relativedelta import relativedelta


@ENVIRONMENT.register_module(force=True)
class StrategyEvaluateEnvironment(Environment):
    """Quick backtest environement"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="strategy_evaluate", description="The name of the evaluation environment.")
    description: str = Field(default="Strategy evaluation environment for strategy testing", description="The description of the strategy evaluation environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the quickbacktest environment including current available signals",
        }
    }, description="The metadata of the quickbacktest environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")


    def __init__(
        self,
        base_dir: str ,
        require_grad: bool = False,
        backtest_start_date: str = "2023-01-01",
        backtest_end_date: str = "2024-01-01",
        **kwargs: Any,
    ):
        
        super().__init__(**kwargs)
        self.base_dir =  Path(assemble_project_path(base_dir))
        logger.info(f"| 🚀 Strategy Evaluate Environment initialized at: {self.base_dir}")

        self.start = datetime.strptime(backtest_start_date, "%Y-%m-%d")
        self.end = datetime.strptime(backtest_end_date, "%Y-%m-%d")
        self.tools = [
    name 
    for name, func in inspect.getmembers(s, inspect.isfunction)
    if func.__module__ == s.__name__ and name[0]!="_"
]
        self.step = 0
        self.STRATEGY_EVALUATION_STORE = []


    async def initialize(self) -> None:
        """Initialize the strategy evaluate environment."""
        try:
            for folders in ["signals","evaluations","strategies",]:
                env_dir = Path(self.base_dir) / folders
                if not env_dir.exists():
                    env_dir.mkdir(parents=True, exist_ok=True)
                dst_1 = env_dir / "__init__.py"
                dst_1.touch(exist_ok=True)
            images_dir = self.base_dir / "images"
            if not images_dir.exists():
                images_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"| 🚀 Strategy Evaluate Environment initialized at: {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize Strategy Evaluate Environment: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup the signal research environment."""
        # try:
        #     for folders in ["strategies", "signals"]:
        #         env_dir = Path(self.base_dir) / folders
        #         if env_dir.exists() and env_dir.is_dir():
        #             shutil.rmtree(env_dir)

        #     if Path(self.base_dir).exists() and Path(self.base_dir).is_dir():
        #         shutil.rmtree(Path(self.base_dir))
        #     logger.info("| 🧹 Signal Research Environment cleanup completed")
        # except Exception as e:
        #     logger.error(f"Failed to cleanup Signal Research Environment: {str(e)}")

        pass

    @environment_manager.action(name="addEvaluation",description="""Add a benchmark to the environment." \
    Add a trading benchmark to the research environment.

            Args:
                module_code (str): The code of the benchmark to add.
                module_name (str): The name of the benchmark to add.
                hypothesis (str): A description of the hypothesis will be included in this benchmark

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information about the signal.

        """)
    
    async def addEvaluation(self, module_code: str, module_name: str, hypothesis: str,**kwargs) -> Dict[str, Any]:
        """Add a trading benchmark to the research environment.

            Args:
                module_code (str): The code of the benchmark to add.
                module_name (str): The name of the benchmark to add.
                hypothesis (str): A description of the hypothesis to be included in this evaluation.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation of the evaluation.

        """
        try:

            module_path = Path(self.base_dir) / "evaluations" / f"{module_name}.py"
            module_code = parse_code_blobs(module_code)
            if module_path.exists():
                raise FileExistsError(f"evaluation {module_name} already exists in Strategy Evaluate Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)
            await self.modifymd(filename = "EvaluationIterations.md", insights = f"## Evaluation: {module_name}\n\n### Hypothesis:\n{hypothesis}", mode="append")
            return {"success": True, "message": f"Evaluation {module_name} added successfully", "extra": {}}

        except Exception as e:
            logger.error(f"Failed to add evaluation {module_name}: {str(e)}")
            return {"success": False, "message": f"Failed to add evaluation {module_name} since {str(e)}", "extra": {"error": str(e)}}

    
        
    @environment_manager.action(name="fixEvaluation",description="""Fix a trading evaluation in the research environment.
            Args:
                module_code (str): The full code of the evaluation to update.
                module_name (str): The name of the evaluation to update.
                bug_fix (str): A description of the bug fix to be made to the evaluation.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation if update module is signal.
        """)
    async def fixEvaluation(self, module_code: str, module_name: str, bug_fix:str,**kwargs) ->Dict[str, Any]:
        """Update a trading evaluation in the research environment.
            Args:
                module_code (str): The full code of the evaluation to update.
                module_name (str): The name of the evaluation to update.
                bug_fix (str): A description of the bug fix to be made to the evaluation.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information if update module is signal.
        """
        try:
            module_path = Path(self.base_dir) / "evaluations" / f"{module_name}.py"
            try:
                module_code = parse_code_blobs(module_code)
            except Exception as e:
                module_code = module_code

            if not module_path.exists():
                raise FileNotFoundError(f"Evaluation {module_name} does not exist in Strategy Evaluate Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)
                
            await self.modifymd(filename = "EvaluationIterations.md", insights = f"## Evaluation: {module_name}\n\n### Bug Fix:\n{bug_fix}", mode="append")
            return {"success": True, "message": f"Evaluation {module_name} updated successfully", "extra": {}}
        except Exception as e:
            logger.error(f"Failed to update evaluation {module_name}: {e}")
            return {"success": False, "message": f"Failed to update evaluation {module_name}", "extra": {"error": str(e)}}


    async def modifymd(self, filename: str, insights: str, mode: Literal["append", "overwrite"], **kwargs) -> Dict[str, Any]:
        """Save the insights of a trading signal to a markdown file in the environment.
            Args:
                filename (str): The name of the markdown file to save the insights to.
                insights (str): The insights to save. Include data to prove.
                mode (Literal["append", "overwrite"]): The mode to save the insights, "append" will add the insights to the end of the markdown file, while "overwrite" will replace the content of the markdown file with the insights.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the saved markdown file if successful.
        """
        try:
            if mode not in ["append", "overwrite"]:
                raise ValueError("mode must be either 'append' or 'overwrite'")
            insights_dir = Path(self.base_dir).parent
            # insights_dir.mkdir(parents=True, exist_ok=True)
            insights_path = insights_dir / filename
            if mode == "overwrite":
                with open(insights_path, "w") as f:
                    f.write(insights)
            else:
                with open(insights_path, "a") as f:
                    f.write(insights)
            logger.info(f"| ✅ Insights for file {filename} saved to {insights_path}.")
            return {"success": True, "message": f"Insights for file {filename} saved successfully.", "extra": {"insights_path": str(insights_path)}}
        except Exception as e:
            logger.error(f"Failed to save insights for file {filename}: {str(e)}")
            return {"success": False, "message": f"Failed to save insights for file {filename} since {str(e)}", "extra": {"error": str(e)}}
        

    @environment_manager.action(name="removeEvaluation",description="""Remove a benchmark from the environment.
            Args:
                module_name (str): The name of the benchmark to remove.

            Returns:
                Dict[str,Any]: The tool state after removing the signal.
        """)
    async def removeEvaluation(self, module_name: str,**kwargs) -> Dict[str,Any]:
        """Remove a trading evaluation from the environment.
            Args:
                module_name (str): The name of the evaluation to remove.

            Returns:
                Dict[str,Any]: The tool state after removing the evaluation. """
        try:
            
            module_type = "evaluations"
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in Strategy Evaluate Environment.")
            module_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove {module_type[:-1]} {module_name}: {e}")
            return {"success": False, "message": f"Failed to remove {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}
    
        logger.info(f"| ✅ {module_type[:-1]} {module_name} removed from Strategy Evaluate Environment.")
        return {"success": True, "message": f"{module_type[:-1]} {module_name} removed successfully", "extra": {}}


    async def listSignal(self, **kwargs) -> Dict[str,Any]:
        """List all signals in the environment.

            Returns:
                Dict[str, Any]: A dictionary with the module type as the key and a list of module names as the value.
        """
        try:
            module_type = "signals"
            env_dir = Path(self.base_dir) / module_type
            modules = {f"{module_type}": []}
            for file in env_dir.glob("*.py"):
                if file.stem not in ["__init__"]:
                    modules[f"{module_type}"].append(file.stem)
            logger.info(f"| ✅ Listed {module_type}: {modules}")
            return modules

        except Exception as e:            
            logger.error(f"Failed to list {module_type}: {e}")
            return {"success": False, "message": f"Failed to list {module_type} since {str(e)}", "extra": {"error": str(e)}}


    async def listEvaluation(self, **kwargs) -> Dict[str,Any]:
        """List all strategy evaluations in the environment.

            Returns:
                Dict[str, Any]: A dictionary with the module type as the key and a list of module names as the value.
        """
        try:
            module_type = "evaluations"
            env_dir = Path(self.base_dir) / module_type
            modules = {f"{module_type}": []}
            for file in env_dir.glob("*.py"):
                if file.stem not in ["__init__"]:
                    modules[f"{module_type}"].append(file.stem)
            logger.info(f"| ✅ Listed {module_type}: {modules}")
            return modules

        except Exception as e:            
            logger.error(f"Failed to list {module_type}: {e}")
            return {"success": False, "message": f"Failed to list {module_type} since {str(e)}", "extra": {"error": str(e)}}
    



    async def listStrategy(self, **kwargs) -> Dict[str,Any]:
        """List all strategies in the environment.

            Returns:
                Dict[str, Any]: A dictionary with the module type as the key and a list of module names as the value.
        """
        try:
            module_type = "strategies"
            env_dir = Path(self.base_dir) / module_type
            modules = {f"{module_type}": []}
            for file in env_dir.glob("*.py"):
                if file.stem not in ["__init__"]:
                    modules[f"{module_type}"].append(file.stem)
            logger.info(f"| ✅ Listed {module_type}: {modules}")
            return modules

        except Exception as e:            
            logger.error(f"Failed to list {module_type}: {e}")
            return {"success": False, "message": f"Failed to list {module_type} since {str(e)}", "extra": {"error": str(e)}}



    @environment_manager.action(name="removePNG",description="""Remove PNG files from the environment.
            Args:
                images (List[str]): A list of image file names to remove.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation.
        """)
    async def removePNG(self, images: List[str], **kwargs) -> Dict[str, Any]:

        """Remove all PNG files from the images directory."""
        try:
            images_dir = self.base_dir / "images"
            if images_dir.exists() and images_dir.is_dir():
                for file in images_dir.glob("*.png"):
                    if file.name in images:
                        file.unlink()
            logger.info("| ✅ All PNG files removed from images directory.")
            return {"success": True, "message": "All PNG files removed successfully", "extra": {}}
        except Exception as e:
            logger.error(f"Failed to remove PNG files: {e}")
            return {"success": False, "message": f"Failed to remove PNG files since {str(e)}", "extra": {"error": str(e)}}


    async def get_state(self,**kwargs) -> Dict[str, Any]:
        """Get the current state of the environment."""
        diagram_path = [a for a in os.listdir(self.base_dir/"images") if a.endswith(".png")]
        self.step+=1
        state = {
            "state": str({
                    "signals": await self.listSignal(),
                    "strategies": await self.listStrategy(),
                    "evaluations": await self.listEvaluation(),
                    "extra_files": diagram_path}),
            "extra":{}
        }


        logger.info(f"| ✅ Strategy Evaluate Environment state retrieved: {state}")
        return state
    
    @environment_manager.action(name="getDocString",description="""Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "signal_evaluations", "strategy_evaluations","strategies"]): The type of the module to get the docstring from.
            Returns:
                Dict[str,Any]: The tool state and docstring.
        """)
    
    async def getDocString(self, module_name: str, module_type: Literal["signals", "evaluations","strategies"], **kwargs) -> Dict[str,Any]:
        """Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "evaluations","strategies"]): The type of the module to get the docstring from.

            Returns:
                Dict[str,Any]: The tool state and docstring.
        """
        try:
            if module_type not in ["signals", "evaluations","strategies"]:
                raise ValueError("module_type must be either 'signals', 'evaluations', or 'strategies'")
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in Strategy Evaluate Environment.")
            module = ClassLoader.load_class(
                file_path=module_path,
                class_name=module_name,
            )
            doc = module.__doc__ if module.__doc__ else "No docstring available."
            del module

            logger.info(f"| ✅ Retrieved docstring for {module_type[:-1]} {module_name}.")
            return {"success": True, "message": f"Retrieved docstring for {module_type[:-1]} {module_name} with docstring {doc}.", "extra": {"docstring": doc}}
        except Exception as e:
            logger.error(f"Failed to get docstring for {module_type[:-1]} {module_name}: {str(e)}")
            return {"success": False, "message": f"Failed to get docstring for {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}

    @environment_manager.action(name="removeModule",description="""Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "evaluations","strategies"]): The type of the module to remove.

            Returns:
                Dict[str,Any]: The tool state after removing the module.
        """)
    async def removeModule(self, module_name: str, module_type: Literal["signals", "evaluations","strategies"], **kwargs) -> Dict[str,Any]:
        """Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "evaluations","strategies"]): The type of the module to remove.
            Returns:
                Dict[str,Any]: The tool state after removing the module.
        """
        try:
            
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in Strategy Evaluate Environment.")
            module_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove {module_type[:-1]} {module_name}: {e}")
            return {"success": False, "message": f"Failed to remove {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}
    
        logger.info(f"| ✅ {module_type[:-1]} {module_name} removed from signal_evaluate Environment.")
        return {"success": True, "message": f"{module_type[:-1]} {module_name} removed successfully", "extra": {}}


    @environment_manager.action(name="runStrategyEvaluation",description= """Backtest a trading signal + strategy using historical data and run the custom strategy evaluation.
            Args:
                strategy_name (str): The name of the strategy module to use.
                signal_names (List[str]): The names of the signal modules to use. Order matters. They will be labeled as signal_1, signal_2...to signal_5.
                evaluation_name (str): The name of the custom strategy evaluation module to use. The custom strategy evaluation module should be added to the environment before running this action.
                strategy_feedback (str): The custom strategy feedback to run after backtesting. Include data to prove.

            Returns:    
                Dict[str, Any]: The backtest result including performance metrics and trade history.
                """)
    async def runStrategyEvaluation(self,strategy_name,signal_names,evaluation_name,start=None,end=None,**kwargs) -> Dict[str, Any]:
        """Backtest a trading signal + strategy using historical data.
                    Args:
                        strategy_name (str): The name of the strategy module to use.
                        signal_names (List[str]): The names of the signal modules to use.
                        evaluation_name (str): The name of the custom strategy evaluation module to use.
                        start (str): The start date of the backtest period in "YYYY-MM-DD" format. Optional, defaults to out-of-sample period start date.
                        end (str): The end date of the backtest period in "YYYY-MM-DD" format. Optional, defaults to out-of-sample period end date.
                    Returns:
                        Dict[str, Any]: The backtest result including performance metrics and trade history.

                    
                    Some metrics to consider when evaluating backtest results:
                    - Cumulative Return (%) - Total return of the strategy over the backtest period.
                    - Sharpe Ratio - Risk-adjusted return measure.
                    - Max Drawdown (%) - Largest peak-to-trough decline in the strategy's equity curve
                    - win_rate (%) - Percentage of profitable trades.
                    - closed_trades - Total number of closed trades during the backtest period.
                    - total_commission (%) - Total commission paid as a percentage of the initial capital.
                    - excess_return_ratio (%) - Return of the strategy above the benchmark return.
                    - max_shortfall (%) - Maximum shortfall from the benchmark.
                """
        try:
            if start is not None:
                start_date = datetime.strptime(start, "%Y-%m-%d")
            else:
                start_date = self.start

            if end is not None:
                end_date = datetime.strptime(end, "%Y-%m-%d")
            else:
                end_date = self.end

            benchmark_result = run_strategy_evaluation(data_dir = "datasets/backtest/binance",
                watermark_dir = "datasets/backtest/binance_state.duckdb",
                venue = "binance_um",
                symbol = "BTCUSDT",
                strategy_module=strategy_name, 
                signal_module=signal_names, 
                evaluation_module=evaluation_name, 
                start=start_date, 
                end=end_date, 
                base_dir=self.base_dir)
            
            logger.info(f"| ✅ Backtest completed using strategy {strategy_name} and signals {signal_names} with results\n: {benchmark_result}.")
            return {
                "success": True,
                "message": f"Backtest completed using strategy {strategy_name} and signals {signal_names} with in-sample results {benchmark_result}.",
                "extra": {"backtest_result": benchmark_result,},
            }
        except Exception as e:
                logger.error(f"Backtest failed: {str(e)}")
                return {
                    "success": False,
                    "message": f"Backtest failed using strategy {strategy_name} and signals {signal_names} since {str(e)}.",
                    "extra": {"error": str(e)},
                    }

    @environment_manager.action(name="backtest",description= """Backtest a trading signal + strategy
                Args:
                    strategy_name (str): The name of the strategy module to use.
                    signal_names (List[str]): The names of the signal modules to use. Order matters. They will be labeled as signal_1, signal_2.
                Returns:    
                    Dict[str, Any]: The backtest result including performance metrics and trade history.
                    """)
    async def backtest(self,strategy_name,signal_names:List[str] = [],start=None,end=None,**kwargs) -> Dict[str, Any]:
            """Backtest a trading signal + strategy using historical data.
                        Args:
                            strategy_name (str): The name of the strategy module to use.
                            signal_names (List[str]): The names of the signal modules to use.
                            start (str): The start date of the backtest period in "YYYY-MM-DD" format. Optional, defaults to out-of-sample period start date.
                            end (str): The end date of the backtest period in "YYYY-MM-DD" format. Optional, defaults to out-of-sample period end date.
                        Returns:
                            Dict[str, Any]: The backtest result including performance metrics and trade history.

                        
                        Some metrics to consider when evaluating backtest results:
                        - Cumulative Return (%) - Total return of the strategy over the backtest period.
                        - Sharpe Ratio - Risk-adjusted return measure.
                        - Max Drawdown (%) - Largest peak-to-trough decline in the strategy's equity curve
                        - win_rate (%) - Percentage of profitable trades.
                        - closed_trades - Total number of closed trades during the backtest period.
                        - total_commission (%) - Total commission paid as a percentage of the initial capital.
                        - excess_return_ratio (%) - Return of the strategy above the benchmark return.
                        - max_shortfall (%) - Maximum shortfall from the benchmark.
                    """
            try:
                if start is not None:
                    start_date = datetime.strptime(start, "%Y-%m-%d")
                else:
                    start_date = self.start

                if end is not None:
                    end_date = datetime.strptime(end, "%Y-%m-%d")
                else:
                    end_date = self.end

                result = run_backtest(
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    strategy_module=strategy_name,
                    signal_modules=signal_names,
                    base_dir=self.base_dir,
                    start=start_date,
                    end=end_date,
                    plot=False
                    )

                self.backtest_history_dir = self.base_dir / "backtest_history.csv"

                with open(self.backtest_history_dir, "a", newline="") as f:

                    file_exists = self.backtest_history_dir.exists()
                    writer = csv.writer(f)
                    if file_exists:
                        if self.start.date() == start_date.date() and self.end.date() == end_date.date():
                            writer.writerow([self.step]+list(result.values())+["Out-of-sample"])
                        else:
                            writer.writerow([self.step]+list(result.values())+["In-sample"])
                    else:
                        writer.writerow(["Step"]+list(result.keys())+["Type"])
                        if self.start.date() == start_date.date() and self.end.date() == end_date.date():
                            writer.writerow([self.step]+list(result.values())+["Out-of-sample"])
                        else:
                            writer.writerow([self.step]+list(result.values())+["In-sample"])
                    

                
                logger.info(f"| ✅ Backtest result is {result}")
                return {
                    "success": True,
                    "message": f"Backtest result is {result}.",
                    "extra": {},
                }
            
            except Exception as e:
                    logger.error(f"Backtest failed: {str(e)}")
                    return {
                        "success": False,
                        "message": f"Backtest failed using strategy {strategy_name} and signal {signal_names} since {str(e)}.",
                        "extra": {"error": str(e)},
                        }


    @environment_manager.action(
        name="saveStrategyEvaluation",
        description="""Save a strategy evaluation in standard format.
                    {
                        "original_hypothesis": "...",
                        "hypothesis_true": true,
                        "recommended_hypothesis": "...",
                        "best_signal_combinations": ["signal_1+signal_3"],
                        "best_strategy_name": "...",
                        "metrics": {
                            "excess_return": 0.0,
                            "information_ratio": 0.0,
                            "risk_to_reward_ratio": 0.0
                        },
                        "hypothesis_evidence": {
                            "summary": "..."
                        }
                    }

                Args:
                    evaluation (Dict[str, Any]): The strategy evaluation to save in standard format.
                Returns:
                    Dict[str, Any]: A dictionary indicating success or failure of the operation and the index of the saved evaluation if successful.
                """,
    )
    async def save_strategy_evaluation(self, evaluation: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Save a strategy evaluation in standard format.

        Expected format:
        {
            "original_hypothesis": str(in the form of equation using s1,s2....s5 to represent the signals),
            "hypothesis_true": bool,
            "recommended_hypothesis": str(in the form of equation using s1,s2....s5 to represent the signals),
            "best_signal_combinations": list[str],
            "best_strategy_name": str,
            "metrics": {
                "excess_return": float,
                "information_ratio": float,
                "risk_to_reward_ratio": float
            },
            "hypothesis_evidence": {
                "summary": str
            }
        }
        """
        try:
            if not isinstance(evaluation, dict):
                raise TypeError("evaluation must be a dict")

            required_top_fields = {
                "original_hypothesis",
                "hypothesis_true",
                "recommended_hypothesis",
                "best_signal_combinations",
                "best_strategy_name",
                "metrics",
                "hypothesis_evidence",
            }

            missing = required_top_fields - set(evaluation.keys())
            if missing:
                raise ValueError(f"evaluation missing fields: {sorted(missing)}")

            if not isinstance(evaluation["original_hypothesis"], str):
                raise TypeError("'original_hypothesis' must be a string")

            if not isinstance(evaluation["hypothesis_true"], bool):
                raise TypeError("'hypothesis_true' must be a bool")

            if not isinstance(evaluation["recommended_hypothesis"], str):
                raise TypeError("'recommended_hypothesis' must be a string")

            if not isinstance(evaluation["best_signal_combinations"], list):
                raise TypeError("'best_signal_combinations' must be a list")

            for i, item in enumerate(evaluation["best_signal_combinations"]):
                if not isinstance(item, str):
                    raise TypeError(f"best_signal_combinations[{i}] must be a string")

            if not isinstance(evaluation["best_strategy_name"], str):
                raise TypeError("'best_strategy_name' must be a string")

            if not isinstance(evaluation["metrics"], dict):
                raise TypeError("'metrics' must be a dict")

            required_metrics_fields = {
                "excess_return",
                "information_ratio",
                "risk_to_reward_ratio",
            }

            missing = required_metrics_fields - set(evaluation["metrics"].keys())
            if missing:
                raise ValueError(f"metrics missing fields: {sorted(missing)}")

            for field in required_metrics_fields:
                if not isinstance(evaluation["metrics"][field], (int, float)):
                    raise TypeError(f"metrics['{field}'] must be numeric")

            if not isinstance(evaluation["hypothesis_evidence"], dict):
                raise TypeError("'hypothesis_evidence' must be a dict")

            required_evidence_fields = {"summary"}
            missing = required_evidence_fields - set(evaluation["hypothesis_evidence"].keys())
            if missing:
                raise ValueError(f"hypothesis_evidence missing fields: {sorted(missing)}")

            if not isinstance(evaluation["hypothesis_evidence"]["summary"], str):
                raise TypeError("'hypothesis_evidence.summary' must be a string")

            self.STRATEGY_EVALUATION_STORE.append(evaluation)
            idx = len(self.STRATEGY_EVALUATION_STORE) - 1

            logger.info(f"| ✅ Saved strategy evaluation at index {idx}")

            return {
                "success": True,
                "message": f"Saved strategy evaluation at index {idx}",
                "extra": {
                    "index": idx,
                    "total": len(self.STRATEGY_EVALUATION_STORE),
                }
            }

        except Exception as e:
            logger.error(f"| ❌ Failed to save strategy evaluation: {str(e)}")

            return {
                "success": False,
                "message": f"Failed to save strategy evaluation: {str(e)}",
                "extra": {}
            }


    async def get_last_strategy_evaluation_result(self, **kwargs) -> Dict[str, Any]:
        """Get the last saved strategy evaluation result.

        Returns:
            Dict[str, Any]: The last saved strategy evaluation.
        """
        return self.STRATEGY_EVALUATION_STORE[-1]