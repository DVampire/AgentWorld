import csv
import json
import os
from typing import Any, Dict, Any, Dict, List, Literal, Optional, Union
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
from src.utils.args_utils import parse_tool_args
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
class SignalEvaluateEnvironment(Environment):
    """Quick backtest environement"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="signal_evaluate", description="The name of the evaluation environment.")
    description: str = Field(default="Signal evaluation environment for strategy testing", description="The description of the signal evaluation environment.")
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
        logger.info(f"| 🚀 Signal Research Environment initialized at: {self.base_dir}")

        self.start = datetime.strptime(backtest_start_date, "%Y-%m-%d")
        self.end = datetime.strptime(backtest_end_date, "%Y-%m-%d")
        self.tools = [
    name 
    for name, func in inspect.getmembers(s, inspect.isfunction)
    if func.__module__ == s.__name__ and name[0]!="_"
]
        self.step = 0
        self.EVALUATION_STORE = []


    async def initialize(self) -> None:
        """Initialize the signal research environment."""
        try:
            for folders in ["signals","evaluations","strategies"]:
                env_dir = Path(self.base_dir) / folders
                if not env_dir.exists():
                    env_dir.mkdir(parents=True, exist_ok=True)
                dst_1 = env_dir / "__init__.py"
                dst_1.touch(exist_ok=True)
            images_dir = self.base_dir / "images"
            if not images_dir.exists():
                images_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"| 🚀 Signal Research Environment initialized at: {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize Signal Research Environment: {str(e)}")
    
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

    @environment_manager.action(name="addEvaluation",description="""Add an evaluation to the environment." \
    Add a trading evaluation to the research environment.

            Args:
                module_code (str): The code of the evaluation to add.
                module_name (str): The name of the evaluation to add.
                hypothesis (str): A description of the hypothesis will be included in this evaluation

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information about the signal.

        """)
    
    async def addEvaluation(self, module_code: str, module_name: str, hypothesis: str,**kwargs) -> Dict[str, Any]:
        """Add a trading benchmark to the research environment.

            Args:
                module_code (str): The code of the evaluation to add.
                module_name (str): The name of the evaluation to add.
                hypothesis (str): A description of the hypothesis to be included in this evaluation.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation of the benchmark.

        """
        try:

            module_path = Path(self.base_dir) / "evaluations" / f"{module_name}.py"

            module_code = parse_code_blobs(module_code)
            if module_path.exists():
                raise FileExistsError(f"evaluation {module_name} already exists in Signal Evaluate Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)
            await self.modifymd(filename = "BenchmarkIterations.md", insights = f"## Benchmark: {module_name}\n\n### Hypothesis:\n{hypothesis}", mode="append")
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
                raise FileNotFoundError(f"Evaluation {module_name} does not exist in SignalEvaluate Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)
                
            await self.modifymd(filename = "BenchmarkIterations.md", insights = f"## Evaluation: {module_name}\n\n### Bug Fix:\n{bug_fix}", mode="append")
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
        

    @environment_manager.action(name="removeEvaluation",description="""Remove a evaluation from the environment.
            Args:
                module_name (str): The name of the benchmark to remove.

            Returns:
                Dict[str,Any]: The tool state after removing the signal.
        """)
    async def removeEvaluation(self, module_name: str,**kwargs) -> Dict[str,Any]:
        """Remove a trading evaluation from the environment.
            Args:
                module_name (str): The name of the evaluation to remove.
                Type (Literal["signal", "strategy"]): The type of the evaluation to remove.
            Returns:
                Dict[str,Any]: The tool state after removing the evaluation.
        """
        try:
            module_type = "evaluations"
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in Signal Evaluate Environment.")
            module_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove {module_type[:-1]} {module_name}: {e}")
            return {"success": False, "message": f"Failed to remove {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}
    
        logger.info(f"| ✅ {module_type[:-1]} {module_name} removed from Signal Evaluate Environment.")
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

    async def listSignalEvaluation(self, **kwargs) -> Dict[str,Any]:
        """List all signal evaluations in the environment.

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
    

    @environment_manager.action(name="runSignalEvaluation",description="""Run a specified signal evaluation in the environment.
            Args:
                evaluation_name (str): The name of the signal evaluation to run.
                signal_names List[str]: The name of the signal to use for the evaluation, which should already exist in the environment. Order matters, they will be assigned as signal_1 ... 5
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the result of
                the signal evaluation.
        """)    
    async def runSignalEvaluation(self, evaluation_name: str,signal_names:List[str], **kwargs) -> Dict[str,Any]:
        """Run a specified signal evaluation in the environment.

            Args:
                evaluation_name (str): The name of the signal evaluation to run.
                signal_names List[str]: The names of the signals to use for the evaluation.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the result of the signal evaluation.
        """
        try:
            start_date = self.start
            end_date = self.end

            run_signal_evaluation_result = run_signal_evaluation(
                data_dir = "datasets/backtest/binance",
                watermark_dir = "datasets/backtest/binance_state.duckdb",
                venue = "binance_um",
                symbol = "BTCUSDT",
                base_dir=self.base_dir,
                start = start_date,
                end = end_date,
                evaluation_module=evaluation_name,
                signal_module=signal_names)

        except Exception as e:
            logger.error(f"Failed to run evaluation {evaluation_name}: {e}")
            return {"success": False, "message": f"Failed to run evaluation {evaluation_name} since {str(e)}", "extra": {"error": str(e)}}

        return {"success": True, "message": f"Evaluation {evaluation_name} executed successfully with result {run_signal_evaluation_result}", "extra": run_signal_evaluation_result}




    # @environment_manager.action(name="transferSignaltoBacktest",description="""Transfer the trading signal to a quick backtest environment for simulation.
    #         Args:
    #             module_names (List[str]): The names of the signal modules to transfer. The name is same as in updateModule and addModule.
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the transferred signal module if successful.
    #     """)
    # async def transferSignaltoBacktest(self, module_names: List[str], target_env: Literal["quick_backtest", "signal_evaluate"] = "quick_backtest", **kwargs) -> Dict[str, Any]:
    #     """Transfer the trading signal to a signal evaluate for hypothesis testing.
    #         Args:
    #             module_names (List[str]): The names of the signal modules to transfer. The name is same as in updateModule and addModule.
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the transferred signal module if successful.
    #     """
    #     try:
    #         for module_name in module_names:
    #             signal_module_path = Path(self.base_dir) / "signals" / f"{module_name}.py"
    #             if not signal_module_path.exists():
    #                 raise FileNotFoundError(f"Signal {module_name} does not exist in Signal Research Environment.")
    #             target_path = Path(self.base_dir).parent / target_env /"signals" /f"{module_name}.py"
    #             if not target_path.parent.exists():
    #                 raise FileNotFoundError(f"Target environment {target_env} does not exist or does not have signals directory.")
    #         shutil.copy2(signal_module_path, target_path)
    #         logger.info(f"| ✅ Signal {module_name} transferred to target environment {target_env} successfully.")
    #         return {"success": True, "message": f"Signal {module_name} transferred to target environment {target_env} successfully.", "extra": {"target_path": str(target_path)}}
    #     except Exception as e:
    #         logger.error(f"Failed to transfer signal {module_name} to target environment {target_env}: {str(e)}")
    #         return {"success": False, "message": f"Failed to transfer signal {module_name} to target environment {target_env} since {str(e)}", "extra": {"error": str(e)}}


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
                    "signal_evaluations": await self.listSignalEvaluation(),
                    "extra_files": diagram_path}),
            "extra":{}
        }


        logger.info(f"| ✅ Signal Research Environment state retrieved: {state}")
        return state
    
    @environment_manager.action(name="getDocString",description="""Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "evaluations"]): The type of the module to get the docstring from.
            Returns:
                Dict[str,Any]: The tool state and docstring.
        """)
    
    async def getDocString(self, module_name: str, module_type: Literal["signals", "evaluations"], **kwargs) -> Dict[str,Any]:
        """Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "evaluations"]): The type of the module to get the docstring from.

            Returns:
                Dict[str,Any]: The tool state and docstring.
        """
        try:
            if module_type not in ["signals", "evaluations"]:
                raise ValueError("module_type must be either 'signals' or 'evaluations'")
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in QuickBacktest Environment.")
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

    # @environment_manager.action(name="addStrategyDesignAdvise",description="""Add strategy design advise to the signal file.
    #         Args:
    #             signal_name (str): The name of the signal to add the strategy design advise to. The signal must exist in the environment.
    #             strategy_design (str): The strategy design advise to add. Include data to prove.
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to
    #             the signal docstring file if successful.
    #     """)
    # async def addStrategyDesignAdvise(self, signal_name: str, strategy_design: str, **kwargs) -> Dict[str, Any]:
    #     """Add strategy design advise to the environment.
    #         Args:
    #             signal_name (str): The name of the signal to add the strategy design advise to. The signal must exist in the environment.
    #             strategy_design (str): The strategy design advise to add. Include data to prove.
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the signal docstring file if successful.
    #     """
    #     try:
    #         module_path = Path(self.base_dir) / "signals" / f"{signal_name}.py"
    #         if not module_path.exists():
    #             raise FileNotFoundError(f"Signal {signal_name} does not exist in Signal Research Environment.")
    #         advice = {"strategy_design_advise": strategy_design}
    #         doc_config = PatchConfig(add_fields=advice)
    #         patch_file(str(module_path), config=doc_config)

    #         return {"success": True, "message": f"Strategy design advise added to signal {signal_name} successfully.", "extra": {}}
        
    #     except Exception as e:
    #         logger.error(f"Failed to add strategy design advise to {signal_name}: {str(e)}")
    #         return {"success": False, "message": f"Failed to add strategy design advise to {signal_name} since {str(e)}", "extra": {"error": str(e)}}


    @environment_manager.action(name="removeModule",description="""Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "signal_benchmarks", "strategy_benchmarks"]): The type of the module to remove.

            Returns:
                Dict[str,Any]: The tool state after removing the module.
        """)
    async def removeModule(self, module_name: str, module_type: Literal["signals", "evaluations"], **kwargs) -> Dict[str,Any]:
        """Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "evaluations","strategies"]): The type of the module to remove.
            Returns:
                Dict[str,Any]: The tool state after removing the module.
        """
        try:
            if module_type not in ["signals", "evaluations"]:
                raise ValueError("module_type must be either 'signals' or 'evaluations'")
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in signal_evaluate Environment.")
            module_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove {module_type[:-1]} {module_name}: {e}")
            return {"success": False, "message": f"Failed to remove {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}
    
        logger.info(f"| ✅ {module_type[:-1]} {module_name} removed from signal_evaluate Environment.")
        return {"success": True, "message": f"{module_type[:-1]} {module_name} removed successfully", "extra": {}}



    @environment_manager.action(name="saveSignalEvaluation",description="""Save a signal evaluation in standard format.
                {
                "signals": [
                    {
                    "name": "...",
                    "original_hypothesis": "...",
                    "hypothesis_true": true,
                    "recommended_hypothesis": "...",
                    "evidence": "..."
                    }
                ],
                "combo": {
                    "original_hypothesis": "...(in the form of equation using s1,s2....s5 to represent the signals)",
                    "hypothesis_true": true,
                    "recommended_hypothesis": "...",
                    "rank_ic": 0.0
                }
                }
                
            Args:
                evaluation (Dict[str, Any]): The signal evaluation to save in standard format.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the index of the saved evaluation if successful.
                
                """)
    
    async def save_signal_evaluation(self, evaluation: (Dict[str, Any]), **kwargs) -> Dict[str, Any]:
        """
        Save a signal evaluation in standard format.

        Expected format:
        {
            "signals": [...],
            "combo": {...}
        }
        """

        


        try:

            # evaluation = parse_tool_args(evaluation)
            # ===== 基础校验 =====
            if not isinstance(evaluation, dict):
                raise TypeError("evaluation must be a dict")

            if "signals" not in evaluation or "combo" not in evaluation:
                raise ValueError("evaluation must contain 'signals' and 'combo'")

            if not isinstance(evaluation["signals"], list):
                raise TypeError("'signals' must be a list")

            if not isinstance(evaluation["combo"], dict):
                raise TypeError("'combo' must be a dict")

            # ===== 结构校验（signal级别）=====
            required_signal_fields = {
                "name",
                "original_hypothesis",
                "hypothesis_true",
                "recommended_hypothesis",
                "evidence",
            }

            for i, signal in enumerate(evaluation["signals"]):
                if not isinstance(signal, dict):
                    raise TypeError(f"signals[{i}] must be a dict")

                missing = required_signal_fields - set(signal.keys())
                if missing:
                    raise ValueError(f"signals[{i}] missing fields: {sorted(missing)}")

            # ===== combo校验 =====
            required_combo_fields = {
                "original_hypothesis",
                "hypothesis_true",
                "recommended_hypothesis",
                "rank_ic",
            }

            missing = required_combo_fields - set(evaluation["combo"].keys())
            if missing:
                raise ValueError(f"combo missing fields: {sorted(missing)}")

            # ===== 保存 =====
            self.EVALUATION_STORE.append(evaluation)
            idx = len(self.EVALUATION_STORE) - 1

            logger.info(f"| ✅ Saved signal evaluation at index {idx}")

            return {
                "success": True,
                "message": f"Saved signal evaluation at index {idx}",
                "extra": {
                    "index": idx,
                    "total": len(self.EVALUATION_STORE)
                }
            }

        except Exception as e:
            logger.error(f"| ❌ Failed to save signal evaluation: {str(e)}")

            return {
                "success": False,
                "message": f"Failed to save signal evaluation: {str(e)}",
                "extra": {}
            }
        
    async def get_last_evaluation_result(self, **kwargs) -> Dict[str, Any]:
        """Get the history of all saved signal evaluations.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation and the list of all saved evaluations if successful.
        """
        return self.EVALUATION_STORE[-1]