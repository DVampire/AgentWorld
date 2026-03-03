import csv
from datetime import datetime
import os
from tracemalloc import start
from tracemalloc import start
from typing import Any, Dict, Any, Dict, List, Literal, Optional, Union
from pydantic import  Field, ConfigDict
from src.logger import logger
from src.environment.server import ecp
from src.environment.types import Environment
from src.registry import ENVIRONMENT
from src.environment.quickbacktest.run import run_backtest,ClassLoader,get_signal_quantile,dict_to_markdown_table
from src.environment.quickbacktest.cst_utils import patch_file,PatchConfig
from src.utils import assemble_project_path,parse_json_blob
from src.utils.utils import parse_code_blobs
from importlib import resources
from pathlib import Path
import shutil
from src.prompt import prompt_manager
from src.environment.quickbacktest.fills_analyzer import analyze_fills
from src.environment.quickbacktest.trade_analyzer import analyze_trades
from src.utils import dedent


_INTERACTION_RULES = """Interaction guidelines:
1. addModue: Use this action to add a new trading module (signal or strategy) to the environment. Provide the module code, name, and type.
2. updateModule: Use this action to update an existing trading module in the environment. Provide the updated module code, name, and type.
3. removeModule: Use this action to remove a trading module from the environment. Provide the module name and type.
4. listModules: Use this action to list all trading modules in the environment. Provide the
    module type (signals or strategies).
5. getDocString: Use this action to get the docstring of a trading module in the environment. Provide the module name and type.
6. backtest: Use this action to backtest a trading signal + strategy using historical data. Provide the strategy and signal module names.

Important !!! Limit trading times per day to avoid sky high transaction costs.!!! MAX 3 trades per day is recommended.
Your are free to rename the class name when adding or updating modules as the file name is the same as the class name, but make sure to use the correct class name when invoking them in backtests.
"""



@ENVIRONMENT.register_module(force=True)
class QuickBacktestEnvironment(Environment):
    """Quick backtest environement"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="quickbacktest", description="The name of the quickbacktest environment.")
    description: str = Field(default="Quick backtest environment for strategy backtesting", description="The description of the quickbacktest environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the quickbacktest environment including backtestresult such as sharpe ratio, annual returns.",
            "interaction_rules": _INTERACTION_RULES,
        }
    }, description="The metadata of the quickbacktest environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")


    def __init__(
        self,
        base_dir: str = "workdir/trading_strategy_agent/environment/quickbacktest",
        backtest_start_date: str = "2025-08-01",
        backtest_end_date: str = "2025-12-31",
        require_grad: bool = False,
        **kwargs: Any,
    ):
        
        super().__init__(**kwargs)
        self.base_dir =  Path(assemble_project_path(base_dir))
        self.last_best_backtest_result: Optional[Dict[str, Any]] = None
        self.last_best_strategy: Optional[str] = None
        self.last_best_signal: Optional[str] = None
        self.start = datetime.strptime(backtest_start_date, "%Y-%m-%d")
        self.end = datetime.strptime(backtest_end_date, "%Y-%m-%d")
        self.backtest_history_dir = Path(self.base_dir).parent / "backtest_history.csv"
        self.step = 0
        

    async def initialize(self) -> None:
        """Initialize the quickbacktest environment."""
        try:
            for folders in ["strategies", "signals"]:
                env_dir = Path(self.base_dir) / folders
                if not env_dir.exists():
                    env_dir.mkdir(parents=True, exist_ok=True)
                dst_1 = env_dir / "__init__.py"
                dst_1.touch(exist_ok=True)
            images_dir = self.base_dir / "images"
            if not images_dir.exists():
                images_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"| 🚀 QuickBacktest Environment initialized at: {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize QuickBacktest Environment: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup the quickbacktest environment."""
        # try:
        #     for folders in ["strategies", "signals"]:
        #         env_dir = Path(self.base_dir) / folders
        #         if env_dir.exists() and env_dir.is_dir():
        #             shutil.rmtree(env_dir)

        #     if Path(self.base_dir).exists() and Path(self.base_dir).is_dir():
        #         shutil.rmtree(Path(self.base_dir))
        #     logger.info("| 🧹 QuickBacktest Environment cleanup completed")
        # except Exception as e:
        #     logger.error(f"Failed to cleanup QuickBacktest Environment: {str(e)}")

        pass


    @ecp.action(name="addStrategy",description="""Add a strategy module to the environment.
        Args:
            strategy_code (str): The code of the strategy module to add.
            strategy_name (str): The name of the strategy module to add.
            improvement (str): The description of the improvement to apply and the reason for the improvement.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.

    """)
    async def addStrategy(self, strategy_code: str, strategy_name: str, improvement, **kwargs) -> Dict[str, Any]:
        """Add a strategy module to the environment."""
        result =  await self.addModule(module_code=strategy_code, module_name=strategy_name, module_type="strategies", **kwargs)

        if result.get("success", False):
            await self.modifymd(filename="StrategyIterations.md", insights=f"### Added strategy {strategy_name}\n\n{improvement}\n\n---\n\n", mode="append")

        return result


    @ecp.action(name="fixStrategy",description=        """Fix a strategy module in the environment.
        Args:   
            strategy_code (str): The full code of the strategy module to update.
            strategy_name (str): The name of the strategy module to update.
            bug_fix (str): The description of the bug to fix and the fix to apply.
        Returns:
            Dict[str, Any]: A dictionary indicating success or failure of the operation.
    """)
    async def fixStrategy(self, strategy_code: str, strategy_name: str, bug_fix: str, **kwargs
        ) -> Dict[str, Any]:
        """Update a strategy module in the environment."""
        result = await self.updateModule(module_code=strategy_code, module_name=strategy_name, module_type="strategies", **kwargs)
        if result.get("success", False):
            await self.modifymd(filename="StrategyIterations.md", insights=f"### Bug fix for strategy {strategy_name}\n\n{bug_fix}\n\n---\n\n", mode="append")
        return result
    

    # @ecp.action(name="addModule",description="""Add a trading module (signal or strategy) to the environment." \
    # Add a trading module (signal or strategy) to the environment.

    #         Args:
    #             module_code (str): The code of the module to add.
    #             module_name (str): The name of the module to add.
    #             module_type (Literal["signals", "strategies"]): The type of the module to add. 

    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operations.

    #     """)
    
    async def addModule(self, module_code: str, module_name: str, module_type: Literal["signals", "strategies"],**kwargs) -> Dict[str, Any]:
        """Add a trading module (signal or strategy) to the environment.

            Args:
                module_code (str): The code of the module to add.
                module_name (str): The name of the module to add.
                module_type (Literal["signals", "strategies"]): The type of the module to add. 

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation.

        """
        try:
            if module_type not in ["signals", "strategies"]:
                raise ValueError("module_type must be either 'signals' or 'strategies'")
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            module_code = parse_code_blobs(module_code)
            if module_path.exists():
                raise FileExistsError(f"{module_type[:-1]} {module_name} already exists in QuickBacktest Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)
            logger.info(f"| ✅ {module_type[:-1]} {module_name} added to QuickBacktest Environment.")
            return {"success": True, "message": f"{module_type[:-1]} {module_name} added successfully", "extra": {}}
        except Exception as e:
            logger.error(f"Failed to add {module_type[:-1]} {module_name}: {str(e)}")
            return {"success": False, "message": f"Failed to add {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}



        # if module_type == "signals":
        #     try:
        #         range_info = await self.getSignalQuantile(module_name)
        #         return {"success": True, "message": f"{module_type[:-1]} {module_name} added successfully with range {range_info}", "extra": {"signal_range": range_info}}
        #     except Exception as e:
        #         logger.warning(f"| ⚠️ Failed to compute quantile values for signal {module_name}: {str(e)}")
        #         return {"success": False, "message": f"Failed to compute quantile values for signal {module_name} since {str(e)}", "extra": {"error": str(e)}}
        # else:

        #     return {"success": True, "message": f"{module_type[:-1]} {module_name} added successfully", "extra": {}}
    




    # @ecp.action(name="saveModule",description=        """Save the current trading modules due to its excellent performance.
    #         Args:
    #             module_name (str): The name of the module to save.
    #             module_type (Literal["signals", "strategies"]): The type of the module to save.
    #         Returns:
    #             None    
    #     """)
    async def saveModule(self,module_name: str, module_type: Literal["signals", "strategies"], **kwargs) -> None:
        """Save the current trading modules due to its excellent performance."""
        if module_type not in ["signals", "strategies"]:
            raise ValueError("module_type must be either 'signals' or 'strategies'")
        module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in QuickBacktest Environment.")
        save_dir = Path(assemble_project_path("saved_modules")) / module_type
        save_dir.mkdir(parents=True, exist_ok=True)
        dst_path = save_dir / f"{module_name}.py"
        shutil.copy2(module_path, dst_path)
        logger.info(f"| ✅ {module_type[:-1]} {module_name} saved to {dst_path}.")
        
        
    # @ecp.action(name="updateModule",description=        """Update a trading module (signal or strategy) in the environment.
    #         Args:
    #             module_code (str): The full code of the module to update.
    #             module_name (str): The name of the module to update.
    #             module_type (Literal["signals", "strategies"]): The type of the module to update.”

    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation.
    #     """)
    async def updateModule(self, module_code: str, module_name: str, module_type: Literal["signals", "strategies"], **kwargs) ->Dict[str, Any]:
        """Update a trading module (signal or strategy) in the environment.
            Args:
                module_code (str): The full code of the module to update.
                module_name (str): The name of the module to update.
                module_type (Literal["signals", "strategies"]): The type of the module to update.”

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation.
        """
        try:
            if module_type not in ["signals", "strategies"]:
                raise ValueError("module_type must be either 'signals' or 'strategies'")
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            try:
                module_code = parse_code_blobs(module_code)
            except Exception as e:
                module_code = module_code

            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in QuickBacktest Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)
        except Exception as e:
            logger.error(f"Failed to update {module_type[:-1]} {module_name}: {e}")
            return {"success": False, "message": f"Failed to update {module_type[:-1]} {module_name}", "extra": {"error": str(e)}}

        logger.info(f"| ✅ {module_type[:-1]} {module_name} updated in QuickBacktest Environment.")

        if module_type == "signals":
            try:
                range_info = await self.getSignalQuantile(module_name)
                logger.info(f"| ✅ Signal {module_name} quantile values updated")
                return {"success": True, "message": f"{module_type[:-1]} {module_name} updated successfully with range {range_info}", "extra": {"singnal_range": range_info}}
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to compute quantile values for updated signal {module_name}: {str(e)}")
                return {"success": False, "message": f"Failed to compute quantile values for updated signal {module_name} since {str(e)}", "extra": {"error": str(e)}}
        else:
            return {"success": True, "message": f"{module_type[:-1]} {module_name} updated successfully", "extra": {}}

        

    @ecp.action(name="removeModule",description="""Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "strategies"]): The type of the module to remove.

            Returns:
                Dict[str,Any]: The tool state after removing the module.
        """)
    async def removeModule(self, module_name: str, module_type: Literal["signals", "strategies"], **kwargs) -> Dict[str,Any]:
        """Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "strategies"]): The type of the module to remove.
            Returns:
                Dict[str,Any]: The tool state after removing the module.
        """
        try:
            if module_type not in ["signals", "strategies"]:
                raise ValueError("module_type must be either 'signals' or 'strategies'")
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in QuickBacktest Environment.")
            module_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove {module_type[:-1]} {module_name}: {e}")
            return {"success": False, "message": f"Failed to remove {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}
    
        logger.info(f"| ✅ {module_type[:-1]} {module_name} removed from QuickBacktest Environment.")
        return {"success": True, "message": f"{module_type[:-1]} {module_name} removed successfully", "extra": {}}

    # @ecp.action(name="listModules",description="""List all trading modules in the environment.
    #         Args:
    #             module_type (Literal["signals", "strategies"]): The type of the modules to list.

    #         Returns:
    #             Dict[str, Any]: A dictionary with the module type as the key and a list of module names as the value.
    #     """)
    async def listModules(self, module_type: Literal["signals", "strategies"], **kwargs) -> Dict[str,Any]:
        """List all trading modules in the environment.
            Args:
                module_type (Literal["signals", "strategies"]): The type of the modules to list.

            Returns:
                Dict[str, Any]: A dictionary with the module type as the key and a list of module names as the value.
        """
        try:
            if module_type not in ["signals", "strategies"]:
                raise ValueError("module_type must be either 'signals' or 'strategies'")
            env_dir = Path(self.base_dir) / module_type
            modules = {f"{module_type}": []}
            for file in env_dir.glob("*.py"):
                if file.stem not in ["__init__"]:
                    modules[f"{module_type}"].append(file.stem)
            logger.info(f"| ✅ Listed {module_type}: {modules}")
            return {"success": True, "message": modules, "extra":modules}

        except Exception as e:            
            logger.error(f"Failed to list {module_type}: {e}")
            return {"success": False, "message": f"Failed to list {module_type} since {str(e)}", "extra": {"error": str(e)}}
            

    # @ecp.action(name="getSignalQuantile",description=        """Get the quantile values of a trading signal using historical data to help design strategy.
    #         Args:
    #             signal_name (str): The name of the signal module to use. The name is same as in updateModule and addModule.
    #         Returns:
    #             Dict[str, Any]: The quantile values of the trading signal.  
    #     """)

    async def getSignalQuantile(self,signal_name: str, **kwargs) -> Dict[str, Any]:
        """
            Get the quantile values of a trading signal using historical data.
            Args:
                signal_name (str): The name of the signal module to use. The name is same as in updateModule and addModule.
            Returns:
                Dict[str, Any]: The quantile values of the trading signal.

        """
        module_path = Path(self.base_dir) / "signals" / f"{signal_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Signal {signal_name} does not exist in QuickBacktest Environment.")
        result = get_signal_quantile(                
                data_dir = "datasets/backtest/binance",
                watermark_dir = "datasets/backtest/binance_state.duckdb",
                venue = "binance_um",
                symbol = "BTCUSDT",
                signal_module=signal_name,
                base_dir=self.base_dir
        )

        logger.info(f"| ✅ Signal {signal_name} quantile values computed")

        doc_config = PatchConfig(add_fields=result)
        patch_file(str(module_path), config=doc_config)
        return result

    @ecp.action(name="removePNG",description="""Remove an image from the environment.
            Args:
                img_name List[str]: The list of names of the images to remove. Don't include file extension.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation.
        """)
    async def removePNG(self, img_names:List[str], **kwargs) -> Dict[str, Any]:
        """Remove an image from the environment.
            Args:
                img_name List[str]: The list of names of the images to remove. Don't include file extension.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation.
        """
        try:
            for img_name in img_names:
                img_path = Path(self.base_dir) / "images" / f"{img_name}.png"
                if not img_path.exists():
                    raise FileNotFoundError(f"Image {img_name} does not exist in Signal Research Environment.")
                img_path.unlink()
                logger.info(f"| ✅ Image {img_name} removed from Signal Research Environment.")
            return {"success": True, "message": f"Images {img_names} removed successfully.", "extra": {}}
        except Exception as e:
            logger.error(f"Failed to remove image {img_names}: {str(e)}")
            return {"success": False, "message": f"Failed to remove image {img_names} since {str(e)}", "extra": {"error": str(e)}}



    # @ecp.action(name="addStrategyInsight",description="""Save the insights of a trading strategy to a markdown file in the environment (APPEND ONLY).
    #         Args:
    #             insights (str): The insights to save. Include data to prove. markdown format
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the saved markdown file if successful.
    #     """)
    # async def addStrategyInsight(self, insights: str,**kwargs) -> Dict[str, Any]:
    #     """Save the insights of a trading strategy to a markdown file in the environment.
    #         Args:
    #             insights (str): The insights to save. Include data to prove. markdown format
    #             mode (Literal["append", "overwrite"]): The mode to save the insights, "append" will add the insights to the end of the markdown file, while "overwrite" will replace the content of the markdown file with the insights.
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the saved markdown file if successful.
    #     """
    #     insights = f"\n\n## Insight at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + insights + "\n\n---\n\n"

    #     return await self.modifymd(filename="StrategyDesignInsights.md", insights=insights, mode="append", **kwargs)



    # @ecp.action(name="retrieveStrategyInsights",description="""Retrieve the insights of a trading strategy from a markdown file in the environment.
    #         Args:
    #             None
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the content of the markdown file if successful.
    #     """)
    # async def retrieveStrategyInsights(self, **kwargs) -> Dict[str, Any]:
    #     """Retrieve the insights of a trading strategy from a markdown file in the environment.
    #         Args:
    #             None
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the content of the markdown file if successful.
    #     """
    #     return await self.read_md(md_name="StrategyDesignInsights.md", **kwargs)



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
        

    @ecp.action(name="read_md",description="""Read a markdown file from the environment.
            Args:   
                md_name (str): The name of the markdown file to read.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the content of the markdown file if successful.
        """)

    async def read_md(self, md_name: str, **kwargs) -> Dict[str, Any]:
        """Read a markdown file from the environment.
            Args:
                md_name (str): The name of the markdown file to read.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the content of the markdown file if successful.
        """
        try:
            md_path = Path(self.base_dir).parent / md_name
            if not md_path.exists():
                raise FileNotFoundError(f"Markdown file {md_name} does not exist in Signal Research Environment.")
            with open(md_path, "r") as f:
                content = f.read()
            logger.info(f"| ✅ Markdown file {md_name} read successfully.")
            return {"success": True, "message": f"{content}.", "extra": {"content": content}}
        except Exception as e:
            logger.error(f"Failed to read markdown file {md_name}: {str(e)}")
            return {"success": False, "message": f"Failed to read markdown file {md_name} since {str(e)}", "extra": {"error": str(e)}}

    
    @ecp.action(name="getDocString",description="""Get the docstring of a trading module in the environment,including rsignal/factor range, crucial for strategy design.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "strategies"]): The type of the module to get the docstring from.

            Returns:
                Dict[str,Any]: The tool state and docstring.
        """)
    async def getDocString(self, module_name: str, module_type: Literal["signals", "strategies"], **kwargs) -> Dict[str,Any]:
        """Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "strategies"]): The type of the module to get the docstring from.

            Returns:
                Dict[str,Any]: The tool state and docstring.
        """
        try:
            if module_type not in ["signals", "strategies"]:
                raise ValueError("module_type must be either 'signals' or 'strategies'")
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


    async def get_state(self,**kwargs) -> Dict[str, Any]:
        """Get the current state of the environment."""
        signals = await self.listModules("signals")
        strategies = await self.listModules("strategies")
        self.step+=1
        diagram_path = [a for a in os.listdir(self.base_dir/"images") if a.endswith(".png")]
        state = {
            "state": str({
                    "signals": signals.get("message",[]),
                    "strategies": strategies.get("message",[]),
                    "extra_files": diagram_path},
                    ),
            "extra":{}
        }

        logger.info(f"| ✅ QuickBacktest Environment state retrieved: {state}")
        return state

    @ecp.action(name="insample_backtest",description= """Backtest a trading signal + strategy using historical data with detailed feedback.
            Args:
                strategy_name (str): The name of the strategy module to use.
                signal_name (str): The name of the signal module to use.
                rolling_window (int): The rolling window size for performance metrics calculation.
                time_periods (Dict[str,[str,str]]): The time periods to backtest in the format 
                {"bear": ["start_date", "end_date"],
                "bull": ["start_date", "end_date"],
                "sideways": ["start_date", "end_date"]
                }
                    where the dates are in "YYYY-MM-DD" format. This allows testing the strategy under different market conditions.

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
        """)
    async def insample_backtest(self,strategy_name,signal_name,time_periods:Dict[str,List[str]],**kwargs) -> Dict[str, Any]:
        """Backtest a trading signal + strategy using historical data.
            Args:
                strategy_name (str): The name of the strategy module to use.
                signal_name (str): The name of the signal module to use.
                rolling_window (int): The rolling window size for performance metrics calculation.
                start (str): The start date of the backtest period in "YYYY-MM-DD" format. Optional, defaults to the environment's start date.
                end (str): The end date of the backtest period in "YYYY-MM-DD" format. Optional, defaults to the environment's end date.
                slippage_perc (float): The slippage percentage to apply to the backtest.

            Returns:
                Dict[str, Any]: The backtest result including performance metrics and trade history.

            """
        try:
            results = {}
            if set(list(time_periods.keys())) !=set(["bear", "bull", "sideways"]):
                raise ValueError(f"Invalid time periods keys: {list(time_periods.keys())}. Must be ['bear', 'bull', 'sideways']")
            for period_name, (start, end) in time_periods.items():
                end = datetime.strptime(end, "%Y-%m-%d")
                start = datetime.strptime(start, "%Y-%m-%d")
                if self.end > end and self.start < end:
                    raise ValueError(f"Insample backtest overlap with out of sample period, please set end date before {self.start.date()}")
                result = run_backtest(
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    strategy_module=strategy_name,
                    signal_module=signal_name,
                    base_dir=self.base_dir,
                    start=start,
                    end = end,
                    slippage_perc=0.0
                )
                report_trades = analyze_trades(trades_csv="trade_logs/trades.csv",out_dir=Path(self.base_dir) / "images",initial=1.0,rolling_window=400,prefix=f"{strategy_name}_{signal_name}_{period_name}",    mark_liq=True,)
                report_fills = analyze_fills(fills_csv="trade_logs/fills.csv",out_dir=Path(self.base_dir) / "images",prefix=f"{strategy_name}_{signal_name}_{period_name}",mark_liq=True)

                results[period_name] = {"backtest_result": result}

                backtest_history_dict = result.copy()
                backtest_history_dict.update({
                        "strategy": strategy_name,
                        "signal": signal_name,
                    })
                
                backtest_history_dict.update({
                    "start": start.date(),
                    "end": end.date(),
                })

                file_exists = self.backtest_history_dir.exists()
                with open(self.backtest_history_dir, "a", newline="") as f:
                    writer = csv.writer(f)
                    if file_exists:
                        writer.writerow([self.step]+list(backtest_history_dict.values())+[period_name])
                    else:
                        writer.writerow(["Step"]+list(backtest_history_dict.keys())+["Type"])
                        writer.writerow([self.step]+list(backtest_history_dict.values())+[period_name])

                backtest_log = Path(self.base_dir).parent / f"backtest_log.md"
                content = f"""
                ## Step {self.step} Backtest Result
                ## Time Period: {period_name}
                ## Strategy: {strategy_name}
                ## Signal: {signal_name}
                ### Backtest Period: {start.date()} to {end.date()}
                ### Type: In-sample Backtest
                ### Backtest Result:
                {dict_to_markdown_table(result)}

                ### Strategy Docsting:
                ```python
                {(await self.getDocString(module_name=strategy_name, module_type="strategies"))["extra"].get("docstring","No docstring available.")}
                ```
                ### Signal Docstring:
                ```python
                {(self.base_dir / "signals" / f"{signal_name}.py").read_text() if (self.base_dir / "signals" / f"{signal_name}.py").exists() else "Signal code not found."}
                ```

                --- 
                """
                try:
                    response = await self.modifymd(filename=backtest_log.name, insights=dedent(content), mode="append")
                except Exception as e:
                    logger.warning(f"Failed to update backtest log: {e}")

            
            logger.info(f"| ✅ In-sample Backtest completed using strategy {strategy_name} and signal {signal_name} with results\n: {results}.")
            return {
                "success": True,
                "message": f"In-sample Backtest completed using strategy {strategy_name} and signal {signal_name} with results {results}.\n Trade report: {report_trades}\n Fill report: {report_fills}",
                "extra": {"results": results},
            }
            
        except Exception as e:
            logger.error(f"Failed to run insample backtest: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to run insample backtest using strategy {strategy_name} and signal {signal_name} since {str(e)}.",
                "extra": {"error": str(e)},
            }
    
    @ecp.action(name="outofsample_backtest",description= """Backtest a trading signal + strategy using historical data with limited feedback.
            Args:
                strategy_name (str): The name of the strategy module to use.
                signal_name (str): The name of the signal module to use.
            Returns:    
                Dict[str, Any]: The backtest result including performance metrics and trade history.
                """)
    async def outofsample_backtest(self,strategy_name,signal_name,**kwargs) -> Dict[str, Any]:
        """Backtest a trading signal + strategy using historical data.
                    Args:
                        strategy_name (str): The name of the strategy module to use.
                        signal_name (str): The name of the signal module to use.
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
                result = run_backtest(
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    strategy_module=strategy_name,
                    signal_module=signal_name,
                    base_dir=self.base_dir,
                    start=self.start,
                    end=self.end
                )
                backtest_history_dict = result.copy()
                backtest_history_dict.update({
                    "strategy": strategy_name,
                    "signal": signal_name,
                })
                
                file_exists = self.backtest_history_dir.exists()

                backtest_history_dict.update({
                "start": self.start.date(),
                "end": self.end.date(),
                    })

                with open(self.backtest_history_dir, "a", newline="") as f:
                    writer = csv.writer(f)
                    if file_exists:
                        writer.writerow([self.step]+list(backtest_history_dict.values())+["Type"])
                    else:
                        writer.writerow(["Step"]+list(backtest_history_dict.keys())+["Type"])
                        writer.writerow([self.step]+list(backtest_history_dict.values())+["Out-of-sample"])

                backtest_log = Path(self.base_dir).parent / f"backtest_log.md"
                content = f"""
                ## Step {self.step} Backtest Result
                ## Strategy: {strategy_name}
                ## Signal: {signal_name}
                ### Backtest Period: {self.start.date()} to {self.end.date()}
                ### Type: Out-of-sample Backtest
                ### Backtest Result:
                {dict_to_markdown_table(result)}

                ### Strategy Docsting:
                ```python
                {(await self.getDocString(module_name=strategy_name, module_type="strategies"))["extra"].get("docstring","No docstring available.")}
                ```
                ### Signal Code:
                ```python
                {(self.base_dir / "signals" / f"{signal_name}.py").read_text() if (self.base_dir / "signals" / f"{signal_name}.py").exists() else "Signal code not found."}
                ```

                --- 
                """
                try:
                    response = await self.modifymd(filename=backtest_log.name, insights=dedent(content), mode="append")
                except Exception as e:
                    logger.warning(f"Failed to update backtest log: {e}")

                
                logger.info(f"| ✅ Backtest completed using strategy {strategy_name} and signal {signal_name} with results\n: {result}.")
                return {
                    "success": True,
                    "message": f"Backtest completed using strategy {strategy_name} and signal {signal_name} with results {result}.\n",
                    "extra": {"backtest_result": result,},
                }
        except Exception as e:
                logger.error(f"Backtest failed: {str(e)}")
                return {
                    "success": False,
                    "message": f"Backtest failed using strategy {strategy_name} and signal {signal_name} since {str(e)}.",
                    "extra": {"error": str(e)},
                    }


    # @ecp.action(name="backtest",description= """Backtest a trading signal + strategy using historical data with detailed feedback.
    #         Args:
    #             strategy_name (str): The name of the strategy module to use.
    #             signal_name (str): The name of the signal module to use.
    #             rolling_window (int): The rolling window size for performance metrics calculation.
    #         Returns:
    #             Dict[str, Any]: The backtest result including performance metrics and trade history.

    #         Some metrics to consider when evaluating backtest results:
    #         - Cumulative Return (%) - Total return of the strategy over the backtest period.
    #         - Sharpe Ratio - Risk-adjusted return measure.
    #         - Max Drawdown (%) - Largest peak-to-trough decline in the strategy's equity curve
    #         - win_rate (%) - Percentage of profitable trades.
    #         - closed_trades - Total number of closed trades during the backtest period.
    #         - total_commission (%) - Total commission paid as a percentage of the initial capital.
    #         - excess_return_ratio (%) - Return of the strategy above the benchmark return.
    #         - max_shortfall (%) - Maximum shortfall from the benchmark.
    #     """)


    async def backtest(self,strategy_name:str = "AgentStrategy",signal_name: str = "AgentSignal",rolling_window: int = 50,**kwargs) -> Dict[str, Any]:
        """Backtest a trading signal + strategy using historical data.
            Args:
                strategy_name (str): The name of the strategy module to use.
                signal_name (str): The name of the signal module to use.
                rolling_window (int): The rolling window size for performance metrics calculation.
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
            result = run_backtest(
                data_dir = "datasets/backtest/binance",
                watermark_dir = "datasets/backtest/binance_state.duckdb",
                venue = "binance_um",
                symbol = "BTCUSDT",
                strategy_module=strategy_name,
                signal_module=signal_name,
                base_dir=self.base_dir,
                start=self.start,
                end=self.end
            )

            report_trades = analyze_trades(trades_csv="trade_logs/trades.csv",out_dir=Path(self.base_dir) / "images",initial=1.0,rolling_window=rolling_window,prefix=f"{strategy_name}_{signal_name}_",    mark_liq=True,)
            report_fills = analyze_fills(fills_csv="trade_logs/fills.csv",out_dir=Path(self.base_dir) / "images",prefix=f"{strategy_name}_{signal_name}_",mark_liq=True)
            backtest_history_dict = result.copy()
            backtest_history_dict.update({
                "strategy": strategy_name,
                "signal": signal_name,
            })
            
            file_exists = self.backtest_history_dir.exists()

            with open(self.backtest_history_dir, "a", newline="") as f:
                writer = csv.writer(f)
                if file_exists:
                    writer.writerow(list(backtest_history_dict.values()))
                else:
                    writer.writerow(list(backtest_history_dict.keys()))
                    writer.writerow(list(backtest_history_dict.values()))




            backtest_log = Path(self.base_dir).parent / f"backtest_log.md"
            content = f"""
            ## Step {self.step} Backtest Result
            ## Strategy: {strategy_name}
            ## Signal: {signal_name}
            ### Backtest Period: {self.start.date()} to {self.end.date()}
            ### Backtest Result:
            {dict_to_markdown_table(result)}

            ### Strategy Docsting:
            ```python
            {(await self.getDocString(module_name=strategy_name, module_type="strategies"))["extra"].get("docstring","No docstring available.")}
            ```
            ### Signal Docstring:
            ```python
            {(self.base_dir / "signals" / f"{signal_name}.py").read_text() if (self.base_dir / "signals" / f"{signal_name}.py").exists() else "Signal code not found."}
            ```

            --- 
            """
            try:
                response = await self.modifymd(filename=backtest_log.name, insights=dedent(content), mode="append")
            except Exception as e:
                logger.warning(f"Failed to update backtest log: {e}")

            
            logger.info(f"| ✅ Backtest completed using strategy {strategy_name} and signal {signal_name} with results\n: {result}.")
            return {
                "success": True,
                "message": f"Backtest completed using strategy {strategy_name} and signal {signal_name} with results {result}.\n Trade report: {report_trades}\n Fill report: {report_fills}",
                "extra": {"backtest_result": result, "trade_report": report_trades, "fill_report": report_fills},
            }
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return {
                "success": False,
                "message": f"Backtest failed using strategy {strategy_name} and signal {signal_name} since {str(e)}.",
                "extra": {"error": str(e)},
                }


        

        