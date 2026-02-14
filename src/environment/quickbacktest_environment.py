from typing import Any, Dict, Any, Dict, List, Literal, Optional, Union
from pydantic import  Field, ConfigDict
from src.logger import logger
from src.environment.server import ecp
from src.environment.types import Environment
from src.registry import ENVIRONMENT
from src.environment.quickbacktest.run import run_backtest,ClassLoader,get_signal_quantile
from src.environment.quickbacktest.cst_utils import patch_file,PatchConfig
from src.utils import assemble_project_path,parse_json_blob
from src.utils.utils import parse_code_blobs
from importlib import resources
from pathlib import Path
import shutil
from src.prompt import prompt_manager



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
        require_grad: bool = False,
        **kwargs: Any,
    ):
        
        super().__init__(**kwargs)
        self.base_dir =  assemble_project_path(base_dir)
        self.last_best_backtest_result: Optional[Dict[str, Any]] = None
        self.last_best_strategy: Optional[str] = None
        self.last_best_signal: Optional[str] = None
        

    async def initialize(self) -> None:
        """Initialize the quickbacktest environment."""
        try:
            for folders in ["strategies", "signals"]:
                env_dir = Path(self.base_dir) / folders
                if not env_dir.exists():
                    env_dir.mkdir(parents=True, exist_ok=True)
                dst_1 = env_dir / "__init__.py"
                dst_1.touch(exist_ok=True)
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

    @ecp.action(name="addModule",description="""Add a trading module (signal or strategy) to the environment." \
    Add a trading module (signal or strategy) to the environment.

            Args:
                module_code (str): The code of the module to add.
                module_name (str): The name of the module to add.
                module_type (Literal["signals", "strategies"]): The type of the module to add. 

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information about the signal.

        """)
    
    async def addModule(self, module_code: str, module_name: str, module_type: Literal["signals", "strategies"],**kwargs) -> Dict[str, Any]:
        """Add a trading module (signal or strategy) to the environment.

            Args:
                module_code (str): The code of the module to add.
                module_name (str): The name of the module to add.
                module_type (Literal["signals", "strategies"]): The type of the module to add. 

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information about the signal.

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
        except Exception as e:
            logger.error(f"Failed to add {module_type[:-1]} {module_name}: {str(e)}")
            return {"success": False, "message": f"Failed to add {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}
        logger.info(f"| ✅ {module_type[:-1]} {module_name} added to QuickBacktest Environment.")


        if module_type == "signals":
            try:
                range_info = await self.getSignalQuantile(module_name)
                return {"success": True, "message": f"{module_type[:-1]} {module_name} added successfully with range {range_info}", "extra": {"signal_range": range_info}}
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to compute quantile values for signal {module_name}: {str(e)}")
                return {"success": False, "message": f"Failed to compute quantile values for signal {module_name} since {str(e)}", "extra": {"error": str(e)}}
        else:

            return {"success": True, "message": f"{module_type[:-1]} {module_name} added successfully", "extra": {}}
    




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
        
        
    @ecp.action(name="updateModule",description=        """Update a trading module (signal or strategy) in the environment.
            Args:
                module_code (str): The full code of the module to update.
                module_name (str): The name of the module to update.
                module_type (Literal["signals", "strategies"]): The type of the module to update.”

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information if update module is signal.
        """)
    async def updateModule(self, module_code: str, module_name: str, module_type: Literal["signals", "strategies"], **kwargs) ->Dict[str, Any]:
        """Update a trading module (signal or strategy) in the environment.
            Args:
                module_code (str): The full code of the module to update.
                module_name (str): The name of the module to update.
                module_type (Literal["signals", "strategies"]): The type of the module to update.”

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information if update module is signal.
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
        state = {
            "state": str({
                    "signals": signals.get("message",[]),
                    "strategies": strategies.get("message",[]),
                    "last_best_backtest_result": self.last_best_backtest_result,
                    "current_best_strategy": self.last_best_strategy,
                    "current_best_signal": self.last_best_signal},
                    ),
            "extra":{}
        }

        logger.info(f"| ✅ QuickBacktest Environment state retrieved: {state}")
        return state

    @ecp.action(name="backtest",description= """Backtest a trading signal + strategy using historical data.
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
        """)
    async def backtest(self,strategy_name:str = "AgentStrategy",signal_name: str = "AgentSignal", **kwargs) -> Dict[str, Any]:
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
            )
            if result.get("cumulative_return (%)",0) > (self.last_best_backtest_result.get("cumulative_return (%)", -float('inf')) if self.last_best_backtest_result else -float('inf')):
                self.last_best_backtest_result = result
                self.last_best_signal = signal_name
                self.last_best_strategy = strategy_name


                logger.info(f"| 🚀 New best backtest result achieved: {result.get('cumulative_return (%)',0)}")
                if float(result.get('cumulative_return (%)',0))>0:
                    await self.saveModule(strategy_name, "strategies")
                    await self.saveModule(signal_name, "signals")

            logger.info(f"| ✅ Backtest completed using strategy {strategy_name} and signal {signal_name} with results\n: {result}.")
            return {
                "success": True,
                "message": f"Backtest completed using strategy {strategy_name} and signal {signal_name} with results {result}.",
                "extra": {"backtest_result": result},
            }
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return {
                "success": False,
                "message": f"Backtest failed using strategy {strategy_name} and signal {signal_name} since {str(e)}.",
                "extra": {"error": str(e)},
                }

        

        