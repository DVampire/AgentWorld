from typing import Any, Dict, Any, Dict, List, Literal, Optional, Union
from pydantic import  Field, ConfigDict
from src.logger import logger
from src.environment.server import ecp
from src.environment.types import Environment
from src.registry import ENVIRONMENT
from src.environment.quickbacktest.run import run_backtest,ClassLoader
from src.utils import assemble_project_path,parse_json_blob
from src.utils.utils import parse_code_blobs
from importlib import resources
from pathlib import Path
import shutil



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
        self.base_dir =  base_dir
        self.last_best_backtest_result: Optional[Dict[str, Any]] = None
        

    async def initialize(self) -> None:
        """Initialize the quickbacktest environment."""
        try:
            for folders in ["strategies", "signals"]:
                env_dir = Path(self.base_dir) / folders
                if not env_dir.exists():
                    env_dir.mkdir(parents=True, exist_ok=True)
                dst_1 = env_dir / "__init__.py"
                dst_1.touch(exist_ok=True)

                dst_2 = env_dir / "base_types.py"
                with resources.as_file(resources.files("quickbacktest").joinpath("base_types.py")) as src:
                    shutil.copy2(src, dst_2)

            logger.info(f"| 🚀 QuickBacktest Environment initialized at: {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize QuickBacktest Environment: {e}")
    
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
        #     logger.error(f"Failed to cleanup QuickBacktest Environment: {e}")

        pass

    @ecp.action(name="addModule",description="""Add a trading module (signal or strategy) to the environment." \
    Add a trading module (signal or strategy) to the environment.

            Args:
                module_code (str): The code of the module to add.
                module_name (str): The name of the module to add.
                module_type (Literal["signals", "strategies"]): The type of the module to add. 

        """)
    
    async def addModule(self, module_code: str, module_name: str, module_type: Literal["signals", "strategies"],**kwargs) -> None:
        """Add a trading module (signal or strategy) to the environment.

            Args:
                module_code (str): The code of the module to add.
                module_name (str): The name of the module to add.
                module_type (Literal["signals", "strategies"]): The type of the module to add. 

        """
        if module_type not in ["signals", "strategies"]:
            raise ValueError("module_type must be either 'signals' or 'strategies'")
        module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
        module_code = parse_code_blobs(module_code)
        if module_path.exists():
            raise FileExistsError(f"{module_type[:-1]} {module_name} already exists in QuickBacktest Environment.")
        with open(module_path, "w") as f:
            f.write(module_code)

        logger.info(f"| ✅ {module_type[:-1]} {module_name} added to QuickBacktest Environment.")

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
                module_code (str): The code of the module to update.
                module_name (str): The name of the module to update.
                module_type (Literal["signals", "strategies"]): The type of the module to update.”
        """)
    async def updateModule(self, module_code: str, module_name: str, module_type: Literal["signals", "strategies"], **kwargs) -> None:
        """Update a trading module (signal or strategy) in the environment.
            Args:
                module_code (str): The code of the module to update.
                module_name (str): The name of the module to update.
                module_type (Literal["signals", "strategies"]): The type of the module to update.”
        """
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
        logger.info(f"| ✅ {module_type[:-1]} {module_name} updated in QuickBacktest Environment.")

    @ecp.action(name="removeModule",description="""Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "strategies"]): The type of the module to remove.
        """)
    async def removeModule(self, module_name: str, module_type: Literal["signals", "strategies"], **kwargs) -> None:
        """Remove a trading module from the environment.
            Args:
                module_name (str): The name of the module to remove.
                module_type (Literal["signals", "strategies"]): The type of the module to remove.
        """
        if module_type not in ["signals", "strategies"]:
            raise ValueError("module_type must be either 'signals' or 'strategies'")
        module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in QuickBacktest Environment.")
        module_path.unlink()
        logger.info(f"| ✅ {module_type[:-1]} {module_name} removed from QuickBacktest Environment.")

    @ecp.action(name="listModules",description="""List all trading modules in the environment.
            Args:
                module_type (Literal["signals", "strategies"]): The type of the modules to list.

            Returns:
                Dict[str, List[str]]: A dictionary with the module type as the key and a list of module names as the value.
        """)
    async def listModules(self, module_type: Literal["signals", "strategies"], **kwargs) -> Dict[str, List[str]]:
        """List all trading modules in the environment.
            Args:
                module_type (Literal["signals", "strategies"]): The type of the modules to list.

            Returns:
                Dict[str, List[str]]: A dictionary with the module type as the key and a list of module names as the value.
        """
        if module_type not in ["signals", "strategies"]:
            raise ValueError("module_type must be either 'signals' or 'strategies'")
        env_dir = Path(self.base_dir) / module_type
        modules = {f"{module_type}": []}
        for file in env_dir.glob("*.py"):
            if file.stem not in ["__init__", "base_types"]:
                modules[f"{module_type}"].append(file.stem)
        return modules
        
    @ecp.action(name="getDocString",description="""Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "strategies"]): The type of the module to get the docstring from.

            Returns:
                str: The docstring of the module.
        """)
    async def getDocString(self, module_name: str, module_type: Literal["signals", "strategies"], **kwargs) -> str:
        """Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.
                module_type (Literal["signals", "strategies"]): The type of the module to get the docstring from.

            Returns:
                str: The docstring of the module.
        """
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
        return doc


    async def get_state(self,**kwargs) -> Dict[str, Any]:
        """Get the current state of the environment."""
        signals = await self.listModules("signals")
        strategies = await self.listModules("strategies")
        state = {
            "state": str({"signals": signals,
                    "strategies": strategies}),

            "extra":{}
        }
        return state

    @ecp.action(name="backtest",description= """Backtest a trading signal + strategy using historical data.
            Args:
                strategy_name (str): The name of the strategy module to use.
                signal_name (str): The name of the signal module to use.

            Returns:
                Dict[str, Any]: The backtest result including performance metrics and trade history.
        """)
    async def backtest(self,strategy_name:str = "AgentStrategy",signal_name: str = "AgentSignal", **kwargs) -> Dict[str, Any]:
        """Backtest a trading signal + strategy using historical data.
            Args:
                strategy_name (str): The name of the strategy module to use.
                signal_name (str): The name of the signal module to use.

            Returns:
                Dict[str, Any]: The backtest result including performance metrics and trade history.
        """
        try:
            result = run_backtest(
                data_dir = "datasets/backtest/binance",
                watermark_dir = "datasets/backtest/binance_state.duckdb",
                venue = "binance_um",
                symbol = "BTCUSDT",
                strategy_module=strategy_name,
                signal_module=signal_name,
                base_dir=self.base_dir
            )
            if result.get("cumulative_return (%)",0) > (self.last_best_backtest_result.get("cumulative_return (%)", -float('inf')) if self.last_best_backtest_result else -float('inf')):
                self.last_best_backtest_result = result
                await self.saveModule(strategy_name, "strategies")
                await self.saveModule(signal_name, "signals")

            return result
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {"error": str(e)}

        

        