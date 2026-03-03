import csv
import json
import os
from typing import Any, Dict, Any, Dict, List, Literal, Optional, Union
from unittest import result
from matplotlib.pylab import roll
from pydantic import  Field, ConfigDict
from src.logger import logger
from src.environment.server import ecp
from src.environment.types import Environment
from src.registry import ENVIRONMENT
from src.environment.quickbacktest.run import get_signal_quantile,ClassLoader
from src.utils import dedent, assemble_project_path
from src.utils.utils import parse_code_blobs
from src.environment.quickbacktest.cst_utils import patch_file,PatchConfig,extract_first_json_object
import src.environment.quickbacktest.signal_research as s
from pathlib import Path
import shutil
from datetime import datetime
from src.prompt import prompt_manager
import inspect


@ENVIRONMENT.register_module(force=True)
class SignalResearchEnvironment(Environment):
    """Quick backtest environement"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="signal_research", description="The name of the research environment.")
    description: str = Field(default="Signal research environment for strategy backtesting", description="The description of the signal research environment.")
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
        **kwargs: Any,
    ):
        
        super().__init__(**kwargs)
        self.base_dir =  Path(assemble_project_path(base_dir))
        logger.info(f"| 🚀 Signal Research Environment initialized at: {self.base_dir}")
        self.tools = [
    name 
    for name, func in inspect.getmembers(s, inspect.isfunction)
    if func.__module__ == s.__name__ and name[0]!="_"
]
        self.step = 0


    async def initialize(self) -> None:
        """Initialize the signal research environment."""
        try:
            for folders in ["signals"]:
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

    @ecp.action(name="addSignal",description="""Add a signal to the environment." \
    Add a trading signal to the research environment.

            Args:
                module_code (str): The code of the signal to add.
                module_name (str): The name of the signal to add.
                improvement (str): A description of the improvement will be included in this signal

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information about the signal.

        """)
    
    async def addSignal(self, module_code: str, module_name: str, improvement: str,**kwargs) -> Dict[str, Any]:
        """Add a trading signal to the research environment.

            Args:
                module_code (str): The code of the signal to add.
                module_name (str): The name of the signal to add.
                improvement (str): A description of the improvement to be included in this signal.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation of the signal.

        """
        try:

            module_path = Path(self.base_dir) / "signals" / f"{module_name}.py"
            module_code = parse_code_blobs(module_code)
            if module_path.exists():
                raise FileExistsError(f"signal {module_name} already exists in Signal Research Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)

            try:
                signal_report = s._signal_analyzer(
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    signal_module=module_name,
                    base_dir=self.base_dir,
                    start = datetime.strptime("2024-01-01 00:00", "%Y-%m-%d %H:%M"),
                    end = datetime.strptime("2025-01-01 00:00", "%Y-%m-%d %H:%M"),
                )


                quantiles = await self.getSignalQuantile(module_name=module_name,start="2024-01-01 00:00",end="2025-01-01 00:00",**kwargs)
                signal_report["quantiles"] = quantiles.get("extra", {}).get("result", {})

                await self.modifymd(filename="SignalIterations.md", insights=f"### Add {module_name}: improvement made are {improvement}\n\n", mode="append", **kwargs)
            except Exception as e:
                logger.error(f"Failed to analyze signal {module_name} after adding: {str(e)}")
                await self.removeSignal(module_name=module_name)
                return {"success": False, "message": f"Signal {module_name} added but failed to analyze the signal since {str(e)} thus signal removed", "extra": {"error": str(e)}}

            return {"success": True, "message": f"Signal {module_name} added successfully to Signal Research Environment with initial report {signal_report}.", "extra": {"report": signal_report}}
        except Exception as e:
            logger.error(f"Failed to add signal {module_name}: {str(e)}")
            return {"success": False, "message": f"Failed to add signal {module_name} since {str(e)}", "extra": {"error": str(e)}}
    
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
        
        
    @ecp.action(name="fixSignal",description="""Fix a trading signal in the research environment.
            Args:
                module_code (str): The full code of the signal to update.
                module_name (str): The name of the signal to update.
                bug_fix (str): A description of the bug fix to be made to the signal.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation if update module is signal.
        """)
    async def fixSignal(self, module_code: str, module_name: str, bug_fix:str,**kwargs) ->Dict[str, Any]:
        """Update a trading signal in the research environment.
            Args:
                module_code (str): The full code of the signal to update.
                module_name (str): The name of the signal to update.
                bug_fix (str): A description of the bug fix to be made to the signal.

            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the range information if update module is signal.
        """
        try:
            module_path = Path(self.base_dir) / "signals" / f"{module_name}.py"
            try:
                module_code = parse_code_blobs(module_code)
            except Exception as e:
                module_code = module_code

            if not module_path.exists():
                raise FileNotFoundError(f"Signal {module_name} does not exist in SignalResearch Environment.")
            with open(module_path, "w") as f:
                f.write(module_code)

            try:
                signal_report = s._signal_analyzer(
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    signal_module=module_name,
                    base_dir=self.base_dir,
                    start = datetime.strptime("2024-01-01 00:00", "%Y-%m-%d %H:%M"),
                    end = datetime.strptime("2025-01-01 00:00", "%Y-%m-%d %H:%M"),
                )
                quantiles = await self.getSignalQuantile(module_name=module_name,start="2024-01-01 00:00",end="2025-01-01 00:00",**kwargs)
                signal_report["quantiles"] = quantiles.get("extra", {}).get("result", {})

                await self.modifymd(filename="SignalIterations.md", insights=f"### Fix {module_name}: fix bug {bug_fix}\n\n", mode="append", **kwargs)

            except Exception as e:
                logger.error(f"Failed to analyze signal {module_name} after update: {str(e)}")
                await self.removeSignal(module_name=module_name)
                return {"success": False, "message": f"Signal {module_name} updated but failed to analyze the signal since {str(e)} thus signal removed", "extra": {"error": str(e)}}



            return {"success": True, "message": f"Signal {module_name} updated successfully in SignalResearch Environment with initial report {signal_report}", "extra": signal_report}
        except Exception as e:
            logger.error(f"Failed to update signal {module_name}: {e}")
            return {"success": False, "message": f"Failed to update signal {module_name}", "extra": {"error": str(e)}}

            
        

    @ecp.action(name="removeSignal",description="""Remove a trading signal from the environment.
            Args:
                module_name (str): The name of the signal to remove.

            Returns:
                Dict[str,Any]: The tool state after removing the signal.
        """)
    async def removeSignal(self, module_name: str, **kwargs) -> Dict[str,Any]:
        """Remove a trading signal from the environment.
            Args:
                module_name (str): The name of the signal to remove.
            Returns:
                Dict[str,Any]: The tool state after removing the signal.
        """
        try:
            module_type = "signals"
            module_path = Path(self.base_dir) / module_type / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"{module_type[:-1]} {module_name} does not exist in QuickBacktest Environment.")
            module_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove {module_type[:-1]} {module_name}: {e}")
            return {"success": False, "message": f"Failed to remove {module_type[:-1]} {module_name} since {str(e)}", "extra": {"error": str(e)}}
    
        logger.info(f"| ✅ {module_type[:-1]} {module_name} removed from QuickBacktest Environment.")
        return {"success": True, "message": f"{module_type[:-1]} {module_name} removed successfully", "extra": {}}

    # @ecp.action(name="listSignals",description="""List all trading signals in the environment.
    #         Args:
    #             module_type (Literal["signals", "strategies"]): The type of the modules to list.

    #         Returns:
    #             Dict[str, Any]: A dictionary with the module type as the key and a list of module names as the value.
    #     """)
    async def listSignals(self, **kwargs) -> Dict[str,Any]:
        """List all trading signals in the environment.
            Args:
                module_type (Literal["signals", "strategies"]): The type of the modules to list.

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
            return {"success": True, "message": modules, "extra":modules}

        except Exception as e:            
            logger.error(f"Failed to list {module_type}: {e}")
            return {"success": False, "message": f"Failed to list {module_type} since {str(e)}", "extra": {"error": str(e)}}
            

    @ecp.action(name="getSignalQuantile",description=        """Get the quantile values of a trading signal using historical data to help design strategy.
            Args:
                module_name (str): The name of the signal module to use.
                start (str): The start date for the historical data, in the form of "YYYY-MM-DD HH:MM".
                end (str): The end date for the historical data, in the form of "YYYY-MM-DD HH:MM".
            Returns:
                Dict[str, Any]: The quantile values of the trading signal.  
        """)

    async def getSignalQuantile(self,module_name: str, start:str,end:str,**kwargs) -> Dict[str, Any]:
        """
            Get the quantile values of a trading signal using historical data.
            Args:
                module_name (str): The name of the signal module to use. The name is same as in updateModule and addModule.
                start (str): The start date for the historical data, in the form of "YYYY-MM-DD HH:MM".
                end (str): The end date for the historical data, in the form of "YYYY-MM-DD HH:MM".
            Returns:
                Dict[str, Any]: The quantile values of the trading signal.

        """
        try:
            module_path = Path(self.base_dir) / "signals" / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"Signal {module_name} does not exist in QuickBacktest Environment.")
            result = get_signal_quantile(                
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    signal_module=module_name,
                    base_dir=self.base_dir,
                    start = datetime.strptime(start, "%Y-%m-%d %H:%M"),
                    end = datetime.strptime(end, "%Y-%m-%d %H:%M"),
            )

            logger.info(f"| ✅ Signal {module_name} quantile values computed")

            doc_config = PatchConfig(add_fields=result)
            patch_file(str(module_path), config=doc_config)

            module = ClassLoader.load_class(
                file_path=module_path,
                class_name=module_name,
            )
            doc = module.__doc__ if module.__doc__ else "No docstring available."
            del module

            signal_information = json.loads(extract_first_json_object(doc).json_text)
            signal_log_dir = Path(self.base_dir).parent / "signals.csv"
            file_exists = signal_log_dir.exists()
            if signal_information is not None:
                with open(signal_log_dir, "a",newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["step","signal","explanation","mean","std","hit_rate"])
                    for info in list(signal_information.values())[:-1]:
                        writer.writerow([self.step, info["name"], info["explanation"],info["range"]["mean"],info["range"]["std"],info["range"]["hit_rate"]])
            return {"success": True, "message": f"Quantile values for signal {module_name} computed successfully with result {result}.", "extra": {"result": result}}
        except Exception as e:
            logger.error(f"Failed to compute quantile values for signal {module_name}: {str(e)}")
            return {"success": False, "message": f"Failed to compute quantile values for signal {module_name} since {str(e)}", "extra": {"error": str(e)}}


    @ecp.action(name="getSignalDocString",description="""Get the docstring of a signal module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.

            Returns:
                Dict[str,Any]: The tool state and docstring.
        """)
    async def getSignalDocString(self, module_name: str, **kwargs) -> Dict[str,Any]:
        """Get the docstring of a trading module in the environment.
            Args:
                module_name (str): The name of the module to get the docstring from.

            Returns:
                Dict[str,Any]: The tool state and docstring.
        """
        try:
            module_type = "signals"
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

    
    async def listanalysisToolcases(self, **kwargs) -> Dict[str, Any]:
        """List all available toolcases for signal analysis in the environment.
            Returns:
                Dict[str, Any]: A dictionary containing the list of available toolcases.
        """
        toolcases = self.tools
        logger.info(f"| ✅ Listed available toolcases for signal analysis: {toolcases}")
        return {"success": True, "message": f"Listed available toolcases for signal analysis: {toolcases}", "extra": {"toolcases": toolcases}}


    @ecp.action(name="analysisToolcase",description="""Run a specific toolcase for signal analysis to get the result and insights. All analysis are based on 1m data.
            Args:
                tool_name (str): The name of the toolcase to run. The toolcases are defined in src.environment.quickbacktest.signal_research.
                signal_module (str): The name of the signal module to use in the toolcase.
                rolling_window (int): The rolling window size for the toolcase analysis.
                horizon (int): The horizon for the toolcase analysis.
                start (str): The start date for the historical data, in the form of "YYYY-MM-DD HH:MM".
                end (str): The end date for the historical data, in the form of "YYYY-MM-DD HH:MM".
                use_close_smoothing (bool): Whether to use close price smoothing in the analysis.(False at default)
                ic_rolling_window (Optional[int]): The rolling window size for IC curve analysis, only used when the toolcase is "get_rolling_ic_curve" at least 600.
                horizons (Optional[list]): The list of horizons for the toolcase analysis, only used when the toolcase is "get_spearman_correlation".
                img_name (str): The name of the image to save the analysis result if the toolcase generates images. Don't include file extension.
                factor (Optional[str]): The factor to plot in the toolcase analysis, only used when the toolcase is "get_rolling_ic_curve", can be "signal" or "factor1" or "factor2", default to "signal".
            Returns:
                Dict[str, Any]: The result and insights from the toolcase analysis.
        """)
    async def analysis_toolcase(self,tool_name,signal_module,rolling_window,start,end,img_name,horizon=5,ic_rolling_window: Optional[int]=None,horizons:Optional[list]=None,use_close_smoothing=False,factor: Optional[str]="signal",**kwargs) -> Dict[str, Any]:
        
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Tool {tool_name} is not available in Signal Research Environment.")
            elif tool_name == "get_ic_curve":
                if horizons is None:
                    raise ValueError("horizons must be provided for get_ic_curve toolcase.")
                result = s.get_ic_curve(
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    signal_module=signal_module,
                    base_dir=self.base_dir,
                    start = datetime.strptime(start, "%Y-%m-%d %H:%M"),
                    end = datetime.strptime(end, "%Y-%m-%d %H:%M"),
                    horizons = horizons,
                    use_smoothing=use_close_smoothing,
                    rolling_window=rolling_window,
                    img_name=img_name,
                )
                logger.info(f"| ✅ Toolcase analysis for {tool_name} completed.")
                return {"success": True, "message": f"Toolcase analysis for {tool_name} completed with result {result}.", "extra": {"result": result}}
            elif tool_name == "get_rolling_ic_curve":
                if ic_rolling_window is None:
                    raise ValueError("ic_rolling_window must be provided for get_rolling_ic_curve toolcase.")
                result = s.get_rolling_ic_curve(
                    data_dir = "datasets/backtest/binance",
                    watermark_dir = "datasets/backtest/binance_state.duckdb",
                    venue = "binance_um",
                    symbol = "BTCUSDT",
                    signal_module=signal_module,
                    base_dir=self.base_dir,
                    start = datetime.strptime(start, "%Y-%m-%d %H:%M"),
                    end = datetime.strptime(end, "%Y-%m-%d %H:%M"),
                    ic_window=ic_rolling_window,
                    use_smoothing=use_close_smoothing,
                    horizon=horizon,
                    rolling_window=rolling_window,
                    img_name=img_name,
                    factor=factor,
                )
                logger.info(f"| ✅ Toolcase analysis for {tool_name} completed.")
                return {"success": True, "message": f"Toolcase analysis for {tool_name} completed with result {result}.", "extra": {"result": result}}

            func = getattr(s, tool_name)
            result = func(
                data_dir = "datasets/backtest/binance",
                watermark_dir = "datasets/backtest/binance_state.duckdb",
                venue = "binance_um",
                symbol = "BTCUSDT",
                signal_module=signal_module,
                base_dir=self.base_dir,
                start = datetime.strptime(start, "%Y-%m-%d %H:%M"),
                end = datetime.strptime(end, "%Y-%m-%d %H:%M"),
                rolling_window=rolling_window,
                horizon=horizon,
                use_smoothing=use_close_smoothing,
                img_name=img_name,
            )
            logger.info(f"| ✅ Toolcase analysis for {tool_name} completed.")
            return {"success": True, "message": f"Toolcase analysis for {tool_name} completed with result {result}.", "extra": {"result": result}}
        except Exception as e:
            logger.error(f"Failed to run toolcase analysis for {tool_name}: {str(e)}")
            return {"success": False, "message": f"Failed to run toolcase analysis for {tool_name} since {str(e)}", "extra": {"error": str(e)}} 
        
    # @ecp.action(name="addSignalLog",description="""Save the insights of a trading signal to a markdown file in the environment. (APPEND ONLY)
    #         Args:
    #             insights (str): The insights to save. Include data to prove. markdown format
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the saved markdown file if successful.
    #     """)
    # async def addSignalLog(self, insights: str,**kwargs) -> Dict[str, Any]:
    #     """Save the insights of a trading signal to a markdown file in the environment.
    #         Args:
    #             filename (str): The name of the markdown file to save the insights to.
    #             insights (str): The insights to save. Include data to prove.
    #             mode (Literal["append", "overwrite"]): The mode to save the insights, "append" will add the insights to the end of the markdown file, while "overwrite" will replace the content of the markdown file with the insights.
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the saved markdown file if successful.
    #     """
    #     insights = f"## Insight at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + insights + "\n---\n"

    #     return await self.modifymd(filename="SignalDesignInsights.md", insights=insights, mode="append", **kwargs)


    # @ecp.action(name="retrieveSignalInsight",description="""Retrieve the insights of a trading signal from a markdown file in the environment.
    #         Args:
    #             None
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to
    #             the saved markdown file if successful.
    #     """)
    # async def retrieveSignalInsights(self, **kwargs) -> Dict[str, Any]:
    #     """Retrieve the insights of a trading signal from a markdown file in the environment.
    #         Args:
    #             None
    #         Returns:
    #             Dict[str, Any]: A dictionary indicating success or failure of the operation and the content of the markdown file if successful.
    #     """
    #     return await self.read_md(md_name="SignalDesignInsights.md", **kwargs)
    
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

    @ecp.action(name="readMD",description="""Read a markdown file from the environment.
            Args:
                md_name (str): The name of the markdown file to read.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the content of
                the markdown file if successful.
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

    @ecp.action(name="transferSignal",description="""Transfer the trading signal to a strategy module for backtesting.
            Args:
                module_name (str): The name of the signal module to transfer. The name is same as in updateModule and addModule.
                target_env (str): The target environment to transfer the signal to, default to "quick_backtest"
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the transferred signal module if successful.
        """)
    async def transferSignal(self, module_name: str, target_env = "quick_backtest",**kwargs) -> Dict[str, Any]:
        """Transfer the trading signal to a strategy module for backtesting.
            Args:
                module_name (str): The name of the signal module to transfer. The name is same as in updateModule and addModule.
            Returns:
                Dict[str, Any]: A dictionary indicating success or failure of the operation and the path to the transferred strategy module if successful.
        """
        try:
            signal_module_path = Path(self.base_dir) / "signals" / f"{module_name}.py"
            if not signal_module_path.exists():
                raise FileNotFoundError(f"Signal {module_name} does not exist in Signal Research Environment.")
            target_path = Path(self.base_dir).parent / target_env /"signals" /f"{module_name}.py"
            if not target_path.parent.exists():
                raise FileNotFoundError(f"Target environment {target_env} does not exist or does not have signals directory.")
            shutil.copy2(signal_module_path, target_path)
            logger.info(f"| ✅ Signal {module_name} transferred to target environment {target_env} successfully.")
            return {"success": True, "message": f"Signal {module_name} transferred to target environment {target_env} successfully.", "extra": {"target_path": str(target_path)}}
        except Exception as e:
            logger.error(f"Failed to transfer signal {module_name} to target environment {target_env}: {str(e)}")
            return {"success": False, "message": f"Failed to transfer signal {module_name} to target environment {target_env} since {str(e)}", "extra": {"error": str(e)}}

    async def get_state(self,**kwargs) -> Dict[str, Any]:
        """Get the current state of the environment."""
        signals = await self.listSignals()
        diagram_path = [a for a in os.listdir(self.base_dir/"images") if a.endswith(".png")]
        self.step+=1
        state = {
            "state": str({
                    "signals": signals.get("message",[]),
                    "signal_analysis_tools": self.tools,
                    "extra_files": diagram_path}),
            "extra":{}
        }


        logger.info(f"| ✅ Signal Research Environment state retrieved: {state}")
        return state
        