"""Trading offline tool for trading offline environment."""

import asyncio
from typing import Optional, Dict, Any, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import config
from src.registry import ENVIRONMENTS, DATASETS


_TRADING_OFFLINE_TOOL_DESCRIPTION = """Trading offline tool for trading offline environment.
Use this tool to trade offline.

Available operations:
1. RESET: Reset the trading environment.
2. STEP: Step the trading environment.
    - action: The action to take. Should be `BUY` or `SELL` or `HOLD`.

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "reset"}
Example: {"operation": "step", "action": "BUY"}
"""

class TradingOfflineToolArgs(BaseModel):
    operation: str = Field(description="The operation to execute")
    action: Optional[str] = Field(description="The action to take. Should be `BUY` or `SELL` or `HOLD`")

class TradingOfflineTool(BaseTool):
    """Trading offline tool for trading offline environment."""

    name: str = "trading_offline"
    description: str = _TRADING_OFFLINE_TOOL_DESCRIPTION
    args_schema: Type[TradingOfflineToolArgs] = TradingOfflineToolArgs
    
    dataset: Any = Field(description="The dataset to use")
    environment: Any = Field(description="The environment to use")
    state: Any = Field(description="The state of the trading environment")
    info: Any = Field(description="The info of the trading environment")
    done: bool = Field(description="The done flag of the trading environment")

    def __init__(self, dataset: Optional[Any] = None, 
                 environment: Optional[Any] = None, 
                 state: Optional[Any] = None, 
                 info: Optional[Any] = None,
                 done: Optional[bool] = False, 
                 **kwargs):
        super().__init__(
            dataset=dataset,
            environment=environment,
            state=state,
            info=info,
            done=done,
            **kwargs
        )
        
        if dataset is None:
            dataset_config = config.trading_offline_tool.dataset
            self.dataset = DATASETS.build(dataset_config)
        else:
            self.dataset = dataset
        if environment is None:
            environment_config = config.trading_offline_tool.environment
            environment_config.update(
                dataset=self.dataset,
            )
            self.environment = ENVIRONMENTS.build(environment_config)
        else:
            self.environment = environment
            
    def _reset(self) -> str:
        self.state, self.info = self.environment.reset()
        self.done = self.info["done"]
        res = ("Reset the trading environment successfully.",
               f"The environment is {'done' if self.done else 'not done'}.",
               f"The state is:\n{self.state['prompt']}")
        return res
        
    def _step(self, action: str) -> str:
        next_state, reward, done, truncted, info = self.environment.step(action)
        self.state = next_state
        self.info = info
        self.done = done
        res = ("Step the trading environment successfully.",
               f"The environment is {'done' if self.done else 'not done'}.",
               f"The state is:\n{self.state['prompt']}")
        return res
    
    async def _arun(self, operation: str, action: Optional[str] = None) -> str:
        if operation == "reset":
            return self._reset()
        elif operation == "step":
            return self._step(action) 
        else:
            raise ValueError(f"Invalid operation: {operation}")

    def _run(self, operation: str, action: Optional[str] = None) -> str:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(operation, action))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in trading offline tool: {str(e)}"
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema
        }