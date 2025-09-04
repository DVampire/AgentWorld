from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from inspect import cleandoc
from langchain.tools import BaseTool, StructuredTool

from src.tools.base import ToolResponse
from src.controller.base import BaseController
from src.registry import CONTROLLERS

_TRADING_OFFLINE_ACTION_DESCRIPTION = """Trading offline tool for trading offline environment.
Use this tool to trade offline.

Available operations:
1. reset: Reset the trading environment.
2. step: Step the trading environment.
    - action: The action to take. Should be `BUY` or `SELL` or `HOLD`.

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "reset"}
Example: {"operation": "step", "action": "BUY"}
"""

class TradingOfflineActionArgs(BaseModel):
    operation: str = Field(description="The operation to execute")
    action: Optional[str] = Field(
        default=None,
        description="The action to take. Should be `BUY` or `SELL` or `HOLD`"
    )

@CONTROLLERS.register_module(force=True)
class TradingOfflineController(BaseController):
    def __init__(self, environment: Any):
        self.environment = environment

        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        
        self.state = None
        self.info = None
        self.done = None
        
    async def initialize(self):
        """Initialize the trading offline controller."""
        await self._register_tools()
        
    async def get_state(self) -> str:
        """Get the state of the trading offline controller."""
        state = cleandoc(f"""<environment_trading_offline_state>
                         Current state: {self.state['prompt']}
                         </environment_trading_offline_state>""")
        return state
    
    async def _action_tool(self, operation: str, action: Optional[str] = None) -> ToolResponse:
        
        if operation == "reset":
            try:
                state, info = await self.environment.reset()
            except Exception as e:
                return f"Error in resetting the trading environment: {str(e)}"
            
            done = info["done"]
            
            result = cleandoc(f"""Reset the trading environment successfully.
            The environment is {'done' if done else 'not done'}.
            """)
            
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
        
        elif operation == "step":
            state, reward, done, truncted, info = await self.environment.step(action)
            result = cleandoc(f"""Step the trading environment successfully.
            Action: {action}
            Reward: {reward}
            The environment is {'done' if done else 'not done'}.
            """)
            
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
        else:
            raise ValueError(f"Invalid operation: {operation}")
    
    async def _register_tools(self):
        
        # register action tool
        action_tool = StructuredTool.from_function(
            name="trading_offline_action",
            description=_TRADING_OFFLINE_ACTION_DESCRIPTION,
            coroutine=self._action_tool,
            args_schema=TradingOfflineActionArgs
        )
        action_tool_config = {
            "name": action_tool.name,
            "description": action_tool.description,
            "args_schema": action_tool.args_schema
        }
        self._tools["trading_offline_action"] = action_tool
        self._tool_configs["trading_offline_action"] = action_tool_config
        
    def list_tools(self) -> List[str]:
        """List all tools."""
        return list(self._tools.keys())
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tool_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool config by name."""
        return self._tool_configs.get(name)
    
    async def init_tools(self):
        """Initialize the tools."""
        await self.initialize()