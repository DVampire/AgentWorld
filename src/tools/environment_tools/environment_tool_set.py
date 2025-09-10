"""Environment tool set for managing environment tools."""
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.tools import StructuredTool

from src.config import config
from src.environments import ecp
from src.tools.base import ToolResponse

class EnvironmentToolSet:
    """Environment tool set containing environment tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        # Note: _load_environment_tools is now async, so it should be called separately
        # or the class should be used with an async factory method
    
    async def initialize(self, env_names: Optional[List[str]] = None):
        """Initialize the tool set by loading all environment tools asynchronously."""
        await self._load_environment_tools(env_names)
    
    async def _load_environment_tools(self, env_names: Optional[List[str]] = None):
        """Load all environment tools asynchronously."""
        if env_names is None:
            return
        for env_name in env_names:
            actions_info = ecp.get_actions(env_name)
            for action_name, action_info in actions_info.items():
                
                async def create_wrapper(action_info, env_name):
                    async def wrapper(**kwargs)->ToolResponse:
                        try:
                            env_instance = ecp.get_environment_info(env_name).env_instance
                            res = await action_info.function(env_instance, **kwargs)
                            return ToolResponse(content=res)
                        except Exception as e:
                            return ToolResponse(content=f"Error in {action_name}: {e}")
                    return wrapper
                
                wrapper_func = await create_wrapper(action_info, env_name)
                
                tool = StructuredTool.from_function(
                    name=action_name,
                    description=action_info.description,
                    func=wrapper_func,
                    coroutine=wrapper_func,
                    args_schema=action_info.args_schema
                )
                
                tool_config = {
                    "name": action_name,
                    "description": action_info.description,
                    "args_schema": action_info.args_schema,
                    "type": env_name,
                }
                
                self._tools[action_name] = tool
                self._tool_configs[action_name] = tool_config
    
    def list_tools(self) -> List[str]:
        """Get all environment tools."""
        return list(self._tools.keys())
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def get_tool_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool configuration by name."""
        return self._tool_configs.get(name)
    
    def list_tool_names(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())
    
    def list_tools_by_category(self, category: str) -> List[BaseTool]:
        """List tools by category."""
        category_tools = []
        for name, config in self._tool_configs.items():
            if config.get("category") == category:
                tool = self._tools.get(name)
                if tool:
                    category_tools.append(tool)
        return category_tools
    
    def list_categories(self) -> List[str]:
        """List all available tool categories."""
        categories = set()
        for config in self._tool_configs.values():
            category = config.get("category")
            if category:
                categories.add(category)
        return list(categories)
    
    def add_tool(self, name: str, tool: BaseTool, config: Optional[Dict[str, Any]] = None):
        """Add a new tool to the environment tool set."""
        self._tools[name] = tool
        self._tool_configs[name] = tool.get_tool_config()
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the environment tool set."""
        if name in self._tools:
            del self._tools[name]
            del self._tool_configs[name]
            return True
        return False
    
    def get_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tools."""
        tools_info = {}
        for name, tool in self._tools.items():
            config = self._tool_configs.get(name, {})
            tools_info[name] = config
        return tools_info

    async def init_tools(self, env_names: Optional[List[str]] = None):
        """Factory method to create and initialize an EnvironmentToolSet asynchronously."""
        await self.initialize(env_names)
