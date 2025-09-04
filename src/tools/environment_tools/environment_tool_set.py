"""Environment tool set for managing environment tools."""
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool

from src.config import config
from src.controller.base import BaseController

class EnvironmentToolSet:
    """Environment tool set containing environment tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        # Note: _load_environment_tools is now async, so it should be called separately
        # or the class should be used with an async factory method
    
    async def initialize(self, controllers: Optional[List[BaseController]] = None):
        """Initialize the tool set by loading all environment tools asynchronously."""
        await self._load_environment_tools(controllers)
    
    async def _load_environment_tools(self, controllers: Optional[List[BaseController]] = None):
        """Load all environment tools asynchronously."""
        if controllers is None:
            return
        for controller in controllers:
            await controller.init_tools()
            tools = controller.list_tools()
            for tool in tools:
                self._tools[tool] = controller.get_tool(tool)
                self._tool_configs[tool] = controller.get_tool_config(tool)
    
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

    async def init_tools(self, controllers: Optional[List[BaseController]] = None):
        """Factory method to create and initialize an EnvironmentToolSet asynchronously."""
        await self.initialize(controllers)
