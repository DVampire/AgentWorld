"""Default tool set for managing built-in tools."""

import asyncio
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool


from .bash import BashTool
from .file import FileTool
from .project import ProjectTool


class DefaultToolSet:
    """Default tool set containing built-in tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        # Note: _load_default_tools is now async, so it should be called separately
        # or the class should be used with an async factory method
    
    async def initialize(self):
        """Initialize the tool set by loading all default tools asynchronously."""
        await self._load_default_tools()
    
    async def _load_default_tools(self):
        """Load all default tools asynchronously."""
        # Load bash tool
        bash_tool = BashTool()
        self._tools["bash"] = bash_tool
        self._tool_configs["bash"] = bash_tool.get_tool_config()

        # Load file tool
        file_tool = FileTool()
        self._tools["file"] = file_tool
        self._tool_configs["file"] = file_tool.get_tool_config()

        # Load project tool
        project_tool = ProjectTool()
        self._tools["project"] = project_tool
        self._tool_configs["project"] = project_tool.get_tool_config()
    
    def list_tools(self) -> List[str]:
        """Get all default tools."""
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
        """Add a new tool to the default tool set."""
        self._tools[name] = tool
        self._tool_configs[name] = tool.get_tool_config()
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the default tool set."""
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

    async def init_tools(self):
        """Factory method to create and initialize a DefaultToolSet asynchronously."""
        await self.initialize()
