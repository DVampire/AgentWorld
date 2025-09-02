"""Tool manager for managing default tools and MCP tools."""

import asyncio
from typing import Dict, List, Any, Optional, Union
from langchain.tools import BaseTool

from src.tools.default_tools.default_tool_set import DefaultToolSet
from src.tools.mcp_tools.mcp_tool_set import MCPToolSet
from src.tools.environment_tools.environment_tool_set import EnvironmentToolSet
from src.logger import logger
from src.utils import Singleton
from src.config import config

class ToolManager(metaclass=Singleton):
    """Unified tool manager for managing default tools and MCP tools."""
    
    def __init__(self):
        self._default_tool_set: Optional[DefaultToolSet] = None
        self._mcp_tool_set: Optional[MCPToolSet] = None
        self._environment_tool_set: Optional[EnvironmentToolSet] = None
        self._all_tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize both default tool set and MCP tool set asynchronously."""
        if self._initialized:
            return
        
        # Initialize default tool set
        try:
            default_tool_set = DefaultToolSet()
            await default_tool_set.init_tools()
            self._default_tool_set = default_tool_set
            logger.info("| ✅ Default tool set initialized successfully")
        except Exception as e:
            logger.error(f"| ⚠️ Failed to initialize default tool set: {e}")
            self._default_tool_set = None
        
        # Initialize environment tool set
        try:
            environment_tool_set = EnvironmentToolSet()
            await environment_tool_set.init_tools()
            self._environment_tool_set = environment_tool_set
            logger.info("| ✅ Environment tool set initialized successfully")
        except Exception as e:
            logger.error(f"| ⚠️ Failed to initialize environment tool set: {e}")
            self._environment_tool_set = None
        
        # Initialize MCP tool set
        try:
            mcp_tool_set = MCPToolSet()
            await mcp_tool_set.init_tools()
            self._mcp_tool_set = mcp_tool_set
            logger.info("| ✅ MCP tool set initialized successfully")
        except Exception as e:
            logger.error(f"| ⚠️ Failed to initialize MCP tool set: {e}")
            self._mcp_tool_set = None
        
        # Merge all tools
        await self._merge_tools()
        
        self._initialized = True
        logger.info(f"| 🎉 Tool manager initialized with {len(self._all_tools)} tools")
    
    async def _merge_tools(self):
        """Merge tools from both tool sets."""
        self._all_tools.clear()
        self._tool_configs.clear()
        
        # Add default tools
        if self._default_tool_set:
            default_tools = self._default_tool_set.list_tools()
            for tool in default_tools:
                self._all_tools[tool] = self._default_tool_set.get_tool(tool)
                config = self._default_tool_set.get_tool_config(tool)
                if config:
                    self._tool_configs[tool] = config
                    
        # Add environment tools
        if self._environment_tool_set:
            environment_tools = self._environment_tool_set.list_tools()
            for tool in environment_tools:
                self._all_tools[tool] = self._environment_tool_set.get_tool(tool)
                config = self._environment_tool_set.get_tool_config(tool)
                if config:
                    self._tool_configs[tool] = config
        
        # Add MCP tools
        if self._mcp_tool_set:
            mcp_tools = self._mcp_tool_set.list_tools()
            for tool in mcp_tools:
                self._all_tools[tool] = self._mcp_tool_set.get_tool(tool)
                config = self._mcp_tool_set.get_tool_config(tool)
                if config:
                    self._tool_configs[tool] = config
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        return self._all_tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        return list(self._all_tools.keys())
    
    def list_tool_args_schemas(self) -> Dict[str, Any]:
        """List all available tool argument schemas."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        return [
            tool.args_schema for tool in self._all_tools.values()
        ]
    
    def get_tool_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool configuration by name."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        return self._tool_configs.get(name)
    
    def get_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tools."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        return self._tool_configs.copy()
    
    def list_tools_by_category(self, category: str) -> List[BaseTool]:
        """List tools by category."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        category_tools = []
        for name, config in self._tool_configs.items():
            if config.get("category") == category:
                tool = self._all_tools.get(name)
                if tool:
                    category_tools.append(tool)
        return category_tools
    
    def list_categories(self) -> List[str]:
        """List all available tool categories."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        categories = set()
        for config in self._tool_configs.values():
            category = config.get("category")
            if category:
                categories.add(category)
        return list(categories)
    
    def add_tool(self, name: str, tool: BaseTool, config: Optional[Dict[str, Any]] = None):
        """Add a new tool to the tool manager."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        self._all_tools[name] = tool
        if config:
            self._tool_configs[name] = config
        else:
            # Try to get config from tool if available
            if hasattr(tool, 'get_tool_config'):
                self._tool_configs[name] = tool.get_tool_config()
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the tool manager."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        if name in self._all_tools:
            del self._all_tools[name]
            if name in self._tool_configs:
                del self._tool_configs[name]
            return True
        return False
    
    async def execute_tool(self, name: str, args: Any, **kwargs) -> Any:
        """Execute a tool by name."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        print(args)
        
        if hasattr(tool, 'ainvoke'):
            res = await tool.ainvoke(input=args)
            return res
        else:
            return tool.invoke(input=args)
    
    async def execute_multiple_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tools concurrently."""
        if not self._initialized:
            raise RuntimeError("Tool manager not initialized. Call initialize() first.")
        
        results = []
        tasks = []
        
        for tool_call in tool_calls:
            name = tool_call.get("name")
            args = tool_call.get("args", [])
            kwargs = tool_call.get("kwargs", {})
            
            if name:
                task = self.execute_tool(name, *args, **kwargs)
                tasks.append((name, task))
        
        # Execute all tools concurrently
        for name, task in tasks:
            try:
                result = await task
                results.append({
                    "name": name,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def get_default_tool_set(self) -> Optional[DefaultToolSet]:
        """Get the default tool set."""
        return self._default_tool_set
    
    def get_mcp_tool_set(self) -> Optional[MCPToolSet]:
        """Get the MCP tool set."""
        return self._mcp_tool_set
    
    def is_initialized(self) -> bool:
        """Check if the tool manager is initialized."""
        return self._initialized
    
    async def init_tools(self):
        """Factory method to create and initialize a ToolManager asynchronously."""
        await self.initialize()


# Global tool manager instance
tool_manager = ToolManager()
