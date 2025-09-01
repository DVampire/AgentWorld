"""MCP tool set for managing local and remote MCP tools."""

from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import StructuredTool

from src.utils import assemble_project_path
from src.tools.mcp_tools.server import MCP_TOOL_ARGS
import os

class MCPToolSet:
    """Tool set for managing local and remote MCP tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        self.client = None
    
    async def initialize(self):
        """Initialize the tool set by loading all MCP tools asynchronously."""
        if self.client is None:
            # Pass environment variables and configuration to MCP server
            env = os.environ.copy()
            
            # Add any additional environment variables needed by MCP tools
            env.update({
                'PYTHONPATH': os.pathsep.join([
                    os.getcwd(),
                    os.path.join(os.getcwd(), 'src'),
                    env.get('PYTHONPATH', '')
                ])
            })
            
            self.client = MultiServerMCPClient(
                {
                    "local_mcp_server": {
                        "transport": "stdio",
                        "command": "python",
                        "args": [
                            assemble_project_path("src/tools/mcp_tools/server.py"),
                        ],
                        "env": env,  # Pass environment variables
                        "cwd": os.getcwd(),  # Set working directory
                    },
                }
            )
        await self._load_mcp_tools()
    
    async def _load_mcp_tools(self):
        """Load all default tools."""
        tools = await self.client.get_tools()
        for tool in tools:
            
            tool = StructuredTool.from_function(
                name=tool.name,
                description=tool.description,
                func=tool.func,
                coroutine=tool.coroutine,
                args_schema=MCP_TOOL_ARGS[tool.name]
            )
            
            self._tools[tool.name] = tool
            # Set default config for MCP tools
            self._tool_configs[tool.name] = {
                "name": tool.name,
                "description": getattr(tool, 'description', 'MCP tool'),
                "type": "mcp",
                "category": "mcp_tools"
            }
    
    
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
        """Factory method to create and initialize an MCPToolSet asynchronously."""
        await self.initialize()
