"""Tool manager for managing all tools."""

import os
import json
from typing import Dict, List, Any, Optional, Union
from langchain.tools import BaseTool
from pathlib import Path

from src.utils import Singleton
from src.tools.custom_tools import CustomToolSet
from src.tools.mcp_tools import MCPToolManager, MCPToolLoader, EXAMPLE_MCP_TOOLS


class ToolManager(metaclass=Singleton):
    """Manager for all tools in the system."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_configs: Dict[str, Dict] = {}
        self.custom_tool_set = CustomToolSet()
        self.mcp_tool_manager = MCPToolManager()
        self.mcp_tool_loader = MCPToolLoader(self.mcp_tool_manager)
        self._load_default_tools()
    
    def _load_default_tools(self):
        """Load default tools."""
        # Load custom tools
        custom_tools = self.custom_tool_set.get_tools()
        for tool in custom_tools:
            self.add_tool(tool.name, tool, {
                "type": "custom",
                "description": tool.description,
                "source": "CustomToolSet"
            })
        
        # Load MCP tools
        mcp_tools = self.mcp_tool_loader.load_tools_from_dict(list(EXAMPLE_MCP_TOOLS.values()))
        for tool in mcp_tools:
            self.add_tool(tool.name, tool, {
                "type": "mcp",
                "description": tool.description,
                "source": "MCPToolManager"
            })
    
    def add_tool(self, name: str, tool: BaseTool, config: Optional[Dict] = None):
        """Add a tool to the manager."""
        self.tools[name] = tool
        self.tool_configs[name] = config or {
            "type": "custom",
            "description": getattr(tool, 'description', 'No description'),
            "source": "manual"
        }
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the manager."""
        if name in self.tools:
            del self.tools[name]
            del self.tool_configs[name]
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_tool_config(self, name: str) -> Optional[Dict]:
        """Get tool configuration by name."""
        return self.tool_configs.get(name)
    
    def list_tools(self, tools_name: List[str] = None) -> List[BaseTool]:
        """List all available tools."""
        if tools_name is None:
            return list(self.tools.values())
        else:
            return [self.tools[name] for name in tools_name if name in self.tools]
    
    def list_tool_names(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())
    
    def list_tools_by_type(self, tool_type: str) -> List[str]:
        """List tools by type (custom, mcp, etc.)."""
        return [name for name, config in self.tool_configs.items() if config.get("type") == tool_type]
    
    def list_custom_tools(self) -> List[str]:
        """List all custom tools."""
        return self.list_tools_by_type("custom")
    
    def list_mcp_tools(self) -> List[str]:
        """List all MCP tools."""
        return self.list_tools_by_type("mcp")
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category (file, web, calculation, etc.)."""
        category_tools = []
        for name, tool in self.tools.items():
            tool_name = name.lower()
            if category.lower() in tool_name or category.lower() in tool.description.lower():
                category_tools.append(tool)
        return category_tools
    
    def get_file_tools(self) -> List[BaseTool]:
        """Get all file-related tools."""
        return self.get_tools_by_category("file")
    
    def get_web_tools(self) -> List[BaseTool]:
        """Get all web-related tools."""
        return self.get_tools_by_category("web")
    
    def get_calculation_tools(self) -> List[BaseTool]:
        """Get all calculation-related tools."""
        return self.get_custom_tool_set().get_tools_by_category("calculation")
    
    def get_custom_tool_set(self) -> CustomToolSet:
        """Get the custom tool set."""
        return self.custom_tool_set
    
    def get_mcp_tool_manager(self) -> MCPToolManager:
        """Get the MCP tool manager."""
        return self.mcp_tool_manager
    
    def add_custom_tool(self, tool: BaseTool):
        """Add a custom tool."""
        self.custom_tool_set.add_tool(tool)
        self.add_tool(tool.name, tool, {
            "type": "custom",
            "description": tool.description,
            "source": "CustomToolSet"
        })
    
    def add_mcp_tool(self, server_name: str, tool_name: str, tool_config: Dict):
        """Add an MCP tool."""
        tool = self.mcp_tool_manager.create_mcp_tool(server_name, tool_name, tool_config)
        if tool:
            self.add_tool(tool.name, tool, {
                "type": "mcp",
                "description": tool.description,
                "source": "MCPToolManager",
                "server": server_name
            })
            return tool
        return None
    
    def register_mcp_server(self, server_name: str, server_config: Dict):
        """Register an MCP server."""
        self.mcp_tool_manager.register_mcp_server(server_name, server_config)
    
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> str:
        """Execute a tool asynchronously."""
        tool = self.get_tool(tool_name)
        if tool:
            try:
                result = await tool.ainvoke(*args, **kwargs)
                return result
            except Exception as e:
                return f"Error executing tool {tool_name}: {str(e)}"
        else:
            return f"Tool {tool_name} not found"
    
    async def execute_multiple_tools(self, tool_calls: List[Dict]) -> List[str]:
        """Execute multiple tools concurrently."""
        tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            args = tool_call.get("args", [])
            kwargs = tool_call.get("kwargs", {})
            
            task = self.execute_tool(tool_name, *args, **kwargs)
            tasks.append(task)
        
        import asyncio
        return await asyncio.gather(*tasks)
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        tool = self.get_tool(name)
        config = self.get_tool_config(name)
        
        if tool and config:
            return {
                "name": name,
                "description": tool.description,
                "type": config.get("type"),
                "source": config.get("source"),
                "args_schema": getattr(tool, 'args_schema', None),
                "return_direct": getattr(tool, 'return_direct', False)
            }
        return None
    
    def validate_tool(self, name: str) -> bool:
        """Validate that a tool exists and is properly configured."""
        tool = self.get_tool(name)
        if not tool:
            return False
        
        # Check if tool has required attributes
        if not hasattr(tool, 'name') or not hasattr(tool, 'description'):
            return False
        
        return True
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get all available tools grouped by type."""
        return {
            "custom": self.list_custom_tools(),
            "mcp": self.list_mcp_tools(),
            "file": [tool.name for tool in self.get_file_tools()],
            "web": [tool.name for tool in self.get_web_tools()],
            "all": self.list_tools()
        }
    
    def export_tools(self, file_path: str):
        """Export all tool configurations to a file."""
        export_data = {
            "tools": {},
            "configs": self.tool_configs
        }
        
        for name, tool in self.tools.items():
            export_data["tools"][name] = {
                "type": self.tool_configs[name].get("type"),
                "description": tool.description,
                "source": self.tool_configs[name].get("source")
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def import_tools(self, file_path: str):
        """Import tools from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        # Note: This is a simplified import that only loads configurations
        # Actual tool recreation would require more complex logic
        self.tool_configs.update(import_data.get("configs", {}))
    
    def reload_tools(self):
        """Reload all tools."""
        self.tools.clear()
        self.tool_configs.clear()
        self._load_default_tools()
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get statistics about all tools."""
        total_tools = len(self.tools)
        custom_tools = len(self.list_custom_tools())
        mcp_tools = len(self.list_mcp_tools())
        
        valid_tools = sum(1 for name in self.tools.keys() if self.validate_tool(name))
        
        return {
            "total_tools": total_tools,
            "custom_tools": custom_tools,
            "mcp_tools": mcp_tools,
            "valid_tools": valid_tools,
            "invalid_tools": total_tools - valid_tools
        }
    
    def search_tools(self, query: str) -> List[str]:
        """Search tools by name or description."""
        query_lower = query.lower()
        matching_tools = []
        
        for name, tool in self.tools.items():
            if (query_lower in name.lower() or 
                query_lower in tool.description.lower()):
                matching_tools.append(name)
        
        return matching_tools
    
tool_manager = ToolManager()
