"""MCP (Model Context Protocol) tools integration."""

from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool, tool
import json
import asyncio
import aiohttp


class MCPToolManager:
    """Manager for MCP tools."""
    
    def __init__(self):
        self.mcp_servers = {}
        self.mcp_tools = {}
    
    def register_mcp_server(self, server_name: str, server_config: Dict):
        """Register an MCP server."""
        self.mcp_servers[server_name] = server_config
    
    def create_mcp_tool(self, server_name: str, tool_name: str, tool_config: Dict) -> Optional[BaseTool]:
        """Create a LangChain tool from MCP tool definition."""
        try:
            @tool
            async def mcp_tool_wrapper(**kwargs):
                """MCP tool wrapper."""
                return await self._call_mcp_tool(server_name, tool_name, kwargs)
            
            # Set tool metadata
            mcp_tool_wrapper.name = tool_name
            mcp_tool_wrapper.description = tool_config.get('description', f'MCP tool: {tool_name}')
            
            # Store the tool
            self.mcp_tools[tool_name] = {
                'server': server_name,
                'config': tool_config,
                'tool': mcp_tool_wrapper
            }
            
            return mcp_tool_wrapper
        except Exception as e:
            print(f"Failed to create MCP tool {tool_name}: {e}")
            return None
    
    async def _call_mcp_tool(self, server_name: str, tool_name: str, parameters: Dict) -> str:
        """Call an MCP tool."""
        server_config = self.mcp_servers.get(server_name)
        if not server_config:
            return f"Error: MCP server {server_name} not found"
        
        try:
            # This is a placeholder implementation
            # In a real implementation, you would use the MCP protocol to communicate with the server
            # Simulate async operation
            await asyncio.sleep(0.1)
            return f"MCP tool {tool_name} called with parameters: {parameters}"
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {str(e)}"
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all MCP tools as LangChain tools."""
        return [tool_info['tool'] for tool_info in self.mcp_tools.values()]
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a specific MCP tool."""
        tool_info = self.mcp_tools.get(tool_name)
        return tool_info['tool'] if tool_info else None
    
    async def execute_mcp_tool_concurrently(self, tool_name: str, **kwargs) -> str:
        """Execute an MCP tool concurrently."""
        tool = self.get_tool(tool_name)
        if tool:
            return await tool.ainvoke(kwargs)
        else:
            return f"MCP tool {tool_name} not found"
    
    async def execute_multiple_mcp_tools(self, tool_calls: List[Dict]) -> List[str]:
        """Execute multiple MCP tools concurrently."""
        tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            kwargs = tool_call.get("kwargs", {})
            
            task = self.execute_mcp_tool_concurrently(tool_name, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)


# Example MCP tool configurations
EXAMPLE_MCP_TOOLS = {
    "file_system": {
        "name": "file_system",
        "description": "File system operations",
        "server": "file_system_server",
        "methods": ["read", "write", "list", "delete"]
    },
    "database": {
        "name": "database",
        "description": "Database operations",
        "server": "database_server", 
        "methods": ["query", "insert", "update", "delete"]
    },
    "web_search": {
        "name": "web_search",
        "description": "Web search capabilities",
        "server": "search_server",
        "methods": ["search", "get_page_content"]
    }
}


def create_mcp_tool_from_config(tool_config: Dict, mcp_manager: MCPToolManager) -> Optional[BaseTool]:
    """Create an MCP tool from configuration."""
    server_name = tool_config.get('server')
    tool_name = tool_config.get('name')
    
    if not server_name or not tool_name:
        return None
    
    return mcp_manager.create_mcp_tool(server_name, tool_name, tool_config)


class MCPToolLoader:
    """Loader for MCP tools from configuration files."""
    
    def __init__(self, mcp_manager: MCPToolManager):
        self.mcp_manager = mcp_manager
    
    async def load_tools_from_config(self, config_path: str) -> List[BaseTool]:
        """Load MCP tools from a configuration file."""
        try:
            # Use asyncio to read file
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, lambda: open(config_path, 'r').read())
            config = json.loads(content)
            
            tools = []
            for tool_config in config.get('tools', []):
                tool = create_mcp_tool_from_config(tool_config, self.mcp_manager)
                if tool:
                    tools.append(tool)
            
            return tools
        except Exception as e:
            print(f"Error loading MCP tools from {config_path}: {e}")
            return []
    
    def load_tools_from_dict(self, tools_config: List[Dict]) -> List[BaseTool]:
        """Load MCP tools from a dictionary configuration."""
        tools = []
        for tool_config in tools_config:
            tool = create_mcp_tool_from_config(tool_config, self.mcp_manager)
            if tool:
                tools.append(tool)
        
        return tools
    
    async def load_tools_from_dict_async(self, tools_config: List[Dict]) -> List[BaseTool]:
        """Load MCP tools from a dictionary configuration asynchronously."""
        # This could be made async if tool creation becomes async
        return self.load_tools_from_dict(tools_config)


class AsyncMCPClient:
    """Async client for MCP server communication."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def call_method(self, method: str, params: Dict = None) -> Dict:
        """Call an MCP method."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }
        
        async with self.session.post(self.server_url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"MCP server error: {response.status}")
    
    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a specific tool."""
        try:
            result = await self.call_method(tool_name, kwargs)
            return result.get("result", "No result")
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"
