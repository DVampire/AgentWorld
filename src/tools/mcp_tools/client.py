"""MCP Client for managing MCP server connections and tools."""

import logging
import os
import asyncio
from typing import Dict, Optional, Any, Type
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.tools.protocol.tool import BaseTool
from src.utils import assemble_project_path
from src.tools.protocol import tcp
from src.logger import logger

# Configure logging
logging.getLogger("mcp.os.posix.utilities").setLevel(logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

class MCPTool(BaseTool):
    """MCP Tool for managing MCP server connections and tools."""
    
    name: str = Field(description="The name of the MCP Tool")
    type: str = Field(description="The type of the MCP Tool")
    description: str = Field(description="The description of the MCP Tool")
    args_schema: Type[BaseModel] = Field(description="The args schema of the MCP Tool")
    metadata: Dict[str, Any] = Field(description="The metadata of the MCP Tool")
    
    tool: BaseTool = Field(description="The tool of the MCP Tool")
    
    def __init__(self, tool: BaseTool, **kwargs):
        """Initialize MCP Tool."""
        super().__init__(tool=tool, **kwargs)
        self.name = tool.name
        self.type = "MCP Tool"
        self.description = tool.description
        self.args_schema = tool.args_schema
        self.metadata = tool.metadata
        self.tool = tool
        
    def _run(self, **kwargs):
        return self.tool._run(**kwargs)
    
    async def _arun(self, **kwargs):
        return await self.tool._arun(**kwargs)


class MCPClient:
    """MCP Client for managing MCP server connections and tools."""
    
    def __init__(self, 
                 servers_config: Optional[Dict[str, Dict[str, Any]]] = None,
                 default_env: Optional[Dict[str, str]] = None):
        """Initialize MCP client.
        
        Args:
            servers_config: Configuration for MCP servers
            default_env: Default environment variables for MCP servers
        """
        self.servers_config = servers_config or self._get_default_servers_config()
        self.default_env = default_env or self._get_default_env()
        self.client: Optional[MultiServerMCPClient] = None
        self._initialized = False
        
        self.initialize()
    
    def _get_default_env(self) -> Dict[str, str]:
        """Get default environment variables for MCP servers.
        
        Returns:
            Dict[str, str]: Default environment variables
        """
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': os.pathsep.join([
                os.getcwd(),
                os.path.join(os.getcwd(), 'src'),
                env.get('PYTHONPATH', '')
            ]),
            # Add process management environment variables
            'MCP_PROCESS_GROUP_MANAGEMENT': '1',
            'MCP_TERMINATE_TIMEOUT': '5.0',
            'MCP_KILL_TIMEOUT': '2.0',
            # Add signal handling preferences
            'MCP_SIGNAL_HANDLING': 'graceful',
            # Reduce warning verbosity
            'MCP_LOG_LEVEL': 'WARNING'
        })
        return env
    
    def _get_default_servers_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default MCP servers configuration.
        
        Returns:
            Dict[str, Dict[str, Any]]: Default servers configuration
        """
        return {
            "local_mcp_server": {
                "transport": "stdio",
                "command": "python",
                "args": [
                    assemble_project_path("src/tools/mcp_tools/server.py"),
                ],
                "env": self._get_default_env(),
                "cwd": os.getcwd(),
            },
        }
    
    def initialize(self) -> None:
        """Initialize the MCP client with configured servers."""
        if self._initialized:
            return
        
        self.client = MultiServerMCPClient(self.servers_config)
        self._initialized = True
        
    def register_tools(self) -> None:
        """Register the tools from the MCP client."""
        if not self._initialized:
            self.initialize()
            
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tools = loop.run_until_complete(self.client.get_tools())
                for tool in tools:
                    tool = MCPTool(tool=tool)
                    tcp.tool(tool=tool)
                    logger.info(f"Registered MCP tool: {tool.name}")
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")

mcp_client = MCPClient()
mcp_client.register_tools()
