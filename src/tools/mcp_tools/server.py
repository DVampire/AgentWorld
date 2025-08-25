from typing import Any, Dict, Callable, Awaitable
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, ToolAnnotations
from mcp.server.fastmcp.server import Context

mcp_server = FastMCP("mcp_server")
MCP_TOOL_ARGS = {}