from pathlib import Path
import sys

root = str(Path(__file__).resolve().parents[3])
sys.path.append(root)

from src.tools.mcp_tools.server import mcp_server, MCP_TOOL_ARGS
from src.tools.mcp_tools.browser import browser

if __name__ == "__main__":
    mcp_server.run(transport="stdio")