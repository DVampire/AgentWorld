"""MCP importer tool.

Converts tools exposed by MCP servers into Autogenesis `Tool` implementations and
registers them into the TCP (`src.tool.server.tcp`) so agents can call them like native tools.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from src.dynamic import dynamic_manager
from src.logger import logger
from src.registry import TOOL
from src.tool.server import tcp
from src.tool.types import Tool, ToolExtra, ToolResponse


_MCP_IMPORTER_DESCRIPTION = """Import MCP server tools and register them into TCP.

This tool:
- Connects to one or more MCP servers (stdio / http / websocket depending on connection config)
- Discovers available tools (name/description/args schema)
- Registers one MCP proxy tool per server into TCP
- Each registered MCP proxy tool routes calls by `action` and `args`

Args:
- connections (dict): A dict mapping `server_name -> connection config`.
  The format follows `langchain_mcp_adapters.client.MultiServerMCPClient` connections, e.g.:
  {
    "math": {"command": "python", "args": ["./servers/math_server.py"], "transport": "stdio"}
  }
- server_name (Optional[str]): If provided, only import tools from this server.
- name_prefix (str): Prefix for registered TCP tool names (default: "mcp").
- override (bool): Whether to override existing TCP tool registrations.
- dry_run (bool): If True, only returns what would be registered.
"""


def _safe_literal(value: Any) -> str:
    """Return a safe Python literal representation for embedding in generated code."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if all(isinstance(x, (type(None), bool, int, float, str)) for x in value):
            return repr(list(value))
        return "None"
    if isinstance(value, dict):
        try:
            json.dumps(value)
            return repr(value)
        except Exception:
            return "None"
    return "None"


def _build_server_proxy_code(
    *,
    tcp_tool_name: str,
    tcp_tool_description: str,
    server_name: str,
    connection: Dict[str, Any],
    available_actions: List[str],
) -> str:
    """Generate a per-server MCP proxy `Tool`.

    The proxy routes calls by:
    - action: MCP tool name
    - args: dict payload for the MCP tool
    """
    class_name = re.sub(r"[^0-9a-zA-Z_]", "_", tcp_tool_name.title()).replace("_", "")
    if not class_name.endswith("Tool"):
        class_name = f"{class_name}Tool"

    connection_literal = _safe_literal(connection)
    if connection_literal == "None":
        connection_literal = "{}"

    actions_literal = _safe_literal(available_actions)
    if actions_literal == "None":
        actions_literal = "[]"

    return f'''from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from src.tool.types import Tool, ToolResponse, ToolExtra


class {class_name}(Tool):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = {tcp_tool_name!r}
    description: str = {tcp_tool_description!r}
    metadata: Dict[str, Any] = Field(default={{
        "mcp": {{
            "server_name": {server_name!r},
            "available_actions": {actions_literal},
        }}
    }})
    require_grad: bool = False

    server_name: str = Field(default={server_name!r})
    connection: Dict[str, Any] = Field(default_factory=lambda: {connection_literal})
    available_actions: List[str] = Field(default_factory=lambda: {actions_literal})

    async def __call__(self,
                       action: str,
                       args: Optional[Dict[str, Any]] = None,
                       **kwargs) -> ToolResponse:
        """
        Route a call to a tool exposed by this MCP server.

        Args:
        action: MCP tool name to call.
        args: Dict payload for the MCP tool.
        """
        try:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                from langchain_mcp_adapters.tools import load_mcp_tools
            except Exception as e:
                return ToolResponse(
                    success=False,
                    message=f"Missing MCP dependency: {{e}}. Install `langchain_mcp_adapters` to use MCP tools.",
                )

            client = MultiServerMCPClient({{
                self.server_name: self.connection
            }}, tool_name_prefix=False)

            async with client.session(self.server_name) as session:
                tools = await load_mcp_tools(
                    session,
                    server_name=self.server_name,
                    tool_name_prefix=False,
                )
                tool = next((t for t in tools if getattr(t, "name", None) == action), None)
                if tool is None:
                    available = [getattr(t, "name", None) for t in tools]
                    return ToolResponse(
                        success=False,
                        message=f"MCP action not found: {{action}} (server={{self.server_name}}). Available: {{available}}",
                    )

                payload = args or {{}}

                result = await tool.ainvoke(payload)

                if isinstance(result, (dict, list)):
                    msg = json.dumps(result, ensure_ascii=False)
                else:
                    msg = str(result)

                return ToolResponse(
                    success=True,
                    message=msg,
                    extra=ToolExtra(data={{"server": self.server_name, "action": action, "args": payload}}),
                )
        except Exception as e:
            return ToolResponse(success=False, message=f"MCP tool call failed: {{e}}")
'''


@TOOL.register_module(force=True)
class MCPImportTool(Tool):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "mcp_importer"
    description: str = _MCP_IMPORTER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")

    async def __call__(
        self,
        connections: Dict[str, Any],
        server_name: Optional[str] = None,
        name_prefix: str = "mcp",
        override: bool = True,
        dry_run: bool = False,
        **kwargs,
    ) -> ToolResponse:
        try:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
            except Exception as e:
                return ToolResponse(
                    success=False,
                    message=f"Missing MCP dependency: {e}. Install `langchain_mcp_adapters` to import MCP tools.",
                )

            if not isinstance(connections, dict) or not connections:
                return ToolResponse(success=False, message="`connections` must be a non-empty dict of server configs.")

            selected_servers = [server_name] if server_name else list(connections.keys())
            missing = [s for s in selected_servers if s not in connections]
            if missing:
                return ToolResponse(success=False, message=f"Unknown server(s) in connections: {missing}")

            client = MultiServerMCPClient(
                {s: connections[s] for s in selected_servers},
                tool_name_prefix=False,
            )

            planned: List[Dict[str, Any]] = []
            registered: List[str] = []

            for s in selected_servers:
                tools = await client.get_tools(server_name=s)
                action_names = [getattr(t, "name", None) or "" for t in tools]
                action_names = [a for a in action_names if a]
                logger.info(f"| MCP importer discovered {len(action_names)} action(s) from server '{s}'")

                tcp_tool_name = f"{name_prefix}_{s}" if name_prefix else s
                tcp_tool_name = re.sub(r"[^0-9a-zA-Z_]", "_", tcp_tool_name)

                show_n = 20
                shown = ", ".join(action_names[:show_n])
                suffix = f" (+{len(action_names) - show_n} more)" if len(action_names) > show_n else ""
                tcp_tool_description = f"[MCP:{s}] MCP server proxy. Call via action+args. Actions: {shown}{suffix}".strip()

                code = _build_server_proxy_code(
                    tcp_tool_name=tcp_tool_name,
                    tcp_tool_description=tcp_tool_description,
                    server_name=s,
                    connection=connections[s],
                    available_actions=action_names,
                )

                planned.append(
                    {
                        "server": s,
                        "tcp_tool": tcp_tool_name,
                        "description": tcp_tool_description,
                        "num_actions": len(action_names),
                    }
                )

                if dry_run:
                    continue

                class_name = dynamic_manager.extract_class_name_from_code(code)
                if not class_name:
                    raise ValueError(f"Failed to extract class name from generated MCP proxy for {tcp_tool_name}")

                tool_cls = dynamic_manager.load_class(code, class_name=class_name, base_class=Tool, context="tool")
                await tcp.register(tool_cls, config={}, override=override, code=code)
                registered.append(tcp_tool_name)

            msg = (
                f"Planned {len(planned)} MCP server tool(s); registered {len(registered)} tool(s)."
                if not dry_run
                else f"Planned {len(planned)} MCP server tool(s) (dry-run)."
            )
            return ToolResponse(
                success=True,
                message=msg,
                extra=ToolExtra(
                    data={
                        "planned": planned,
                        "registered": registered,
                        "servers": selected_servers,
                        "name_prefix": name_prefix,
                    }
                ),
            )

        except Exception as e:
            logger.error(f"| ❌ MCP importer failed: {e}")
            return ToolResponse(success=False, message=f"MCP importer failed: {e}")

