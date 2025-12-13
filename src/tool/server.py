"""TCP Server

Server implementation for the Tool Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
import os
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.tool.context import ToolContextManager
from src.tool.types import Tool, ToolConfig, ToolResponse
from src.utils import assemble_project_path

class TCPServer(BaseModel):
    """TCP Server for managing tool registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the TCP Server."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, ToolConfig] = {}  # tool_name -> ToolConfig

        
    async def initialize(self, tool_names: Optional[List[str]] = None):
        """Initialize tools by names using tool context manager with concurrent support.
        
        Args:
            tool_names: List of tool names to initialize. If None, initialize all registered tools.
        """
        
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "tool"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "tool.json")
        logger.info(f"| 📁 TCP Server base directory: {self.base_dir} and save path: {self.save_path}")
        
        # Initialize tool context manager
        self.tool_context_manager = ToolContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            auto_discover=True,
            model_name="openrouter/text-embedding-3-large",
        )
        await self.tool_context_manager.initialize()
        
        # Auto-discover tools if needed
        if self.tool_context_manager.auto_discover:
            await self.tool_context_manager.discover()
            self.tool_context_manager.auto_discover = False
        
        tools_to_init = tool_names if tool_names is not None else list(self.tool_context_manager._tool_configs.keys())
        
        logger.info(f"| 🛠️ Initializing {len(tools_to_init)} tools with context manager...")
        
        # Prepare initialization tasks for concurrent execution
        async def init_tool(tool_name: str):
            # Get tool config from context manager (discover() registers tools here)
            tool_config = self.tool_context_manager._tool_configs.get(tool_name)
            if tool_config is None:
                # Also check registered_configs for manually registered tools
                tool_config = self._registered_configs.get(tool_name)
                if tool_config is None:
                    logger.warning(f"| ⚠️ Tool {tool_name} not found in registered configs")
                    return
            
            # Skip if already initialized
            if tool_config.instance is not None:
                logger.debug(f"| ⏭️ Tool {tool_name} already initialized")
                return
            
            logger.debug(f"| 🔧 Initializing tool: {tool_name}")
            
            # Get tool config from global config if available
            global_config = config.get(f"{tool_name}_tool", {})
            if global_config:
                # Merge with existing config
                tool_config.config = {**tool_config.config, **global_config}
            
            # Create tool instance and store it in context manager
            try:
                await self.tool_context_manager.build(tool_config)
                # Sync to registered_configs for consistency
                self._registered_configs[tool_name] = tool_config
                logger.debug(f"| ✅ Tool {tool_name} initialized")
            except Exception as e:
                logger.error(f"| ❌ Failed to initialize tool {tool_name}: {e}")
        
        # Initialize tools concurrently
        init_tasks = [init_tool(tool_name) for tool_name in tools_to_init]
        await asyncio.gather(*init_tasks, return_exceptions=True)
        
        logger.info("| ✅ Tools initialization completed")
    
    async def register(self, tool: Union[Tool, Type[Tool]], *, override: bool = False, **kwargs: Any) -> ToolConfig:
        """Register a tool class or instance asynchronously.
        
        Args:
            tool: Tool class or instance to register
            override: Whether to override existing registration
            **kwargs: Configuration for tool initialization
            
        Returns:
            ToolConfig: Tool configuration
        """
        tool_config = await self.tool_context_manager.register(tool, override=override, **kwargs)
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def list(self, include_disabled: bool = False) -> List[str]:
        """List all registered tools
        
        Args:
            include_disabled: Whether to include disabled tools
            
        Returns:
            List[str]: List of tool names
        """
        return await self.tool_context_manager.list(include_disabled=include_disabled)
    
    
    async def get(self, tool_name: str) -> Tool:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool: Tool instance or None if not found
        """
        tool = await self.tool_context_manager.get(tool_name)
        return tool
    
    async def get_info(self, tool_name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolConfig: Tool configuration or None if not found
        """
        return await self.tool_context_manager.get_info(tool_name)
    
    async def cleanup(self):
        """Cleanup all tools"""
        await self.tool_context_manager.cleanup()
        self._registered_configs.clear()
    
    async def update(self, tool_name: str, tool: Union[Tool, Type[Tool]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> ToolConfig:
        """Update an existing tool with new configuration and create a new version
        
        Args:
            tool_name: Name of the tool to update
            tool: New tool class or instance with updated implementation
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for tool initialization
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        tool_config = await self.tool_context_manager.update(
            tool_name, tool, new_version, description, **kwargs
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def copy(self, tool_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> ToolConfig:
        """Copy an existing tool
        
        Args:
            tool_name: Name of the tool to copy
            new_name: New name for the copied tool. If None, uses original name.
            new_version: New version for the copied tool. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            ToolConfig: New tool configuration
        """
        tool_config = await self.tool_context_manager.copy(
            tool_name, new_name, new_version, **override_config
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def unregister(self, tool_name: str) -> bool:
        """Unregister a tool
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.tool_context_manager.unregister(tool_name)
        if success and tool_name in self._registered_configs:
            del self._registered_configs[tool_name]
        return success
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all tool configurations to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        return await self.tool_context_manager.save_to_json(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load tool configurations from JSON
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically initialize tools after loading
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        success = await self.tool_context_manager.load_from_json(file_path, auto_initialize)
        if success:
            # Sync registered_configs
            for tool_name in await self.tool_context_manager.list(include_disabled=True):
                tool_config = self.tool_context_manager._tool_configs.get(tool_name)
                if tool_config:
                    self._registered_configs[tool_name] = tool_config
        return success
    
    async def restore_version(self, tool_name: str, version: str, auto_initialize: bool = True) -> Optional[ToolConfig]:
        """Restore a specific version of a tool from history
        
        Args:
            tool_name: Name of the tool
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored tool
            
        Returns:
            ToolConfig of the restored version, or None if not found
        """
        tool_config = await self.tool_context_manager.restore_version(tool_name, version, auto_initialize)
        if tool_config:
            self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def __call__(self, name: str, input: Dict[str, Any]) -> ToolResponse:
        """Call a tool by name
        
        Args:
            name: Tool name
            input: Input for the tool
            
        Returns:
            ToolResponse: Tool result
        """
        return await self.tool_context_manager(name, input)


# Global TCP server instance
tcp = TCPServer()
