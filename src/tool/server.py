"""TCP Server

Server implementation for the Tool Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
from src.logger import logger
from src.config import config
from src.tool.context import ToolContextManager
from src.tool.types import Tool, ToolConfig

class TCPServer:
    """TCP Server for managing tool registration and execution with lazy loading."""
    
    def __init__(self):
        self._registered_configs: Dict[str, ToolConfig] = {}  # tool_name -> ToolConfig
        self.tool_context_manager = ToolContextManager(
            auto_discover=True,
            model_name="openrouter/text-embedding-3-large"
        )
        
    async def initialize(self, tool_names: Optional[List[str]] = None):
        """Initialize tools by names using tool context manager with concurrent support.
        
        Args:
            tool_names: List of tool names to initialize. If None, initialize all registered tools.
        """
        # Initialize tool context manager
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
    
    async def to_text(self, tool_name: str) -> str:
        """Convert tool information to string
        
        Args:
            tool_name: Tool name
            
        Returns:
            str: Tool information string
        """
        return await self.tool_context_manager.to_text(tool_name)
    
    async def to_function_call(self, tool_name: str) -> Dict[str, Any]:
        """Convert tool information to function call
        
        Args:
            tool_name: Tool name
            
        Returns:
            Dict[str, Any]: Function call
        """
        return await self.tool_context_manager.to_function_call(tool_name)
    
    async def get(self, tool_name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolConfig: Tool configuration or None if not found
        """
        return await self.tool_context_manager.get(tool_name)
    
    async def cleanup(self):
        """Cleanup all tools"""
        await self.tool_context_manager.cleanup()
        self._registered_configs.clear()


# Global TCP server instance
tcp = TCPServer()
