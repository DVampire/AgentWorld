"""Tool Context Manager for managing tool lifecycle and resources."""

import atexit
from typing import Any, Dict, Optional, Callable
from src.logger import logger

class ToolContext:
    """Context manager for individual tool lifecycle."""
    
    def __init__(self, tool_name: str, tool_factory: Callable, tool_manager: 'ToolContextManager'):
        """Initialize tool context.
        
        Args:
            tool_name: Name of the tool
            tool_factory: Function to create the tool instance
            tool_manager: Reference to the tool context manager
        """
        self.tool_name = tool_name
        self.tool_factory = tool_factory
        self.tool_manager = tool_manager
        self.tool_instance: Optional[Any] = None
    
    async def __aenter__(self) -> Any:
        """Async context manager entry - create tool."""
        logger.debug(f"| 🔧 Creating tool: {self.tool_name}")
        
        # Create tool instance
        self.tool_instance = await self.tool_factory()
        
        # Store in manager
        self.tool_manager._tool_instances[self.tool_name] = self.tool_instance
        
        logger.debug(f"| ✅ Tool ready: {self.tool_name}")
        return self.tool_instance
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup tool."""
        if self.tool_instance:
            logger.debug(f"| 🧹 Cleaning up tool: {self.tool_name}")
            
            self.tool_manager._tool_instances.pop(self.tool_name, None)
            
            logger.debug(f"| ✅ Tool cleaned up: {self.tool_name}")
    
    def __enter__(self) -> Any:
        """Sync context manager entry - create tool."""
        logger.debug(f"| 🔧 Creating tool (sync): {self.tool_name}")
        
        # Create tool instance
        self.tool_instance = self.tool_factory()
        
        # Store in manager
        self.tool_manager._tool_instances[self.tool_name] = self.tool_instance
        
        logger.debug(f"| ✅ Tool ready (sync): {self.tool_name}")
        return self.tool_instance
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit - cleanup tool."""
        if self.tool_instance:
            logger.debug(f"| 🧹 Cleaning up tool (sync): {self.tool_name}")
            
            # Remove from manager
            self.tool_manager._tool_instances.pop(self.tool_name, None)
            
            logger.debug(f"| ✅ Tool cleaned up (sync): {self.tool_name}")

class ToolContextManager:
    """Global context manager for all tools."""
    
    def __init__(self):
        """Initialize the tool context manager."""
        self._tool_instances: Dict[str, Any] = {}
        self._cleanup_registered = False
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
            
    def invoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke a tool.
        
        Args:
            tool_name: Name of the tool
            input: Input for the tool
            **kwargs: Keyword arguments for the tool
        """
        
        if name in self._tool_instances:
            instance = self._tool_instances[name]
            return instance.invoke(input, **kwargs)
        else:
            raise ValueError(f"Tool {name} not found")
    
    async def ainvoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke a tool.
        
        Args:
            tool_name: Name of the tool
            input: Input for the tool
            **kwargs: Keyword arguments for the tool
        """
        if name in self._tool_instances:
            instance = self._tool_instances[name]
            return await instance.ainvoke(input, **kwargs)
        else:
            raise ValueError(f"Tool {name} not found")
    
    def tool_context(self, tool_name: str, tool_factory: Callable) -> ToolContext:
        """Create a tool context for managing tool lifecycle.
        
        Args:
            tool_name: Name of the tool
            tool_factory: Function to create the tool instance
            
        Returns:
            ToolContext instance
        """
        tool_context = ToolContext(tool_name, tool_factory, self)
        return tool_context
    
    def get(self, tool_name: str) -> Any: # type: ignore
        """Get a tool instance.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance
        """
        # If tool is already active, return existing instance
        if tool_name in self._tool_instances:
            return self._tool_instances[tool_name]
        else:
            raise ValueError(f"Tool {tool_name} not found")
    
    def create_tool(self, tool_name: str, tool_factory: Callable) -> Any:
        """Create a tool instance and store it.
        
        Args:
            tool_name: Name of the tool
            tool_factory: Function to create the tool instance
            
        Returns:
            Tool instance
        """
        if tool_name in self._tool_instances:
            return self._tool_instances[tool_name]
        
        # Create new tool instance
        try:
            tool_instance = tool_factory()
            self._tool_instances[tool_name] = tool_instance
            logger.debug(f"| 🔧 Tool {tool_name} created and stored")
            return tool_instance
        except Exception as e:
            logger.error(f"| ❌ Failed to create tool {tool_name}: {e}")
            raise
    
    def cleanup(self):
        """Cleanup all active tools."""
        logger.info("| 🧹 Cleaning up all tools...")
        
        # Get list of active tool names to avoid dict modification during iteration
        active_tools = list(self._tool_instances.keys())
        
        for tool_name in active_tools:
            self._tool_instances.pop(tool_name, None)
        
        logger.info("| ✅ All tools cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
