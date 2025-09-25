"""TCP Server

Server implementation for the Tool Context Protocol.
"""
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel

from src.logger import logger
from src.config import config
from src.tools.protocol.tool import BaseTool, WrappedTool
from src.tools.protocol.types import ToolInfo
from src.tools.protocol.context import ToolContextManager

class TCPServer:
    """TCP Server for managing tool registration and execution"""
    
    def __init__(self):
        self._registered_tools: Dict[str, ToolInfo] = {}  # tool_name -> ToolInfo
        self.tool_context_manager = ToolContextManager()
    
    def tool(self, tool: Union[WrappedTool, Type[BaseTool]] = None):
        """Register a tool class or tool instance with decorator support"""
        # If tool_instance is provided, register it directly
        if tool is not None:
            return self._register_tool_instance(tool=tool)
        
        # Otherwise, return a decorator for class registration
        def decorator(cls: Type[BaseTool]):
            return self._register_tool_class(cls)
        return decorator
    
    def _register_tool_class(self, 
                           cls: Type[BaseTool]):
        """Register a tool class
        
        Args:
            cls: Tool class to register
        """
        # Get class-level attributes for registration
        model_fields = cls.model_fields
        
        name = model_fields['name'].default
        description = model_fields['description'].default
        args_schema = model_fields['args_schema'].default
        metadata = model_fields['metadata'].default
        type = model_fields['type'].default
        
        # Create ToolInfo with lazy instance creation
        tool_info = ToolInfo(
            name=name,
            type=type,
            description=description,
            args_schema=args_schema,
            metadata=metadata if metadata is not None else {},
            cls=cls,  # Store the class for lazy instantiation
            instance=None
        )
        
        self._registered_tools[name] = tool_info
        
        return cls
    
    def _register_tool_instance(self, tool: WrappedTool):
        """Register a tool instance directly
        
        Args:
            tool: Tool instance to register
        """
        name = tool.name
        type = tool.type
        description = tool.description
        args_schema = tool.args_schema
        metadata = tool.metadata
        
        # Create ToolInfo and store it
        tool_info = ToolInfo(
            name=name,
            type=type,
            description=description,
            args_schema=args_schema,
            metadata=metadata if metadata is not None else {},
            cls=WrappedTool,
            instance=tool
        )
        
        self._registered_tools[name] = tool_info
    
    async def initialize(self):
        """Initialize tools by names using tool context manager
        
        Args:
            env_names: List of environment names
        """
        logger.info(f"| 🛠️ Initializing {len(self._registered_tools)} tools with context manager...")
        
        for tool_name, tool_info in self._registered_tools.items():
            # Check if tool instance needs initialization
            if tool_info.instance is None and tool_info.cls is not None:
                logger.debug(f"| 🔧 Initializing tool: {tool_name}")
                
                # Create tool factory function
                def tool_factory():
                    tool_config = config.get(f"{tool_name}_tool", None)
                    if tool_config:
                        return tool_info.cls(**tool_config)
                    else:
                        return tool_info.cls()
                
                # Create tool instance and store it in context manager
                await self.tool_context_manager.build(tool_info, tool_factory)
                logger.debug(f"| ✅ Tool {tool_name} initialized")
            else:
                logger.debug(f"| ⏭️ Tool {tool_name} already initialized or no class available")
        
        logger.info("| ✅ Tools initialization completed")
    
    async def ainvoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke a tool with context management.
        
        Args:
            name: Name of the tool to invoke
            input: Input for the tool
            **kwargs: Keyword arguments for the tool
        Returns:
            Tool execution result
        """
        return await self.tool_context_manager.ainvoke(name, input, **kwargs)
    
    async def register(self, tool_info: ToolInfo):
        """Add a tool to the tool context manager
        
        Args:
            tool_info: Tool information
        """
        if tool_info.name not in self._registered_tools:
            self._registered_tools[tool_info.name] = tool_info
        await self.tool_context_manager.register(tool_info)
    
    def args_schemas(self) -> Dict[str, Type[BaseModel]]:
        """List all registered tool args schemas
        
        Returns:
            Dict[str, Type[BaseModel]]: Dictionary of tool args schemas
        """
        return self.tool_context_manager.args_schemas()
    
    def invoke(self, name: str, input: Any, **kwargs) -> Any:
        """Synchronous invoke a tool using tool context manager.
        
        Args:
            name: Name of the tool to invoke
            input: Input for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            Tool execution result
        """
        return self.tool_context_manager.invoke(name, input, **kwargs)
    
    def list(self) -> List[str]:
        """List all registered tools
        
        Returns:
            List[ToolInfo]: List of tool information
        """
        return self.tool_context_manager.list()
    
    def to_string(self, tool_name: str) -> str:
        """Convert tool information to string
        
        Args:
            tool_name: Tool name
            
        Returns:
            str: Tool information string
        """
        return self.tool_context_manager.to_string(tool_name)
    
    def get_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get tool information by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolInfo: Tool information or None if not found
        """
        return self.tool_context_manager.get_info(tool_name)
    
    def get(self, tool_name: str) -> Optional[Any]:
        """Get tool instance by name using tool context manager
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tool_context_manager.get(tool_name).instance
        
    async def cleanup(self):
        """Cleanup all tools"""
        await self.tool_context_manager.cleanup()


# Global TCP server instance
tcp = TCPServer()
