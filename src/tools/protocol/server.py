"""TCP Server

Server implementation for the Tool Context Protocol.
"""
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import create_model, BaseModel, Field
from langchain.tools import BaseTool, StructuredTool
import inflection

from src.logger import logger
from src.tools.protocol.types import ToolInfo
from src.tools.protocol.context import ToolContextManager

class TCPServer:
    """TCP Server for managing tool registration and execution"""
    
    def __init__(self):
        self._registered_tools: Dict[str, ToolInfo] = {}  # tool_name -> ToolInfo
        self.tool_context_manager = ToolContextManager()
    
    def tool(self, tool: Union[BaseTool, Type[BaseTool]] = None):
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
        type = metadata.get('type', name) if metadata else name
        config = metadata.get('config', None)
        
        # Create ToolInfo with lazy instance creation
        tool_info = ToolInfo(
            name=name,
            type=type,
            description=description,
            args_schema=args_schema,
            metadata=metadata,
            cls=cls,  # Store the class for lazy instantiation
            config=config,
            instance=None
        )
        
        self._registered_tools[name] = tool_info
        
        return cls
    
    def _register_tool_instance(self, tool: BaseTool):
        """Register a tool instance directly
        
        Args:
            tool: Tool instance to register
        """
        
        args_schema_name = inflection.camelize(tool.name) + 'InputArgs'
        fields = {}
        # Get required fields
        required_fields = tool.args_schema.get('required', [])
        
        for field_name, field_info in tool.args_schema['properties'].items():
            # Convert string type to actual Python type
            field_type_str = field_info.get('type', 'string')
            if field_type_str == 'string':
                field_type = str
            elif field_type_str == 'integer':
                field_type = int
            elif field_type_str == 'number':
                field_type = float
            elif field_type_str == 'boolean':
                field_type = bool
            elif field_type_str == 'array':
                field_type = list
            else:
                field_type = str  # Default to string
            
            # Get description
            field_description = field_info.get('description', f"Parameter {field_name}")
            
            # Handle default values and required fields
            if 'default' in field_info:
                # Has default value
                fields[field_name] = (field_type, Field(default=field_info['default'], description=field_description))
            elif field_name in required_fields:
                # Required field without default
                fields[field_name] = (field_type, Field(description=field_description))
            else:
                # Optional field without default
                fields[field_name] = (field_type, Field(default=None, description=field_description))
        
        args_schema = create_model(args_schema_name, **fields)
        
        name = tool.name
        description = tool.description
        metadata = tool.metadata
        type = metadata.get('type', name) if metadata else name
        config = metadata.get('config', None)
        
        tool = StructuredTool(
            name=name,
            description=description,
            args_schema=args_schema,
            metadata=metadata,
            func=tool._run,
            coroutine=tool._arun
        )
        
        # Create ToolInfo and store it
        tool_info = ToolInfo(
            name=name,
            type=type,
            description=description,
            args_schema=args_schema,
            metadata=metadata,
            cls=StructuredTool,
            config=config,
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
                    if tool_info.config:
                        return tool_info.cls(**tool_info.config)
                    else:
                        return tool_info.cls()
                
                # Create tool instance and store it in context manager
                tool_instance = await self.tool_context_manager.build(tool_name, tool_factory)
                tool_info.instance = tool_instance
                logger.debug(f"| ✅ Tool {tool_name} initialized")
            else:
                logger.debug(f"| ⏭️ Tool {tool_name} already initialized or no class available")
        
        logger.info("| ✅ Tools initialization completed")
    
    def list(self) -> List[str]:
        """List all registered tools
        
        Returns:
            List[ToolInfo]: List of tool information
        """
        names = [name for name in self._registered_tools.keys()]
        return names
    
    def args_schemas(self) -> List[Type[BaseModel]]:
        """List all registered tool args schemas
        
        Returns:
            List[Type[BaseModel]]: List of tool args schemas
        """
        return [tool_info.args_schema for tool_info in self._registered_tools.values()]
    
    def to_string(self, tool_info: ToolInfo) -> str:
        """Convert tool information to string
        
        Returns:
            str: Tool information string
        """
        return f"Tool: {tool_info.name}\nDescription: {tool_info.description}\nArgs Schema: {tool_info.args_schema}"
    
    def get_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get tool information by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolInfo: Tool information or None if not found
        """
        return self._registered_tools.get(tool_name)
    
    def get(self, tool_name: str) -> Optional[Any]:
        """Get tool instance by name using tool context manager
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._registered_tools.get(tool_name).instance
    
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
        
    async def cleanup(self):
        """Cleanup all tools"""
        self.tool_context_manager.cleanup()


# Global TCP server instance
tcp = TCPServer()
