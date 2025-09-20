"""TCP Server

Server implementation for the Tool Context Protocol.
"""

import inspect
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, create_model, Field
from langchain.tools import BaseTool
import inflection
from langchain.tools import StructuredTool

from src.tools.protocol.types import ToolInfo

class TCPServer:
    """TCP Server for managing tool registration and execution"""
    
    def __init__(self):
        self._registered_tools: Dict[str, ToolInfo] = {}  # tool_name -> ToolInfo
    
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
        instance = cls()
        
        name = instance.name
        description = instance.description
        args_schema = instance.args_schema
        metadata = instance.metadata
        type = metadata.get('type', cls.__name__.lower())
        
        # Create ToolInfo and store it
        tool_info = ToolInfo(
            name=name,
            type=type,
            description=description,
            instance=instance,
            args_schema=args_schema,
            metadata=metadata
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
        type = metadata.get('type', tool.name.lower())
        
        tool = StructuredTool(
            name=name,
            description=description,
            args_schema=args_schema,
            func=tool._run,
            coroutine=tool._arun
        )
        
        # Create ToolInfo and store it
        tool_info = ToolInfo(
            name=name,
            type=type,
            description=description,
            instance=tool,
            args_schema=args_schema,
            metadata=metadata
        )
        
        self._registered_tools[name] = tool_info
        
        return tool
    
    def _parse_init_signature(self, cls: Type[Any]) -> Optional[Type[BaseModel]]:
        """Parse __init__ method signature to generate initialization schema
        
        Args:
            cls: Tool class
            
        Returns:
            Type[BaseModel]: Initialization schema or None if no parameters
        """
        try:
            sig = inspect.signature(cls.__init__)
            fields = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                # Get the type annotation
                field_type = param.annotation
                if field_type == inspect.Parameter.empty:
                    field_type = str  # Default to string if no annotation
                
                # Get description from docstring or create default
                if isinstance(field_type, str):
                    type_name = field_type
                else:
                    type_name = getattr(field_type, '__name__', str(field_type))
                field_description = f"Parameter {param_name} of type {type_name}"
                
                # Handle default values
                if param.default != inspect.Parameter.empty:
                    fields[param_name] = (field_type, Field(default=param.default, description=field_description))
                else:
                    fields[param_name] = (field_type, Field(description=field_description))
            
            if not fields:
                return None
                
            # Create init schema class
            init_schema_name = inflection.camelize(cls.__name__) + 'InitArgs'
            init_schema = create_model(init_schema_name, **fields)
            
            return init_schema
            
        except Exception as e:
            print(f"Warning: Failed to parse init signature for {cls.__name__}: {e}")
            return None
    
    def _create_tool_instance(self, 
                              cls: Type[Any], 
                              init_schema: Optional[Type[BaseModel]] = None,
                              tool_name: str = None,
                              tool_description: str = None,
                              tool_args_schema: Optional[Type[BaseModel]] = None):
        """Create tool instance with default parameters
        
        Args:
            cls: Tool class
            init_schema: Initialization schema
            
        Returns:
            Tool instance
        """
        if init_schema:
            # Create instance with default values from schema
            default_values = {}
            for field_name, field_info in init_schema.model_fields.items():
                if field_info.default is not None:
                    default_values[field_name] = field_info.default
            instance = cls(**default_values)
            
        else:
            # Create instance without parameters
            instance = cls()
            
        tool = StructuredTool(
                name=tool_name,
                description=tool_description,
                args_schema=tool_args_schema,
                func=instance._run,
                coroutine=instance._arun
            )
            
        return tool
        
    def list_tools(self) -> List[ToolInfo]:
        """List all registered tools
        
        Returns:
            List[ToolInfo]: List of tool information
        """
        return list(self._registered_tools.values())
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get tool information by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolInfo: Tool information or None if not found
        """
        return self._registered_tools.get(tool_name)
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get tool instance by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._registered_tools.get(tool_name).instance


# Global TCP server instance
tcp = TCPServer()
