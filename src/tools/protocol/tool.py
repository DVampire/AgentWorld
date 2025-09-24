from typing import Any, Dict, Type, Union
from pydantic import BaseModel, Field, create_model, ConfigDict
import inflection
from langchain.tools import BaseTool as LangchainBaseTool, StructuredTool as LangchainStructuredTool

class BaseTool(LangchainBaseTool):
    """Base tool for the Tool Context Protocol."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the tool.")
    type: str = Field(description="The type of the tool.")
    description: str = Field(description="The description of the tool.")
    args_schema: Type[BaseModel] = Field(description="The args schema of the tool.")
    metadata: Dict[str, Any] = Field(description="The metadata of the tool.")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class WrappedTool(BaseTool):
    """MCP Tool for managing MCP server connections and tools."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the Wrapped Tool")
    type: str = Field(description="The type of the Wrapped Tool")
    description: str = Field(description="The description of the Wrapped Tool")
    args_schema: Type[BaseModel] = Field(description="The args schema of the Wrapped Tool")
    metadata: Dict[str, Any] = Field(description="The metadata of the Wrapped Tool")
    
    def __init__(self, tool: Union[BaseTool, LangchainStructuredTool, LangchainBaseTool], **kwargs):
        """Initialize Wrapped Tool."""
        
        args_schema = tool.args_schema
        
        if isinstance(args_schema, dict):
            args_schema_name = inflection.camelize(tool.name) + 'InputArgs'
            fields = {}
            
            required_fields = args_schema.get('required', [])
            
            for field_name, field_info in args_schema['properties'].items():
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
        type = "Wrapped Tool"
        description = tool.description
        metadata = tool.metadata
        
        super().__init__(
            name=name,
            type=type,
            description=description,
            args_schema=args_schema,
            metadata=metadata if metadata is not None else {},
            **kwargs
            )
        
        self.tool = tool
        
    def _run(self, **kwargs):
        # Handle different tool types
        if hasattr(self.tool, '_run'):
            # For Langchain tools, add config parameter if missing
            if 'config' not in kwargs and hasattr(self.tool, '_run'):
                # Check if the tool's _run method requires config
                import inspect
                sig = inspect.signature(self.tool._run)
                if 'config' in sig.parameters:
                    kwargs['config'] = None
            return self.tool._run(**kwargs)
    
    async def _arun(self, **kwargs):
        # Handle different tool types
        if hasattr(self.tool, '_arun'):
            # For Langchain tools, add config parameter if missing
            if 'config' not in kwargs and hasattr(self.tool, '_arun'):
                # Check if the tool's _arun method requires config
                import inspect
                sig = inspect.signature(self.tool._arun)
                if 'config' in sig.parameters:
                    kwargs['config'] = None
            return await self.tool._arun(**kwargs)