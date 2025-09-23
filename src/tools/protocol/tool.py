from typing import Any, Dict, Type
from pydantic import BaseModel, Field, create_model
import inflection
from langchain.tools import BaseTool as LangchainBaseTool

class BaseTool(LangchainBaseTool):
    """Base tool for the Tool Context Protocol."""
    name: str = Field(description="The name of the tool.")
    type: str = Field(description="The type of the tool.")
    description: str = Field(description="The description of the tool.")
    args_schema: Type[BaseModel] = Field(description="The args schema of the tool.")
    metadata: Dict[str, Any] = Field(description="The metadata of the tool.")

class WrappedTool(BaseTool):
    """MCP Tool for managing MCP server connections and tools."""
    
    name: str = Field(description="The name of the Wrapped Tool")
    type: str = Field(description="The type of the Wrapped Tool")
    description: str = Field(description="The description of the Wrapped Tool")
    args_schema: Type[BaseModel] = Field(description="The args schema of the Wrapped Tool")
    metadata: Dict[str, Any] = Field(description="The metadata of the Wrapped Tool")
    
    tool: BaseTool = Field(description="The tool of the Wrapped Tool")
    
    def __init__(self, tool: BaseTool, **kwargs):
        """Initialize Wrapped Tool."""
        super().__init__(tool=tool, **kwargs)
        self.name = tool.name
        self.type = "Wrapped Tool"
        self.description = tool.description
        
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
        self.args_schema = args_schema
        
        self.metadata = tool.metadata
        self.tool = tool
        
    def _run(self, **kwargs):
        return self.tool._run(**kwargs)
    
    async def _arun(self, **kwargs):
        return await self.tool._arun(**kwargs)