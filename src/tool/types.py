from __future__ import annotations

import inspect
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, get_type_hints

from pydantic import BaseModel, ConfigDict, Field


from src.utils import (
    PYTHON_TYPE_FIELD,
    default_parameters_schema,
    parse_docstring_descriptions,
    annotation_to_types,
    build_args_schema,
    build_function_calling,
    build_text_representation,
)

class ToolResponse(BaseModel):
    """Response for a tool call."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    success: bool = Field(description="Whether the tool call was successful")
    message: str = Field(description="The message from the tool call")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra data from the tool call")
    
    def __str__(self) -> str:
        return f"ToolResponse(success={self.success}, message={self.message}, extra={self.extra})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        from pydantic import BaseModel
        
        def serialize_value(value: Any) -> Any:
            """Recursively serialize Pydantic models and other nested structures."""
            if isinstance(value, BaseModel):
                return value.model_dump(**kwargs)
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                return value
        
        result = {
            "success": self.success,
            "message": self.message,
            "extra": serialize_value(self.extra) if self.extra is not None else None
        }
        return result
    
    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump())

class Tool(BaseModel):
    """Base class for all tools that can be exposed through function calling."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    enabled: bool = Field(default=True, description="Whether the tool is enabled")

    @staticmethod
    def default_parameters_schema() -> Dict[str, Any]:
        return default_parameters_schema()
        
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = value
        
    @property
    def is_enabled(self) -> bool:
        return bool(self.enabled)

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        self.enabled = bool(value)

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        schema = self._build_parameter_schema()
        return deepcopy(schema)

    async def __call__(self, input: Dict[str, Any], **kwargs) -> ToolResponse:
        """
        Execute the tool asynchronously.
        
        Args:
            input (Dict[str, Any]): The input to the tool.
            
        Returns:
            ToolResponse: The response from the tool call.
        """
        raise NotImplementedError("Tool subclasses must implement __call__")
    
    @property
    def function_calling(self) -> Dict[str, Any]:
        """Return the OpenAI-compatible function-calling representation."""
        schema = self.parameter_schema
        return build_function_calling(self.name, self.description, schema)
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return a BaseModel type for the tool's input parameters.
        
        The model name will be `{tool_name}Input` (e.g., `bashInput`, `python_interpreterInput`).
        
        Returns:
            Type[BaseModel]: A Pydantic BaseModel class for the tool's input parameters
        """
        schema = self.parameter_schema
        return build_args_schema(self.name, schema)
    
    @property
    def text(self) -> str:
        """
        Return the text representation of the tool.
        
        Example:
        ```
        Tool: python_interpreter
        Description: A tool that can execute Python code.
        Parameters:
            - code (str): Python code to execute. (default: None)
        ```
        """
        schema = self.parameter_schema
        return build_text_representation(self.name, self.description, schema, entity_type="Tool")

    def _build_parameter_schema(self) -> Dict[str, Any]:
        """Build parameter schema from function signature and docstring."""
        try:
            signature = inspect.signature(self.__class__.__call__)
        except (TypeError, ValueError):
            return self.default_parameters_schema()

        # Get type hints
        try:
            hints = get_type_hints(self.__class__.__call__)
        except Exception:
            hints = {}

        # Get docstring for descriptions
        docstring = inspect.getdoc(self.__class__.__call__) or ""
        doc_descriptions = parse_docstring_descriptions(docstring)

        properties = {}
        required = []
        
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            
            # Skip generic "input" parameter
            if name == "input" and len(signature.parameters) == 2:  # self + input
                continue
            
            # Skip VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) parameters
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            
            # Get type annotation
            annotation = hints.get(name, param.annotation)
            json_type, python_type = annotation_to_types(annotation)
            
            # Determine if required
            is_required = param.default is inspect._empty
            
            # Build schema
            schema: Dict[str, Any] = {
                "type": json_type,
                "description": doc_descriptions.get(name, ""),
            }
            schema[PYTHON_TYPE_FIELD] = python_type
            
            if not is_required:
                schema["default"] = param.default
            
            properties[name] = schema
            if is_required:
                required.append(name)

        if not properties:
            return self.default_parameters_schema()

        result: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            result["required"] = required
        return result

class ToolConfig(BaseModel):
    """Tool configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    enabled: bool = Field(default=True, description="Whether the tool is enabled")
    version: str = Field(default="1.0.0", description="Version of the tool")
    
    cls: Optional[Type[Tool]] = Field(default=None, description="The class of the tool")
    config: Optional[Dict[str, Any]] = Field(default={}, description="The initialization configuration of the tool")
    instance: Optional[Tool] = Field(default=None, description="The instance of the tool")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="The metadata of the tool")
    code: Optional[str] = Field(default=None, description="Source code for dynamically generated tool classes (used when cls cannot be imported from a module)")
    
    # Default representations
    function_calling: Optional[Dict[str, Any]] = Field(default=None, description="Default function calling representation")
    text: Optional[str] = Field(default=None, description="Default text representation")
    args_schema: Optional[Type[BaseModel]] = Field(default=None, description="Default args schema (BaseModel type)")

__all__ = [
    "Tool",
    "ToolResponse",
    "ToolConfig",
]

