from __future__ import annotations

import inspect
import re
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, get_type_hints, Type, Union
import inflection

from pydantic import BaseModel, ConfigDict, Field, create_model

PYTHON_TYPE_FIELD = "x-python-type"
JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "string": "str",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}

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
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        
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
        # Remove x-python-type field from schema as it's only for internal use
        schema = self.parameter_schema
        cleaned_schema = self._remove_python_type_field(schema)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": cleaned_schema,
            },
        }
        
    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return a BaseModel type for the tool's input parameters.
        
        The model name will be `{tool_name}Input` (e.g., `bashInput`, `python_interpreterInput`).
        
        Returns:
            Type[BaseModel]: A Pydantic BaseModel class for the tool's input parameters
        """
        import inflection
        
        schema = self.parameter_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        # Generate model name: {tool_name}Input (PascalCase)
        model_name = inflection.camelize(self.name) + "Input"
        
        # If no properties, return a simple empty model
        if not properties:
            return create_model(
                model_name,
                __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow")
            )
        
        # Build field definitions for create_model
        field_definitions = {}
        for param_name, param_info in properties.items():
            # Get Python type from x-python-type or convert from JSON type
            python_type_str = param_info.get(PYTHON_TYPE_FIELD)
            if python_type_str:
                # Parse type string (e.g., "str", "Optional[str]", "List[str]")
                python_type = self._parse_type_string(python_type_str)
            else:
                json_type = param_info.get("type", "string")
                python_type = self._json_type_to_python_type(json_type)
            
            # Check if field is required
            is_required = param_name in required
            
            # Get default value if exists
            if "default" in param_info:
                default_value = param_info["default"]
            elif is_required:
                default_value = ...  # Required field, no default
            else:
                default_value = None  # Optional field, default to None
            
            # Create Field with description
            description = param_info.get("description", "")
            if is_required and default_value is ...:
                # Required field without default
                field_definitions[param_name] = (
                    python_type,
                    Field(description=description)
                )
            else:
                # Optional field or field with default
                field_definitions[param_name] = (
                    Optional[python_type] if not is_required else python_type,
                    Field(default=default_value, description=description)
                )
        
        # Create the model
        return create_model(
            model_name,
            __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
            **field_definitions
        )
    
    def _parse_type_string(self, type_str: str) -> Type:
        """Parse a type string (e.g., "str", "Optional[str]", "List[str]") to Python type.
        
        Args:
            type_str: Type string to parse
            
        Returns:
            Python type
        """
        # Handle common type strings
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "dict": dict,
            "list": list,
            "Any": Any,
        }
        
        # Check if it's a direct mapping
        if type_str in type_mapping:
            return type_mapping[type_str]
        
        # Handle Optional[Type]
        if type_str.startswith("Optional[") and type_str.endswith("]"):
            inner_type_str = type_str[9:-1]
            inner_type = self._parse_type_string(inner_type_str)
            return Optional[inner_type]
        
        # Handle List[Type]
        if type_str.startswith("List[") and type_str.endswith("]"):
            inner_type_str = type_str[5:-1]
            inner_type = self._parse_type_string(inner_type_str)
            return List[inner_type]
        
        # Handle Dict[K, V]
        if type_str.startswith("Dict[") and type_str.endswith("]"):
            return dict
        
        # Default to Any if can't parse
        return Any
    
    def _json_type_to_python_type(self, json_type: str) -> Type:
        """Convert JSON schema type to Python type.
        
        Args:
            json_type: JSON schema type (e.g., "string", "integer", "number")
            
        Returns:
            Python type
        """
        type_mapping = {
            "integer": int,
            "number": float,
            "string": str,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        return type_mapping.get(json_type, str)
        
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
        properties = schema.get("properties", {})
        if not properties:
            return f"Tool: {self.name}\nDescription: {self.description}\nParameters: None"

        required = set(schema.get("required", []))
        text = f"Tool: {self.name}\nDescription: {self.description}\nParameters:\n"
        for param, info in properties.items():
            raw_type = info.get("type", "string")
            type_label = info.get(PYTHON_TYPE_FIELD) or JSON_TO_PYTHON_TYPE.get(raw_type, raw_type)
            if param not in required and not str(type_label).startswith("Optional["):
                type_label = f"Optional[{type_label}]"
            default = info.get("default", "N/A")
            text += f"    - {param} ({type_label}): {info.get('description', '')} (default: {default})\n"
        return text

    def _remove_python_type_field(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove x-python-type field from schema recursively."""
        from copy import deepcopy
        cleaned = deepcopy(schema)
        
        # Remove from top level if present
        cleaned.pop(PYTHON_TYPE_FIELD, None)
        
        # Remove from properties
        if "properties" in cleaned:
            for prop_name, prop_info in cleaned["properties"].items():
                if isinstance(prop_info, dict):
                    prop_info.pop(PYTHON_TYPE_FIELD, None)
        
        return cleaned

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
        doc_descriptions = self._parse_docstring_descriptions(docstring)

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
            json_type, python_type = self._annotation_to_types(annotation)
            
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

    def _parse_docstring_descriptions(self, docstring: str) -> Dict[str, str]:
        """Parse parameter descriptions from Google-style docstring."""
        if not docstring:
            return {}
        
        descriptions = {}
        lines = inspect.cleandoc(docstring).splitlines()
        in_args = False
        
        for line in lines:
            stripped = line.strip()
            if not in_args:
                if stripped.lower().startswith("args"):
                    in_args = True
                continue
            
            # Stop at other sections
            if stripped.lower().startswith(("returns:", "yields:", "raises:", "examples:")):
                break
            
            # Parse parameter line: "param_name (type): description"
            match = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
            if match:
                param_name = match.group(1)
                description = match.group(2).strip()
                descriptions[param_name] = description
        
        return descriptions

    def _annotation_to_types(self, annotation: Any) -> Tuple[str, str]:
        """Convert Python type annotation to JSON type and Python type string."""
        if annotation is inspect._empty or annotation is None:
            return "string", "Any"
        
        # Handle basic types
        if annotation is str or annotation == str:
            return "string", "str"
        if annotation is int or annotation == int:
            return "integer", "int"
        if annotation is float or annotation == float:
            return "number", "float"
        if annotation is bool or annotation == bool:
            return "boolean", "bool"
        if annotation is dict or annotation == dict:
            return "object", "dict"
        if annotation is list or annotation == list:
            return "array", "list"
        
        # Handle typing types
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        
        # Optional[Type] or Union[Type, None]
        if origin is Union and len(args) == 2 and type(None) in args:
            inner_type = args[0] if args[1] is type(None) else args[1]
            json_type, python_type = self._annotation_to_types(inner_type)
            return json_type, f"Optional[{python_type}]"
        
        # List[Type]
        if origin is list or (hasattr(annotation, "__origin__") and "List" in str(annotation)):
            return "array", "list"
        
        # Dict[K, V]
        if origin is dict or (hasattr(annotation, "__origin__") and "Dict" in str(annotation)):
            return "object", "dict"
        
        # Default fallback
        type_str = str(annotation).replace("typing.", "")
        return "string", type_str

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

