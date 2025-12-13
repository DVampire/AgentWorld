"""Environment Context Protocol (ECP) Types

Core type definitions for the Environment Context Protocol.
"""

import inspect
import json
import re
import uuid
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Optional, Union, Type, Callable, List, Tuple, get_type_hints

import inflection
from pydantic import BaseModel, Field, ConfigDict, create_model

PYTHON_TYPE_FIELD = "x-python-type"
JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "string": "str",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}


def _default_parameters_schema() -> Dict[str, Any]:
    """Default empty parameters schema for actions."""
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def _parse_docstring_descriptions(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from Google-style docstrings."""
    if not docstring:
        return {}

    descriptions: Dict[str, str] = {}
    lines = inspect.cleandoc(docstring).splitlines()
    in_args = False

    for line in lines:
        stripped = line.strip()
        if not in_args:
            if stripped.lower().startswith("args"):
                in_args = True
            continue

        if stripped.lower().startswith(("returns:", "yields:", "raises:", "examples:")):
            break

        match = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
        if match:
            param_name = match.group(1)
            description = match.group(2).strip()
            descriptions[param_name] = description

    return descriptions


def _annotation_to_types(annotation: Any) -> Tuple[str, str]:
    """Convert Python type annotation to JSON type and Python type string."""
    if annotation is inspect._empty or annotation is None:
        return "string", "Any"

    basic_map = {
        str: ("string", "str"),
        int: ("integer", "int"),
        float: ("number", "float"),
        bool: ("boolean", "bool"),
        dict: ("object", "dict"),
        list: ("array", "list"),
    }
    if annotation in basic_map:
        return basic_map[annotation]

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin is Union and len(args) == 2 and type(None) in args:
        inner_type = args[0] if args[1] is type(None) else args[1]
        json_type, python_type = _annotation_to_types(inner_type)
        return json_type, f"Optional[{python_type}]"

    if origin is list or (hasattr(annotation, "__origin__") and "List" in str(annotation)):
        return "array", "list"

    if origin is dict or (hasattr(annotation, "__origin__") and "Dict" in str(annotation)):
        return "object", "dict"

    type_str = str(annotation).replace("typing.", "")
    return "string", type_str


def _parse_type_string(type_str: str) -> Type:
    """Parse a type string (e.g., 'str', 'Optional[str]') to Python type."""
    mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "dict": dict,
        "list": list,
        "Any": Any,
    }
    if type_str in mapping:
        return mapping[type_str]
    if type_str.startswith("Optional[") and type_str.endswith("]"):
        inner = type_str[9:-1]
        return Optional[_parse_type_string(inner)]  # type: ignore[index]
    if type_str.startswith("List[") and type_str.endswith("]"):
        inner = type_str[5:-1]
        return List[_parse_type_string(inner)]  # type: ignore[index]
    if type_str.startswith("Dict[") and type_str.endswith("]"):
        return dict
    return Any


def _json_type_to_python_type(json_type: str) -> Type:
    """Convert JSON schema type to Python type."""
    mapping = {
        "integer": int,
        "number": float,
        "string": str,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(json_type, str)


def _remove_python_type_field(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove x-python-type field from schema recursively."""
    cleaned = deepcopy(schema)
    cleaned.pop(PYTHON_TYPE_FIELD, None)
    if "properties" in cleaned:
        for prop_info in cleaned["properties"].values():
            if isinstance(prop_info, dict):
                prop_info.pop(PYTHON_TYPE_FIELD, None)
    return cleaned


def _build_parameter_schema_from_callable(func: Callable) -> Dict[str, Any]:
    """Build parameter schema from a callable's signature and docstring."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return _default_parameters_schema()

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    docstring = inspect.getdoc(func) or ""
    doc_descriptions = _parse_docstring_descriptions(docstring)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if name == "input" and len(signature.parameters) == 2:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = hints.get(name, param.annotation)
        json_type, python_type = _annotation_to_types(annotation)

        is_required = param.default is inspect._empty
        schema: Dict[str, Any] = {
            "type": json_type,
            "description": doc_descriptions.get(name, ""),
            PYTHON_TYPE_FIELD: python_type,
        }
        if not is_required:
            schema["default"] = param.default

        properties[name] = schema
        if is_required:
            required.append(name)

    if not properties:
        return _default_parameters_schema()

    result: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _build_args_model_from_schema(action_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a parameter schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    model_name = inflection.camelize(action_name) + "Input"

    if not properties:
        return create_model(
            model_name,
            __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
        )

    field_definitions: Dict[str, Any] = {}
    for param_name, param_info in properties.items():
        python_type_str = param_info.get(PYTHON_TYPE_FIELD)
        if python_type_str:
            python_type = _parse_type_string(python_type_str)
        else:
            json_type = param_info.get("type", "string")
            python_type = _json_type_to_python_type(json_type)

        is_required = param_name in required
        if "default" in param_info:
            default_value = param_info["default"]
        elif is_required:
            default_value = ...  # Required
        else:
            default_value = None

        description = param_info.get("description", "")
        if is_required and default_value is ...:
            field_definitions[param_name] = (
                python_type,
                Field(description=description),
            )
        else:
            field_definitions[param_name] = (
                Optional[python_type] if not is_required else python_type,
                Field(default=default_value, description=description),
            )

    return create_model(
        model_name,
        __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
        **field_definitions,
    )


def _build_text_representation(name: str, description: str, schema: Dict[str, Any]) -> str:
    """Build a human-readable text representation of an action."""
    properties = schema.get("properties", {})
    if not properties:
        return f"Action: {name}\nDescription: {description}\nParameters: None"

    required = set(schema.get("required", []))
    text = f"Action: {name}\nDescription: {description}\nParameters:\n"
    for param, info in properties.items():
        raw_type = info.get("type", "string")
        type_label = info.get(PYTHON_TYPE_FIELD) or JSON_TO_PYTHON_TYPE.get(raw_type, raw_type)
        if param not in required and not str(type_label).startswith("Optional["):
            type_label = f"Optional[{type_label}]"
        default = info.get("default", "N/A")
        text += f"    - {param} ({type_label}): {info.get('description', '')} (default: {default})\n"
    return text


class Environment(BaseModel):
    """Base abstract class for ECP environments"""
    
    name: str = Field(description="The name of the environment.")
    description: str = Field(description="The description of the environment.")
    metadata: Dict[str, Any] = Field(description="The metadata of the environment.")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        extra="allow"
    )
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register Environment subclasses"""
        super().__init_subclass__(**kwargs)
        # No need to manually track classes here - we'll use __subclasses__() in initialize()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize actions dictionary for this instance
        self.actions: Dict[str, ActionConfig] = {}
        
        # Register all actions marked with @ecp.action decorator
        from src.environment.server import ecp
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_action_name'):
                action_name = attr._action_name
                if action_name not in self.actions:
                    action_config = ActionConfig(
                        env_name=self.name,
                        name=action_name,
                        description=getattr(attr, '_action_description', ''),
                        function=attr,
                        metadata=getattr(attr, '_metadata', {})
                    )
                    # function_calling, text, and args_schema are computed on-demand via properties
                    self.actions[action_name] = action_config
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the state of the environment"""
        raise NotImplementedError("Get state method not implemented")

class ECPErrorCode(Enum):
    """ECP error codes"""
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    ENVIRONMENT_NOT_FOUND = -32001
    ACTION_NOT_FOUND = -32002
    ACTION_EXECUTION_ERROR = -32003


class ECPError(BaseModel):
    """ECP error structure"""
    code: ECPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None


class ECPRequest(BaseModel):
    """ECP request structure"""
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None


class ECPResponse(BaseModel):
    """ECP response structure"""
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ECPError] = None


class ECPNotification(BaseModel):
    """ECP notification structure"""
    method: str
    params: Optional[Dict[str, Any]] = None

class ActionConfig(BaseModel):
    """Action configuration (equivalent to MCP tool)"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    env_name: str = Field(description="The name of the environment this action belongs to")
    id: Optional[int] = Field(default=None, description="Unique identifier for the action")
    name: str = Field(description="The name of the action")
    description: str = Field(description="The description of the action")
    function: Optional[Callable] = Field(default=None, description="The function implementing the action")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the action")
    enabled: bool = Field(default=True, description="Whether the action is enabled")
    version: str = Field(default="1.0.0", description="Version of the action")
    
    cls: Optional[Any] = Field(default=None, description="The class of the action")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The initialization configuration of the action")
    
    @property
    def parameter_schema(self) -> Dict[str, Any]:
        """Get the parameter schema for this action."""
        schema = self._build_parameter_schema()
        return deepcopy(schema)
    
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
        """Return a BaseModel type for the action's input parameters.
        
        The model name will be `{action_name}Input` (e.g., `clickInput`, `typeInput`).
        
        Returns:
            Type[BaseModel]: A Pydantic BaseModel class for the action's input parameters
        """
        schema = self.parameter_schema
        return _build_args_model_from_schema(self.name, schema)
    
    @property
    def text(self) -> str:
        """Return the text representation of the action."""
        schema = self.parameter_schema
        return _build_text_representation(self.name, self.description, schema)
    
    def _build_parameter_schema(self) -> Dict[str, Any]:
        """Build parameter schema from function signature and docstring."""
        if self.function is None:
            return _default_parameters_schema()
        
        return _build_parameter_schema_from_callable(self.function)
    
    def _remove_python_type_field(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove x-python-type field from schema recursively."""
        return _remove_python_type_field(schema)


class EnvironmentConfig(BaseModel):
    """Environment configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the environment")
    rules: str = Field(description="The rules of the environment")
    description: str = Field(description="The description of the environment")
    version: str = Field(default="1.0.0", description="Version of the environment")
    actions: Dict[str, ActionConfig] = Field(default_factory=dict, description="Dictionary of actions available in this environment")
    cls: Optional[Type[Environment]] = Field(default=None, description="The class of the environment")
    config: Optional[Dict[str, Any]] = Field(default={}, description="The initialization configuration of the environment")
    instance: Optional[Any] = Field(default=None, description="The instance of the environment")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the environment")
    
    def __str__(self):
        return f"EnvironmentConfig(name={self.name}, description={self.description}, actions={len(self.actions)})"
    
    def __repr__(self):
        return self.__str__()
    
class ScreenshotInfo(BaseModel):
    """Screenshot information"""
    transformed: bool = Field(default=False, description="Whether the screenshot has been transformed")
    screenshot: str = Field(default="Screenshot base64")
    screenshot_path: str = Field(default="Screenshot path")
    screenshot_description: str = Field(default="Screenshot description")
    transform_info: Optional[Dict[str, Any]] = Field(default=None, description="Transform information")

class ActionResult(BaseModel):
    """Action result"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    success: bool = Field(description="Whether the action was successful")
    message: str = Field(description="The message of the action result")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra information of the action result")

    def __str__(self) -> str:
        return f"ActionResult(success={self.success}, message={self.message}, extra={self.extra})"

    def __repr__(self) -> str:
        return self.__str__()

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        from pydantic import BaseModel

        def serialize_value(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump(**kwargs)
            if isinstance(value, list):
                return [serialize_value(item) for item in value]
            if isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            return value

        return {
            "success": self.success,
            "message": self.message,
            "extra": serialize_value(self.extra) if self.extra is not None else None,
        }

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump())
    
class EnvironmentState(BaseModel):
    """Environment state"""
    state: str = Field(default="State", description="The state of the environment")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra information of the state")