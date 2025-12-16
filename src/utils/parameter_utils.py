"""Parameter schema utilities for tools and environments.

Shared utilities for building parameter schemas, parsing docstrings,
converting types, and creating Pydantic models from schemas.
"""

import inspect
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable, get_type_hints

import inflection
from pydantic import BaseModel, ConfigDict, Field, create_model

# Constants
PYTHON_TYPE_FIELD = "x-python-type"
JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "string": "str",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}


def default_parameters_schema() -> Dict[str, Any]:
    """Default empty parameters schema."""
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def parse_docstring_descriptions(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from Google-style docstrings.
    
    Args:
        docstring: The docstring to parse
        
    Returns:
        Dictionary mapping parameter names to descriptions
    """
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


def annotation_to_types(annotation: Any) -> Tuple[str, str]:
    """Convert Python type annotation to JSON type and Python type string.
    
    Args:
        annotation: Python type annotation
        
    Returns:
        Tuple of (json_type, python_type_string)
    """
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
        json_type, python_type = annotation_to_types(inner_type)
        return json_type, f"Optional[{python_type}]"

    if origin is list or (hasattr(annotation, "__origin__") and "List" in str(annotation)):
        return "array", "list"

    if origin is dict or (hasattr(annotation, "__origin__") and "Dict" in str(annotation)):
        return "object", "dict"

    type_str = str(annotation).replace("typing.", "")
    return "string", type_str


def parse_type_string(type_str: str) -> Type:
    """Parse a type string (e.g., 'str', 'Optional[str]', 'List[str]', 'Dict[str, Any]') to Python type.
    
    Supports both Python type names (str, int) and JSON schema type names (string, integer).
    Also handles 'typing.' prefix and detailed Dict[K, V] parsing.
    
    Args:
        type_str: Type string to parse
        
    Returns:
        Python type
    """
    # Remove typing. prefix if present
    type_str = type_str.replace("typing.", "").strip()
    
    # Handle common types (both Python and JSON schema names)
    mapping = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
        "dict": dict,
        "object": dict,
        "list": list,
        "array": list,
        "Any": Any,
    }
    if type_str in mapping:
        return mapping[type_str]
    
    # Handle Optional[Type]
    if type_str.startswith("Optional[") and type_str.endswith("]"):
        inner = type_str[9:-1].strip()
        return Optional[parse_type_string(inner)]  # type: ignore[index]
    
    # Handle List[Type]
    if type_str.startswith("List[") and type_str.endswith("]"):
        inner = type_str[5:-1].strip()
        return List[parse_type_string(inner)]  # type: ignore[index]
    
    # Handle Dict[K, V] - parse key and value types if provided
    if type_str.startswith("Dict[") and type_str.endswith("]"):
        inner = type_str[5:-1].strip()
        # Try to parse Dict[K, V] format
        if "," in inner:
            parts = inner.split(",", 1)
            if len(parts) == 2:
                key_type = parse_type_string(parts[0].strip())
                value_type = parse_type_string(parts[1].strip())
                return Dict[key_type, value_type]  # type: ignore[index]
        # Fallback to generic dict if parsing fails
        return dict
    
    # Default to Any if can't parse
    return Any


def json_type_to_python_type(json_type: str) -> Type:
    """Convert JSON schema type to Python type.
    
    Args:
        json_type: JSON schema type (e.g., "string", "integer", "number")
        
    Returns:
        Python type
    """
    mapping = {
        "integer": int,
        "number": float,
        "string": str,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(json_type, str)


def remove_python_type_field(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove x-python-type field from schema recursively.
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Cleaned schema dictionary
    """
    cleaned = deepcopy(schema)
    cleaned.pop(PYTHON_TYPE_FIELD, None)
    if "properties" in cleaned:
        for prop_info in cleaned["properties"].values():
            if isinstance(prop_info, dict):
                prop_info.pop(PYTHON_TYPE_FIELD, None)
    return cleaned


def build_args_schema(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a parameter schema.
    
    Args:
        name: Name for the model (will be converted to PascalCase + "Input")
        schema: Parameter schema dictionary
        
    Returns:
        Pydantic BaseModel class
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    model_name = inflection.camelize(name) + "Input"

    if not properties:
        return create_model(
            model_name,
            __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
        )

    field_definitions: Dict[str, Any] = {}
    for param_name, param_info in properties.items():
        python_type_str = param_info.get(PYTHON_TYPE_FIELD)
        if python_type_str:
            python_type = parse_type_string(python_type_str)
        else:
            json_type = param_info.get("type", "string")
            python_type = json_type_to_python_type(json_type)

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


def build_function_calling(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build OpenAI-compatible function-calling representation.
    
    Args:
        name: Name of the tool/action
        description: Description of the tool/action
        schema: Parameter schema dictionary (will be cleaned by removing x-python-type fields)
        
    Returns:
        OpenAI-compatible function-calling dictionary
    """
    # Remove x-python-type field from schema as it's only for internal use
    cleaned_schema = remove_python_type_field(schema)
    
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": cleaned_schema,
        },
    }


def build_text_representation(name: str, description: str, schema: Dict[str, Any], entity_type: str = "Tool") -> str:
    """Build a human-readable text representation.
    
    Args:
        name: Name of the tool/action
        description: Description
        schema: Parameter schema dictionary
        entity_type: Type of entity ("Tool" or "Action")
        
    Returns:
        Human-readable text representation
    """
    properties = schema.get("properties", {})
    if not properties:
        return f"{entity_type}: {name}\nDescription: {description}\nParameters: None"

    required = set(schema.get("required", []))
    text = f"{entity_type}: {name}\nDescription: {description}\nParameters:\n"
    for param, info in properties.items():
        raw_type = info.get("type", "string")
        type_label = info.get(PYTHON_TYPE_FIELD) or JSON_TO_PYTHON_TYPE.get(raw_type, raw_type)
        if param not in required and not str(type_label).startswith("Optional["):
            type_label = f"Optional[{type_label}]"
        default = info.get("default", "N/A")
        text += f"    - {param} ({type_label}): {info.get('description', '')} (default: {default})\n"
    return text

