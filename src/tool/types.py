from __future__ import annotations

import ast
import inspect
import re
import json
import uuid
from enum import Enum
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, get_type_hints, Type, Union

from pydantic import BaseModel, ConfigDict, Field

PYTHON_TYPE_FIELD = "x-python-type"
JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "string": "str",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}
UNSET = object()

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

    async def __call__(self, input: Dict[str, Any]) -> ToolResponse:
        """
        Execute the tool asynchronously.
        
        Args:
            input (Dict[str, Any]): The input to the tool.
            
        Returns:
            ToolResponse: The response from the tool call.
        """
        raise NotImplementedError("Tool subclasses must implement __call__")
    
    def to_function_call(self) -> Dict[str, Any]:
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
        
    def to_text(self) -> str:
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
        docstring = inspect.getdoc(self.__class__.__call__) or ""
        doc_params = _parse_parameters_from_docstring(docstring)
        if _only_generic_input(doc_params):
            doc_params = []
        sig_params = _parse_parameters_from_signature(self.__class__.__call__)
        parsed = _merge_parameter_definitions(doc_params, sig_params)
        if not parsed:
            return self.default_parameters_schema()

        properties = {}
        required = []
        for arg in parsed:
            schema: Dict[str, Any] = {
                "type": arg["type"],
                "description": arg["description"],
            }
            if PYTHON_TYPE_FIELD in arg:
                schema[PYTHON_TYPE_FIELD] = arg[PYTHON_TYPE_FIELD]
            if "default" in arg:
                schema["default"] = arg["default"]
            if "enum" in arg and arg["enum"]:
                schema["enum"] = arg["enum"]
            properties[arg["name"]] = schema
            if arg.get("required"):
                required.append(arg["name"])

        result: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            result["required"] = required
        return result


def _parse_parameters_from_docstring(docstring: str) -> List[Dict[str, Any]]:
    """Parse the Args section of a Google-style docstring."""
    if not docstring:
        return []

    lines = inspect.cleandoc(docstring).splitlines()
    args_lines: List[str] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if not collecting:
            if stripped.lower().startswith("args"):
                collecting = True
            continue
        else:
            if stripped.lower().startswith(("returns:", "yields:", "raises:", "examples:", "note:", "notes:")):
                break
            if stripped == "":
                args_lines.append(line)
                continue
            if not line.startswith((" ", "\t")):
                break
            args_lines.append(line)

    entries = _consolidate_arg_entries(args_lines)
    parsed: List[Dict[str, Any]] = []
    for entry in entries:
        parsed.append(_build_arg_schema(entry))
    return parsed


def _consolidate_arg_entries(lines: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    param_pattern = re.compile(
        r"^(?P<name>[a-zA-Z_][\w]*)\s*(?:\((?P<typeinfo>[^)]*)\))?\s*:\s*(?P<desc>.*)$"
    )

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if current:
                current["desc_lines"].append("")
            continue
        match = param_pattern.match(stripped)
        if match:
            if current:
                entries.append(current)
            current = {
                "name": match.group("name"),
                "typeinfo": match.group("typeinfo") or "",
                "desc_lines": [match.group("desc").strip()],
            }
        else:
            if current:
                current["desc_lines"].append(stripped)
    if current:
        entries.append(current)
    return entries


def _build_arg_schema(entry: Dict[str, Any]) -> Dict[str, Any]:
    description = " ".join(line.strip() for line in entry.get("desc_lines", []) if line is not None).strip()
    typeinfo = entry.get("typeinfo", "")
    json_type, required, default, enum_values, python_label = _interpret_typeinfo(typeinfo)

    if default is UNSET:
        default_from_desc = _extract_default_from_description(description)
        if default_from_desc is not UNSET:
            default = default_from_desc

    schema = {
        "name": entry["name"],
        "type": json_type,
        "description": description,
        "required": required,
    }
    schema[PYTHON_TYPE_FIELD] = python_label or json_type
    if default is not UNSET:
        schema["default"] = default
    if enum_values:
        schema["enum"] = enum_values
    return schema


def _interpret_typeinfo(typeinfo: str) -> Tuple[str, bool, Any, Optional[List[Any]], str]:
    if not typeinfo:
        return "string", True, UNSET, None, "Any"

    parts = [part.strip() for part in typeinfo.split(",") if part.strip()]
    base = parts[0]
    required = True
    default: Any = UNSET
    enum_values: Optional[List[Any]] = None

    normalized_base = base.replace("typing.", "")
    python_label = normalized_base or "Any"
    optional_match = re.fullmatch(r"Optional\[(.+)\]", normalized_base, re.IGNORECASE)
    if optional_match:
        normalized_base = optional_match.group(1)
        required = False
        python_label = f"Optional[{normalized_base}]"

    literal_match = re.fullmatch(r"Literal\[(.+)\]", normalized_base, re.IGNORECASE)
    if literal_match:
        enum_values = _parse_literal_list(literal_match.group(1))
        normalized_base = "string"
        if enum_values:
            python_label = f"Literal[{', '.join(repr(v) for v in enum_values)}]"

    json_type = _map_type_name(normalized_base)

    for meta in parts[1:]:
        lower = meta.lower()
        if lower == "optional":
            required = False
        elif lower == "required":
            required = True
        elif lower.startswith("default="):
            default = _safe_literal_eval(meta.split("=", 1)[1].strip())

    if not python_label:
        python_label = normalized_base or "Any"
    return json_type, required, default, enum_values, python_label


def _map_type_name(type_name: str) -> str:
    name = type_name.strip().lower()
    if name.startswith("optional["):
        inner = name[len("optional[") : -1]
        return _map_type_name(inner)
    if any(token in name for token in ("list", "sequence", "tuple", "iterable")):
        return "array"
    if any(token in name for token in ("dict", "mapping")):
        return "object"
    if name in {"str", "string", "text"}:
        return "string"
    if name in {"int", "integer"}:
        return "integer"
    if name in {"float", "double", "number"}:
        return "number"
    if name in {"bool", "boolean"}:
        return "boolean"
    if name in {"any"}:
        return "object"
    return "string"


def _parse_literal_list(values: str) -> List[Any]:
    raw_items = [item.strip() for item in values.split(",") if item.strip()]
    parsed_items = []
    for item in raw_items:
        parsed_items.append(_safe_literal_eval(item))
    return parsed_items


def _extract_default_from_description(description: str) -> Any:
    if not description:
        return UNSET
    default_patterns = [
        r"[Dd]efaults?\s+to\s+([^.;]+)",
        r"[Dd]efault(?:\s+value)?\s*[:=]\s*([^.;]+)",
    ]
    for pattern in default_patterns:
        match = re.search(pattern, description)
        if match:
            value = match.group(1).strip()
            return _safe_literal_eval(value)
    return UNSET


def _safe_literal_eval(value: str) -> Any:
    value = value.strip().strip(".")
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "none":
            return None
        return value


def _parse_parameters_from_signature(func) -> List[Dict[str, Any]]:
    """Fallback parser that inspects the function signature."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []

    hints = get_type_hints(func)
    parsed: List[Dict[str, Any]] = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        typeinfo = _format_annotation(hints.get(name, param.annotation))
        json_type, required, default, enum_values, python_label = _interpret_typeinfo(typeinfo)
        if param.default is not inspect._empty:
            default = param.default

        schema = {
            "name": name,
            "type": json_type,
            "description": "",
            "required": required,
        }
        schema[PYTHON_TYPE_FIELD] = python_label or json_type
        if default is not UNSET:
            schema["default"] = default
        if enum_values:
            schema["enum"] = enum_values
        parsed.append(schema)

    return parsed


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect._empty or annotation is None:
        return ""
    if isinstance(annotation, type):
        return annotation.__name__
    ann_repr = repr(annotation)
    return ann_repr.replace("typing.", "")


def _only_generic_input(parsed: List[Dict[str, Any]]) -> bool:
    if len(parsed) != 1:
        return False
    arg = parsed[0]
    return arg["name"] == "input" and arg["type"] == "object"


def _merge_parameter_definitions(
    doc_params: List[Dict[str, Any]],
    sig_params: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not doc_params and not sig_params:
        return []

    doc_map = {param["name"]: param for param in doc_params}
    sig_map = {param["name"]: param for param in sig_params}

    ordered_names: List[str] = []
    for name in doc_map.keys():
        if name not in ordered_names:
            ordered_names.append(name)
    for name in sig_map.keys():
        if name not in ordered_names:
            ordered_names.append(name)

    merged: List[Dict[str, Any]] = []
    for name in ordered_names:
        doc_entry = doc_map.get(name)
        sig_entry = sig_map.get(name)

        if doc_entry:
            entry = dict(doc_entry)
        elif sig_entry:
            entry = dict(sig_entry)
        else:
            continue

        if sig_entry:
            entry.setdefault("type", sig_entry.get("type"))
            entry.setdefault("description", sig_entry.get("description", ""))
            entry.setdefault(PYTHON_TYPE_FIELD, sig_entry.get(PYTHON_TYPE_FIELD))

            if "default" in sig_entry:
                entry["default"] = sig_entry["default"]

            if "required" not in entry:
                entry["required"] = sig_entry.get("required", True)
            else:
                if sig_entry.get("required") is False:
                    entry["required"] = False

            if sig_entry.get("enum") and "enum" not in entry:
                entry["enum"] = sig_entry["enum"]

        if "required" not in entry:
            entry["required"] = True

        merged.append(entry)

    return merged

class ToolConfig(BaseModel):
    """Tool configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    id: int = Field(description="Unique identifier for the tool")
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    enabled: bool = Field(default=True, description="Whether the tool is enabled")
    version: str = Field(default="1.0.0", description="Version of the tool")
    
    cls: Optional[Type[Tool]] = Field(default=None, description="The class of the tool")
    config: Optional[Dict[str, Any]] = Field(default={}, description="The initialization configuration of the tool")
    instance: Optional[Tool] = Field(default=None, description="The instance of the tool")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="The metadata of the tool")

__all__ = [
    "Tool",
    "ToolResponse",
    "ToolConfig",
]

