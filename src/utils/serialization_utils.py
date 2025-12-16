"""Serialization utilities for Pydantic models and types."""
import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.logger import logger
from src.utils.parameter_utils import parse_type_string


def serialize_args_schema(args_schema: Type[BaseModel]) -> Optional[Dict[str, Any]]:
    """Serialize a BaseModel type to a dictionary with class name and field information.
    
    Args:
        args_schema: BaseModel class type
        
    Returns:
        Dictionary with class_name and fields info, or None if serialization fails
    """
    try:
        schema_info = {
            "class_name": args_schema.__name__,
            "fields": {}
        }
        
        # Extract field information from model_fields
        for field_name, field_info in args_schema.model_fields.items():
            field_data = {
                "type": str(field_info.annotation) if hasattr(field_info, 'annotation') else "Any",
                "required": field_info.is_required() if hasattr(field_info, 'is_required') else True,
            }
            
            # Add description if available
            if hasattr(field_info, 'description') and field_info.description:
                field_data["description"] = field_info.description
            
            # Add default value if available
            if hasattr(field_info, 'default') and field_info.default is not ...:
                if field_info.default is not None:
                    # Try to serialize default value
                    try:
                        json.dumps(field_info.default)
                        field_data["default"] = field_info.default
                    except (TypeError, ValueError):
                        field_data["default"] = None
                else:
                    field_data["default"] = None
            
            schema_info["fields"][field_name] = field_data
        
        return schema_info
    except Exception as e:
        logger.warning(f"| ⚠️ Failed to serialize args_schema {args_schema.__name__}: {e}")
        return None


def deserialize_args_schema(schema_info: Dict[str, Any]) -> Optional[Type[BaseModel]]:
    """Deserialize a BaseModel type from saved schema information.
    
    Args:
        schema_info: Dictionary with class_name and fields info
        
    Returns:
        BaseModel class type, or None if deserialization fails
    """
    try:
        from typing import get_origin, get_args, Any as TypingAny
        
        class_name = schema_info.get("class_name")
        fields_info = schema_info.get("fields", {})
        
        if not class_name:
            return None
        
        # Build field definitions for create_model
        field_definitions = {}
        for field_name, field_data in fields_info.items():
            # Parse type string (e.g., "str", "Optional[str]", "List[str]")
            type_str = field_data.get("type", "Any")
            python_type = parse_type_string(type_str)
            
            # Get default value
            default_value = field_data.get("default")
            is_required = field_data.get("required", True)
            
            if default_value is None and not is_required:
                # Optional field
                from typing import Optional
                python_type = Optional[python_type] if python_type != TypingAny else TypingAny
                default_value = None
            elif default_value is None and is_required:
                default_value = ...  # Required field
            
            # Create Field with description if available
            description = field_data.get("description", "")
            if description:
                field_definitions[field_name] = (python_type, Field(default=default_value, description=description))
            else:
                field_definitions[field_name] = (python_type, default_value)
        
        # Create the model dynamically
        model = create_model(
            class_name,
            __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
            **field_definitions
        )
        
        return model
    except Exception as e:
        logger.warning(f"| ⚠️ Failed to deserialize args_schema: {e}")
        return None

