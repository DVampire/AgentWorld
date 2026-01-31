"""Environment Context Protocol (ECP) Types

Core type definitions for the Environment Context Protocol.
"""

import json
import uuid
import inflection
from enum import Enum
from typing import Any, Dict, Optional, Union, Type, Callable
from pydantic import BaseModel, Field, ConfigDict

from src.dynamic import dynamic_manager
from src.utils import generate_unique_id


class EnvironmentContext(BaseModel):
    id: str = Field(default_factory=lambda: generate_unique_id(prefix="environment"), description="The unique identifier for the environment context")

class Environment(BaseModel):
    """Base abstract class for ECP environments"""
    
    name: str = Field(description="The name of the environment.")
    description: str = Field(description="The description of the environment.")
    metadata: Dict[str, Any] = Field(description="The metadata of the environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")
    
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
    
    def get_rules(self) -> str:
        """Generate environment rules from environment instance.
        
        Returns:
            str: Generated environment rules
        """
        metadata = self.metadata if self.metadata else {}
        has_vision = metadata.get('has_vision', False)
        additional_rules = metadata.get('additional_rules', None)
        env_name = self.name
        actions = self.actions
        
        # Start building the rules
        rules_parts = [f"<environment_{inflection.underscore(env_name)}>"]
        
        # Add state section
        rules_parts.append("<state>")
        if additional_rules and 'state' in additional_rules:
            rules_parts.append(additional_rules['state'])
        else:
            rules_parts.append(f"The environment state about {env_name}.")
        rules_parts.append("</state>")
        
        # Add vision section
        rules_parts.append("<vision>")
        if additional_rules and 'vision' in additional_rules:
            rules_parts.append(additional_rules['vision'])
        else:
            if has_vision:
                rules_parts.append("The environment vision information.")
            else:
                rules_parts.append("No vision available.")
        rules_parts.append("</vision>")
        
        # Add additional rules if provided (for backward compatibility)
        if additional_rules and 'additional_rules' in additional_rules:
            rules_parts.append("<additional_rules>")
            rules_parts.append(additional_rules['additional_rules'])
            rules_parts.append("</additional_rules>")
        
        # Add interaction section with actions
        rules_parts.append("<interaction>")
        
        if additional_rules and 'interaction' in additional_rules:
            # Use custom interaction rules
            rules_parts.append(additional_rules['interaction'])
        else:
            # Use default interaction rules
            rules_parts.append("Available actions:")
            
            # Sort actions by name for consistent output
            sorted_actions = sorted(actions.items(), key=lambda x: x[0])
            
            for i, (action_name, action_config) in enumerate(sorted_actions, 1):
                rules_parts.append(f"{i}. {action_name}: {action_config.description}")
            
            rules_parts.append("Input format: JSON string with action-specific parameters.")
            rules_parts.append("Example: {\"name\": \"action_name\", \"args\": {\"action-specific parameters\"}}")
        
        rules_parts.append("</interaction>")
        
        # Close the environment tag
        rules_parts.append(f"</environment_{inflection.underscore(env_name)}>")
        
        return "\n".join(rules_parts)

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
    name: str = Field(description="The name of the action")
    description: str = Field(description="The description of the action")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the action")
    version: str = Field(default="1.0.0", description="Version of the action")
    
    function: Optional[Callable] = Field(default=None, description="The function implementing the action")
    code: Optional[str] = Field(default=None, description="The source code of the action")
    
    # Default representations
    args_schema: Optional[Type[BaseModel]] = Field(default=None, description="Default args schema (BaseModel type)")
    function_calling: Optional[Dict[str, Any]] = Field(default=None, description="Default function calling representation")
    text: Optional[str] = Field(default=None, description="Default text representation")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        
        result = {
            "env_name": self.env_name,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "version": self.version,
            
            "function": f"<{self.function.__name__}>",
            "code": self.code,
            
            "args_schema": dynamic_manager.serialize_args_schema(self.args_schema) if self.args_schema else None,
            "function_calling": self.function_calling,
            "text": self.text,
        }
        
        return result
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'ActionConfig':
        """Validate the model from a dictionary."""
        env_name = data.get("env_name")
        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata")
        version = data.get("version")
        
        code = data.get("code")
        function = None
        
        args_schema = dynamic_manager.deserialize_args_schema(data.get("args_schema"))
        function_calling = data.get("function_calling")
        text = data.get("text")
        
        return cls(env_name=env_name,
            name=name,
            description=description,
            metadata=metadata,
            version=version,
            function=function,
            code=code,
            args_schema=args_schema,
            function_calling=function_calling,
            text=text
        )

class EnvironmentConfig(BaseModel):
    """Environment configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the environment")
    description: str = Field(description="The description of the environment")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the environment")
    rules: str = Field(description="The rules of the environment")
    version: str = Field(default="1.0.0", description="Version of the environment")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")
    
    cls: Optional[Type[Environment]] = Field(default=None, description="The class of the environment")
    config: Optional[Dict[str, Any]] = Field(default={}, description="The initialization configuration of the environment")
    instance: Optional[Any] = Field(default=None, description="The instance of the environment")
    code: Optional[str] = Field(default=None, description="Source code for dynamically generated environment classes (used when cls cannot be imported from a module)")
    
    actions: Dict[str, ActionConfig] = Field(default_factory=dict, description="Dictionary of actions available in this environment")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        result = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "rules": self.rules,
            "version": self.version,
            "require_grad": self.require_grad,
            
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
            
            "actions": {name: action_config.model_dump() for name, action_config in self.actions.items()},
        }
        
        return result
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        """Validate the model from a dictionary."""
        
        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata")
        rules = data.get("rules")
        version = data.get("version")
        require_grad = data.get("require_grad", False)
        
        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code, 
                        class_name=class_name,
                        base_class=Environment,
                        context="environment"
                    )
                except Exception as e:
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None
            
        config = data.get("config")
        instance = data.get("instance", None)
        
        actions = {name: ActionConfig.model_validate(action_config) for name, action_config in data.get("actions", {}).items()}
        
        # If cls_ is loaded, restore function references for actions from the class
        if cls_ is not None:
            for action_name, action_config in actions.items():
                # First try direct attribute access (most common case where action_name == method_name)
                if hasattr(cls_, action_name):
                    attr = getattr(cls_, action_name)
                    if hasattr(attr, '_action_name') and getattr(attr, '_action_name') == action_name:
                        action_config.function = attr
                        continue
        
        return cls(name=name,
            description=description,
            metadata=metadata,
            rules=rules,
            version=version,
            require_grad=require_grad,
            cls=cls_,
            config=config,
            instance=instance,
            code=code,
            actions=actions
            )
    
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