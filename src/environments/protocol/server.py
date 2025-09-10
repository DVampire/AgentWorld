"""ECP Server

Server implementation for the Environment Context Protocol with decorator support.
"""

import asyncio
import json
import inspect
from typing import Any, Dict, List, Optional, Callable, Type
from datetime import datetime
from pydantic import BaseModel, create_model, Field
import inflection

from src.environments.protocol.types import (
    EnvironmentInfo, 
    ActionInfo, 
)
from src.environments.protocol.environment import BaseEnvironment


class ECPServer:
    """ECP Server for managing environments and actions with decorator support"""
    
    def __init__(self):
        self._registered_environments: Dict[str, EnvironmentInfo] = {}  # env_name -> EnvironmentInfo
    
    def environment(self, 
                    name: str = None,
                    env_type: Optional[str] = None, 
                    description: Optional[str] = "", 
                    rules: Optional[str] = ""
                    ):
        """Decorator to register an environment class
        
        Args:
            name: Environment name (defaults to class name)
            env_type: Environment type (defaults to class name)
            description: Environment description
            rules: Environment rules
        """
        def decorator(cls: Type[BaseEnvironment]):
            env_name = name or cls.__name__
            env_type_name = env_type or cls.__name__.lower()  # Use class name as environment type
            
            # Store environment metadata
            cls._env_name = env_name
            cls._env_type = env_type_name
            cls._env_description = description
            cls._env_rules = rules
            
            # Collect all actions from the class
            actions = {}
            for attr_name in dir(cls):
                attr = getattr(cls, attr_name)
                if hasattr(attr, '_action_name'):
                    # Set the environment name for this action
                    attr._env_name = env_name
                    
                    # Set env_name as the default value for the env_name field
                    args_schema = getattr(attr, '_args_schema')
                    # Create a new model with updated env_name default
                    fields = {}
                    for field_name, field_info in args_schema.model_fields.items():
                        if field_name == 'env_name':
                            fields[field_name] = (field_info.annotation, Field(default=env_name, description=field_info.description))
                        else:
                            fields[field_name] = (field_info.annotation, field_info)
                    
                    # Recreate the model with the correct default
                    new_args_schema = create_model(args_schema.__name__, **fields)
                    attr._args_schema = new_args_schema
                    
                    # Create ActionInfo from the decorated method
                    action_info = ActionInfo(
                        env_name=env_name,
                        name=getattr(attr, '_action_name'),
                        description=getattr(attr, '_action_description', ''),
                        args_schema=new_args_schema,
                        function=getattr(attr, '_action_function', None),
                        metadata=getattr(attr, '_metadata', None)
                    )
                    
                    actions[getattr(attr, '_action_name')] = action_info
            
            # Create EnvironmentInfo and store it
            env_info = EnvironmentInfo(
                name=env_name,
                type=env_type_name,
                description=description,
                rules=rules,
                actions=actions,
                env_class=cls,
                env_config=None,
                env_instance=None,
                metadata=None
            )
            
            self._registered_environments[env_name] = env_info
            
            return cls
        return decorator
    
    def action(self, 
               name: str = None, 
               description: str = ""):
        """Decorator to register an action (tool) for an environment
        
        Args:
            name: Action name (defaults to function name)
            description: Action description
        """
        def decorator(func: Callable):
            action_name = name or func.__name__
            
            # Parse function signature to generate schemas
            sig = inspect.signature(func)
            args_schema = self._parse_function_signature(sig, action_name, func)
            
            # Store action metadata (env_name will be set later by environment decorator)
            func._action_name = action_name
            func._action_description = description
            func._args_schema = args_schema
            func._action_function = func
            func._metadata = None
            
            return func
        return decorator
    
    def _parse_function_signature(self, sig: inspect.Signature, action_name: str, func: Callable = None) -> Type[BaseModel]:
        """Parse function signature to generate args and output type
        
        Args:
            sig: Function signature
            action_name: Action name
            func: Function object (for docstring parsing)
            
        Returns:
            Type[BaseModel]: args_schema_class
        """
        # Parse docstring for parameter descriptions
        param_descriptions = {}
        if func and func.__doc__:
            import re
            # Parse docstring for Args section
            docstring = func.__doc__
            args_section = re.search(r'Args:\s*\n(.*?)(?:\n\s*\n|\n\s*Returns:|\n\s*$)', docstring, re.DOTALL)
            if args_section:
                args_text = args_section.group(1)
                # Parse individual parameter descriptions
                param_pattern = r'(\w+)\s*\([^)]*\):\s*([^\n]+)'
                for match in re.finditer(param_pattern, args_text):
                    param_name = match.group(1)
                    param_desc = match.group(2).strip()
                    param_descriptions[param_name] = param_desc
        
        # Parse parameters for args schema
        fields = {}
        fields["env_name"] = (str, Field(default=None, description="The environment name"))
        fields["action_name"] = (str, Field(default=action_name, description="The action name to execute"))
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Get the type annotation
            field_type = param.annotation
            if field_type == inspect.Parameter.empty:
                field_type = str  # Default to string if no annotation
            
            # Get description from docstring or create default
            field_description = param_descriptions.get(param_name, f"Parameter {param_name} of type {field_type.__name__}")
            
            # Handle default values
            if param.default != inspect.Parameter.empty:
                fields[param_name] = (field_type, Field(default=param.default, description=field_description))
            else:
                fields[param_name] = (field_type, Field(description=field_description))
        
        # Create args schema class with camelCase naming
        args_schema_name = inflection.camelize(action_name) + 'InputArgs'
        args_schema = create_model(args_schema_name, **fields)

        return args_schema
    
    def build_environment(self, env_name: str, env_config: Optional[Dict[str, Any]] = None):
        """Build an environment by name
        
        Args:
            env_name: Environment name
            
        Returns:
            BaseEnvironment: Environment instance or None if not found
        """
        env_info = self._registered_environments.get(env_name)
        env_info.env_config = env_config
        env_info.env_instance = env_info.env_class(**env_config)
        
        self._registered_environments[env_name] = env_info
        
        return env_info.env_instance
    
    async def get_state(self, env_name: str) -> Optional[str]:
        """Get the state of an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            str: State of the environment or None if not found
        """
        env = self._registered_environments.get(env_name).env_instance
        if not env:
            raise ValueError(f"Environment '{env_name}' not found")
        return await env.get_state()
    
    async def call_action(self, env_name: str, action_name: str, **kwargs) -> Any:
        """Call an action by name with arguments
        
        Args:
            env_name: Environment name
            action_name: Action name
            arguments: Action arguments
            
        Returns:
            Any: Action result
        """
        env_info = self._registered_environments.get(env_name)
        if not env_info:
            raise ValueError(f"Environment '{env_name}' not found")
        
        # Find the action
        action_info = env_info.actions.get(action_name)
        
        if not action_info:
            raise ValueError(f"Action '{action_name}' not found in environment '{env_name}'")
        
        if not action_info.function:
            raise ValueError(f"Action '{action_name}' has no function implementation")
        
        # Get environment instance
        env_instance = env_info.env_instance
        if not env_instance:
            # Create instance if not exists
            env_instance = env_info.env_class()
            env_info.env_instance = env_instance
        
        # Call the action function
        if asyncio.iscoroutinefunction(action_info.function):
            result = await action_info.function(env_instance, **kwargs)
        else:
            result = action_info.function(env_instance, **kwargs)
        
        return result
    
    def get_registered_environments(self) -> List[EnvironmentInfo]:
        """Get list of registered environments
        
        Returns:
            List[EnvironmentInfo]: List of registered environment information
        """
        return list(self._registered_environments.values())
    
    def get_environment_info(self, env_name: str) -> Optional[EnvironmentInfo]:
        """Get environment information by type
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentInfo: Environment information or None if not found
        """
        return self._registered_environments.get(env_name)
    
    def get_action_info(self, env_name: str, action_name: str) -> Optional[ActionInfo]:
        """Get action information by name
        
        Args:
            env_name: Environment name
            action_name: Action name
            
        Returns:
            ActionInfo: Action information or None if not found
        """
        env_info = self._registered_environments.get(env_name)
        if not env_info:
            raise ValueError(f"Environment '{env_name}' not found")
        return env_info.actions.get(action_name)
    
    def get_actions(self, env_name: str) -> Optional[Dict[str, ActionInfo]]:
        """Get all actions for an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            Dict[str, ActionInfo]: Dictionary of actions or None if not found
        """
        return self._registered_environments.get(env_name).actions
    
    def list_actions(self) -> Dict[str, Dict[str, ActionInfo]]:
        """List all actions for all environments
        
        Returns:
            Dict[str, Dict[str, ActionInfo]]: Dictionary of environments and their actions
        """
        actions = {}
        for env_name in self._registered_environments:
            actions[env_name] = self._registered_environments.get(env_name).actions
        return actions
    
ecp = ECPServer()