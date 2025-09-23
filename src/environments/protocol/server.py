"""ECP Server

Server implementation for the Environment Context Protocol with decorator support.
"""

import inspect
from typing import Any, Dict, List, Optional, Callable, Type, get_origin, get_args
from pydantic import BaseModel, create_model, Field
import inflection
import typing

from src.config import config
from src.logger import logger
from src.environments.protocol.types import EnvironmentInfo, ActionInfo
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol.context import EnvironmentContextManager

class ECPServer:
    """ECP Server for managing environments and actions with decorator support"""
    
    def __init__(self):
        self._registered_environments: Dict[str, EnvironmentInfo] = {}  # env_name -> EnvironmentInfo
        self.environment_context_manager = EnvironmentContextManager()
    
    def environment(self, 
                    name: str = None,
                    type: str = None, 
                    description: str = "", 
                    has_vision: bool = False,
                    additional_rules: Optional[Dict[str, str]] = None,
                    metadata: Optional[Dict[str, Any]] = None
                    ):
        """Decorator to register an environment class
        
        Args:
            name (str): Environment name (defaults to class name)
            env_type (str): Environment type (defaults to class name)
            description (str): Environment description
            has_vision (bool): Whether the environment has vision capabilities
            additional_rules (Dict[str, str]): Dictionary with custom rules for 'state', 'vision', 'interaction'
        """
        def decorator(cls: Type[BaseEnvironment]):
            env_name = name or cls.__name__
            env_type = type or cls.__name__.lower()
            
            # Store environment metadata
            cls._env_name = env_name
            cls._env_type = env_type
            cls._env_description = description
            cls._has_vision = has_vision
            cls._additional_rules = additional_rules
            cls._metadata = metadata
            
            # Collect all actions from the class
            actions = {}
            for attr_name in dir(cls):
                attr = getattr(cls, attr_name)
                if hasattr(attr, '_action_name'):
                    # Set the environment name for this action
                    attr._env_name = env_name
                    
                    # Create ActionInfo from the decorated method
                    action_info = ActionInfo(
                        env_name=env_name,
                        name=getattr(attr, '_action_name'),
                        type=getattr(attr, '_action_type', ''),
                        description=getattr(attr, '_action_description', ''),
                        args_schema=getattr(attr, '_args_schema', None),
                        function=getattr(attr, '_action_function', None),
                        metadata=getattr(attr, '_metadata', None)
                    )
                    
                    actions[getattr(attr, '_action_name')] = action_info
            
            # Generate rules
            final_rules = self._generate_environment_rules(
                env_name,
                env_type,
                description,
                actions,
                has_vision,
                additional_rules
            )
            
            # Create EnvironmentInfo and store it
            env_info = EnvironmentInfo(
                name=env_name,
                type=env_type,
                description=description,
                rules=final_rules,
                actions=actions,
                cls=cls,
                instance=None,
                metadata=metadata
            )
            
            self._registered_environments[env_name] = env_info
            
            return cls
        return decorator
    
    def action(self, 
               name: str = None, 
               type: str = None,
               description: str = "",
               metadata: Optional[Dict[str, Any]] = None):
        """Decorator to register an action (tool) for an environment
        
        Args:
            name: Action name (defaults to function name)
            type: Action type (defaults to function name)
            description: Action description,
            metadata: Action metadata
        """
        def decorator(func: Callable):
            action_name = name or func.__name__
            
            # Parse function signature to generate schemas
            sig = inspect.signature(func)
            args_schema = self._parse_function_signature(sig, action_name, func)
            
            # Store action metadata (env_name will be set later by environment decorator)
            func._action_name = action_name
            func._action_type = type
            func._action_description = description
            func._args_schema = args_schema
            func._action_function = func
            func._metadata = metadata
            
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
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Get the type annotation
            field_type = param.annotation
            if field_type == inspect.Parameter.empty:
                field_type = str  # Default to string if no annotation
            
            # Get description from docstring or create default
            if isinstance(field_type, str):
                type_name = field_type
            else:
                type_name = getattr(field_type, '__name__', str(field_type))
            field_description = param_descriptions.get(param_name, f"Parameter {param_name} of type {type_name}")
            
            # Handle default values
            if param.default != inspect.Parameter.empty:
                fields[param_name] = (field_type, Field(default=param.default, description=field_description))
            else:
                fields[param_name] = (field_type, Field(description=field_description))
        
        # Create args schema class with camelCase naming
        args_schema_name = inflection.camelize(action_name) + 'InputArgs'
        args_schema = create_model(args_schema_name, **fields)

        return args_schema
    
    def _get_type_string(self, func: Callable, param_name: str) -> str:
        """Get the type string for a parameter from function signature
        
        Args:
            func: The function object
            param_name: The parameter name
            
        Returns:
            str: The type string (e.g., 'str', 'Optional[str]', 'List[str]')
        """
        if not func:
            return 'unknown'
            
        try:
            sig = inspect.signature(func)
            param = sig.parameters.get(param_name)
            if not param:
                return 'unknown'
                
            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                return 'unknown'
            
            # Convert type annotation to string
            result = self._format_type_annotation(annotation)
            return result
            
        except Exception:
            return 'unknown'
    
    def _format_type_annotation(self, annotation) -> str:
        """Format type annotation to readable string
        
        Args:
            annotation: The type annotation
            
        Returns:
            str: Formatted type string
        """
        
        # Handle basic types
        if annotation is str:
            return 'str'
        elif annotation is int:
            return 'int'
        elif annotation is float:
            return 'float'
        elif annotation is bool:
            return 'bool'
        elif annotation is list:
            return 'List[Any]'
        elif annotation is dict:
            return 'Dict[str, Any]'
        
        # Handle typing constructs
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        if origin is typing.Union:
            # Handle Optional types (Union[SomeType, NoneType])
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                inner_type = self._format_type_annotation(non_none_type)
                return f"Optional[{inner_type}]"
            else:
                # Handle other Union types
                type_strs = [self._format_type_annotation(arg) for arg in args]
                return f"Union[{', '.join(type_strs)}]"
        
        elif origin is list:
            if args:
                return f"List[{self._format_type_annotation(args[0])}]"
            else:
                return 'List[Any]'
        
        elif origin is dict:
            if len(args) >= 2:
                key_type = self._format_type_annotation(args[0])
                value_type = self._format_type_annotation(args[1])
                return f"Dict[{key_type}, {value_type}]"
            else:
                return 'Dict[str, Any]'
        
        elif origin is tuple:
            if args:
                type_strs = [self._format_type_annotation(arg) for arg in args]
                return f"Tuple[{', '.join(type_strs)}]"
            else:
                return 'Tuple'
        
        elif origin is set:
            if args:
                return f"Set[{self._format_type_annotation(args[0])}]"
            else:
                return 'Set[Any]'
        
        # Handle built-in types and standard library types
        elif hasattr(annotation, '__name__'):
            # Check if it's a built-in type or from standard library
            if self._is_builtin_or_standard_type(annotation):
                return annotation.__name__
            else:
                # Custom type - use Any
                return 'Any'
        
        # Handle typing module types
        elif hasattr(annotation, '__module__') and annotation.__module__ == 'typing':
            return str(annotation).replace('typing.', '')
        
        # Fallback for custom types - use Any
        else:
            return 'Any'
    
    def _is_builtin_or_standard_type(self, annotation) -> bool:
        """Check if annotation is a built-in or standard library type
        
        Args:
            annotation: The type annotation
            
        Returns:
            bool: True if it's a built-in or standard library type
        """
        # Built-in types
        builtin_types = {str, int, float, bool, list, dict, tuple, set, bytes, bytearray}
        if annotation in builtin_types:
            return True
        
        # Check if it's from builtins module
        if hasattr(annotation, '__module__'):
            module = annotation.__module__
            if module in ('builtins', '__builtin__'):
                return True
            
            # Standard library modules
            standard_modules = (
                'collections', 'datetime', 'decimal', 'fractions', 'pathlib',
                'uuid', 'enum', 'dataclasses', 'typing', 'abc', 'io', 'os',
                'sys', 'json', 'pickle', 'copy', 'itertools', 'functools',
                'operator', 'math', 'random', 'statistics', 'time', 'calendar'
            )
            if any(module.startswith(std_mod) for std_mod in standard_modules):
                return True
        
        return False
    
    def _generate_environment_rules(self, 
                                   env_name: str, 
                                   env_type: str, 
                                   description: str, 
                                   actions: Dict[str, ActionInfo],
                                   has_vision: bool = False,
                                   additional_rules: Optional[Dict[str, str]] = None) -> str:
        """Generate environment rules from actions and metadata
        
        Args:
            env_name: Environment name
            env_type: Environment type
            description: Environment description
            actions: Dictionary of actions
            has_vision: Whether environment has vision capabilities
            additional_rules: Dictionary with custom rules for 'state', 'vision', 'interaction'
            
        Returns:
            str: Generated environment rules
        """
        # Start building the rules
        rules_parts = [f"<environment_{env_type}>"]
        
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
            
            for i, (action_name, action_info) in enumerate(sorted_actions, 1):
                rules_parts.append(f"{i}. {action_name}: {action_info.description}")
                
                # Add parameter information if available
                if action_info.args_schema:
                    try:
                        schema = action_info.args_schema.model_json_schema()
                        properties = schema.get('properties', {})
                        required = schema.get('required', [])
                        
                        for param_name, param_info in properties.items():
                            # Get the original type annotation from the function signature
                            param_type_str = self._get_type_string(action_info.function, param_name)
                            param_desc = param_info.get('description', f'Parameter {param_name}')
                            is_required = param_name in required
                            
                            
                            if is_required:
                                rules_parts.append(f"    - {param_name} ({param_type_str}): {param_desc}")
                            else:
                                default_val = param_info.get('default', 'None')
                                rules_parts.append(f"    - {param_name} ({param_type_str}): {param_desc} (default: {default_val})")
                                
                    except Exception:
                        # If schema parsing fails, just add basic info
                        rules_parts.append(f"    - Parameters: See function signature")
            
            rules_parts.append("Input format: JSON string with action-specific parameters.")
            rules_parts.append("Example: {\"name\": \"action_name\", \"args\": {\"action-specific parameters\"}}")
        
        rules_parts.append("</interaction>")
        
        # Close the environment tag
        rules_parts.append(f"</environment_{env_type}>")
        
        return "\n".join(rules_parts)
    
    async def initialize(self, env_names: List[str]):
        """Initialize environments by names
        
        Args:
            env_names: List of environment names
        """
        logger.info(f"| 🎮 Initializing {len(self._registered_environments)} environments with context manager...")
        
        for env_name, env_info in self._registered_environments.items():
            if env_name in env_names:
                logger.debug(f"| 🔧 Initializing environment: {env_name}")
                
                def environment_factory():
                    env_config = config.get(f"{env_name}_environment", None)
                    if env_config:
                        return env_info.cls(**env_config)
                    else:
                        return env_info.cls()
                
                await self.environment_context_manager.build(env_info, environment_factory)
                logger.debug(f"| ✅ Environment {env_name} initialized")
            else:
                logger.info(f"| ⏭️ Environment {env_name} not found")
                
        logger.info("| ✅ Environments initialization completed")
        
    
    async def get_state(self, env_name: str) -> Optional[Dict[str, Any]]:
        """Get the state of an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            Dict[str, Any]: State of the environment or None if not found
        """
        env = self._registered_environments.get(env_name).instance
        if not env:
            raise ValueError(f"Environment '{env_name}' not found")
        return await env.get_state()
    
    def list(self) -> List[str]:
        """Get list of registered environments
        
        Returns:
            List[EnvironmentInfo]: List of registered environment information
        """
        return [name for name in self._registered_environments.keys()]
    
    def get_info(self, env_name: str) -> Optional[EnvironmentInfo]:
        """Get environment information by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentInfo: Environment information or None if not found
        """
        return self._registered_environments.get(env_name)
    
    def get(self, env_name: str) -> Optional[BaseEnvironment]:
        """Get environment information by type
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentInfo: Environment information or None if not found
        """
        return self._registered_environments.get(env_name).instance
    
    def invoke(self, name: str, action: str, input: Any, **kwargs) -> Any:
        """Invoke an environment action using context manager.
        
        Args:
            name: Name of the environment
            action: Name of the action
            input: Input for the action
            **kwargs: Keyword arguments for the action
            
        Returns:
            Action result
        """
        return self.environment_context_manager.invoke(name, action, input, **kwargs)
    
    async def ainvoke(self, name: str, action: str, input: Any, **kwargs) -> Any:
        """Invoke an environment action asynchronously using context manager.
        
        Args:
            name: Name of the environment
            action: Name of the action
            input: Input for the action
            **kwargs: Keyword arguments for the action
            
        Returns:
            Action result
        """
        return await self.environment_context_manager.ainvoke(name, action, input, **kwargs)
    
    def cleanup(self):
        """Cleanup all environments using context manager."""
        self.environment_context_manager.cleanup()
    
ecp = ECPServer()