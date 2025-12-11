"""Environment Context Manager for managing environment lifecycle and resources."""

import asyncio
import atexit
from typing import Any, Dict, Callable, Optional, List

from src.logger import logger
from src.environment.types import Environment, EnvironmentConfig
from typing import Union, Type, Any

class EnvironmentContextManager:
    """Global context manager for all environments."""
    
    def __init__(self):
        """Initialize the environment context manager."""
        self._environment_configs: Dict[str, EnvironmentConfig] = {}  # Store environment metadata
        self._cleanup_registered = False
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
            
    def invoke(self, name: str, action: str, input: Any, **kwargs) -> Any:
        """Invoke an environment action.
        
        Args:
            name: Name of the environment
            action: Name of the action
            input: Input for the action
            **kwargs: Keyword arguments for the action
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.ainvoke(name, action, input, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in synchronous execution: {str(e)}"
    
    async def ainvoke(self, name: str, action: str, input: Dict[str, Any], **kwargs) -> Any:
        """Invoke an environment action asynchronously.
        
        Args:
            name: Name of the environment
            action: Name of the action
            input: Input for the action
            **kwargs: Keyword arguments for the action
        """
        
        if name in self._environment_configs:
            env_config = self._environment_configs[name]
            
            instance = env_config.instance
            action_config = env_config.actions.get(action)
            if action_config is None:
                raise ValueError(f"Action {action} not found in environment {name}")
            action_function = action_config.function
            
            return await action_function(instance, **input)
        else:
            raise ValueError(f"Environment {name} not found")
    
    async def initialize(self):
        """Initialize the environment context manager."""
        pass
    
    async def build(self, 
              env_config: EnvironmentConfig,
              env_factory: Callable,
              **kwargs
              ) -> EnvironmentConfig:
        """Create and store an environment instance.
        
        Args:
            env_config: Environment configuration
            env_factory: Function to create the environment instance
            
        Returns:
            EnvironmentConfig: Environment configuration
        """
        if env_config.name in self._environment_configs:
            existing_config = self._environment_configs[env_config.name]
            # If instance already exists, return it
            if existing_config.instance is not None:
                return existing_config
            # Otherwise, update config and create instance
            existing_config.config = env_config.config
            existing_config.cls = env_config.cls
            env_config = existing_config
        
        try:
            # Create environment instance
            instance = env_factory()
            
            # Initialize environment
            if hasattr(instance, "initialize"):
                await instance.initialize()
            
            # Store instance
            env_config.instance = instance
            
            # Store metadata
            self._environment_configs[env_config.name] = env_config
            
            logger.info(f"| ✅ Environment {env_config.name} created and stored")
            return env_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to create environment {env_config.name}: {e}")
            raise
        
    async def register(self, env: Union[Environment, Type[Environment]], *, override: bool = False, **kwargs: Any) -> EnvironmentConfig:
        """Register an environment class or instance.
        
        Args:
            env: Environment class or instance
            override: Whether to override existing registration
            **kwargs: Configuration for environment initialization
            
        Returns:
            EnvironmentConfig: Environment configuration
        """
        # Create temporary instance to get name and description
        try:
            if isinstance(env, Environment):
                env_name = env.name
                env_description = env.description
                env_type = env.type
                env_cls = type(env)
                env_instance = env
            elif isinstance(env, type) and issubclass(env, Environment):
                # Try to create temporary instance
                try:
                    temp_instance = env(**kwargs)
                    env_name = temp_instance.name
                    env_description = temp_instance.description
                    env_type = temp_instance.type
                    env_cls = env
                    env_instance = None
                except Exception:
                    # If instantiation fails, try to get from class attributes
                    env_name = getattr(env, 'name', None)
                    env_description = getattr(env, 'description', '')
                    env_type = getattr(env, 'type', '')
                    env_cls = env
                    env_instance = None
                    
                    if not env_name:
                        raise ValueError(f"Environment class {env.__name__} has no name")
            else:
                raise TypeError(f"Expected Environment instance or subclass, got {type(env)!r}")
            
            if not env_name:
                raise ValueError("Environment.name cannot be empty.")
            
            if env_name in self._environment_configs and not override:
                raise ValueError(f"Environment '{env_name}' already registered. Use override=True to replace it.")
            
            # Collect actions from the class or instance
            actions = {}
            target = env_cls if env_instance is None else type(env_instance)
            for attr_name in dir(target):
                attr = getattr(target, attr_name)
                if hasattr(attr, '_action_name'):
                    action_name = getattr(attr, '_action_name')
                    from src.environment.types import ActionConfig
                    action_config = ActionConfig(
                        env_name=env_name,
                        name=action_name,
                        type=getattr(attr, '_action_type', ''),
                        description=getattr(attr, '_action_description', ''),
                        args_schema=getattr(attr, '_args_schema', None),
                        function=getattr(attr, '_action_function', None),
                        metadata=getattr(attr, '_metadata', {})
                    )
                    actions[action_name] = action_config
            
            # Create EnvironmentConfig
            if env_instance is not None:
                # Registering an instance
                env_config = EnvironmentConfig(
                    name=env_name,
                    type=env_type,
                    description=env_description,
                    rules="",  # Will be generated by server
                    args_schema=getattr(env_instance, 'args_schema', None),
                    actions=actions,
                    cls=env_cls,
                    config={},
                    instance=env_instance,
                    metadata=getattr(env_instance, 'metadata', {})
                )
            else:
                # Registering a class - store config for lazy loading
                env_config = EnvironmentConfig(
                    name=env_name,
                    type=env_type,
                    description=env_description,
                    rules="",  # Will be generated by server
                    args_schema=getattr(env_cls, 'args_schema', None),
                    actions=actions,
                    cls=env_cls,
                    config=kwargs,
                    instance=None,  # Will be created on initialize
                    metadata={}
                )
            
            # Store environment config
            self._environment_configs[env_name] = env_config
            
            logger.debug(f"| 📝 Registered environment config: {env_name}")
            
            return env_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register environment: {e}")
            raise
        
    async def get_state(self, env_name: str) -> Optional[Dict[str, Any]]:
        """Get the state of an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            Optional[Dict[str, Any]]: State of the environment or None if not found
        """
        env_config = self._environment_configs.get(env_name)
        if not env_config or not env_config.instance:
            raise ValueError(f"Environment '{env_name}' not found")
        return await env_config.instance.get_state()
        
    def list(self) -> List[str]:
        """Get list of registered environments
        
        Returns:
            List[str]: List of registered environment names
        """
        return [name for name in self._environment_configs.keys()]
    
    def get_info(self, env_name: str) -> Optional[EnvironmentConfig]:
        """Get environment configuration by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentConfig: Environment configuration or None if not found
        """
        return self._environment_configs.get(env_name)
    
    def get(self, env_name: str) -> Optional[Environment]:
        """Get environment instance by name
        
        Args:
            env_name: Environment name
            
        Returns:
            Environment: Environment instance or None if not found
        """
        env_config = self._environment_configs.get(env_name)
        if env_config:
            return env_config.instance
        return None
    
    def cleanup(self):
        """Cleanup all environment instances and resources."""
        try:
            # Clear instances and configs
            self._environment_configs.clear()
            logger.info("| 🧹 Environment context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during environment context manager cleanup: {e}")
