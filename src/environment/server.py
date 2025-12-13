"""ECP Server
Server implementation for the Environment Context Protocol with decorator support.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Callable, Type, Union
from pydantic import BaseModel, ConfigDict, Field
import inflection

from src.config import config
from src.logger import logger
from src.environment.types import EnvironmentConfig, ActionConfig, Environment
from src.environment.context import EnvironmentContextManager
from src.utils import assemble_project_path

class ECPServer(BaseModel):
    """ECP Server for managing environments and actions with decorator support"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the environments")
    save_path: str = Field(default=None, description="The path to save the environments")
    
    def __init__(self, **kwargs):
        """Initialize the ECP Server."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, EnvironmentConfig] = {}  # env_name -> EnvironmentConfig
        self._pending_registrations: List[Type[Environment]] = []  # Classes to register on initialize
        self._all_environment_classes: List[Type[Environment]] = []  # All discovered environment classes
    
    async def initialize(self, env_names: Optional[List[str]] = None):
        """Initialize environments by names using environment context manager with concurrent support.
        
        Args:
            env_names: List of environment names to initialize. If None, initialize all registered environments.
        """
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "environment"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "environment.json")
        logger.info(f"| 📁 ECP Server base directory: {self.base_dir} and save path: {self.save_path}")
        
        # Initialize environment context manager (this will trigger discovery if auto_discover is True)
        self.environment_context_manager = EnvironmentContextManager(base_dir=self.base_dir, save_path=self.save_path)
        await self.environment_context_manager.initialize()
        
        # Sync registered_configs from context manager after discovery
        for env_name in self.environment_context_manager.list():
            env_config = self.environment_context_manager.get_info(env_name)
            if env_config and env_name not in self._registered_configs:
                self._registered_configs[env_name] = env_config
        
        environments_to_init = env_names if env_names is not None else list(self._registered_configs.keys())
        
        logger.info(f"| 🎮 Initializing {len(environments_to_init)} environments with context manager...")
        
        # Prepare initialization tasks for concurrent execution
        async def init_environment(env_name: str):
            # Get environment config
            env_config = self._registered_configs.get(env_name)
            if env_config is None:
                logger.warning(f"| ⚠️ Environment {env_name} not found in registered configs")
                return
            
            # Skip if already initialized
            if env_config.instance is not None:
                logger.debug(f"| ⏭️ Environment {env_name} already initialized")
                return
            
            logger.debug(f"| 🔧 Initializing environment: {env_name}")
            
            # Get environment config from global config if available
            global_config = config.get(f"{env_name}_environment", {})
            if global_config:
                # Merge with existing config
                env_config.config = {**env_config.config, **global_config}
            
            # Create environment instance and store it in context manager
            try:
                def environment_factory():
                    if env_config.config:
                        return env_config.cls(**env_config.config)
                    else:
                        return env_config.cls()
                
                await self.environment_context_manager.build(env_config, environment_factory)
                
                # Update actions from instance
                if env_config.instance and hasattr(env_config.instance, 'actions'):
                    env_config.actions = env_config.instance.actions.copy()
                
                # Sync to registered_configs for consistency
                self._registered_configs[env_name] = env_config
                logger.debug(f"| ✅ Environment {env_name} initialized")
            except Exception as e:
                logger.error(f"| ❌ Failed to initialize environment {env_name}: {e}")
        
        # Initialize environments concurrently
        init_tasks = [init_environment(env_name) for env_name in environments_to_init]
        await asyncio.gather(*init_tasks, return_exceptions=True)
        
        logger.info("| ✅ Environments initialization completed")
    
    def action(self, 
               name: str = None, 
               description: str = "",
               metadata: Optional[Dict[str, Any]] = None):
        """Decorator to register an action (tool) for an environment
        
        Actions will be registered to the environment instance's actions dictionary during instantiation.
        
        Args:
            name: Action name (defaults to function name)
            description: Action description
            metadata: Action metadata
        """
        def decorator(func: Callable):
            action_name = name or func.__name__
            
            # Store action metadata for registration in instance's __init__
            func._action_name = action_name
            func._action_description = description
            func._action_function = func
            func._metadata = metadata if metadata is not None else {}
            
            return func
        return decorator
    
    async def register(self, env: Union[Environment, Type[Environment]], *, override: bool = False, **kwargs: Any) -> EnvironmentConfig:
        """Register an environment class or instance asynchronously.
        
        Args:
            env: Environment class or instance to register
            override: Whether to override existing registration
            **kwargs: Configuration for environment initialization
            
        Returns:
            EnvironmentConfig: Environment configuration
        """
        env_config = await self.environment_context_manager.register(env, override=override, **kwargs)
        self._registered_configs[env_config.name] = env_config
        return env_config
    
    async def get_state(self, env_name: str) -> Optional[Dict[str, Any]]:
        """Get the state of an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            Optional[Dict[str, Any]]: State of the environment or None if not found
        """
        return await self.environment_context_manager.get_state(env_name)
    
    def list(self) -> List[str]:
        """Get list of registered environments
        
        Returns:
            List[str]: List of registered environment names
        """
        return self.environment_context_manager.list()
    
    def get_info(self, env_name: str) -> Optional[EnvironmentConfig]:
        """Get environment configuration by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentConfig: Environment configuration or None if not found
        """
        return self.environment_context_manager.get_info(env_name)
    
    def get(self, env_name: str) -> Optional[Environment]:
        """Get environment instance by name
        
        Args:
            env_name: Environment name
            
        Returns:
            Environment: Environment instance or None if not found
        """
        return self.environment_context_manager.get(env_name)
    
    async def __call__(self, name: str, action: str, input: Dict[str, Any]) -> Any:
        """Call an environment action
        
        Args:
            name: Name of the environment
            action: Name of the action
            input: Input for the action
            
        Returns:
            Action result
        """
        return await self.environment_context_manager(name, action, input)
    
    async def update(self, env_name: str, env: Union[Environment, Type[Environment]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> EnvironmentConfig:
        """Update an existing environment with new configuration and create a new version
        
        Args:
            env_name: Name of the environment to update
            env: New environment class or instance with updated implementation
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for environment initialization
            
        Returns:
            EnvironmentConfig: Updated environment configuration
        """
        env_config = await self.environment_context_manager.update(
            env_name, env, new_version, description, **kwargs
        )
        self._registered_configs[env_config.name] = env_config
        return env_config
    
    async def copy(self, env_name: str, new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, **override_config) -> EnvironmentConfig:
        """Copy an existing environment
        
        Args:
            env_name: Name of the environment to copy
            new_name: New name for the copied environment. If None, uses original name.
            **override_config: Configuration overrides
            
        Returns:
            EnvironmentConfig: New environment configuration
        """
        env_config = await self.environment_context_manager.copy(env_name, new_name, new_version, **override_config)
        self._registered_configs[env_config.name] = env_config
        return env_config
    
    async def unregister(self, env_name: str) -> bool:
        """Unregister an environment
        
        Args:
            env_name: Name of the environment to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.environment_context_manager.unregister(env_name)
        if success and env_name in self._registered_configs:
            del self._registered_configs[env_name]
        return success
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all environment configurations to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        return await self.environment_context_manager.save_to_json(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load environment configurations from JSON
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically initialize environments after loading
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        success = await self.environment_context_manager.load_from_json(file_path, auto_initialize)
        if success:
            # Sync registered_configs
            for env_name in self.environment_context_manager.list():
                env_config = self.environment_context_manager.get_info(env_name)
                if env_config:
                    self._registered_configs[env_name] = env_config
        return success
    
    def cleanup(self):
        """Cleanup all environments using context manager."""
        self.environment_context_manager.cleanup()
    
ecp = ECPServer()