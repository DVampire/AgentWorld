"""ECP Server

Server implementation for the Environment Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union, Callable
import asyncio
import os
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.environment.context import EnvironmentContextManager
from src.environment.types import Environment, EnvironmentConfig
from src.utils import assemble_project_path

class ECPServer(BaseModel):
    """ECP Server for managing environment registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the environments")
    save_path: str = Field(default=None, description="The path to save the environments")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the ECP Server."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, EnvironmentConfig] = {}  # env_name -> EnvironmentConfig

        
    async def initialize(self, env_names: Optional[List[str]] = None):
        """Initialize environments by names using environment context manager with concurrent support.
        
        Args:
            env_names: List of environment names to initialize. If None, initialize all registered environments.
            model_name: The model to use for the environments
            embedding_model_name: The model to use for the environment embeddings
        """
        
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "environment"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "environment.json")
        logger.info(f"| 📁 ECP Server base directory: {self.base_dir} and save path: {self.save_path}")
        
        # Initialize environment context manager
        self.environment_context_manager = EnvironmentContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            model_name="openrouter/gpt-4.1",
            embedding_model_name="openrouter/text-embedding-3-large"
        )
        await self.environment_context_manager.initialize(env_names=env_names)
        
        # Sync registered_configs from context manager after initialization
        env_list = await self.environment_context_manager.list()
        for env_name in env_list:
            env_config = await self.environment_context_manager.get_info(env_name)
            if env_config and env_name not in self._registered_configs:
                self._registered_configs[env_name] = env_config
        
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
    
    async def list(self, include_disabled: bool = False) -> List[str]:
        """List all registered environments
        
        Args:
            include_disabled: Whether to include disabled environments (not used for environments, kept for compatibility)
            
        Returns:
            List[str]: List of environment names
        """
        return await self.environment_context_manager.list(include_disabled=include_disabled)
    
    
    async def get(self, env_name: str) -> Optional[Environment]:
        """Get environment instance by name
        
        Args:
            env_name: Environment name
            
        Returns:
            Environment: Environment instance or None if not found
        """
        return await self.environment_context_manager.get(env_name)
    
    async def get_info(self, env_name: str) -> Optional[EnvironmentConfig]:
        """Get environment configuration by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentConfig: Environment configuration or None if not found
        """
        return await self.environment_context_manager.get_info(env_name)
    
    async def get_state(self, env_name: str) -> Optional[Dict[str, Any]]:
        """Get the state of an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            Optional[Dict[str, Any]]: State of the environment or None if not found
        """
        return await self.environment_context_manager.get_state(env_name)
    
    async def cleanup(self):
        """Cleanup all environments"""
        await self.environment_context_manager.cleanup()
        self._registered_configs.clear()
    
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
            new_version: New version for the copied environment. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            EnvironmentConfig: New environment configuration
        """
        env_config = await self.environment_context_manager.copy(
            env_name, new_name, new_version, **override_config
        )
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
    
    async def restore(self, env_name: str, version: str, auto_initialize: bool = True) -> Optional[EnvironmentConfig]:
        """Restore a specific version of an environment from history
        
        Args:
            env_name: Name of the environment
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored environment
            
        Returns:
            EnvironmentConfig of the restored version, or None if not found
        """
        env_config = await self.environment_context_manager.restore(env_name, version, auto_initialize)
        if env_config:
            self._registered_configs[env_config.name] = env_config
        return env_config
    
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


# Global ECP server instance
ecp = ECPServer()
