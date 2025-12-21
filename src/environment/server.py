"""ECP Server

Server implementation for the Environment Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union, Callable
import os
from pydantic import BaseModel, ConfigDict, Field
import inspect

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
    contract_path: str = Field(default=None, description="The path to save the environment contract")
    
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
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 ECP Server base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")
        
        # Initialize environment context manager
        self.environment_context_manager = EnvironmentContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
            model_name="openrouter/gemini-3-flash-preview",
            embedding_model_name="openrouter/text-embedding-3-large"
        )
        await self.environment_context_manager.initialize(env_names=env_names)
        
        logger.info("| ✅ Environments initialization completed")
        
    async def get_contract(self) -> str:
        """Get the contract for all environments"""
        return await self.environment_context_manager.load_contract()
    
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
            
            func._action_name = action_name
            func._action_description = description
            func._action_function = func
            func._action_metadata = metadata if metadata is not None else {}
            
            return func
        return decorator
    
    async def register(self, 
                       env_cls: Type[Environment],
                       env_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> EnvironmentConfig:
        """Register an environment class asynchronously.
        
        Args:
            env_cls: Environment class to register
            env_config_dict: Configuration dict for environment initialization
            override: Whether to override existing registration
            version: Optional version string
            
        Returns:
            EnvironmentConfig: Environment configuration
        """
        env_config = await self.environment_context_manager.register(
            env_cls, 
            env_config_dict=env_config_dict, 
            override=override,
            version=version
        )
        self._registered_configs[env_config.name] = env_config
        return env_config
    
    async def list(self) -> List[str]:
        """List all registered environments
        
        Returns:
            List[str]: List of environment names
        """
        return await self.environment_context_manager.list()
    
    
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
    
    async def update(self, 
                     env_cls: Type[Environment],
                     env_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> EnvironmentConfig:
        """Update an existing environment with new configuration and create a new version
        
        Args:
            env_cls: New environment class with updated implementation
            env_config_dict: Configuration dict for environment initialization
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            EnvironmentConfig: Updated environment configuration
        """
        env_config = await self.environment_context_manager.update(
            env_cls, env_config_dict=env_config_dict, new_version=new_version, description=description
        )
        self._registered_configs[env_config.name] = env_config
        return env_config
    
    async def copy(self, 
                  env_name: str,
                  new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, 
                  new_config: Optional[Dict[str, Any]] = None) -> EnvironmentConfig:
        """Copy an existing environment
        
        Args:
            env_name: Name of the environment to copy
            new_name: New name for the copied environment. If None, uses original name.
            new_version: New version for the copied environment. If None, increments version.
            new_config: New configuration dict for the copied environment. If None, uses original config.
            
        Returns:
            EnvironmentConfig: New environment configuration
        """
        env_config = await self.environment_context_manager.copy(
            env_name, new_name, new_version, new_config
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
            name (str): Name of the environment
            action (str): Name of the action
            input (Dict[str, Any]): Input for the action
            
        Returns:
            Any: Action result
        """
        return await self.environment_context_manager(name, action, input)


# Global ECP server instance
ecp = ECPServer()
