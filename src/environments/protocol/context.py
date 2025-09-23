"""Environment Context Manager for managing environment lifecycle and resources."""

import asyncio
import atexit
from typing import Any, Dict, Callable, Optional, List

from src.logger import logger
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol.types import EnvironmentInfo

class EnvironmentContextManager:
    """Global context manager for all environments."""
    
    def __init__(self):
        """Initialize the environment context manager."""
        self._environment_info: Dict[str, EnvironmentInfo] = {}  # Store environment metadata
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
        
        if name in self._environment_info:
            env_info = self._environment_info[name]
            
            instance = env_info.instance
            action_function = env_info.actions.get(action).function
            
            return await action_function(instance, **input, **kwargs)  # type: ignore
        else:
            raise ValueError(f"Environment {name} not found")
    
    async def build(self, 
              env_info: EnvironmentInfo,
              env_factory: Callable,
              **kwargs
              ) -> EnvironmentInfo:
        """Create and store an environment instance.
        
        Args:
            env_info: Environment information
            env_factory: Function to create the environment instance
            
        Returns:
            EnvironmentInfo: Environment information
        """
        if env_info.name in self._environment_info:
            return self._environment_info[env_info.name]
        
        try:
            # Create environment instance
            instance = env_factory()
            
            # Store instance
            env_info.instance = instance
            
            # Store metadata
            self._environment_info[env_info.name] = env_info
            
            logger.info(f"| ✅ Environment {env_info.name} created and stored")
            return instance
            
        except Exception as e:
            logger.error(f"| ❌ Failed to create environment {env_info.name}: {e}")
            raise
        
    async def get_state(self, env_name: str) -> Optional[Dict[str, Any]]:
        """Get the state of an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            Optional[Dict[str, Any]]: State of the environment or None if not found
        """
        env = self._environment_info.get(env_name).instance
        if not env:
            raise ValueError(f"Environment '{env_name}' not found")
        return await env.get_state()
        
    def list(self) -> List[str]:
        """Get list of registered environments
        
        Returns:
            List[EnvironmentInfo]: List of registered environment information
        """
        return [name for name in self._environment_info.keys()]
    
    def get_info(self, env_name: str) -> Optional[EnvironmentInfo]:
        """Get environment information by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentInfo: Environment information or None if not found
        """
        return self._environment_info.get(env_name)
    
    def get(self, env_name: str) -> Optional[BaseEnvironment]:
        """Get environment information by type
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentInfo: Environment information or None if not found
        """
        return self._environment_info.get(env_name).instance
    
    def cleanup(self):
        """Cleanup all environment instances and resources."""
        try:
            # Clear instances and info
            self._environment_info.clear()
            logger.info("| 🧹 Environment context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during environment context manager cleanup: {e}")
