"""Agent Context Manager for managing agent lifecycle and resources."""

import asyncio
import atexit
from typing import Any, Dict, Callable

from src.logger import logger
from src.agents.protocol.types import AgentInfo

class AgentContextManager:
    """Global context manager for all agents."""
    
    def __init__(self):
        """Initialize the agent context manager."""
        self._agent_info: Dict[str, AgentInfo] = {}  # Store agent metadata
        self._cleanup_registered = False
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
            
    def invoke(self, name: str, method: str, input: Any, **kwargs) -> Any:
        """Invoke an agent method.
        
        Args:
            name: Name of the agent
            method: Name of the method
            input: Input for the method
            **kwargs: Keyword arguments for the method
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.ainvoke(name, method, input, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in synchronous execution: {str(e)}"
    
    async def ainvoke(self, name: str, method: str, input: Dict[str, Any], **kwargs) -> Any:
        """Invoke an agent method asynchronously.
        
        Args:
            name: Name of the agent
            method: Name of the method
            input: Input for the method
            **kwargs: Keyword arguments for the method
        """
        
        if name in self._agent_info:
            agent_info = self._agent_info[name]
            
            instance = agent_info.instance
            if hasattr(instance, method):
                method_func = getattr(instance, method)
                return await method_func(**input, **kwargs)  # type: ignore
            else:
                raise ValueError(f"Method {method} not found in agent {name}")
        else:
            raise ValueError(f"Agent {name} not found")
    
    async def build(self, 
              agent_info: AgentInfo,
              agent_factory: Callable,
              **kwargs
              ) -> AgentInfo:
        """Create and store an agent instance.
        
        Args:
            agent_info: Agent information
            agent_factory: Function to create the agent instance
            
        Returns:
            AgentInfo: Agent information
        """
        if agent_info.name in self._agent_info:
            return self._agent_info[agent_info.name]
        
        try:
            # Create agent instance
            instance = agent_factory()
            
            # Store instance
            agent_info.instance = instance
            
            # Store metadata
            self._agent_info[agent_info.name] = agent_info
            
            logger.info(f"| ✅ Agent {agent_info.name} created and stored")
            return agent_info
            
        except Exception as e:
            logger.error(f"| ❌ Failed to create agent {agent_info.name}: {e}")
            raise
    
    def cleanup(self):
        """Cleanup all agent instances and resources."""
        try:
            # Clear instances and info
            self._agent_info.clear()
            logger.info("| 🧹 Agent context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during agent context manager cleanup: {e}")
