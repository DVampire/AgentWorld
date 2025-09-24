"""Agent Context Manager for managing agent lifecycle and resources."""

import asyncio
import atexit
from typing import Any, Dict, Callable, Optional, List

from src.logger import logger
from src.agents.protocol.types import AgentInfo
from src.agents.protocol.agent import BaseAgent

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
            
    def invoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke an agent method.
        
        Args:
            name: Name of the agent
            input: Input for the agent
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.ainvoke(name, input, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in synchronous execution: {str(e)}"
    
    async def ainvoke(self, name: str, input: Dict[str, Any], **kwargs) -> Any:
        """Invoke an agent method asynchronously.
        
        Args:
            name: Name of the agent
            input: Input for the agent
            **kwargs: Keyword arguments for the agent
        """
        
        if name in self._agent_info:
            agent_info = self._agent_info[name]
            
            instance = agent_info.instance
            
            return await instance.ainvoke(**input, **kwargs)
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
        
    async def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent instance by name
        
        Args:
            name: Name of the agent
        """
        return self._agent_info.get(name).instance if self._agent_info.get(name) else None
    
    def get_info(self, name: str) -> Optional[AgentInfo]:
        """Get an agent information by name
        
        Args:
            name: Name of the agent
        """
        return self._agent_info.get(name)
    
    def list(self) -> List[str]:
        """Get list of registered agents
        
        Returns:
            List[str]: List of agent names
        """
        return [name for name in self._agent_info.keys()]
    
    def cleanup(self):
        """Cleanup all agent instances and resources."""
        try:
            # Clear instances and info
            self._agent_info.clear()
            logger.info("| 🧹 Agent context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during agent context manager cleanup: {e}")
