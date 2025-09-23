"""ACP Server

Server implementation for the Agent Context Protocol.
"""

from typing import Any, Dict, List, Optional, Type

from src.agents.protocol.types import AgentInfo
from src.agents.protocol.agent import BaseAgent
from src.agents.protocol.context import AgentContextManager

class ACPServer:
    """ACP Server for managing agent registration and lifecycle"""
    
    def __init__(self):
        self._registered_agents: Dict[str, AgentInfo] = {}  # agent_name -> AgentInfo
        self.agent_context_manager = AgentContextManager()
    
    def agent(self, 
              name: str = None,
              type: str = None, 
              description: str = "",
              metadata: Optional[Dict[str, Any]] = None
              ):
        """Decorator to register an agent class
        
        Args:
            name (str): Agent name (defaults to class name)
            type (str): Agent type (defaults to class name)
            description (str): Agent description
            metadata (Dict[str, Any]): Additional metadata
        """
        def decorator(cls: Type[BaseAgent]):
            agent_name = name or cls.__name__
            agent_type = type or cls.__name__.lower()
            
            # Store agent metadata
            cls._agent_name = agent_name
            cls._agent_type = agent_type
            cls._agent_description = description
            cls._agent_metadata = metadata
            
            # Create AgentInfo and store it
            agent_info = AgentInfo(
                name=agent_name,
                type=agent_type,
                description=description,
                cls=cls,
                instance=None,
                metadata=metadata
            )
            
            self._registered_agents[agent_name] = agent_info
            
            return cls
        return decorator
    
    def get_registered_agents(self) -> List[AgentInfo]:
        """Get list of registered agents
        
        Returns:
            List[AgentInfo]: List of registered agent information
        """
        return list(self._registered_agents.values())
    
    def get_agent_info(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            AgentInfo: Agent information or None if not found
        """
        return self._registered_agents.get(agent_name)
    
    
    def list_agents(self) -> Dict[str, AgentInfo]:
        """List all registered agents
        
        Returns:
            Dict[str, AgentInfo]: Dictionary of agent names and their information
        """
        return self._registered_agents.copy()
    
    def list(self) -> List[str]:
        """Get list of registered agent names
        
        Returns:
            List[str]: List of registered agent names
        """
        return [name for name in self._registered_agents.keys()]
    
    def get_info(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            AgentInfo: Agent information or None if not found
        """
        return self._registered_agents.get(agent_name)
    
    def get(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent instance by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            BaseAgent: Agent instance or None if not found
        """
        agent_info = self._registered_agents.get(agent_name)
        return agent_info.instance if agent_info else None
    
    def invoke(self, name: str, method: str, input: Any, **kwargs) -> Any:
        """Invoke an agent method using context manager.
        
        Args:
            name: Name of the agent
            method: Name of the method
            input: Input for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Method result
        """
        return self.agent_context_manager.invoke(name, method, input, **kwargs)
    
    async def ainvoke(self, name: str, method: str, input: Any, **kwargs) -> Any:
        """Invoke an agent method asynchronously using context manager.
        
        Args:
            name: Name of the agent
            method: Name of the method
            input: Input for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Method result
        """
        return await self.agent_context_manager.ainvoke(name, method, input, **kwargs)
    
    def cleanup(self):
        """Cleanup all agents using context manager."""
        self.agent_context_manager.cleanup()


# Global ACP server instance
acp = ACPServer()
