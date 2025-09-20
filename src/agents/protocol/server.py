"""ACP Server

Server implementation for the Agent Context Protocol.
"""

from typing import Any, Dict, List, Optional, Type

from src.agents.protocol.types import AgentInfo
from src.agents.base_agent import BaseAgent

class ACPServer:
    """ACP Server for managing agent registration and capabilities"""
    
    def __init__(self):
        self._registered_agents: Dict[str, AgentInfo] = {}  # agent_name -> AgentInfo
    
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
                type=type,
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


# Global ACP server instance
acp = ACPServer()
