"""ACP Server

Server implementation for the Agent Context Protocol.
"""

from typing import Any, Dict, List, Optional, Type

from src.agents.protocol.types import AgentInfo
from src.agents.protocol.agent import BaseAgent
from src.agents.protocol.context import AgentContextManager
from src.config import config
from src.logger import logger

class ACPServer:
    """ACP Server for managing agent registration and lifecycle"""
    
    def __init__(self):
        self._registered_agents: Dict[str, AgentInfo] = {}  # agent_name -> AgentInfo
        self.agent_context_manager = AgentContextManager()
    
    def agent(self):
        """Decorator to register an agent class
        
        Args:
            name (str): Agent name (defaults to class name)
            type (str): Agent type (defaults to class name)
            description (str): Agent description
            metadata (Dict[str, Any]): Additional metadata
        """
        def decorator(cls: Type[BaseAgent]):
            model_fields = cls.model_fields
        
            name = model_fields['name'].default
            description = model_fields['description'].default
            args_schema = model_fields['args_schema'].default
            metadata = model_fields['metadata'].default
            type = model_fields['type'].default
            
            # Create AgentInfo and store it
            agent_info = AgentInfo(
                name=name,
                type=type,
                description=description,
                args_schema=args_schema,
                cls=cls,
                instance=None,
                metadata=metadata if metadata is not None else {}
            )
            
            self._registered_agents[name] = agent_info
            
            return cls
        return decorator
    
    async def initialize(self, agent_names: List[str]):
        """Initialize environments by names
        
        Args:
            agent_names (List[str]): List of agent names
        """
        logger.info(f"| 🎮 Initializing {len(self._registered_agents)} agents with context manager...")
        
        for agent_name, agent_info in self._registered_agents.items():
            if agent_name in agent_names:
                logger.debug(f"| 🔧 Initializing agent: {agent_name}")
                
                def agent_factory():
                    agent_config = config.get(f"{agent_name}_agent", None)
                    if agent_config:
                        return agent_info.cls(**agent_config)
                    else:
                        return agent_info.cls()
                
                await self.agent_context_manager.build(agent_info, agent_factory)
                logger.debug(f"| ✅ Agent {agent_name} initialized")
            else:
                logger.info(f"| ⏭️ Agent {agent_name} not found")
                
        logger.info("| ✅ Agents initialization completed")
    
    def get_info(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            AgentInfo: Agent information or None if not found
        """
        return self.agent_context_manager.get_info(agent_name)
    
    
    def list(self) -> Dict[str, AgentInfo]:
        """List all registered agents
        
        Returns:
            Dict[str, AgentInfo]: Dictionary of agent names and their information
        """
        return self.agent_context_manager.list()
    
    def get_info(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            AgentInfo: Agent information or None if not found
        """
        return self.agent_context_manager.get_info(agent_name)
    
    def get(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent instance by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            BaseAgent: Agent instance or None if not found
        """
        agent_info = self.agent_context_manager.get(agent_name)
        return agent_info.instance if agent_info else None
    
    def invoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke an agent method using context manager.
        
        Args:
            name: Name of the agent
            input: Input for the agent
            
        Returns:
            Agent result
        """
        return self.agent_context_manager.invoke(name, input, **kwargs)
    
    async def ainvoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke an agent method asynchronously using context manager.
        
        Args:
            name: Name of the agent
            input: Input for the agent
            
        Returns:
            Agent result
        """
        return await self.agent_context_manager.ainvoke(name, input, **kwargs)
    
    def cleanup(self):
        """Cleanup all agents using context manager."""
        self.agent_context_manager.cleanup()


# Global ACP server instance
acp = ACPServer()
