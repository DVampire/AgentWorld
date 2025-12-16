"""ACP Server

Server implementation for the Agent Context Protocol with lazy loading support.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.logger import logger
from src.agent.types import AgentConfig, Agent
from src.agent.context import AgentContextManager
from src.utils import assemble_project_path

class ACPServer(BaseModel):
    """ACP Server for managing agent registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the ACP Server."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, AgentConfig] = {}  # agent_name -> AgentConfig

        
    async def initialize(self, agent_names: Optional[List[str]] = None):
        """Initialize agents by names using agent context manager with concurrent support.
        
        Args:
            agent_names: List of agent names to initialize. If None, initialize all registered agents.
        """
        
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "agent.json")
        logger.info(f"| 📁 ACP Server base directory: {self.base_dir} and save path: {self.save_path}")
        
        # Initialize agent context manager
        self.agent_context_manager = AgentContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            model_name="openrouter/gpt-4.1",
            embedding_model_name="openrouter/text-embedding-3-large",
        )
        await self.agent_context_manager.initialize(agent_names=agent_names)
        
        # Sync registered_configs from context manager after initialization
        agent_list = await self.agent_context_manager.list()
        for agent_name in agent_list:
            agent_config = await self.agent_context_manager.get_info(agent_name)
            if agent_config and agent_name not in self._registered_configs:
                self._registered_configs[agent_name] = agent_config
        
        logger.info("| ✅ Agents initialization completed")
        
    async def register(self, 
                       agent: Union[Agent, Type[Agent]],
                       config: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> AgentConfig:
        """Register an agent class or instance asynchronously.
        
        Args:
            agent: Agent class or instance to register
            config: Configuration dict for agent initialization (required when agent is a class)
            override: Whether to override existing registration
            version: Optional version string
            
        Returns:
            AgentConfig: Agent configuration
        """
        agent_config = await self.agent_context_manager.register(
            agent, 
            agent_config_dict=config, 
            override=override,
            version=version
        )
        self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    async def get_info(self, agent_name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            AgentConfig: Agent configuration or None if not found
        """
        return await self.agent_context_manager.get_info(agent_name)
    
    async def list(self) -> List[str]:
        """List all registered agents
            
        Returns:
            List[str]: List of agent names
        """
        return await self.agent_context_manager.list()
    
    
    async def get(self, agent_name: str) -> Optional[Agent]:
        """Get agent instance by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            Agent: Agent instance or None if not found
        """
        agent = await self.agent_context_manager.get(agent_name)
        return agent
    
    async def cleanup(self):
        """Cleanup all agents"""
        await self.agent_context_manager.cleanup()
        self._registered_configs.clear()
    
    async def update(self, 
                     agent_name: str, agent: Union[Agent, Type[Agent]], 
                     config: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> AgentConfig:
        """Update an existing agent with new configuration and create a new version
        
        Args:
            agent_name: Name of the agent to update
            agent: New agent class or instance with updated implementation
            config: Configuration dict for agent initialization (required when agent is a class)
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            AgentConfig: Updated agent configuration
        """
        agent_config = await self.agent_context_manager.update(
            agent_name, agent, agent_config_dict=config, new_version=new_version, description=description
        )
        self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    async def copy(self, agent_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> AgentConfig:
        """Copy an existing agent
        
        Args:
            agent_name: Name of the agent to copy
            new_name: New name for the copied agent. If None, uses original name.
            new_version: New version for the copied agent. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            AgentConfig: New agent configuration
        """
        agent_config = await self.agent_context_manager.copy(
            agent_name, new_name, new_version, **override_config
        )
        self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    async def unregister(self, agent_name: str) -> bool:
        """Unregister an agent
        
        Args:
            agent_name: Name of the agent to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.agent_context_manager.unregister(agent_name)
        if success and agent_name in self._registered_configs:
            del self._registered_configs[agent_name]
        return success
    
    async def restore(self, agent_name: str, version: str, auto_initialize: bool = True) -> Optional[AgentConfig]:
        """Restore a specific version of an agent from history
        
        Args:
            agent_name: Name of the agent
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored agent
            
        Returns:
            AgentConfig of the restored version, or None if not found
        """
        agent_config = await self.agent_context_manager.restore(agent_name, version, auto_initialize)
        if agent_config:
            self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    async def __call__(self, name: str, input: Dict[str, Any], **kwargs) -> Any:
        """Call an agent method using context manager.
        
        Args:
            name: Name of the agent
            input: Input for the agent
            **kwargs: Keyword arguments for the agent
            
        Returns:
            Agent result
        """
        return await self.agent_context_manager(name, input, **kwargs)


# Global ACP server instance
acp = ACPServer()
