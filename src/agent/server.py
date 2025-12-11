"""ACP Server

Server implementation for the Agent Context Protocol.
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
    """ACP Server for managing agent registration and lifecycle"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents")
    
    def __init__(self, **kwargs):
        """Initialize the ACP Server."""
        super().__init__(**kwargs)
        self._registered_agents: Dict[str, AgentConfig] = {}  # agent_name -> AgentConfig
    
    async def initialize(self, agent_names: Optional[List[str]] = None):
        """Initialize agents by names using agent context manager with concurrent support.
        
        Args:
            agent_names: List of agent names to initialize. If None, initialize all registered agents.
        """
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "agent.json")
        logger.info(f"| 📁 ACP Server base directory: {self.base_dir} and save path: {self.save_path}")
        
        # Initialize agent context manager (this will trigger discovery if auto_discover is True)
        self.agent_context_manager = AgentContextManager(base_dir=self.base_dir, save_path=self.save_path)
        await self.agent_context_manager.initialize()
        
        # Sync registered_agents from context manager after discovery
        all_registered_agents = await self.agent_context_manager.list()
        for agent_name in all_registered_agents:
            agent_config = self.agent_context_manager._agent_configs.get(agent_name)
            if agent_config and agent_name not in self._registered_agents:
                self._registered_agents[agent_name] = agent_config
        
        # Use provided agent_names if specified, otherwise initialize all registered agents
        agents_to_init = agent_names if agent_names is not None else list(self._registered_agents.keys())
        
        logger.info(f"| 🎮 Initializing {len(agents_to_init)} agents with context manager...")
        
        # Prepare initialization tasks for concurrent execution
        async def init_agent(agent_name: str):
            # Get agent config
            agent_config = self._registered_agents.get(agent_name)
            if agent_config is None:
                logger.warning(f"| ⚠️ Agent {agent_name} not found in registered agents")
                return
            
            # Skip if already initialized
            if agent_config.instance is not None:
                logger.debug(f"| ⏭️ Agent {agent_name} already initialized")
                return
            
            logger.debug(f"| 🔧 Initializing agent: {agent_name}")
            
            # Get agent config from global config if available
            global_config = config.get(f"{agent_name}_agent", {})
            if global_config:
                # Merge with existing config
                agent_config.metadata = {**agent_config.metadata, **global_config}
            
            # Create agent instance and store it in context manager
            try:
                def agent_factory():
                    agent_config_dict = config.get(f"{agent_name}_agent", {})
                    # Agent requires workdir
                    if 'workdir' not in agent_config_dict:
                        agent_config_dict['workdir'] = config.workdir
                    if agent_config_dict:
                        return agent_config.cls(**agent_config_dict)
                    else:
                        return agent_config.cls(workdir=config.workdir)
                
                await self.agent_context_manager.build(agent_config, agent_factory)
                
                # Sync to registered_agents for consistency
                self._registered_agents[agent_name] = agent_config
                logger.debug(f"| ✅ Agent {agent_name} initialized")
            except Exception as e:
                logger.error(f"| ❌ Failed to initialize agent {agent_name}: {e}")
        
        # Initialize agents concurrently
        init_tasks = [init_agent(agent_name) for agent_name in agents_to_init]
        await asyncio.gather(*init_tasks, return_exceptions=True)
        
        logger.info("| ✅ Agents initialization completed")
        
    async def register(self, agent: Union[Agent, Type[Agent]], *, override: bool = False, **kwargs: Any) -> AgentConfig:
        """Register an agent class or instance asynchronously.
        
        Args:
            agent: Agent class or instance to register
            override: Whether to override existing registration
            **kwargs: Configuration for agent initialization
            
        Returns:
            AgentConfig: Agent configuration
        """
        agent_config = await self.agent_context_manager.register(agent, override=override, **kwargs)
        self._registered_agents[agent_config.name] = agent_config
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
        self._registered_agents[agent_config.name] = agent_config
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
        agent_config = await self.get_info(agent_name)
        return agent_config.instance if agent_config else None
    
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
    
    async def update(self, agent_name: str, agent: Union[Agent, Type[Agent]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> AgentConfig:
        """Update an existing agent with new configuration and create a new version
        
        Args:
            agent_name: Name of the agent to update
            agent: New agent class or instance with updated implementation
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for agent initialization
            
        Returns:
            AgentConfig: Updated agent configuration
        """
        agent_config = await self.agent_context_manager.update(agent_name, agent, new_version, description, **kwargs)
        self._registered_agents[agent_config.name] = agent_config
        return agent_config
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all agent configurations to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        return await self.agent_context_manager.save_to_json(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load agent configurations from JSON
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically initialize agents after loading
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        success = await self.agent_context_manager.load_from_json(file_path, auto_initialize)
        if success:
            # Sync registered_agents
            agent_names = await self.agent_context_manager.list()
            for agent_name in agent_names:
                agent_config = self.agent_context_manager._agent_configs.get(agent_name)
                if agent_config:
                    self._registered_agents[agent_name] = agent_config
        return success
    
    async def cleanup(self):
        """Cleanup all agents using context manager."""
        if hasattr(self, 'agent_context_manager') and self.agent_context_manager is not None:
            self.agent_context_manager.cleanup()


# Global ACP server instance
acp = ACPServer()
