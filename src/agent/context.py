"""Agent Context Manager for managing agent lifecycle and resources with lazy loading."""
import asyncio
import os
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union, Tuple, TYPE_CHECKING
from datetime import datetime
import inflection
import json
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from src.logger import logger
from src.config import config
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.utils import (
    assemble_project_path,
    gather_with_concurrency,
    file_lock,
    generate_unique_id
)
from src.agent.types import Agent, AgentConfig, AgentContext
from src.version import version_manager
from src.dynamic import dynamic_manager
from src.registry import AGENT


class AgentContextManager(BaseModel):
    """Global context manager for all agents with lazy loading and version history."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents configuration JSON")
    contract_path: str = Field(default=None, description="The path to save the agent contract")

    def __init__(
        self,
        base_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        contract_path: Optional[str] = None,
        model_name: str = "openrouter/gemini-3-flash-preview",
        embedding_model_name: str = "openrouter/text-embedding-3-large",
        **kwargs: Any,
    ):
        """Initialize the agent context manager.

        Args:
            base_dir: Base directory for storing agent data
            save_path: Path to save agent configurations
            contract_path: Path to save agent contract
            model_name: The model name used for embedding text (via FaissService)
            embedding_model_name: The embedding model name (kept for parity with tools)
        """
        super().__init__(**kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| 📁 Agent context manager base directory: {self.base_dir}.")
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "agent.json")
        logger.info(f"| 📁 Agent context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Agent context manager contract path: {self.contract_path}.")

        # Current active configs (latest version)
        self._agent_configs: Dict[str, AgentConfig] = {}
        # Agent version history, e.g., {"agent_name": {"1.0.0": AgentConfig, ...}}
        self._agent_history_versions: Dict[str, Dict[str, AgentConfig]] = {}

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        self._cleanup_registered = False
        self._faiss_service: Optional[FaissService] = None
        self._variables_lock = asyncio.Lock()  # Lock for get/set trainable variables

    async def initialize(self, agent_names: Optional[List[str]] = None) -> None:
        """Initialize the agent context manager and all registered agents."""

        # Register agent-related symbols for auto-injection in dynamic code
        dynamic_manager.register_symbol("AGENT", AGENT)
        dynamic_manager.register_symbol("Agent", Agent)
        dynamic_manager.register_symbol("AgentConfig", AgentConfig)

        # Register agent context provider for automatic import injection
        def agent_context_provider():
            return {
                "AGENT": AGENT,
                "Agent": Agent,
                "AgentConfig": AgentConfig,
            }

        dynamic_manager.register_context_provider("agent", agent_context_provider)

        # Initialize Faiss service for agent embedding
        self._faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name,
        )

        # Load agents from AGENT registry
        agent_configs: Dict[str, AgentConfig] = {}
        registry_agent_configs: Dict[str, AgentConfig] = await self._load_from_registry()
        agent_configs.update(registry_agent_configs)

        # Load agents from code JSON (including older versions / dynamic agents)
        code_agent_configs: Dict[str, AgentConfig] = await self._load_from_code()

        # Merge code configs with registry configs, only override if code version is strictly greater
        for agent_name, code_config in code_agent_configs.items():
            if agent_name in agent_configs:
                registry_config = agent_configs[agent_name]
                if (
                    version_manager.compare_versions(
                        code_config.version, registry_config.version
                    )
                    > 0
                ):
                    logger.info(
                        f"| 🔄 Overriding agent {agent_name} from registry "
                        f"(v{registry_config.version}) with code version (v{code_config.version})"
                    )
                    agent_configs[agent_name] = code_config
                else:
                    logger.info(
                        f"| 📌 Keeping agent {agent_name} from registry (v{registry_config.version}), "
                        f"code version (v{code_config.version}) is not greater"
                    )
                    # If versions are equal, update the history with registry config (which has real class, not dynamic)
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        # Replace the code config in history with registry config to preserve real class reference
                        if agent_name in self._agent_history_versions:
                            self._agent_history_versions[agent_name][registry_config.version] = registry_config
            else:
                agent_configs[agent_name] = code_config

        # Filter agents by names if provided
        if agent_names is not None:
            agent_configs = {name: agent_configs[name] for name in agent_names if name in agent_configs}

        # Build all agents concurrently with a concurrency limit
        names = list(agent_configs.keys())
        tasks = [self.build(agent_configs[name]) for name in names]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for agent_name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize agent {agent_name}: {result}")
                continue
            self._agent_configs[agent_name] = result
            logger.info(f"| 🎮 Agent {agent_name} initialized")

        # Save agent configs to json file
        await self.save_to_json()
        # Save contract to file
        await self.save_contract(agent_names=agent_names)

        # Register async cleanup callback
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True

        logger.info("| ✅ Agents initialization completed")

    async def _load_from_registry(self) -> Dict[str, AgentConfig]:
        """Load agents from AGENT registry."""

        agent_configs: Dict[str, AgentConfig] = {}

        async def register_agent_class(agent_cls: Type[Agent]):
            """Register an agent class synchronously.
            
            Args:
                agent_cls: Agent class to register
            """
            try:
                # Get agent config from global config
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})
                agent_require_grad = agent_config_dict.get("require_grad", False) if agent_config_dict and "require_grad" in agent_config_dict else False
                
                # Get agent properties from agent class
                agent_name = agent_cls.model_fields['name'].default
                agent_description = agent_cls.model_fields['description'].default
                agent_metadata = agent_cls.model_fields['metadata'].default
                
                # Get or generate version from version_manager
                agent_version = await version_manager.get_version("agent", agent_name)
                
                # Get full module source code
                agent_code = dynamic_manager.get_full_module_source(agent_cls)
                
                agent_parameters = dynamic_manager.get_parameters(agent_cls)
                agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
                agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
                agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)
                
                # Create agent config (AgentConfig.id is auto-incremented internally if needed)
                agent_config = AgentConfig(
                    name=agent_name,
                    description=agent_description,
                    version=agent_version,
                    require_grad=agent_require_grad,
                    cls=agent_cls,
                    config=agent_config_dict,
                    instance=None,
                    function_calling=agent_function_calling,
                    text=agent_text,
                    args_schema=agent_args_schema,
                    metadata=agent_metadata,
                    code=agent_code,
                )
                
                # Store agent config
                agent_configs[agent_name] = agent_config
                
                # Store in version history (by version string)
                if agent_name not in self._agent_history_versions:
                    self._agent_history_versions[agent_name] = {}
                self._agent_history_versions[agent_name][agent_version] = agent_config
                
                # Register version to version manager
                await version_manager.register_version("agent", agent_name, agent_version)
                
                logger.info(f"| 📝 Registered agent: {agent_name} ({agent_cls.__name__})")
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register agent class {agent_cls.__name__}: {e}")
                raise

        import src.agent  # noqa: F401

        agent_classes = list(AGENT._module_dict.values())
        logger.info(f"| 🔍 Discovering {len(agent_classes)} agents from AGENT registry")

        tasks = [register_agent_class(agent_cls) for agent_cls in agent_classes]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(
            f"| ✅ Discovered and registered {success_count}/{len(agent_classes)} agents from AGENT registry"
        )

        return agent_configs

    async def _load_from_code(self):
        """Load agents from code files.
        
        JSON file content example:
        {
            "metadata": {
                "saved_at": str,  # "YYYY-MM-DD HH:MM:SS"
                "num_agents": int,  # total agent count
                "num_versions": int  # total version count
            },
            "agents": {
                "agent_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": {
                            "name": str,
                            "description": str,
                            "metadata": dict,
                            "version": str,
                            "cls": Type[Agent],
                            "config": dict,
                            "instance": Agent, # will be built when needed
                            "function_calling": dict, 
                            "text": str, 
                            "args_schema": BaseModel,
                            "code": str
                        },
                        ...
                    }
                }
            }
        }
        """
        
        agent_configs: Dict[str, AgentConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Agent config file not found at {self.save_path}, skipping code-based loading")
            return agent_configs
        
        # Load all agent configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse agent config JSON from {self.save_path}: {e}")
            return agent_configs
        
        metadata = load_data.get("metadata", {})
        agents_data = load_data.get("agents", {})

        async def register_agent_class(agent_name: str, agent_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, AgentConfig], Optional[AgentConfig]]]:
            """Load all versions for a single agent from JSON."""
            try:
                current_version = agent_data.get("current_version", "1.0.0")
                versions = agent_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Agent {agent_name} has no versions")
                    return None
                
                version_map: Dict[str, AgentConfig] = {}
                current_agent_config: Optional[AgentConfig] = None
                
                for _, version_data in versions.items():
                    agent_config = AgentConfig.model_validate(version_data)
                    version = agent_config.version
                    version_map[version] = agent_config
                    
                    if version == current_version:
                        current_agent_config = agent_config
                
                return agent_name, version_map, current_agent_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load agent {agent_name} from code JSON: {e}")
                return None

        # Launch loading of each agent concurrently with a concurrency limit
        tasks = [
            register_agent_class(agent_name, agent_data) for agent_name, agent_data in agents_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            agent_name, version_map, current_agent_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._agent_history_versions[agent_name] = version_map
            # Active config: the one corresponding to current_version
            if current_agent_config is not None:
                agent_configs[agent_name] = current_agent_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Agent {agent_name} current_version not found, using last available version")
                agent_configs[agent_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for agent_config in version_map.values():
                await version_manager.register_version("agent", agent_name, agent_config.version)
            
        logger.info(f"| 📂 Loaded {len(agent_configs)} agents from {self.save_path}")
        return agent_configs

    async def _store(self, agent_config: AgentConfig):
        """Add agent information to the embedding index.
        
        Args:
            agent_config: Agent configuration
        """
        if self._faiss_service is None:
            return
            
        try:
            # Create comprehensive text representation
            agent_text = f"Agent: {agent_config.name}\nDescription: {agent_config.description}"
            
            # Add to FAISS index
            request = FaissAddRequest(
                texts=[agent_text],
                metadatas=[{
                    "name": agent_config.name,
                    "description": agent_config.description
                }]
            )
            
            await self._faiss_service.add_documents(request)
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add agent {agent_config.name} to FAISS index: {e}")

    async def build(self, agent_config: AgentConfig) -> AgentConfig:
        """Create an agent instance and store it.
        
        Args:
            agent_config: Agent configuration
            
        Returns:
            AgentConfig: Agent configuration with instance
        """
        if agent_config.name in self._agent_configs:
            existing_config = self._agent_configs[agent_config.name]
            if existing_config.instance is not None:
                return existing_config
        
        # Create new agent instance
        try:
            # cls should already be loaded (either from registry or from code in _load_from_code)
            if agent_config.cls is None:
                raise ValueError(f"Cannot create agent {agent_config.name}: no class provided. Class should be loaded during initialization.")
            
            # Instantiate agent instance
            agent_instance = agent_config.cls(**agent_config.config) if agent_config.config else agent_config.cls()
            
            # Initialize agent if it has an initialize method
            if hasattr(agent_instance, "initialize"):
                await agent_instance.initialize()
            
            agent_config.instance = agent_instance
            
            # Store agent metadata
            self._agent_configs[agent_config.name] = agent_config
            
            logger.info(f"| 🔧 Agent {agent_config.name} created and stored")
            
            return agent_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create agent {agent_config.name}: {e}")
            raise

    async def register(
        self,
        agent_cls: Type[Agent],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        override: bool = False,
        version: Optional[str] = None,
    ) -> AgentConfig:
        """Register an agent class.

        This will:
        - Create (or reuse) an agent instance
        - Create an `AgentConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager` and FAISS index
        """
        
        try:
            if agent_config_dict is None:
                # Fallback to global config by class name
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})
            
            # Instantiate agent immediately (register is a runtime operation)
            try:
                agent_instance = agent_cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {agent_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__} with provided config: {e}")
            
            agent_name = agent_instance.name
            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict and "require_grad" in agent_config_dict else agent_instance.require_grad
            
            # Get or generate version from version_manager
            if version is None:
                agent_version = await version_manager.get_version("agent", agent_name)
            else:
                agent_version = version
                
            # Get agent code
            agent_code = dynamic_manager.get_source_code(agent_cls)
            if not agent_code:
                logger.warning(f"| ⚠️ Agent {agent_name} is dynamic but source code cannot be extracted")
            
            # Get agent parameters
            agent_parameters = dynamic_manager.get_parameters(agent_cls)
            agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)
            
            # --- Build AgentConfig ---
            agent_config = AgentConfig(
                name=agent_name,
                description=agent_description,
                metadata=agent_metadata,
                version=agent_version,
                require_grad=agent_require_grad,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )
            
            # --- Persist current config and history ---
            self._agent_configs[agent_name] = agent_config
            
            # Store in dict-based history (for quick lookup by version)
            if agent_name not in self._agent_history_versions:
                self._agent_history_versions[agent_name] = {}
            self._agent_history_versions[agent_name][agent_config.version] = agent_config
            
            # Register version in version manager
            await version_manager.register_version("agent", agent_name, agent_config.version)
            
            # Add to FAISS index
            await self._store(agent_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📝 Registered agent config: {agent_name}: {agent_config.version}")
            return agent_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to register agent: {e}")
            raise

    async def get(self, agent_name: str) -> Optional[Agent]:
        """Get agent configuration by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            Agent: Agent instance or None if not found
        """
        agent_config = self._agent_configs.get(agent_name)
        if agent_config is None:
            return None
        return agent_config.instance if agent_config.instance is not None else None
    
    async def get_info(self, agent_name: str) -> Optional[AgentConfig]:
        """Get agent info by name
        
        Args:
            agent_name: Agent name
            
        Returns:
            AgentConfig: Agent info or None if not found
        """
        return self._agent_configs.get(agent_name)
    
    async def list(self) -> List[str]:
        """Get list of registered agents
        
        Returns:
            List[str]: List of agent names
        """
        return [name for name in self._agent_configs.keys()]

    async def update(
        self,
        agent_cls: Type[Agent],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        new_version: Optional[str] = None,
        description: Optional[str] = None,
        code: Optional[str] = None,
    ) -> AgentConfig:
        """Update an existing agent with new configuration and create a new version
        
        Args:
            agent_cls: New agent class with updated implementation
            agent_config_dict: Configuration dict for agent initialization
                   If None, will try to get from global config
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            code: Optional source code string. If provided, uses this instead of extracting from agent_cls.
                  This is useful when agent_cls is dynamically created from code string.
            
        Returns:
            AgentConfig: Updated agent configuration
        """
        try:
            if agent_config_dict is None:
                # Fallback to global config by class name
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})
            
            # Instantiate agent immediately (update is a runtime operation)
            try:
                agent_instance = agent_cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {agent_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__} with provided config: {e}")
            
            agent_name = agent_instance.name
            
            # Check if agent exists
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found. Use register() to register a new agent.")
            
            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict else agent_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                # Get current version from version_manager and generate next patch version
                new_version = await version_manager.generate_next_version("agent", agent_name, "patch")
            
            # Get agent code - use provided code if available (for dynamically created classes)
            if code is not None:
                agent_code = code
            else:
                agent_code = dynamic_manager.get_source_code(agent_cls)
                if not agent_code:
                    logger.warning(f"| ⚠️ Agent {agent_name} is dynamic but source code cannot be extracted")
            
            # Get agent parameters and build properties using dynamic_manager methods
            agent_parameters = dynamic_manager.get_parameters(agent_cls)
            agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)
            
            # --- Build AgentConfig ---
            updated_config = AgentConfig(
                name=agent_name,  # Keep same name
                description=agent_description,
                metadata=agent_metadata,
                version=new_version,
                require_grad=agent_require_grad,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )
            
            # Update the agent config (replaces current version)
            self._agent_configs[agent_name] = updated_config
            
            # Store in version history
            if agent_name not in self._agent_history_versions:
                self._agent_history_versions[agent_name] = {}
            self._agent_history_versions[agent_name][updated_config.version] = updated_config
            
            # Register new version record to version manager
            await version_manager.register_version(
                "agent", 
                agent_name, 
                new_version,
                description=description or f"Updated from {original_config.version}"
            )
            
            # Update embedding index
            await self._store(updated_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 🔄 Updated agent {agent_name} from v{original_config.version} to v{new_version}")
            return updated_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to update agent: {e}")
            raise

    async def copy(
        self,
        agent_name: str,
        new_name: Optional[str] = None,
        new_version: Optional[str] = None,
        new_config: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """Copy an existing agent configuration
        
        Args:
            agent_name: Name of the agent to copy
            new_name: New name for the copied agent. If None, uses original name.
            new_version: New version for the copied agent. If None, increments version.
            new_config: New configuration dict for the copied agent. If None, uses original config.
            
        Returns:
            AgentConfig: New agent configuration
        """
        try:
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found")
            
            if original_config.cls is None:
                raise ValueError(f"Cannot copy agent {agent_name}: no class provided")
            
            # Determine new name
            if new_name is None:
                new_name = agent_name
            
            # Prepare config dict (merge original config with new config)
            agent_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # Merge new config into original config
                agent_config_dict.update(new_config)
            
            # Instantiate agent instance (copy is a runtime operation)
            try:
                agent_instance = original_config.cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {original_config.cls.__name__} with provided config: {e}")
            
            # Apply name override if provided (after instantiation)
            if new_name != agent_name:
                agent_instance.name = new_name
            
            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict and "require_grad" in agent_config_dict else agent_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                if new_name == agent_name:
                    # If copying with same name, get next version from version_manager
                    new_version = await version_manager.generate_next_version("agent", new_name, "patch")
                else:
                    # If copying with different name, get or generate version for new name
                    new_version = await version_manager.get_version("agent", new_name)
            
            # Get agent code
            agent_code = dynamic_manager.get_source_code(original_config.cls)
            if not agent_code:
                logger.warning(f"| ⚠️ Agent {new_name} is dynamic but source code cannot be extracted")
            
            # Get agent parameters and build properties using dynamic_manager methods
            agent_parameters = dynamic_manager.get_parameters(original_config.cls)
            agent_function_calling = dynamic_manager.build_function_calling(new_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(new_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(new_name, agent_parameters)
            
            # --- Build AgentConfig ---
            new_agent_config = AgentConfig(
                name=new_name,
                description=agent_description,
                metadata=agent_metadata,
                version=new_version,
                require_grad=agent_require_grad,
                cls=original_config.cls,
                config=agent_config_dict,
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )
            
            # Register new agent
            self._agent_configs[new_name] = new_agent_config
            
            # Store in version history
            if new_name not in self._agent_history_versions:
                self._agent_history_versions[new_name] = {}
            self._agent_history_versions[new_name][new_version] = new_agent_config
            
            # Register version record to version manager
            await version_manager.register_version(
                "agent", 
                new_name, 
                new_version,
                description=f"Copied from {agent_name}@{original_config.version}"
            )
            
            # Register to embedding index
            await self._store(new_agent_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📋 Copied agent {agent_name}@{original_config.version} to {new_name}@{new_version}")
            return new_agent_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to copy agent: {e}")
            raise

    async def unregister(self, agent_name: str) -> bool:
        """Unregister an agent
        
        Args:
            agent_name: Name of the agent to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if agent_name not in self._agent_configs:
            logger.warning(f"| ⚠️ Agent {agent_name} not found")
            return False
        
        agent_config = self._agent_configs[agent_name]
        
        # Remove from configs
        del self._agent_configs[agent_name]

        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered agent {agent_name}@{agent_config.version}")
        return True

    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all agent configurations with version history to JSON.
        
        Only saves basic configuration fields (name, description, version, config, etc.).
        Instance is not saved as it's runtime state and will be recreated via build() on load.
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            # Ensure parent directory exists
            parent_dir = os.path.dirname(file_path)
            if parent_dir:  # Only create if there's a directory component
                os.makedirs(parent_dir, exist_ok=True)
            
            # Prepare save data - save all versions for each agent
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_agents": len(self._agent_configs),
                    "num_versions": sum(len(versions) for versions in self._agent_history_versions.values()),
                },
                "agents": {}
            }
            
            for agent_name, version_map in self._agent_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, agent_config in version_map.items():
                        config_dict = agent_config.model_dump()
                        versions_data[agent_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if agent_name in self._agent_configs:
                        current_config = self._agent_configs[agent_name]
                        if current_config is not None:
                            current_version = current_config.version
                    
                    # If not found in active configs, use latest version from history
                    if current_version is None and version_map:
                        # Find latest version by comparing version strings
                        latest_version_str = None
                        for version_str in version_map.keys():
                            if latest_version_str is None:
                                latest_version_str = version_str
                            elif version_manager.compare_versions(version_str, latest_version_str) > 0:
                                latest_version_str = version_str
                        current_version = latest_version_str

                    save_data["agents"][agent_name] = {
                        "versions": versions_data,
                        "current_version": current_version,
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize agent {agent_name}: {e}")
                    continue

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.info(
                f"| 💾 Saved {len(self._agent_configs)} agents with version history to {file_path}"
            )
            return str(file_path)

    async def load_from_json(
        self, file_path: Optional[str] = None, auto_initialize: bool = True
    ) -> bool:
        """Load agent configurations with version history from JSON.
        
        Loads basic configuration only (instance is not saved, must be created via build()).
        Only the latest version will be instantiated by default if auto_initialize=True.
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically create instance via build() after loading
            
        Returns:
            True if loaded successfully, False otherwise
        """
        
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Agent file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                agents_data = load_data.get("agents", {})
                loaded_count = 0
                
                for agent_name, agent_data in agents_data.items():
                    try:
                        # Expected format: multiple versions stored as a dict {version_str: config_dict}
                        versions_data = agent_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Agent {agent_name} has invalid format for 'versions' (expected dict), skipping")
                            continue
                        
                        current_version_str = agent_data.get("current_version")
                        
                        # Load all versions
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for version_str, config_dict in versions_data.items():
                            # Ensure version field is present
                            if "version" not in config_dict:
                                config_dict["version"] = version_str
                            
                            try:
                                agent_config = AgentConfig.model_validate(config_dict)
                                version_configs.append(agent_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load agent config for {agent_name}@{version_str}: {e}")
                                continue
                            
                            # Track latest version
                            if latest_config is None or (
                                current_version_str and agent_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    version_manager.compare_versions(agent_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = agent_config
                                latest_version = agent_config.version
                        
                        # Store all versions in history (dict-based)
                        self._agent_history_versions[agent_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }
                        
                        # Only set latest version as active
                        if latest_config:
                            self._agent_configs[agent_name] = latest_config
                            
                            # Register all versions to version manager (only version records)
                            for agent_config in version_configs:
                                await version_manager.register_version("agent", agent_name, agent_config.version)
                            
                            # Create instance if requested (instance is not saved in JSON, must be created via build)
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load agent {agent_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} agents with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load agents from {file_path}: {e}")
                return False

    async def restore(
        self, agent_name: str, version: str, auto_initialize: bool = True
    ) -> Optional[AgentConfig]:
        """Restore a specific version of an agent from history
        
        Args:
            agent_name: Name of the agent
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored agent
            
        Returns:
            AgentConfig of the restored version, or None if not found
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if agent_name in self._agent_history_versions:
            version_config = self._agent_history_versions[agent_name].get(version)
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for agent {agent_name}")
            return None
        
        # Create a copy to avoid modifying the history
        restored_config = AgentConfig(**version_config.model_dump())
        
        # Set as current active config
        self._agent_configs[agent_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("agent", agent_name)
        if version_history:
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("agent", agent_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("agent", agent_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored agent {agent_name} to version {version}")
        return restored_config
    
    async def save_contract(self, agent_names: Optional[List[str]] = None):
        """Save the contract for an agent"""
        contract = []
        if agent_names is not None:
            for index, agent_name in enumerate(agent_names):
                agent_info = await self.get_info(agent_name)
                text = agent_info.text
                contract.append(f"{index + 1:04d}\n{text}\n")
        else:
            for index, agent_name in enumerate(self._agent_configs.keys()):
                agent_info = await self.get_info(agent_name)
                text = agent_info.text
                contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} agents contract to {self.contract_path}")
        
    async def load_contract(self) -> str:
        """Load the contract for an agent"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve similar agents using FAISS similarity search.
        
        Args:
            query: Query string to search for
            k: Number of results to return (default: 4)
            
        Returns:
            List of dictionaries containing agent information with similarity scores
        """
        if self._faiss_service is None:
            logger.warning("| ⚠️ FAISS service not initialized, cannot retrieve agents")
            return []
        
        try:
            from src.environment.faiss.types import FaissSearchRequest
            
            request = FaissSearchRequest(
                query=query,
                k=k,
                fetch_k=k * 5  # Fetch more candidates before filtering
            )
            
            result = await self._faiss_service.search_similar(request)
            
            if not result.success:
                logger.warning(f"| ⚠️ FAISS search failed: {result.message}")
                return []
            
            # Extract documents and scores from result
            documents = []
            if result.extra and "documents" in result.extra:
                docs = result.extra["documents"]
                scores = result.extra.get("scores", [])
                
                for doc, score in zip(docs, scores):
                    # Extract agent name from metadata
                    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                    agent_name = metadata.get("name", "")
                    
                    # Get agent config if available
                    agent_config = None
                    if agent_name and agent_name in self._agent_configs:
                        agent_config = self._agent_configs[agent_name]
                    
                    documents.append({
                        "name": agent_name,
                        "description": metadata.get("description", ""),
                        "score": float(score),
                        "content": doc.get("page_content", "") if isinstance(doc, dict) else str(doc),
                        "config": agent_config.model_dump() if agent_config else None
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"| ❌ Error retrieving agents: {e}")
            return []
    
    async def get_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get variables from agents, where each agent's class source code is used as the variable value.
        
        Args:
            agent_name (Optional[str]): Name of a specific agent. If None, returns variables for all agents.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping agent names to Variable objects. Each Variable has:
                - name: agent name
                - type: "agent_code"
                - description: agent description
                - require_grad: agent's require_grad value
                - variables: agent's class source code (as string value)
        """
        # Lazy import to avoid circular dependency
        from src.optimizer.types import Variable
        
        variables: Dict[str, Variable] = {}
        
        if agent_name is not None:
            # Get specific agent
            agent_config = await self.get_info(agent_name)
            if agent_config is None:
                logger.warning(f"| ⚠️ Agent {agent_name} not found")
                return variables
            
            agent_configs = {agent_name: agent_config}
        else:
            # Get all agents
            agent_configs = self._agent_configs
        
        for name, agent_config in agent_configs.items():
            # Get agent code
            agent_code = ""
            if agent_config.cls is not None:
                agent_code = dynamic_manager.get_full_module_source(agent_config.cls) or ""
            elif agent_config.code:
                agent_code = agent_config.code
            
            # Create Variable for this agent
            variable = Variable(
                name=name,
                type="agent_code",
                description=agent_config.description or f"Code for agent {name}",
                require_grad=agent_config.require_grad,
                template=None,
                variables=agent_code  # Store code as the variable value
            )
            variables[name] = variable
        
        return variables
    
    async def get_trainable_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get trainable variables from agents, filtering out agents with require_grad=False.
        
        Only returns variables for agents where require_grad=True.
        
        Args:
            agent_name (Optional[str]): Name of a specific agent. If None, returns variables for all trainable agents.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping agent names to Variable objects for trainable agents.
        """
        async with self._variables_lock:
            all_variables = await self.get_variables(agent_name=agent_name)
            trainable_variables = {name: var for name, var in all_variables.items() if var.require_grad}
            return trainable_variables
    
    async def set_variables(self, agent_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> AgentConfig:
        """Set variable values in an agent and create a new version.
        
        Args:
            agent_name: Name of the agent to update
            variable_updates: Dictionary mapping variable names to new values.
                For agents, this is typically {"code": new_code_string}
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            AgentConfig: Updated agent configuration
        """
        async with self._variables_lock:
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found. Use register() to register a new agent.")
            
            # For agents, variable_updates format is {"name": "agent_name", "variables": "agent code"}
            # Extract the new code from "variables" field
            if "variables" not in variable_updates:
                raise ValueError(f"variable_updates must contain 'variables' field with agent code, got: {list(variable_updates.keys())}")
            
            new_code = variable_updates["variables"]
            if not isinstance(new_code, str):
                raise ValueError(f"Agent code must be a string, got {type(new_code)}")
            
            # Load agent class from code
            class_name = dynamic_manager.extract_class_name_from_code(new_code)
            if not class_name:
                raise ValueError(f"Cannot extract class name from code")
            
            try:
                agent_cls = dynamic_manager.load_class(
                    new_code,
                    class_name=class_name,
                    base_class=Agent,
                    context="agent"
                )
            except Exception as e:
                logger.error(f"| ❌ Failed to load agent class from code: {e}")
                raise ValueError(f"Failed to load agent class from code: {e}")
            
            # Use update() function to handle version management and persistence
            # Pass the code directly to avoid re-extracting from dynamically created class
            update_description = description or f"Updated code for {agent_name}"
            return await self.update(
                agent_cls=agent_cls,
                agent_config_dict=original_config.config,
                new_version=new_version,
                description=update_description,
                code=new_code  # Pass code directly since agent_cls is dynamically created
            )

    async def cleanup(self):
        """Cleanup all active agents."""
        try:
            # Clear all agent configs and version history
            self._agent_configs.clear()
            self._agent_history_versions.clear()
                
            # Clean up Faiss service (async)
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            logger.info("| 🧹 Agent context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during agent context manager cleanup: {e}")
            
    async def __call__(self, name: str, input: Dict[str, Any], ctx: AgentContext = None, **kwargs) -> Any:
        """Call an agent by name
        
        Args:
            name: Agent name
            input: Input for the agent
            ctx: Agent context
        Returns:
            Agent result
        """
        if ctx is None:
            ctx = AgentContext()
        
        agent_info = await self.get_info(name)
        
        # Agent args
        agent_args = {
            "ctx": ctx,
        }
        
        version = agent_info.version
        agent_instance = agent_info.instance
        logger.info(f"| ✅ Using agent {name}@{version}")
        
        return await agent_instance(**input, **agent_args)

