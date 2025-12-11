"""Agent Context Manager for managing agent lifecycle and resources."""

import asyncio
import atexit
import importlib
import pkgutil
import inflection
import os
import json
from datetime import datetime
from typing import Any, Dict, Callable, Optional, List, Union, Type
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import assemble_project_path
from src.utils.file_utils import file_lock
from src.agent.types import AgentConfig, Agent

class AgentContextManager(BaseModel):
    """Global context manager for all agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents")
    
    DEFAULT_DISCOVERY_PACKAGES: List[str] = [
        "src.agent",
    ]
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 auto_discover: bool = True, 
                 **kwargs):
        """Initialize the agent context manager.
        
        Args:
            base_dir: Base directory for storing agent data
            save_path: Path to save agent configurations
            auto_discover: Whether to automatically discover and register agents from packages
        """
        super().__init__(**kwargs)
        
        # Set up paths
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "agents"))
        os.makedirs(self.base_dir, exist_ok=True)
        
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "agents.json")
        
        self._agent_configs: Dict[str, AgentConfig] = {}  # Store agent metadata
        self._cleanup_registered = False
        self.auto_discover = auto_discover
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
            
    async def __call__(self, name: str, input: Dict[str, Any], **kwargs) -> Any:
        """Call an agent method.
        
        Args:
            name: Name of the agent
            input: Input for the agent
            **kwargs: Keyword arguments for the agent
            
        Returns:
            Agent result
        """
        if name in self._agent_configs:
            agent_config = self._agent_configs[name]
            
            instance = agent_config.instance
            if instance is None:
                raise ValueError(f"Agent {name} is not initialized")
            
            return await instance(**input, **kwargs)
        else:
            raise ValueError(f"Agent {name} not found")
    
    async def build(self, 
              agent_config: AgentConfig,
              agent_factory: Callable,
              **kwargs
              ) -> AgentConfig:
        """Create and store an agent instance.
        
        Args:
            agent_config: Agent configuration
            agent_factory: Function to create the agent instance
            
        Returns:
            AgentConfig: Agent configuration
        """
        # Check if agent already exists in _agent_configs
        if agent_config.name in self._agent_configs:
            existing_config = self._agent_configs[agent_config.name]
            # If instance already exists, return it
            if existing_config.instance is not None:
                return existing_config
            # Otherwise, use the existing config but create instance
            agent_config = existing_config
        
        try:
            # Create agent instance
            instance = agent_factory()
            
            # Initialize agent
            if hasattr(instance, "initialize"):
                await instance.initialize()
            
            # Store instance
            agent_config.instance = instance
            agent_config.name = instance.name
            agent_config.description = instance.description
            
            # Store metadata
            self._agent_configs[agent_config.name] = agent_config
            
            logger.info(f"| ✅ Agent {agent_config.name} created and stored")
            return agent_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to create agent {agent_config.name}: {e}")
            raise
        
    async def initialize(self):
        """Initialize the agent context manager."""
        if self.auto_discover:
            await self.discover()
    
    async def _collect_agent_classes(self, packages: List[str]) -> List[Type[Agent]]:
        """Collect all Agent subclasses from packages.
        
        Args:
            packages: List of package names to scan
            
        Returns:
            List of Agent subclasses
        """
        agent_classes = []
        imported_modules = set()
        
        for package_name in packages:
            try:
                # Import the package
                package = importlib.import_module(package_name)
                imported_modules.add(package_name)
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to import package {package_name}: {e}")
                continue
            
            # Walk through all modules in the package
            package_path = getattr(package, "__path__", None)
            if not package_path:
                continue
            
            # Collect module names first
            module_names = [
                module_name for _, module_name, _ in pkgutil.walk_packages(package_path, package.__name__ + ".")
                if module_name not in imported_modules
            ]
            
            # Import modules concurrently
            async def import_module(module_name: str):
                try:
                    module = importlib.import_module(module_name)
                    imported_modules.add(module_name)
                    
                    # Find all Agent subclasses in the module
                    found_classes = []
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (isinstance(obj, type) and 
                            issubclass(obj, Agent) and 
                            obj is not Agent):
                            found_classes.append(obj)
                    return found_classes
                except Exception as e:
                    logger.debug(f"| ⚠️ Failed to import module {module_name}: {e}")
                    return []
            
            # Import all modules concurrently
            import_tasks = [import_module(module_name) for module_name in module_names]
            results = await asyncio.gather(*import_tasks, return_exceptions=True)
            
            # Collect all agent classes
            for result in results:
                if isinstance(result, list):
                    for agent_cls in result:
                        if agent_cls not in agent_classes:
                            agent_classes.append(agent_cls)
        
        return agent_classes
    
    async def _register_agent_class(self, agent_cls: Type[Agent], override: bool = False):
        """Register an agent class.
        
        Args:
            agent_cls: Agent class to register
            override: Whether to override existing registration
        """
        agent_name = None
        try:
            # Get agent config from global config
            agent_config_key = inflection.underscore(agent_cls.__name__)
            agent_config = config.get(f"{agent_config_key}_agent", {})
            
            # Try to create temporary instance to get name and description
            try:
                # Agent requires workdir, so we need to provide it
                temp_instance = agent_cls(workdir=config.workdir, **agent_config)
                agent_name = temp_instance.name
                agent_type = temp_instance.type
                agent_description = temp_instance.description
                args_schema = temp_instance.args_schema
                metadata = temp_instance.metadata
            except Exception:
                # If instantiation fails, try to get from model_fields
                try:
                    temp_instance = agent_cls(workdir=config.workdir)
                    agent_name = temp_instance.name
                    agent_type = temp_instance.type
                    agent_description = temp_instance.description
                    args_schema = temp_instance.args_schema
                    metadata = temp_instance.metadata
                except Exception:
                    # If still fails, try to get from model_fields
                    if hasattr(agent_cls, 'model_fields') and 'name' in agent_cls.model_fields:
                        field_info = agent_cls.model_fields['name']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            # Check if default is a string (not None and not undefined)
                            if isinstance(default_value, str) and default_value:
                                agent_name = default_value
                    
                    if hasattr(agent_cls, 'model_fields') and 'type' in agent_cls.model_fields:
                        field_info = agent_cls.model_fields['type']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            if isinstance(default_value, str):
                                agent_type = default_value
                            else:
                                agent_type = 'Agent'
                    else:
                        agent_type = 'Agent'
                    
                    if hasattr(agent_cls, 'model_fields') and 'description' in agent_cls.model_fields:
                        field_info = agent_cls.model_fields['description']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            if isinstance(default_value, str):
                                agent_description = default_value
                            else:
                                agent_description = ''
                    else:
                        agent_description = ''
                    
                    if hasattr(agent_cls, 'model_fields') and 'args_schema' in agent_cls.model_fields:
                        field_info = agent_cls.model_fields['args_schema']
                        if hasattr(field_info, 'default'):
                            args_schema = field_info.default
                        else:
                            args_schema = None
                    else:
                        args_schema = None
                    
                    if hasattr(agent_cls, 'model_fields') and 'metadata' in agent_cls.model_fields:
                        field_info = agent_cls.model_fields['metadata']
                        if hasattr(field_info, 'default'):
                            metadata = field_info.default
                        else:
                            metadata = {}
                    else:
                        metadata = {}
                    
                    if not agent_name:
                        logger.warning(f"| ⚠️ Agent class {agent_cls.__name__} has no name, skipping")
                        return
            
            if not agent_name:
                logger.warning(f"| ⚠️ Agent class {agent_cls.__name__} has empty name, skipping")
                return
            
            if agent_name in self._agent_configs and not override:
                logger.debug(f"| ⚠️ Agent {agent_name} already registered, skipping")
                return
            
            # Register the agent class
            await self.register(agent_cls, override=override, **agent_config)
            
            logger.debug(f"| 📝 Registered agent: {agent_name} ({agent_cls.__name__})")
            
        except Exception as e:
            agent_name_str = agent_name or agent_cls.__name__
            logger.warning(f"| ⚠️ Failed to register agent class {agent_cls.__name__} (name: {agent_name_str}): {e}")
            import traceback
            logger.debug(f"| Traceback: {traceback.format_exc()}")
            raise
    
    async def discover(self, packages: Optional[List[str]] = None):
        """Discover and register all Agent subclasses from specified packages.
        
        Args:
            packages: List of package names to scan. Defaults to DEFAULT_DISCOVERY_PACKAGES.
        """
        packages = packages or self.DEFAULT_DISCOVERY_PACKAGES
        
        logger.info(f"| 🔍 Discovering agents from packages: {packages}")
        
        # Collect all Agent subclasses
        agent_classes = await self._collect_agent_classes(packages)
        
        # Register each agent class concurrently
        registration_tasks = [
            self._register_agent_class(agent_cls) for agent_cls in agent_classes
        ]
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Count successful registrations
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(agent_classes)} agents")
    
    async def register(self, agent: Union[Agent, Type[Agent]], *, override: bool = False, **kwargs: Any) -> AgentConfig:
        """Register an agent class or instance.
        
        Args:
            agent: Agent class or instance
            override: Whether to override existing registration
            **kwargs: Configuration for agent initialization
            
        Returns:
            AgentConfig: Agent configuration
        """
        # Create temporary instance to get name and description
        try:
            if isinstance(agent, Agent):
                agent_name = agent.name
                agent_type = agent.type
                agent_description = agent.description
                args_schema = agent.args_schema
                metadata = agent.metadata
                agent_cls = type(agent)
                agent_instance = agent
            elif isinstance(agent, type) and issubclass(agent, Agent):
                # Try to create temporary instance
                try:
                    # BaseAgent requires workdir
                    temp_instance = agent(workdir=config.workdir, **kwargs)
                    agent_name = temp_instance.name
                    agent_type = temp_instance.type
                    agent_description = temp_instance.description
                    args_schema = temp_instance.args_schema
                    metadata = temp_instance.metadata
                    agent_cls = agent
                    agent_instance = None
                except Exception as inst_exc:
                    # If instantiation fails, try to get from model_fields
                    logger.debug(f"| ⚠️ Failed to instantiate {agent.__name__} for registration: {inst_exc}")
                    agent_name = None
                    agent_type = 'Agent'
                    agent_description = ''
                    args_schema = None
                    metadata = {}
                    
                    # Try to get name from model_fields default value
                    if hasattr(agent, 'model_fields') and 'name' in agent.model_fields:
                        field_info = agent.model_fields['name']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            if isinstance(default_value, str) and default_value:
                                agent_name = default_value
                                logger.debug(f"| 📝 Got agent_name from model_fields: {agent_name}")
                    
                    # Try to get type from model_fields
                    if hasattr(agent, 'model_fields') and 'type' in agent.model_fields:
                        field_info = agent.model_fields['type']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            if isinstance(default_value, str):
                                agent_type = default_value
                    
                    # Try to get description from model_fields default value
                    if hasattr(agent, 'model_fields') and 'description' in agent.model_fields:
                        field_info = agent.model_fields['description']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            if isinstance(default_value, str):
                                agent_description = default_value
                    
                    # Try to get args_schema from model_fields
                    if hasattr(agent, 'model_fields') and 'args_schema' in agent.model_fields:
                        field_info = agent.model_fields['args_schema']
                        if hasattr(field_info, 'default'):
                            args_schema = field_info.default
                    
                    # Try to get metadata from model_fields
                    if hasattr(agent, 'model_fields') and 'metadata' in agent.model_fields:
                        field_info = agent.model_fields['metadata']
                        if hasattr(field_info, 'default'):
                            metadata = field_info.default or {}
                    
                    agent_cls = agent
                    agent_instance = None
                    
                    if not agent_name:
                        raise ValueError(f"Agent class {agent.__name__} has no name (failed to instantiate and no default in model_fields)")
            else:
                raise TypeError(f"Expected Agent instance or subclass, got {type(agent)!r}")
            
            if not agent_name:
                raise ValueError("Agent.name cannot be empty.")
            
            if agent_name in self._agent_configs and not override:
                raise ValueError(f"Agent '{agent_name}' already registered. Use override=True to replace it.")
            
            # Get or generate version from version_manager
            version = await version_manager.get_version("agent", agent_name)
            
            # Create AgentConfig
            agent_config = AgentConfig(
                name=agent_name,
                type=agent_type,
                description=agent_description,
                version=version,
                args_schema=args_schema,
                cls=agent_cls,
                instance=agent_instance,
                config=kwargs if kwargs else {},
                metadata=metadata if metadata is not None else {}
            )
            
            # Store metadata
            self._agent_configs[agent_name] = agent_config
            
            # Register version record to version manager
            await version_manager.register_version("agent", agent_name, agent_config.version)
            
            logger.debug(f"| 📝 Registered agent: {agent_name} v{agent_config.version}")
            return agent_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register agent: {e}")
            raise
        
    async def get(self, name: str) -> Optional[Agent]:
        """Get an agent instance by name
        
        Args:
            name: Name of the agent
        """
        return self._agent_configs.get(name).instance if self._agent_configs.get(name) else None
    
    async def get_info(self, name: str) -> Optional[AgentConfig]:
        """Get an agent configuration by name
        
        Args:
            name: Name of the agent
        """
        return self._agent_configs.get(name)
    
    async def list(self) -> List[str]:
        """Get list of registered agents
        
        Returns:
            List[str]: List of agent names
        """
        return [name for name in self._agent_configs.keys()]
    
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
        original_config = self._agent_configs.get(agent_name)
        if original_config is None:
            raise ValueError(f"Agent {agent_name} not found. Use register() to register a new agent.")
        
        # Get new agent info
        if isinstance(agent, Agent):
            new_description = agent.description
            agent_cls = type(agent)
            agent_instance = agent
        elif isinstance(agent, type) and issubclass(agent, Agent):
            try:
                temp_instance = agent(workdir=config.workdir, **kwargs)
                new_description = temp_instance.description
                agent_cls = agent
                agent_instance = None
            except Exception:
                new_description = getattr(agent, 'description', original_config.description)
                agent_cls = agent
                agent_instance = None
        else:
            raise TypeError(f"Expected Agent instance or subclass, got {type(agent)!r}")
        
        # Determine new version from version_manager
        if new_version is None:
            # Get current version from version_manager and generate next patch version
            new_version = await version_manager.generate_next_version("agent", agent_name, "patch")
        
        # Get metadata
        if agent_instance is not None:
            metadata = getattr(agent_instance, 'metadata', {})
        else:
            metadata = original_config.metadata if original_config else {}
        
        # Create updated config
        if agent_instance is not None:
            updated_config = AgentConfig(
                name=agent_name,
                type=original_config.type,
                description=description or new_description,
                version=new_version,
                args_schema=original_config.args_schema,
                cls=agent_cls,
                config={},
                instance=agent_instance,
                metadata=metadata
            )
        else:
            updated_config = AgentConfig(
                name=agent_name,
                type=original_config.type,
                description=description or new_description,
                version=new_version,
                args_schema=original_config.args_schema,
                cls=agent_cls,
                config=kwargs,
                instance=None,
                metadata=metadata
            )
        
        # Store updated config
        self._agent_configs[agent_name] = updated_config
        
        # Register version record to version manager
        await version_manager.register_version("agent", agent_name, updated_config.version)
        
        logger.info(f"| 📝 Updated agent: {agent_name} v{updated_config.version}")
        return updated_config
    
    async def copy(self, agent_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> AgentConfig:
        """Copy an existing agent configuration
        
        Args:
            agent_name: Name of the agent to copy
            new_name: New name for the copied agent. If None, uses original name.
            new_version: New version for the copied agent. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            AgentConfig: New agent configuration
        """
        original_config = self._agent_configs.get(agent_name)
        if original_config is None:
            raise ValueError(f"Agent {agent_name} not found")
        
        # Determine new name
        if new_name is None:
            new_name = agent_name
        
        # Determine new version from version_manager
        if new_version is None:
            if new_name == agent_name:
                # If copying with same name, get next version from version_manager
                new_version = await version_manager.generate_next_version("agent", new_name, "patch")
            else:
                # If copying with different name, get or generate version for new name
                new_version = await version_manager.get_or_generate_version("agent", new_name)
        
        # Create copy of config
        new_config_dict = original_config.model_dump()
        new_config_dict["name"] = new_name
        new_config_dict["version"] = new_version
        
        # Apply overrides
        if override_config:
            if "type" in override_config:
                new_config_dict["type"] = override_config.pop("type")
            if "description" in override_config:
                new_config_dict["description"] = override_config.pop("description")
            if "args_schema" in override_config:
                new_config_dict["args_schema"] = override_config.pop("args_schema")
            if "metadata" in override_config:
                if "metadata" in new_config_dict:
                    new_config_dict["metadata"].update(override_config.pop("metadata"))
                else:
                    new_config_dict["metadata"] = override_config.pop("metadata")
            # Merge remaining overrides into config
            if "config" in new_config_dict:
                new_config_dict["config"].update(override_config)
            else:
                new_config_dict["config"] = override_config
        
        # Clear instance (will be created on demand)
        new_config_dict["instance"] = None
        
        new_config = AgentConfig(**new_config_dict)
        
        # Register new agent
        self._agent_configs[new_name] = new_config
        
        # Register version record to version manager
        await version_manager.register_version(
            "agent", 
            new_name, 
            new_version,
            description=f"Copied from {agent_name}@{original_config.version}"
        )
        
        logger.info(f"| 📋 Copied agent {agent_name}@{original_config.version} to {new_name}@{new_version}")
        return new_config
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all agent configurations to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            # Ensure directory exists
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Prepare save data
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "agent_count": len(self._agent_configs),
                },
                "agents": {}
            }
            
            for agent_name, agent_config in self._agent_configs.items():
                try:
                    # Serialize agent config (excluding non-serializable cls and instance)
                    config_dict = agent_config.model_dump(mode="json", exclude={"cls", "instance", "args_schema"})
                    
                    # Store class path if available
                    if agent_config.cls is not None:
                        config_dict["cls_path"] = f"{agent_config.cls.__module__}.{agent_config.cls.__name__}"
                    
                    save_data["agents"][agent_name] = config_dict
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize agent {agent_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._agent_configs)} agents to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load agent configurations from JSON
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically initialize agents after loading
            
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
                        # Try to load class if cls_path is available
                        cls = None
                        if "cls_path" in agent_data:
                            module_path, class_name = agent_data["cls_path"].rsplit(".", 1)
                            try:
                                module = importlib.import_module(module_path)
                                cls = getattr(module, class_name)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load class {agent_data['cls_path']} for {agent_name}: {e}")
                        
                        # Remove cls_path from dict before creating AgentConfig
                        agent_data_copy = agent_data.copy()
                        agent_data_copy.pop("cls_path", None)
                        
                        # Create AgentConfig
                        agent_config = AgentConfig(**agent_data_copy)
                        if cls is not None:
                            agent_config.cls = cls
                        
                        # Register agent config
                        self._agent_configs[agent_name] = agent_config
                        
                        # Register version to version manager
                        await version_manager.register_version("agent", agent_name, agent_config.version)
                        
                        # Initialize if requested
                        if auto_initialize and agent_config.cls is not None:
                            def agent_factory():
                                agent_config_dict = agent_config.config or {}
                                # Agent requires workdir
                                if 'workdir' not in agent_config_dict:
                                    agent_config_dict['workdir'] = config.workdir
                                if agent_config_dict:
                                    return agent_config.cls(**agent_config_dict)
                                else:
                                    return agent_config.cls(workdir=config.workdir)
                            await self.build(agent_config, agent_factory)
                        
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load agent {agent_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} agents from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load agents from {file_path}: {e}")
                return False
    
    def cleanup(self):
        """Cleanup all agent instances and resources."""
        try:
            # Clear instances and configs
            self._agent_configs.clear()
            logger.info("| 🧹 Agent context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during agent context manager cleanup: {e}")
