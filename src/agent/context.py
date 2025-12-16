"""Agent Context Manager for managing agent lifecycle and resources with lazy loading."""

import ast
import inspect
import json
import os
from asyncio_atexit import register as async_atexit_register
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import inflection
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.dynamic import dynamic_manager
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.logger import logger
from src.registry import AGENT
from src.utils import (
    assemble_project_path,
    deserialize_args_schema,
    gather_with_concurrency,
    serialize_args_schema,
)
from src.utils.file_utils import file_lock
from src.version import version_manager
from src.agent.types import Agent, AgentConfig


class AgentContextManager(BaseModel):
    """Global context manager for all agents with lazy loading and version history."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(
        default=None, description="The base directory to use for the agents"
    )
    save_path: str = Field(
        default=None, description="The path to save the agents configuration JSON"
    )

    def __init__(
        self,
        base_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        model_name: str = "openrouter/gpt-4.1",
        embedding_model_name: str = "openrouter/text-embedding-3-large",
        **kwargs: Any,
    ):
        """Initialize the agent context manager.

        Args:
            base_dir: Base directory for storing agent data
            save_path: Path to save agent configurations
            model_name: The model name used for embedding text (via FaissService)
            embedding_model_name: The embedding model name (kept for parity with tools)
        """
        super().__init__(**kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)

        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "agent.json")

        logger.info(
            f"| 📁 Agent context manager base directory: {self.base_dir} and save path: {self.save_path}"
        )

        # Current active configs (latest version)
        self._agent_configs: Dict[str, AgentConfig] = {}
        # Agent version history, e.g., {"agent_name": {"1.0.0": AgentConfig, ...}}
        self._agent_history_versions: Dict[str, Dict[str, AgentConfig]] = {}

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        self._cleanup_registered = False
        self._faiss_service: Optional[FaissService] = None

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

        # Register async cleanup callback
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True

        logger.info("| ✅ Agents initialization completed")

    async def _load_from_registry(self) -> Dict[str, AgentConfig]:
        """Load agents from AGENT registry."""

        agent_configs: Dict[str, AgentConfig] = {}

        async def register_agent_class(agent_cls: Type[Agent]) -> None:
            agent_name: Optional[str] = None
            try:
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_init_config = config.get(f"{agent_config_key}_agent", {})

                # Try to create temporary instance to get name/description/etc.
                try:
                    temp_instance = agent_cls(workdir=config.workdir, **agent_init_config)
                    agent_name = temp_instance.name
                    agent_description = temp_instance.description
                    args_schema = getattr(temp_instance, "args_schema", None)
                    metadata = getattr(temp_instance, "metadata", {}) or {}
                except Exception:
                    # Fallback to model_fields, similar to previous implementation
                    if hasattr(agent_cls, "model_fields") and "name" in agent_cls.model_fields:
                        field_info = agent_cls.model_fields["name"]
                        default_value = getattr(field_info, "default", None)
                        if isinstance(default_value, str) and default_value:
                            agent_name = default_value

                    if hasattr(agent_cls, "model_fields") and "description" in agent_cls.model_fields:
                        field_info = agent_cls.model_fields["description"]
                        default_value = getattr(field_info, "default", None)
                        if isinstance(default_value, str):
                            agent_description = default_value
                        else:
                            agent_description = ""
                    else:
                        agent_description = ""

                    if hasattr(agent_cls, "model_fields") and "args_schema" in agent_cls.model_fields:
                        field_info = agent_cls.model_fields["args_schema"]
                        args_schema = getattr(field_info, "default", None)
                    else:
                        args_schema = None

                    if hasattr(agent_cls, "model_fields") and "metadata" in agent_cls.model_fields:
                        field_info = agent_cls.model_fields["metadata"]
                        metadata = getattr(field_info, "default", {}) or {}
                    else:
                        metadata = {}

                    if not agent_name:
                        logger.warning(
                            f"| ⚠️ Agent class {agent_cls.__name__} has no name, skipping"
                        )
                        return

                if not agent_name:
                    logger.warning(
                        f"| ⚠️ Agent class {agent_cls.__name__} has empty name, skipping"
                    )
                    return

                # Get or generate version
                agent_version = await version_manager.get_version("agent", agent_name)

                # Get full module source code
                agent_code = self._get_full_module_source(agent_cls)

                # Build AgentConfig (instance is not created here)
                agent_config = AgentConfig(
                    name=agent_name,
                    description=agent_description,
                    version=agent_version,
                    cls=agent_cls,
                    config=agent_init_config,
                    instance=None,
                    metadata=metadata,
                    function_calling=None,
                    text=None,
                    args_schema=args_schema,
                    code=agent_code,
                )

                agent_configs[agent_name] = agent_config

                # Store in history by version
                if agent_name not in self._agent_history_versions:
                    self._agent_history_versions[agent_name] = {}
                self._agent_history_versions[agent_name][agent_version] = agent_config

                logger.info(f"| 📝 Registered agent from registry: {agent_name} ({agent_cls.__name__})")

            except Exception as e:
                agent_name_str = agent_name or agent_cls.__name__
                logger.error(
                    f"| ❌ Failed to register agent class {agent_cls.__name__} (name: {agent_name_str}): {e}"
                )
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

    async def _load_from_code(self) -> Dict[str, AgentConfig]:
        """Load agents from JSON code file (multi-version history)."""

        agent_configs: Dict[str, AgentConfig] = {}

        if not os.path.exists(self.save_path):
            logger.info(
                f"| 📂 Agent config file not found at {self.save_path}, skipping code-based loading"
            )
            return agent_configs

        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(
                f"| ⚠️ Failed to parse agent config JSON from {self.save_path}: {e}"
            )
            return agent_configs

        metadata = load_data.get("metadata", {})
        agents_data = load_data.get("agents", {})

        async def register_agent_from_json(
            agent_name: str, agent_data: Dict[str, Any]
        ) -> Optional[Tuple[str, Dict[str, AgentConfig], Optional[AgentConfig]]]:
            """Load all versions for a single agent from JSON."""
            try:
                current_version = agent_data.get("current_version", "1.0.0")
                versions = agent_data.get("versions", {})

                if not versions:
                    logger.warning(f"| ⚠️ Agent {agent_name} has no versions in JSON")
                    return None

                version_map: Dict[str, AgentConfig] = {}
                current_config: Optional[AgentConfig] = None

                for version_str, version_data in versions.items():
                    name = version_data.get("name", "")
                    description = version_data.get("description", "")
                    enabled_version = version_data.get("version", version_str)

                    code = version_data.get("code", None)
                    config_dict = version_data.get("config", {}) or {}

                    # Dynamic class handling
                    if code:
                        cls = dynamic_manager.load_class(
                                code,
                                base_class=Agent,
                                context="agent",
                            )

                    instance = version_data.get("instance", None)
                    metadata_val = version_data.get("metadata", {}) or {}
                    function_calling = version_data.get("function_calling", None)
                    text = version_data.get("text", None)

                    # Restore args_schema from saved schema info (if present)
                    args_schema = None
                    args_schema_info = version_data.get("args_schema")
                    if args_schema_info:
                        try:
                            args_schema = deserialize_args_schema(args_schema_info)
                        except Exception as e:
                            logger.warning(
                                f"| ⚠️ Failed to restore args_schema for {agent_name}@{version_str}: {e}"
                            )

                    agent_config = AgentConfig(
                        name=name,
                        description=description,
                        version=enabled_version,
                        cls=cls,
                        config=config_dict,
                        instance=instance,
                        metadata=metadata_val,
                        function_calling=function_calling,
                        text=text,
                        args_schema=args_schema,
                        code=code,
                    )

                    version_map[enabled_version] = agent_config
                    if enabled_version == current_version:
                        current_config = agent_config

                return agent_name, version_map, current_config
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to load agent {agent_name} from code JSON: {e}"
                )
                return None

        tasks = [
            register_agent_from_json(agent_name, agent_data)
            for agent_name, agent_data in agents_data.items()
        ]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            agent_name, version_map, current_config = result
            if not version_map:
                continue

            self._agent_history_versions[agent_name] = version_map

            if current_config is not None:
                agent_configs[agent_name] = current_config
            else:
                logger.warning(
                    f"| ⚠️ Agent {agent_name} current_version not found, using last available version"
                )
                agent_configs[agent_name] = list(version_map.values())[-1]

        logger.info(f"| 📂 Loaded {len(agent_configs)} agents from {self.save_path}")
        return agent_configs

    async def _store(self, agent_config: AgentConfig) -> None:
        """Add agent information to the embedding index."""
        if self._faiss_service is None:
            return

        try:
            agent_text = (
                f"Agent: {agent_config.name}\nDescription: {agent_config.description}"
            )
            request = FaissAddRequest(
                texts=[agent_text],
                metadatas=[
                    {
                        "name": agent_config.name,
                        "description": agent_config.description,
                    }
                ],
            )
            await self._faiss_service.add_documents(request)
        except Exception as e:
            logger.warning(
                f"| ⚠️ Failed to add agent {agent_config.name} to FAISS index: {e}"
            )

    async def build(self, agent_config: AgentConfig) -> AgentConfig:
        """Create an agent instance and store it."""

        if agent_config.name in self._agent_configs:
            existing_config = self._agent_configs[agent_config.name]
            if existing_config.instance is not None:
                return existing_config

        try:
            if agent_config.cls is None:
                raise ValueError(
                    f"Cannot create agent {agent_config.name}: no class provided. "
                    "Class should be loaded during initialization."
                )

            init_config = dict(agent_config.config or {})
            # Merge with global config if available
            agent_config_key = inflection.underscore(agent_config.cls.__name__)
            global_config = config.get(agent_config_key, {})
            if global_config:
                init_config.update(global_config)
            if "workdir" not in init_config:
                init_config["workdir"] = config.workdir

            agent_instance = agent_config.cls(**init_config)
            agent_config.instance = agent_instance

            # Initialize the agent instance
            if hasattr(agent_instance, 'initialize'):
                try:
                    await agent_instance.initialize()
                except Exception as e:
                    logger.warning(
                        f"| ⚠️ Failed to initialize agent instance {agent_config.name}: {e}"
                    )

            # Lazy compute function_calling, text, and args_schema if not already set
            if (
                agent_config.function_calling is None
                or agent_config.text is None
                or agent_config.args_schema is None
            ):
                try:
                    agent_config.function_calling = agent_instance.function_calling
                    agent_config.text = agent_instance.text
                    agent_config.args_schema = agent_instance.args_schema
                except Exception as e:
                    logger.debug(
                        f"| ⚠️ Failed to get properties from agent instance {agent_config.name}: {e}"
                    )

            self._agent_configs[agent_config.name] = agent_config
            logger.info(f"| 🎮 Agent {agent_config.name} created and stored")
            return agent_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create agent {agent_config.name}: {e}")
            raise

    async def register(
        self,
        agent: Union[Agent, Type[Agent]],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        override: bool = False,
        version: Optional[str] = None,
    ) -> AgentConfig:
        """Register an agent class or instance.

        This will:
        - Create (or reuse) an agent instance
        - Create an `AgentConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager` and FAISS index
        """

        try:
            if isinstance(agent, Agent):
                agent_instance = agent
                agent_cls = type(agent_instance)
                agent_name = agent_instance.name
                agent_description = agent_instance.description
                function_calling = agent_instance.function_calling
                text = agent_instance.text
                args_schema = agent_instance.args_schema
                metadata = getattr(agent_instance, "metadata", {}) or {}
                agent_config_dict = {}
                agent_version = version or getattr(agent_instance, "version", "1.0.0")
            else:
                agent_cls = agent
                if agent_config_dict is None:
                    agent_config_key = inflection.underscore(agent_cls.__name__)
                    agent_config_dict = config.get(f"{agent_config_key}_agent", {})

                init_config = dict(agent_config_dict)
                if "workdir" not in init_config:
                    init_config["workdir"] = config.workdir

                try:
                    agent_instance = agent_cls(**init_config)
                except Exception as e:
                    logger.error(
                        f"| ❌ Failed to create agent instance for {agent_cls.__name__}: {e}"
                    )
                    raise ValueError(
                        f"Failed to instantiate agent {agent_cls.__name__} with provided config: {e}"
                    )

                agent_name = agent_instance.name
                agent_description = agent_instance.description
                function_calling = agent_instance.function_calling
                text = agent_instance.text
                args_schema = agent_instance.args_schema
                metadata = getattr(agent_instance, "metadata", {}) or {}

                if version is None:
                    agent_version = await version_manager.get_version("agent", agent_name)
                else:
                    agent_version = version

            if not agent_name:
                raise ValueError("Agent.name cannot be empty.")

            if agent_name in self._agent_configs and not override:
                raise ValueError(
                    f"Agent '{agent_name}' already registered. Use override=True to replace it."
                )

            agent_code = None
            if dynamic_manager.is_dynamic_class(agent_cls):
                agent_code = dynamic_manager.get_class_source_code(agent_cls)
                if not agent_code:
                    logger.warning(
                        f"| ⚠️ Agent {agent_name} is dynamic but source code cannot be extracted"
                    )

            agent_config = AgentConfig(
                name=agent_name,
                description=agent_description,
                version=agent_version,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=agent_instance,
                metadata=metadata,
                function_calling=function_calling,
                text=text,
                args_schema=args_schema,
                code=agent_code,
            )

            self._agent_configs[agent_name] = agent_config

            if agent_name not in self._agent_history_versions:
                self._agent_history_versions[agent_name] = {}
            self._agent_history_versions[agent_name][agent_config.version] = agent_config

            await version_manager.register_version(
                "agent", agent_name, agent_config.version
            )

            await self._store(agent_config)
            await self.save_to_json()

            logger.info(f"| 📝 Registered agent config: {agent_name}: {agent_config.version}")
            return agent_config

        except Exception as e:
            logger.error(f"| ❌ Failed to register agent: {e}")
            raise

    async def get(self, name: str) -> Optional[Agent]:
        """Get an agent instance by name."""
        agent_config = self._agent_configs.get(name)
        if agent_config is None:
            return None
        return agent_config.instance if agent_config.instance is not None else None

    async def get_info(self, name: str) -> Optional[AgentConfig]:
        """Get an agent configuration by name."""
        return self._agent_configs.get(name)

    async def list(self) -> List[str]:
        """Get list of registered agents."""
        return list(self._agent_configs.keys())

    async def update(
        self,
        agent_name: str,
        agent: Union[Agent, Type[Agent]],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        new_version: Optional[str] = None,
        description: Optional[str] = None,
    ) -> AgentConfig:
        """Update an existing agent with new configuration and create a new version."""

        original_config = self._agent_configs.get(agent_name)
        if original_config is None:
            raise ValueError(
                f"Agent {agent_name} not found. Use register() to register a new agent."
            )

        if isinstance(agent, Agent):
            agent_instance = agent
            agent_cls = type(agent_instance)
            new_description = agent_instance.description
        elif isinstance(agent, type) and issubclass(agent, Agent):
            agent_cls = agent
            init_config = agent_config_dict or {}
            if "workdir" not in init_config:
                init_config["workdir"] = config.workdir
            try:
                temp_instance = agent_cls(**init_config)
                new_description = temp_instance.description
                agent_instance = None
            except Exception:
                new_description = getattr(agent_cls, "description", original_config.description)
                agent_instance = None
        else:
            raise TypeError(f"Expected Agent instance or subclass, got {type(agent)!r}")

        if new_version is None:
            new_version = await version_manager.generate_next_version(
                "agent", agent_name, "patch"
            )

        if agent_instance is not None:
            metadata = getattr(agent_instance, "metadata", {}) or {}
        else:
            metadata = original_config.metadata or {}

        if agent_instance is not None:
            updated_config = AgentConfig(
                name=agent_name,
                description=description or new_description,
                version=new_version,
                cls=agent_cls,
                config={},
                instance=agent_instance,
                metadata=metadata,
                function_calling=getattr(agent_instance, "function_calling", None),
                text=getattr(agent_instance, "text", None),
                args_schema=getattr(agent_instance, "args_schema", None),
                code=original_config.code,
            )
        else:
            updated_config = AgentConfig(
                name=agent_name,
                description=description or new_description,
                version=new_version,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=None,
                metadata=metadata,
                function_calling=original_config.function_calling,
                text=original_config.text,
                args_schema=original_config.args_schema,
                code=original_config.code,
            )

        self._agent_configs[agent_name] = updated_config

        if agent_name not in self._agent_history_versions:
            self._agent_history_versions[agent_name] = {}
        self._agent_history_versions[agent_name][updated_config.version] = updated_config

        await version_manager.register_version(
            "agent",
            agent_name,
            new_version,
            description=description or f"Updated from {original_config.version}",
        )

        await self._store(updated_config)
        await self.save_to_json()

        logger.info(
            f"| 🔄 Updated agent {agent_name} from v{original_config.version} to v{new_version}"
        )
        return updated_config

    async def copy(
        self,
        agent_name: str,
        new_name: Optional[str] = None,
        new_version: Optional[str] = None,
        **override_config: Any,
    ) -> AgentConfig:
        """Copy an existing agent configuration."""

        original_config = self._agent_configs.get(agent_name)
        if original_config is None:
            raise ValueError(f"Agent {agent_name} not found")

        if new_name is None:
            new_name = agent_name

        if new_version is None:
            if new_name == agent_name:
                new_version = await version_manager.generate_next_version(
                    "agent", new_name, "patch"
                )
            else:
                new_version = await version_manager.get_or_generate_version(
                    "agent", new_name
                )

        new_config_dict = original_config.model_dump()
        new_config_dict["name"] = new_name
        new_config_dict["version"] = new_version

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

            if "config" in new_config_dict:
                new_config_dict["config"].update(override_config)
            else:
                new_config_dict["config"] = override_config

        new_config_dict["instance"] = None

        new_config = AgentConfig(**new_config_dict)

        self._agent_configs[new_name] = new_config

        if new_name not in self._agent_history_versions:
            self._agent_history_versions[new_name] = {}
        self._agent_history_versions[new_name][new_version] = new_config

        await version_manager.register_version(
            "agent",
            new_name,
            new_version,
            description=f"Copied from {agent_name}@{original_config.version}",
        )

        await self._store(new_config)
        await self.save_to_json()

        logger.info(
            f"| 📋 Copied agent {agent_name}@{original_config.version} to {new_name}@{new_version}"
        )
        return new_config

    async def unregister(self, agent_name: str) -> bool:
        """Unregister an agent."""
        if agent_name not in self._agent_configs:
            logger.warning(f"| ⚠️ Agent {agent_name} not found")
            return False

        agent_config = self._agent_configs[agent_name]
        del self._agent_configs[agent_name]

        await self.save_to_json()

        logger.info(f"| 🗑️ Unregistered agent {agent_name}@{agent_config.version}")
        return True

    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all agent configurations with version history to JSON."""

        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            save_data: Dict[str, Any] = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_agents": len(self._agent_configs),
                    "num_versions": sum(
                        len(versions) for versions in self._agent_history_versions.values()
                    ),
                },
                "agents": {},
            }

            for agent_name, version_map in self._agent_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for version_str, agent_config in version_map.items():
                        config_dict = agent_config.model_dump(
                            mode="json", exclude={"cls", "instance", "args_schema"}
                        )

                        if agent_config.args_schema is not None:
                            args_schema_info = serialize_args_schema(
                                agent_config.args_schema
                            )
                            if args_schema_info:
                                config_dict["args_schema"] = args_schema_info

                        versions_data[agent_config.version] = config_dict

                    current_version = None
                    if agent_name in self._agent_configs:
                        current_config = self._agent_configs[agent_name]
                        if current_config is not None:
                            current_version = current_config.version

                    if current_version is None and version_map:
                        latest_version_str = None
                        for version_str in version_map.keys():
                            if latest_version_str is None:
                                latest_version_str = version_str
                            elif (
                                version_manager.compare_versions(
                                    version_str, latest_version_str
                                )
                                > 0
                            ):
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
        """Load agent configurations with version history from JSON."""

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
                        versions_data = agent_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(
                                f"| ⚠️ Agent {agent_name} has invalid format for 'versions' (expected dict), skipping"
                            )
                            continue

                        current_version_str = agent_data.get("current_version")

                        version_configs: List[AgentConfig] = []
                        latest_config: Optional[AgentConfig] = None
                        latest_version: Optional[str] = None

                        for version_str, config_dict in versions_data.items():
                            cls = None
                            config_dict_copy = config_dict.copy()

                            if "code" in config_dict_copy and config_dict_copy["code"]:
                                try:
                                    class_name = config_dict_copy.get("config", {}).get(
                                        "type"
                                    )
                                    if not class_name:
                                        tree = ast.parse(config_dict_copy["code"])
                                        for node in ast.walk(tree):
                                            if isinstance(node, ast.ClassDef):
                                                class_name = node.name
                                                break

                                    if class_name:
                                        cls = dynamic_manager.load_class(
                                            config_dict_copy["code"],
                                            class_name,
                                            Agent,
                                            context="agent",
                                        )
                                        logger.debug(
                                            f"| ✅ Loaded agent class {class_name} from code for {agent_name}"
                                        )
                                    else:
                                        logger.warning(
                                            f"| ⚠️ Cannot determine class name from code for {agent_name}"
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"| ⚠️ Failed to load class from code for {agent_name}: {e}"
                                    )

                            if "version" not in config_dict_copy:
                                config_dict_copy["version"] = version_str

                            args_schema = None
                            if "args_schema" in config_dict_copy:
                                args_schema_info = config_dict_copy.pop("args_schema")
                                try:
                                    args_schema = deserialize_args_schema(args_schema_info)
                                except Exception as e:
                                    logger.warning(
                                        f"| ⚠️ Failed to restore args_schema for {agent_name}@{version_str}: {e}"
                                    )

                            config_dict_copy.pop("cls", None)

                            agent_config = AgentConfig(**config_dict_copy)

                            if cls is not None:
                                agent_config.cls = cls
                            if args_schema is not None:
                                agent_config.args_schema = args_schema

                            version_configs.append(agent_config)

                            if latest_config is None or (
                                current_version_str
                                and agent_config.version == current_version_str
                            ) or (
                                not current_version_str
                                and (
                                    latest_version is None
                                    or version_manager.compare_versions(
                                        agent_config.version, latest_version
                                    )
                                    > 0
                                )
                            ):
                                latest_config = agent_config
                                latest_version = agent_config.version

                        self._agent_history_versions[agent_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }

                        if latest_config:
                            self._agent_configs[agent_name] = latest_config

                            for agent_config in version_configs:
                                await version_manager.register_version(
                                    "agent", agent_name, agent_config.version
                                )

                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)

                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load agent {agent_name}: {e}")
                        continue

                logger.info(
                    f"| 📂 Loaded {loaded_count} agents with version history from {file_path}"
                )
                return True

            except Exception as e:
                logger.error(f"| ❌ Failed to load agents from {file_path}: {e}")
                return False

    def _get_full_module_source(self, cls: Type[Agent]) -> str:
        """Get the full source code of the module containing the class."""
        try:
            module = inspect.getmodule(cls)
            if module is None:
                return inspect.getsource(cls)

            file_path = inspect.getfile(module)
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, TypeError, IOError, AttributeError) as e:
            logger.debug(
                f"| ⚠️ Failed to read module file for {cls.__name__}, falling back to inspect.getsource: {e}"
            )
            try:
                return inspect.getsource(cls)
            except Exception:
                logger.warning(f"| ⚠️ Failed to get source code for {cls.__name__}")
                return ""

    async def restore(
        self, agent_name: str, version: str, auto_initialize: bool = True
    ) -> Optional[AgentConfig]:
        """Restore a specific version of an agent from history."""

        version_config = None
        if agent_name in self._agent_history_versions:
            version_config = self._agent_history_versions[agent_name].get(version)

        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for agent {agent_name}")
            return None

        restored_config = AgentConfig(**version_config.model_dump())
        self._agent_configs[agent_name] = restored_config

        version_history = await version_manager.get_version_history("agent", agent_name)
        if version_history:
            version_history.current_version = version

        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)

        await self.save_to_json()

        logger.info(f"| 🔄 Restored agent {agent_name} to version {version}")
        return restored_config

    async def cleanup(self) -> None:
        """Cleanup all active agents and resources."""
        try:
            self._agent_configs.clear()
            self._agent_history_versions.clear()

            if self._faiss_service is not None:
                await self._faiss_service.cleanup()

            logger.info("| 🧹 Agent context manager cleaned up")
        except Exception as e:
            logger.error(f"| ❌ Error during agent context manager cleanup: {e}")

    async def __call__(self, name: str, input: Dict[str, Any], **kwargs: Any) -> Any:
        """Call an agent by name."""

        agent = await self.get(name)
        if agent is None:
            raise ValueError(f"Agent {name} not found")
        return await agent(**input, **kwargs)

