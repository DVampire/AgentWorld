"""Environment Context Manager for managing environment lifecycle and resources with lazy loading."""

import os
import json
import asyncio
import inflection
from datetime import datetime
from typing import Any, Dict, Callable, Optional, List, Union, Type, Tuple
from pydantic import BaseModel, ConfigDict, Field
from asyncio_atexit import register as async_atexit_register

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import assemble_project_path, gather_with_concurrency
from src.utils.file_utils import file_lock
from src.environment.types import Environment, EnvironmentConfig, ActionConfig
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.dynamic import dynamic_manager
from src.registry import ENVIRONMENT

class EnvironmentContextManager(BaseModel):
    """Global context manager for all environments with lazy loading support."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the environments")
    save_path: str = Field(default=None, description="The path to save the environments")
    contract_path: str = Field(default=None, description="The path to save the environment contract")
            
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None, 
                 contract_path: Optional[str] = None,
                 model_name: str = "openrouter/gemini-3-flash-preview",
                 embedding_model_name: str = "openrouter/text-embedding-3-large",
                 **kwargs):
        """Initialize the environment context manager.
        
        Args:
            base_dir: Base directory for storing environment data
            save_path: Path to save environment configurations
            contract_path: Path to save environment contract
            model_name: The model to use for the environments
            embedding_model_name: The model to use for the environment embeddings
        """
        super().__init__(**kwargs)
        
        # Set up paths
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "environment"))
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| 📁 Environment context manager base directory: {self.base_dir}.")    
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "environment.json")
        logger.info(f"| 📁 Environment context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Environment context manager contract path: {self.contract_path}.")

        self._environment_configs: Dict[str, EnvironmentConfig] = {}  # Current active configs (latest version)
        # Environment version history, e.g., {"env_name": {"1.0.0": EnvironmentConfig, "1.0.1": EnvironmentConfig}}
        self._environment_history_versions: Dict[str, Dict[str, EnvironmentConfig]] = {}
        
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        
        self._cleanup_registered = False
        self._faiss_service = None
        
    async def initialize(self, env_names: Optional[List[str]] = None):
        """Initialize the environment context manager."""
        
        # Register environment-related symbols for auto-injection in dynamic code
        dynamic_manager.register_symbol("ENVIRONMENT", ENVIRONMENT)
        dynamic_manager.register_symbol("Environment", Environment)
        dynamic_manager.register_symbol("EnvironmentConfig", EnvironmentConfig)
        dynamic_manager.register_symbol("ActionConfig", ActionConfig)
        
        # Register environment context provider for automatic import injection
        def environment_context_provider():
            """Provide environment-related imports for dynamic environment classes."""
            return {
                "ENVIRONMENT": ENVIRONMENT,
                "Environment": Environment,
                "EnvironmentConfig": EnvironmentConfig,
                "ActionConfig": ActionConfig,
            }
        dynamic_manager.register_context_provider("environment", environment_context_provider)
        
        # Initialize Faiss service for environment embedding
        self._faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name
        )
        
        # Load environments from ENVIRONMENT registry
        env_configs = {}
        registry_env_configs: Dict[str, EnvironmentConfig] = await self._load_from_registry()
        env_configs.update(registry_env_configs)
        
        # Load environments from code
        code_configs: Dict[str, EnvironmentConfig] = await self._load_from_code()
        
        # Merge code configs with registry configs, only override if code version is strictly greater
        for env_name, code_config in code_configs.items():
            if env_name in env_configs:
                registry_config = env_configs[env_name]
                # Compare versions: only override if code version is strictly greater
                if version_manager.compare_versions(code_config.version, registry_config.version) > 0:
                    logger.info(f"| 🔄 Overriding environment {env_name} from registry (v{registry_config.version}) with code version (v{code_config.version})")
                    env_configs[env_name] = code_config
                else:
                    logger.info(f"| 📌 Keeping environment {env_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater")
                    # If versions are equal, update the history with registry config (which has real class, not dynamic)
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        # Replace the code config in history with registry config to preserve real class reference
                        if env_name in self._environment_history_versions:
                            self._environment_history_versions[env_name][registry_config.version] = registry_config
            else:
                # New environment from code, add it
                env_configs[env_name] = code_config
        
        # Filter environments by names if provided
        if env_names is not None:
            env_configs = {name: env_configs[name] for name in env_names if name in env_configs}
        
        # Build all environments concurrently with a concurrency limit
        env_names_list = list(env_configs.keys())
        tasks = [
            self.build(env_configs[name]) for name in env_names_list
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for env_name, result in zip(env_names_list, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize environment {env_name}: {result}")
                continue
            self._environment_configs[env_name] = result
            logger.info(f"| 🎮 Environment {env_name} initialized")
        
        # Save environment configs to json file
        await self.save_to_json()
        # Save contract to file
        await self.save_contract(env_names=env_names)
        
        # Register cleanup callback
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True
        
        logger.info(f"| ✅ Environments initialization completed")
    
    async def _load_from_registry(self):
        """Load environments from ENVIRONMENT registry."""
        
        env_configs: Dict[str, EnvironmentConfig] = {}
        
        async def register_environment_class(env_cls: Type[Environment]):
            """Register an environment class.
            
            Args:
                env_cls: Environment class to register
            """
            try:
                env_config_key = inflection.underscore(env_cls.__name__)
                env_config_dict= config.get(env_config_key, {})
                
                # Get environment properties from environment class
                env_name = env_cls.model_fields['name'].default
                env_description = env_cls.model_fields['description'].default
                env_metadata = env_cls.model_fields['metadata'].default
                
                # Get or generate version from version_manager
                env_version = await version_manager.get_version("environment", env_name)
                
                # Get full module source code
                env_code = dynamic_manager.get_full_module_source(env_cls)
                
                # Build actions from environment class
                env_actions = {}
                for attr_name in dir(env_cls):
                    attr = getattr(env_cls, attr_name)
                    if hasattr(attr, '_action_name'):
                        action_name = getattr(attr, '_action_name')
                        action_description = getattr(attr, '_action_description', '')
                        action_function = getattr(attr, '_action_function', None)
                        action_metadata = getattr(attr, '_action_metadata', {})
                        
                        action_version = await version_manager.get_version("action", action_name)
                        
                        action_code = dynamic_manager.get_source_code(attr)
                        if not action_code:
                            logger.warning(f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted")
                        
                        action_parameters = dynamic_manager.get_parameters(action_function)
                        action_function_calling = dynamic_manager.build_function_calling(action_name, action_description, action_parameters)
                        action_text = dynamic_manager.build_text_representation(action_name, action_description, action_parameters)
                        action_args_schema = dynamic_manager.build_args_schema(action_name, action_parameters)
                        
                        action_config = ActionConfig(
                            env_name=env_name,
                            name=action_name,
                            description=action_description,
                            function=action_function,
                            metadata=action_metadata,
                            version=action_version,
                            code=action_code,
                            function_calling=action_function_calling,
                            text=action_text,
                            args_schema=action_args_schema,
                        )
                        
                        env_actions[action_name] = action_config
                        
                        
                # Build environment config
                env_config = EnvironmentConfig(
                    name=env_name,
                    description=env_description,
                    metadata=env_metadata,
                    version=env_version,
                    cls=env_cls,
                    config=env_config_dict,
                    instance=None,
                    code=env_code,
                    actions=env_actions,
                    rules="",  # Will be generated when needed
                )
                
                env_configs[env_name] = env_config
                
                # Store in dict-based history (for quick lookup by version)
                if env_name not in self._environment_history_versions:
                    self._environment_history_versions[env_name] = {}
                self._environment_history_versions[env_name][env_version] = env_config
                
                # Register version to version manager
                await version_manager.register_version("environment", env_name, env_version)
                
                logger.info(f"| 📝 Registered environment: {env_name} ({env_cls.__name__})")

            except Exception as e:
                logger.error(f"| ❌ Failed to register environment class {env_cls.__name__}: {e}")
                raise
            
        import src.environment  # noqa: F401
        
        # Get all registered environment classes from ENVIRONMENT registry
        environment_classes = list(ENVIRONMENT._module_dict.values())
        
        logger.info(f"| 🔍 Discovering {len(environment_classes)} environments from ENVIRONMENT registry")
        
        # Register each environment class concurrently with a concurrency limit
        tasks = [
            register_environment_class(env_cls) for env_cls in environment_classes
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(environment_classes)} environments from ENVIRONMENT registry")
        
        return env_configs
    
    async def _load_from_code(self):
        """Load environments from code files.
        
        JSON file content example:
        {
            "metadata": {
                "saved_at": str,  # "YYYY-MM-DD HH:MM:SS"
                "num_environments": int,  # total environment count
                "num_versions": int  # total version count
            },
            "environments": {
                "env_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": {
                            "name": str,
                            "description": str,
                            "metadata": dict,
                            "version": str,
                            "cls": Type[Environment], # will be loaded from code
                            "config": dict,
                            "instance": Environment, # will be built when needed
                            "code": str,
                            "actions": dict, # will be built when needed
                            "rules": str,
                        },
                        ...
                    }
                }
            }
        }
        """
        
        env_configs: Dict[str, EnvironmentConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Environment config file not found at {self.save_path}, skipping code-based loading")
            return env_configs
        
        # Load all environment configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse environment config JSON from {self.save_path}: {e}")
            return env_configs
        
        metadata = load_data.get("metadata", {})
        environments_data = load_data.get("environments", {})

        async def register_environment_class(env_name: str, env_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, EnvironmentConfig], Optional[EnvironmentConfig]]]:
            """Load all versions for a single environment from JSON."""
            try:
                current_version = env_data.get("current_version", "1.0.0")
                versions = env_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Environment {env_name} has no versions")
                    return None
                
                version_map: Dict[str, EnvironmentConfig] = {}
                current_config: Optional[EnvironmentConfig] = None  # Active config for current_version
                
                for _, version_data in versions.items():
                    env_config = EnvironmentConfig.model_validate(version_data)
                    version = env_config.version
                    version_map[version] = env_config
                    
                    if version == current_version:
                        current_config = env_config
                        
                return env_name, version_map, current_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load environment {env_name} from code JSON: {e}")
                return None

        # Launch loading of each environment concurrently with a concurrency limit
        tasks = [
            register_environment_class(env_name, env_data) for env_name, env_data in environments_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            env_name, version_map, current_environment_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._environment_history_versions[env_name] = version_map
            # Active config: the one corresponding to current_version
            if current_environment_config is not None:
                env_configs[env_name] = current_environment_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Environment {env_name} current_version not found, using last available version")
                env_configs[env_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for env_config in version_map.values():
                await version_manager.register_version("environment", env_name, env_config.version)
            
        logger.info(f"| 📂 Loaded {len(env_configs)} environments from {self.save_path}")
        return env_configs
    
    async def _store(self, env_config: EnvironmentConfig):
        """Add environment information to the embedding index.
        
        Args:
            env_config: Environment configuration
        """
        if self._faiss_service is None:
            return
            
        try:
            # Create comprehensive text representation
            env_text = f"Environment: {env_config.name}\nDescription: {env_config.description}"
            
            # Add action descriptions if available
            if env_config.actions:
                action_descriptions = [f"{name}: {action.description}" for name, action in env_config.actions.items()]
                if action_descriptions:
                    env_text += f"\nActions: {'; '.join(action_descriptions)}"
            
            # Add to FAISS index
            request = FaissAddRequest(
                texts=[env_text],
                metadatas=[{
                    "name": env_config.name,
                    "description": env_config.description,
                    "version": env_config.version
                }]
            )
            
            await self._faiss_service.add_documents(request)
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add environment {env_config.name} to FAISS index: {e}")
    
    async def build(self, env_config: EnvironmentConfig) -> EnvironmentConfig:
        """Build an environment instance from config (internal helper, similar to tool's build).
        
        Args:
            env_config: Environment configuration
            
        Returns:
            EnvironmentConfig: Environment configuration with instance
        """
        if env_config.name in self._environment_configs:
            existing_config = self._environment_configs[env_config.name]
            if existing_config.instance is not None:
                return existing_config
        
        try:
            if env_config.cls is None:
                raise ValueError(f"Cannot create environment {env_config.name}: no class provided. Class should be loaded during initialization.")
            
            env_instance = env_config.cls(**env_config.config) if env_config.config else env_config.cls()
            
            # Initialize environment if it has an initialize method
            if hasattr(env_instance, "initialize"):
                await env_instance.initialize()
                
            env_config.instance = env_instance
            
            # Generate rules if not already generated
            if not env_config.rules:
                env_config.rules = env_instance.get_rules()
            
            # Store metadata
            self._environment_configs[env_config.name] = env_config
            
            logger.info(f"| ✅ Environment {env_config.name} created and stored")
            
            return env_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create environment {env_config.name}: {e}")
            raise
    
    async def register(self, 
                       env_cls: Type[Environment], 
                       env_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> EnvironmentConfig:
        """Register an environment class.
        
        This will:
        - Create an environment instance
        - Create an `EnvironmentConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager` and FAISS index
        
        Args:
            env_cls: Environment class
            env_config_dict: Configuration dict for environment initialization.
                           If None, will try to get from global config or use empty dict.
            override: Whether to override existing registration
            version: Optional version string. If None, auto-generates from version_manager.
            
        Returns:
            EnvironmentConfig: Environment configuration
        """
        try:
            if env_config_dict is None:
                # Fallback to global config by class name
                env_config_key = inflection.underscore(env_cls.__name__)
                env_config_dict = getattr(config, env_config_key, {}) if hasattr(config, env_config_key) else {}
            
            # Instantiate environment immediately (register is a runtime operation)
            try:
                env_instance = env_cls(**env_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create environment instance for {env_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate environment {env_cls.__name__} with provided config: {e}")
            
            env_name = env_instance.name
            env_description = env_instance.description
            env_metadata = getattr(env_instance, 'metadata', {})
            
            if not env_name:
                raise ValueError("Environment.name cannot be empty.")
            
            if env_name in self._environment_configs and not override:
                raise ValueError(f"Environment '{env_name}' already registered. Use override=True to replace it.")
            
            # Get or generate version from version_manager
            if version is None:
                env_version = await version_manager.get_version("environment", env_name)
            else:
                env_version = version
            
            # Get environment code
            env_code = dynamic_manager.get_full_module_source(env_cls)
            
            # Build actions from environment class (same as _load_from_registry)
            actions = {}
            for attr_name in dir(env_cls):
                attr = getattr(env_cls, attr_name)
                if hasattr(attr, '_action_name'):
                    action_name = getattr(attr, '_action_name')
                    action_description = getattr(attr, '_action_description', '')
                    action_function = getattr(attr, '_action_function', None)
                    action_metadata = getattr(attr, '_action_metadata', {})
                    
                    action_version = await version_manager.get_version("action", action_name)
                    
                    action_code = dynamic_manager.get_source_code(attr)
                    if not action_code:
                        logger.warning(f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted")
                    
                    action_parameters = dynamic_manager.get_parameters(action_function)
                    action_function_calling = dynamic_manager.build_function_calling(action_name, action_description, action_parameters)
                    action_text = dynamic_manager.build_text_representation(action_name, action_description, action_parameters)
                    action_args_schema = dynamic_manager.build_args_schema(action_name, action_parameters)
                    
                    action_config = ActionConfig(
                        env_name=env_name,
                        name=action_name,
                        description=action_description,
                        function=action_function,
                        metadata=action_metadata,
                        version=action_version,
                        code=action_code,
                        function_calling=action_function_calling,
                        text=action_text,
                        args_schema=action_args_schema,
                    )
                    
                    actions[action_name] = action_config
            
            # Get rules from instance
            env_rules = env_instance.get_rules() if hasattr(env_instance, 'get_rules') else ""
            
            # --- Build EnvironmentConfig ---
            env_config = EnvironmentConfig(
                name=env_name,
                description=env_description,
                rules=env_rules,
                version=env_version,
                actions=actions,
                cls=env_cls,
                config=env_config_dict or {},
                instance=env_instance,
                metadata=env_metadata,
                code=env_code
            )
            
            # --- Persist current config and history ---
            self._environment_configs[env_name] = env_config
            
            # Store in dict-based history (for quick lookup by version)
            if env_name not in self._environment_history_versions:
                self._environment_history_versions[env_name] = {}
            self._environment_history_versions[env_name][env_config.version] = env_config
            
            # Register version in version manager
            await version_manager.register_version("environment", env_name, env_config.version)
            
            # Add to FAISS index
            await self._store(env_config)
            
            # Persist to JSON
            await self.save_to_json()
            
            logger.info(f"| 📝 Registered environment config: {env_name}: {env_config.version}")
            return env_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to register environment: {e}")
            raise
        
    async def get(self, env_name: str) -> Optional[Environment]:
        """Get environment instance by name
        
        Args:
            env_name: Environment name
            
        Returns:
            Environment: Environment instance or None if not found
        """
        env_config = self._environment_configs.get(env_name)
        if env_config:
            return env_config.instance
        return None
    
    async def get_info(self, env_name: str) -> Optional[EnvironmentConfig]:
        """Get environment configuration by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentConfig: Environment configuration or None if not found
        """
        return self._environment_configs.get(env_name)
        
    async def get_state(self, env_name: str) -> Optional[Dict[str, Any]]:
        """Get the state of an environment
        
        Args:
            env_name: Environment name
            
        Returns:
            Optional[Dict[str, Any]]: State of the environment or None if not found
        """
        env_config = self._environment_configs.get(env_name)
        if not env_config or not env_config.instance:
            raise ValueError(f"Environment '{env_name}' not found")
        return await env_config.instance.get_state()
        
    async def list(self) -> List[str]:
        """Get list of registered environments
        
        Args:
            include_disabled: Whether to include disabled environments (not used for environments, kept for compatibility)
            
        Returns:
            List[str]: List of registered environment names
        """
        return [name for name in self._environment_configs.keys()]
    
    
    async def update(self, 
                     env_cls: Type[Environment],
                     env_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> EnvironmentConfig:
        """Update an existing environment with new configuration and create a new version
        
        Args:
            env_cls: New environment class with updated implementation
            env_config_dict: Configuration dict for environment initialization
                   If None, will try to get from global config
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            EnvironmentConfig: Updated environment configuration
        """
        try:
            if env_config_dict is None:
                # Fallback to global config by class name
                env_config_key = inflection.underscore(env_cls.__name__)
                env_config_dict = getattr(config, env_config_key, {}) if hasattr(config, env_config_key) else {}
            
            # Instantiate environment immediately (update is a runtime operation)
            try:
                env_instance = env_cls(**env_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create environment instance for {env_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate environment {env_cls.__name__} with provided config: {e}")
            
            env_name = env_instance.name
            
            # Check if environment exists
            original_config = self._environment_configs.get(env_name)
            if original_config is None:
                raise ValueError(f"Environment {env_name} not found. Use register() to register a new environment.")
            
            env_description = env_instance.description
            env_metadata = getattr(env_instance, 'metadata', {})
            
            # Determine new version from version_manager
            if new_version is None:
                # Get current version from version_manager and generate next patch version
                new_version = await version_manager.generate_next_version("environment", env_name, "patch")
            
            # Get environment code
            env_code = dynamic_manager.get_full_module_source(env_cls)
            
            # Build actions from environment class (same as register)
            actions = {}
            for attr_name in dir(env_cls):
                attr = getattr(env_cls, attr_name)
                if hasattr(attr, '_action_name'):
                    action_name = getattr(attr, '_action_name')
                    action_description = getattr(attr, '_action_description', '')
                    action_function = getattr(attr, '_action_function', None)
                    action_metadata = getattr(attr, '_action_metadata', {})
                    
                    action_version = await version_manager.get_version("action", action_name)
                    
                    action_code = dynamic_manager.get_source_code(attr)
                    if not action_code:
                        logger.warning(f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted")
                    
                    action_parameters = dynamic_manager.get_parameters(action_function)
                    action_function_calling = dynamic_manager.build_function_calling(action_name, action_description, action_parameters)
                    action_text = dynamic_manager.build_text_representation(action_name, action_description, action_parameters)
                    action_args_schema = dynamic_manager.build_args_schema(action_name, action_parameters)
                    
                    action_config = ActionConfig(
                        env_name=env_name,
                        name=action_name,
                        description=action_description,
                        function=action_function,
                        metadata=action_metadata,
                        version=action_version,
                        code=action_code,
                        function_calling=action_function_calling,
                        text=action_text,
                        args_schema=action_args_schema,
                    )
                    
                    actions[action_name] = action_config
            
            # Get rules from instance
            env_rules = env_instance.get_rules() if hasattr(env_instance, 'get_rules') else ""
            
            # --- Build EnvironmentConfig ---
            updated_config = EnvironmentConfig(
                name=env_name,  # Keep same name
                description=env_description,
                rules=env_rules,
                version=new_version,
                actions=actions,
                cls=env_cls,
                config=env_config_dict or {},
                instance=env_instance,
                metadata=env_metadata,
                code=env_code
            )
            
            # Update the environment config (replaces current version)
            self._environment_configs[env_name] = updated_config
            
            # Store in version history
            if env_name not in self._environment_history_versions:
                self._environment_history_versions[env_name] = {}
            self._environment_history_versions[env_name][updated_config.version] = updated_config
            
            # Register new version record to version manager
            await version_manager.register_version(
                "environment", 
                env_name, 
                new_version,
                description=description or f"Updated from {original_config.version}"
            )
            
            # Update embedding index
            await self._store(updated_config)
            
            # Persist to JSON
            await self.save_to_json()
            
            logger.info(f"| 🔄 Updated environment {env_name} from v{original_config.version} to v{new_version}")
            return updated_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to update environment: {e}")
            raise
    
    async def copy(self, 
                  env_name: str,
                  new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, 
                  new_config: Optional[Dict[str, Any]] = None) -> EnvironmentConfig:
        """Copy an existing environment configuration
        
        Args:
            env_name: Name of the environment to copy
            new_name: New name for the copied environment. If None, uses original name.
            new_version: New version for the copied environment. If None, increments version.
            new_config: New configuration dict for the copied environment. If None, uses original config.
            
        Returns:
            EnvironmentConfig: New environment configuration
        """
        try:
            original_config = self._environment_configs.get(env_name)
            if original_config is None:
                raise ValueError(f"Environment {env_name} not found")
            
            if original_config.cls is None:
                raise ValueError(f"Cannot copy environment {env_name}: no class provided")
            
            # Determine new name
            if new_name is None:
                new_name = env_name
            
            # Prepare config dict (merge original config with new config)
            env_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # Merge new config into original config
                env_config_dict.update(new_config)
            
            # Instantiate environment instance (copy is a runtime operation)
            try:
                env_instance = original_config.cls(**env_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create environment instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate environment {original_config.cls.__name__} with provided config: {e}")
            
            # Apply name override if provided (after instantiation)
            if new_name != env_name:
                env_instance.name = new_name
            
            env_description = env_instance.description
            env_metadata = getattr(env_instance, 'metadata', {})
            
            # Determine new version from version_manager
            if new_version is None:
                if new_name == env_name:
                    # If copying with same name, get next version from version_manager
                    new_version = await version_manager.generate_next_version("environment", new_name, "patch")
                else:
                    # If copying with different name, get or generate version for new name
                    new_version = await version_manager.get_version("environment", new_name)
            
            # Get environment code
            env_code = dynamic_manager.get_full_module_source(original_config.cls)
            
            # Build actions from environment class (same as register)
            actions = {}
            for attr_name in dir(original_config.cls):
                attr = getattr(original_config.cls, attr_name)
                if hasattr(attr, '_action_name'):
                    action_name = getattr(attr, '_action_name')
                    action_description = getattr(attr, '_action_description', '')
                    action_function = getattr(attr, '_action_function', None)
                    action_metadata = getattr(attr, '_action_metadata', {})
                    
                    action_version = await version_manager.get_version("action", action_name)
                    
                    action_code = dynamic_manager.get_source_code(attr)
                    if not action_code:
                        logger.warning(f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted")
                    
                    action_parameters = dynamic_manager.get_parameters(action_function)
                    action_function_calling = dynamic_manager.build_function_calling(action_name, action_description, action_parameters)
                    action_text = dynamic_manager.build_text_representation(action_name, action_description, action_parameters)
                    action_args_schema = dynamic_manager.build_args_schema(action_name, action_parameters)
                    
                    action_config = ActionConfig(
                        env_name=new_name,
                        name=action_name,
                        description=action_description,
                        function=action_function,
                        metadata=action_metadata,
                        version=action_version,
                        code=action_code,
                        function_calling=action_function_calling,
                        text=action_text,
                        args_schema=action_args_schema,
                    )
                    
                    actions[action_name] = action_config
            
            # Get rules from instance
            env_rules = env_instance.get_rules() if hasattr(env_instance, 'get_rules') else ""
            
            # --- Build EnvironmentConfig ---
            copied_config = EnvironmentConfig(
                name=new_name,
                description=env_description,
                rules=env_rules,
                version=new_version,
                actions=actions,
                cls=original_config.cls,
                config=env_config_dict,
                instance=env_instance,
                metadata=env_metadata,
                code=env_code
            )
            
            # Register new environment
            self._environment_configs[new_name] = copied_config
            
            # Store in version history
            if new_name not in self._environment_history_versions:
                self._environment_history_versions[new_name] = {}
            self._environment_history_versions[new_name][new_version] = copied_config
            
            # Register version record to version manager
            await version_manager.register_version(
                "environment", 
                new_name, 
                new_version,
                description=f"Copied from {env_name}@{original_config.version}"
            )
            
            # Register to embedding index
            await self._store(copied_config)
            
            # Persist to JSON
            await self.save_to_json()
            
            logger.info(f"| 📋 Copied environment {env_name}@{original_config.version} to {new_name}@{new_version}")
            return copied_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to copy environment: {e}")
            raise
    
    async def unregister(self, env_name: str) -> bool:
        """Unregister an environment
        
        Args:
            env_name: Name of the environment to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if env_name not in self._environment_configs:
            logger.warning(f"| ⚠️ Environment {env_name} not found")
            return False
        
        env_config = self._environment_configs[env_name]
        
        # Remove from configs
        del self._environment_configs[env_name]

        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered environment {env_name}@{env_config.version}")
        return True
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all environment configurations with version history to JSON.
        
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
            
            # Prepare save data - save all versions for each environment
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_environments": len(self._environment_configs),
                    "num_versions": sum(len(versions) for versions in self._environment_history_versions.values()),
                },
                "environments": {}
            }
            
            for env_name, version_map in self._environment_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, env_config in version_map.items():
                        config_dict = env_config.model_dump()
                        versions_data[env_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if env_name in self._environment_configs:
                        current_config = self._environment_configs[env_name]
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
                    
                    save_data["environments"][env_name] = {
                        "versions": versions_data,
                        "current_version": current_version
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize environment {env_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._environment_configs)} environments with version history to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load environment configurations with version history from JSON.
        
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
                logger.warning(f"| ⚠️ Environment file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                environments_data = load_data.get("environments", {})
                loaded_count = 0
                
                for env_name, env_data in environments_data.items():
                    try:
                        # Expected format: multiple versions stored as a dict {version_str: config_dict}
                        versions_data = env_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Environment {env_name} has invalid format for 'versions' (expected dict), skipping")
                            continue
                        
                        current_version_str = env_data.get("current_version")
                        
                        # Load all versions
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for version_str, config_dict in versions_data.items():
                            # Ensure version field is present
                            if "version" not in config_dict:
                                config_dict["version"] = version_str
                            
                            try:
                                env_config = EnvironmentConfig.model_validate(config_dict)
                                version_configs.append(env_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load environment config for {env_name}@{version_str}: {e}")
                                continue
                            
                            # Track latest version
                            if latest_config is None or (
                                current_version_str and env_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    version_manager.compare_versions(env_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = env_config
                                latest_version = env_config.version
                        
                        # Store all versions in history (dict-based)
                        self._environment_history_versions[env_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }
                        
                        # Only set latest version as active
                        if latest_config:
                            self._environment_configs[env_name] = latest_config
                            
                            # Register all versions to version manager (only version records)
                            for env_config in version_configs:
                                await version_manager.register_version("environment", env_name, env_config.version)
                            
                            # Create instance if requested (instance is not saved in JSON, must be created via build)
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load environment {env_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} environments with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load environments from {file_path}: {e}")
                return False
    
    
    async def restore(self, env_name: str, version: str, auto_initialize: bool = True) -> Optional[EnvironmentConfig]:
        """Restore a specific version of an environment from history
        
        Args:
            env_name: Name of the environment
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored environment
            
        Returns:
            EnvironmentConfig of the restored version, or None if not found
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if env_name in self._environment_history_versions:
            version_config = self._environment_history_versions[env_name].get(version)
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for environment {env_name}")
            return None
        
        # Create a copy to avoid modifying the history
        restored_config = EnvironmentConfig(**version_config.model_dump())
        
        # Set as current active config
        self._environment_configs[env_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("environment", env_name)
        if version_history:
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("environment", env_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("environment", env_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored environment {env_name} to version {version}")
        return restored_config
    
    async def save_contract(self, env_names: Optional[List[str]] = None):
        """Save the contract for an environment"""
        contract = []
        if env_names is not None:
            for index, env_name in enumerate(env_names):
                env_info = await self.get_info(env_name)
                if env_info is None:
                    continue
                text = env_info.rules
                contract.append(f"{index + 1:04d}\n{text}\n")
        else:
            for index, env_name in enumerate(self._environment_configs.keys()):
                env_info = await self.get_info(env_name)
                text = env_info.rules
                contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} environments contract to {self.contract_path}")
    
    async def load_contract(self) -> str:
        """Load the contract for an environment"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
    async def cleanup(self):
        """Cleanup all active environments."""
        try:
            # Cleanup all instances
            for env_name, env_config in self._environment_configs.items():
                if env_config.instance and hasattr(env_config.instance, "cleanup"):
                    try:
                        await env_config.instance.cleanup()
                    except Exception as e:
                        logger.warning(f"| ⚠️ Error cleaning up environment {env_name} instance: {e}")
            
            # Clear all environment configs and version history
            self._environment_configs.clear()
            self._environment_history_versions.clear()
            
            # Clean up Faiss service (async)
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            
            logger.info("| 🧹 Environment context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during environment context manager cleanup: {e}")
            
    async def __call__(self, name: str, action: str, input: Dict[str, Any]) -> Any:
        """Call an environment action
        
        Args:
            name: Name of the environment
            action: Name of the action
            input: Input for the action
            
        Returns:
            Action result
        """
        if name in self._environment_configs:
            env_config = self._environment_configs[name]
            
            instance = env_config.instance
            if instance is None:
                raise ValueError(f"Environment {name} is not initialized")
            
            action_config = env_config.actions.get(action)
            if action_config is None:
                raise ValueError(f"Action {action} not found in environment {name}")
            action_function = action_config.function
            
            # Check if action_function is a bound method (already has self bound)
            # Bound methods have __self__ attribute, unbound methods don't
            if hasattr(action_function, '__self__'):
                # Bound method: call directly without passing instance
                return await action_function(**input)
            else:
                # Unbound method: pass instance as first argument
                return await action_function(instance, **input)
        else:
            raise ValueError(f"Environment {name} not found")
