"""Environment Context Manager for managing environment lifecycle and resources with lazy loading."""

import os
import json
import asyncio
import inflection
import inspect
from datetime import datetime
from typing import Any, Dict, Callable, Optional, List, Union, Type, Tuple
from pydantic import BaseModel, ConfigDict, Field
from asyncio_atexit import register as async_atexit_register

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import assemble_project_path, gather_with_concurrency
from src.utils.file_utils import file_lock
from src.utils import serialize_args_schema, deserialize_args_schema
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
            else:
                # New environment from code, add it
                env_configs[env_name] = code_config
        
        # Filter environments by names if provided
        if env_names is not None:
            env_configs = {name: env_configs[name] for name in env_names if name in env_configs}
        
        # Build all environments concurrently with a concurrency limit
        env_names_list = list(env_configs.keys())
        tasks = [
            self._build_environment(env_configs[name]) for name in env_names_list
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
                # Get environment config from global config
                # Follow the same pattern as tools:
                #   tool:  inflection.underscore(ToolCls.__name__) -> config.get(key, {})
                #   env :  inflection.underscore(EnvironmentCls.__name__) -> config.get(key, {})
                #
                # For example, FileSystemEnvironment -> "file_system_environment",
                # which matches the key used in `configs/tool_calling_agent.py`.
                env_config_key = inflection.underscore(env_cls.__name__)
                env_config_dict_raw = config.get(env_config_key, {})
                
                # Filter out None values from config to avoid passing None to constructors
                env_config_dict = {k: v for k, v in env_config_dict_raw.items() if v is not None} if env_config_dict_raw else {}
                
                # Get environment properties from environment class
                env_name = env_cls.model_fields['name'].default
                env_description = env_cls.model_fields['description'].default
                
                # Get or generate version from version_manager
                env_version = await version_manager.get_version("environment", env_name)
                
                # Get full module source code (including imports)
                # This ensures all imports are preserved when saving/loading from JSON
                env_code = self._get_full_module_source(env_cls)
                
                # Collect actions from class methods marked with @ecp.action
                actions = {}
                for attr_name in dir(env_cls):
                    attr = getattr(env_cls, attr_name)
                    if hasattr(attr, '_action_name'):
                        action_name = getattr(attr, '_action_name')
                        action_config = ActionConfig(
                            env_name=env_name,
                            name=action_name,
                            description=getattr(attr, '_action_description', ''),
                            function=getattr(attr, '_action_function', None),
                            metadata=getattr(attr, '_metadata', {})
                        )
                        actions[action_name] = action_config
                
                # Get metadata
                metadata = {}
                
                # Create environment config
                env_config = EnvironmentConfig(
                    name=env_name,
                    description=env_description,
                    rules="",  # Will be generated when needed
                    version=env_version,
                    actions=actions,
                    cls=env_cls,
                    config=env_config_dict,
                    instance=None,
                    metadata=metadata,
                    code=env_code
                )
                
                # Store environment config
                env_configs[env_name] = env_config
                
                # Store in version history (by version string)
                if env_name not in self._environment_history_versions:
                    self._environment_history_versions[env_name] = {}
                self._environment_history_versions[env_name][env_version] = env_config
                
                logger.info(f"| 📝 Registered environment: {env_name} ({env_cls.__name__})")
                
                return env_name
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register environment class {env_cls.__name__}: {e}")
                return None
            
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
                            "version": str,
                            "cls": Type[Environment], # will be loaded from code
                            "config": dict,
                            "instance": Environment, # will be built when needed
                            "metadata": dict,
                            "actions": dict, # will be built when needed
                            "rules": str,
                            "code": str
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
                
                for version_str, version_data in versions.items():
                    name = version_data.get("name", "")
                    description = version_data.get("description", "")
                    version = version_data.get("version", version_str)
                    rules = version_data.get("rules", "")
                    code = version_data.get("code", None)
                    config_dict = version_data.get("config", {})
                    metadata = version_data.get("metadata", {})
                    actions_data = version_data.get("actions", {})
                    instance = version_data.get("instance", None)
                    
                    cls = None
                    if code:
                        # Dynamic class: load class from source code
                        # Extract class name from config
                        class_name = config_dict.get("type")
                        if not class_name:
                            # Try to extract from code
                            class_name = dynamic_manager.extract_class_name_from_code(code)
                        
                        if class_name:
                            try:
                                # Use context="environment" for automatic import injection
                                cls = dynamic_manager.load_class(
                                    code, 
                                    class_name=class_name,
                                    base_class=Environment,
                                    context="environment"
                                )
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load dynamic class for {env_name}@{version}: {e}")
                    
                    # Restore actions from saved data
                    actions = {}
                    if actions_data:
                        for action_name, action_data in actions_data.items():
                            if isinstance(action_data, dict):
                                # Restore args_schema from saved schema info (if present)
                                args_schema = None
                                args_schema_info = action_data.get("args_schema")
                                if args_schema_info:
                                    try:
                                        args_schema = deserialize_args_schema(args_schema_info)
                                    except Exception as e:
                                        logger.warning(f"| ⚠️ Failed to restore args_schema for action {action_name}@{version}: {e}")
                                
                                # Remove args_schema from dict before creating ActionConfig
                                action_data_copy = action_data.copy()
                                action_data_copy.pop("args_schema", None)
                                
                                # Restore ActionConfig from dict
                                action_config = ActionConfig(**action_data_copy)
                                
                                # Set args_schema after creation to bypass validation
                                if args_schema is not None:
                                    action_config.args_schema = args_schema
                                
                                actions[action_name] = action_config
                    
                    # Get code from version_data if available
                    code = version_data.get("code", None)
                    
                    # Create EnvironmentConfig
                    env_config = EnvironmentConfig(
                        name=name,
                        description=description,
                        rules=rules,
                        version=version,
                        cls=cls,
                        config=config_dict,
                        instance=instance,
                        metadata=metadata,
                        actions=actions,
                        code=code,
                    )
                    
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
            env_name, version_map, current_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._environment_history_versions[env_name] = version_map
            # Active config: the one corresponding to current_version
            if current_config is not None:
                env_configs[env_name] = current_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Environment {env_name} current_version not found, using last available version")
                env_configs[env_name] = list(version_map.values())[-1]
            
        logger.info(f"| 📂 Loaded {len(env_configs)} environments from {self.save_path}")
        return env_configs
    
    async def _build_environment(self, env_config: EnvironmentConfig) -> EnvironmentConfig:
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
            
            # Filter out None values from config to avoid passing None to constructors
            filtered_config = {}
            if env_config.config:
                filtered_config = {k: v for k, v in env_config.config.items() if v is not None}
            
            # Create instance from cls
            if filtered_config:
                instance = env_config.cls(**filtered_config)
            else:
                instance = env_config.cls()
            
            # Initialize environment
            if hasattr(instance, "initialize"):
                await instance.initialize()
            
            # Collect actions from instance's actions dictionary
            if hasattr(instance, 'actions') and isinstance(instance.actions, dict):
                env_config.actions = instance.actions.copy()
            
            # Store instance
            env_config.instance = instance
            
            # Generate rules if not already generated
            if not env_config.rules:
                env_config.rules = instance.rules
            
            # Store metadata
            self._environment_configs[env_config.name] = env_config
            
            logger.info(f"| ✅ Environment {env_config.name} created and stored")
            return env_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create environment {env_config.name}: {e}")
            raise
    
    async def build(self, 
              env_config: EnvironmentConfig,
              env_factory: Optional[Callable] = None,
              **kwargs
              ) -> EnvironmentConfig:
        """Create and store an environment instance.
        
        Args:
            env_config: Environment configuration
            env_factory: Function to create the environment instance
            
        Returns:
            EnvironmentConfig: Environment configuration
        """
        if env_config.name in self._environment_configs:
            existing_config = self._environment_configs[env_config.name]
            # If instance already exists, return it
            if existing_config.instance is not None:
                return existing_config
            # Otherwise, update config and create instance
            existing_config.config = env_config.config
            existing_config.cls = env_config.cls
            env_config = existing_config
        
        try:
            # Create environment instance
            if env_config.cls is None:
                raise ValueError(f"Cannot create environment {env_config.name}: no class provided. Class should be loaded during initialization.")
            
            # Filter out None values from config to avoid passing None to constructors
            filtered_config = {}
            if env_config.config:
                filtered_config = {k: v for k, v in env_config.config.items() if v is not None}
            
            # Use factory if provided and valid, otherwise create from cls
            if env_factory and callable(env_factory):
                try:
                    instance = env_factory()
                    if instance is None:
                        # Factory returned None, fall back to cls
                        if filtered_config:
                            instance = env_config.cls(**filtered_config)
                        else:
                            instance = env_config.cls()
                except Exception:
                    # Factory failed, fall back to cls
                    if filtered_config:
                        instance = env_config.cls(**filtered_config)
                    else:
                        instance = env_config.cls()
            else:
                if filtered_config:
                    instance = env_config.cls(**filtered_config)
                else:
                    instance = env_config.cls()
            
            # Initialize environment
            if hasattr(instance, "initialize"):
                await instance.initialize()
            
            # Collect actions from instance's actions dictionary
            if hasattr(instance, 'actions') and isinstance(instance.actions, dict):
                # Update env_config with actions from instance
                env_config.actions = instance.actions.copy()
            
            # Store instance
            env_config.instance = instance
            
            # Generate rules if not already generated
            if not env_config.rules:
                env_config.rules = instance.rules
            
            # Store metadata
            self._environment_configs[env_config.name] = env_config
            
            logger.info(f"| ✅ Environment {env_config.name} created and stored")
            return env_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create environment {env_config.name}: {e}")
            raise
    
    async def register(self, env: Union[Environment, Type[Environment]], *, override: bool = False, **kwargs: Any) -> EnvironmentConfig:
        """Register an environment class or instance.
        
        Args:
            env: Environment class or instance
            override: Whether to override existing registration
            **kwargs: Configuration for environment initialization
            
        Returns:
            EnvironmentConfig: Environment configuration
        """
        # Create temporary instance to get name and description
        try:
            if isinstance(env, Environment):
                env_name = env.name
                env_description = env.description
                env_cls = type(env)
                env_instance = env
            elif isinstance(env, type) and issubclass(env, Environment):
                # Try to create temporary instance
                try:
                    temp_instance = env(**kwargs)
                    env_name = temp_instance.name
                    env_description = temp_instance.description
                    env_cls = env
                    env_instance = None
                except Exception as inst_exc:
                    # If instantiation fails, try to get from model_fields
                    logger.debug(f"| ⚠️ Failed to instantiate {env.__name__} for registration: {inst_exc}")
                    env_name = None
                    env_description = ''
                    
                    # Try to get name from model_fields default value
                    if hasattr(env, 'model_fields') and 'name' in env.model_fields:
                        field_info = env.model_fields['name']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            # Check if default is a string (not None and not undefined)
                            if isinstance(default_value, str) and default_value:
                                env_name = default_value
                                logger.debug(f"| 📝 Got env_name from model_fields: {env_name}")
                    
                    # Try to get description from model_fields default value
                    if hasattr(env, 'model_fields') and 'description' in env.model_fields:
                        field_info = env.model_fields['description']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            # Check if default is a string
                            if isinstance(default_value, str):
                                env_description = default_value
                    
                    env_cls = env
                    env_instance = None
                    
                    if not env_name:
                        raise ValueError(f"Environment class {env.__name__} has no name (failed to instantiate and no default in model_fields)")
            else:
                raise TypeError(f"Expected Environment instance or subclass, got {type(env)!r}")
            
            if not env_name:
                raise ValueError("Environment.name cannot be empty.")
            
            if env_name in self._environment_configs and not override:
                raise ValueError(f"Environment '{env_name}' already registered. Use override=True to replace it.")
            
            # Collect actions - if instance exists, use its actions dictionary, otherwise collect from class
            actions = {}
            if env_instance is not None:
                # If registering an instance, use its actions dictionary
                if hasattr(env_instance, 'actions') and isinstance(env_instance.actions, dict):
                    actions = env_instance.actions.copy()
            else:
                # If registering a class, collect actions from class methods marked with @ecp.action
                target = env_cls
                for attr_name in dir(target):
                    attr = getattr(target, attr_name)
                    if hasattr(attr, '_action_name'):
                        action_name = getattr(attr, '_action_name')
                        action_config = ActionConfig(
                            env_name=env_name,
                            name=action_name,
                            description=getattr(attr, '_action_description', ''),
                            function=getattr(attr, '_action_function', None),
                            metadata=getattr(attr, '_metadata', {})
                        )
                        actions[action_name] = action_config
            
            # Get metadata
            if env_instance is not None:
                metadata = getattr(env_instance, 'metadata', {})
            else:
                metadata = {}
            
            # Get or generate version from version_manager
            version = await version_manager.get_version("environment", env_name)
            
            # Dynamic code handling
            env_code = None
            if dynamic_manager.is_dynamic_class(env_cls):
                env_code = dynamic_manager.get_class_source_code(env_cls)
                if not env_code:
                    logger.warning(f"| ⚠️ Environment {env_name} is dynamic but source code cannot be extracted")
            else:
                # Get full module source code for non-dynamic classes
                env_code = self._get_full_module_source(env_cls)
            
            # Create EnvironmentConfig
            if env_instance is not None:
                # Registering an instance
                env_config = EnvironmentConfig(
                    name=env_name,
                    description=env_description,
                    rules="",  # Will be generated when needed
                    version=version,
                    actions=actions,
                    cls=env_cls,
                    config={},
                    instance=env_instance,
                    metadata=metadata,
                    code=env_code
                )
            else:
                # Registering a class - store config for lazy loading
                # Filter out None values from kwargs
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None} if kwargs else {}
                env_config = EnvironmentConfig(
                    name=env_name,
                    description=env_description,
                    rules="",  # Will be generated when needed
                    version=version,
                    actions=actions,
                    cls=env_cls,
                    config=filtered_kwargs,
                    instance=None,  # Will be created on initialize
                    metadata=metadata,
                    code=env_code
                )
            
            # Generate rules if we have instance, or try to create temporary instance
            if env_config.instance:
                env_config.rules = env_config.instance.rules
            elif env_config.cls and env_config.config is not None:
                # Try to create temporary instance to generate rules
                try:
                    temp_instance = env_config.cls(**env_config.config)
                    env_config.rules = temp_instance.rules
                except Exception as e:
                    logger.debug(f"| ⚠️ Failed to create temporary instance for rules generation: {e}")
                    env_config.rules = ""  # Leave empty, will be generated when instance is created
            
            # Store environment config
            self._environment_configs[env_name] = env_config
            
            # Store in version history (by version string)
            if env_name not in self._environment_history_versions:
                self._environment_history_versions[env_name] = {}
            self._environment_history_versions[env_name][env_config.version] = env_config
            
            # Register version record to version manager
            await version_manager.register_version("environment", env_name, env_config.version)
            
            # Add to FAISS index
            await self._store(env_config)
            
            logger.debug(f"| 📝 Registered environment config: {env_name} v{env_config.version}")
            
            return env_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register environment: {e}")
            raise
    
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
        
    async def list(self, include_disabled: bool = False) -> List[str]:
        """Get list of registered environments
        
        Args:
            include_disabled: Whether to include disabled environments (not used for environments, kept for compatibility)
            
        Returns:
            List[str]: List of registered environment names
        """
        return [name for name in self._environment_configs.keys()]
    
    async def get_info(self, env_name: str) -> Optional[EnvironmentConfig]:
        """Get environment configuration by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentConfig: Environment configuration or None if not found
        """
        return self._environment_configs.get(env_name)
    
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
    
    async def update(self, env_name: str, env: Union[Environment, Type[Environment]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> EnvironmentConfig:
        """Update an existing environment with new configuration and create a new version
        
        Args:
            env_name: Name of the environment to update
            env: New environment class or instance with updated implementation
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for environment initialization
            
        Returns:
            EnvironmentConfig: Updated environment configuration
        """
        original_config = self._environment_configs.get(env_name)
        if original_config is None:
            raise ValueError(f"Environment {env_name} not found. Use register() to register a new environment.")
        
        # Get new environment info
        if isinstance(env, Environment):
            new_description = env.description
            env_cls = type(env)
            env_instance = env
        elif isinstance(env, type) and issubclass(env, Environment):
            try:
                temp_instance = env(**kwargs)
                new_description = temp_instance.description
                env_cls = env
                env_instance = None
            except Exception:
                new_description = getattr(env, 'description', original_config.description)
                env_cls = env
                env_instance = None
        else:
            raise TypeError(f"Expected Environment instance or subclass, got {type(env)!r}")
        
        # Determine new version from version_manager
        if new_version is None:
            # Get current version from version_manager and generate next patch version
            new_version = await version_manager.generate_next_version("environment", env_name, "patch")
        
        # Collect actions
        actions = {}
        if env_instance is not None:
            if hasattr(env_instance, 'actions') and isinstance(env_instance.actions, dict):
                actions = env_instance.actions.copy()
        else:
            target = env_cls
            for attr_name in dir(target):
                attr = getattr(target, attr_name)
                if hasattr(attr, '_action_name'):
                    action_name = getattr(attr, '_action_name')
                    action_config = ActionConfig(
                        env_name=env_name,
                        name=action_name,
                        description=getattr(attr, '_action_description', ''),
                        function=getattr(attr, '_action_function', None),
                        metadata=getattr(attr, '_metadata', {})
                    )
                    actions[action_name] = action_config
        
        # Get metadata
        if env_instance is not None:
            metadata = getattr(env_instance, 'metadata', {})
        else:
            metadata = original_config.metadata if original_config else {}
        
        # Dynamic code handling
        env_code = None
        if dynamic_manager.is_dynamic_class(env_cls):
            env_code = dynamic_manager.get_class_source_code(env_cls)
            if not env_code:
                logger.warning(f"| ⚠️ Environment {env_name} is dynamic but source code cannot be extracted")
        else:
            env_code = self._get_full_module_source(env_cls)
        
        # Create updated config
        if env_instance is not None:
            updated_config = EnvironmentConfig(
                name=env_name,
                description=description or new_description,
                rules="",  # Will be generated below
                version=new_version,
                actions=actions,
                cls=env_cls,
                config={},
                instance=env_instance,
                metadata=metadata,
                code=env_code
            )
        else:
            # Filter out None values from kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None} if kwargs else {}
            updated_config = EnvironmentConfig(
                name=env_name,
                description=description or new_description,
                rules="",  # Will be generated below
                version=new_version,
                actions=actions,
                cls=env_cls,
                config=filtered_kwargs,
                instance=None,
                metadata=metadata,
                code=env_code
            )
        
        # Generate rules
        if updated_config.instance:
            updated_config.rules = updated_config.instance.rules
        elif updated_config.cls and updated_config.config is not None:
            # Try to create temporary instance to generate rules
            try:
                temp_instance = updated_config.cls(**updated_config.config)
                updated_config.rules = temp_instance.rules
            except Exception as e:
                logger.debug(f"| ⚠️ Failed to create temporary instance for rules generation: {e}")
                updated_config.rules = ""  # Leave empty, will be generated when instance is created
        else:
            updated_config.rules = ""  # Leave empty, will be generated when instance is created
        
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
        
        logger.info(f"| 🔄 Updated environment {env_name} from v{original_config.version} to v{new_version}")
        return updated_config
    
    async def copy(self, env_name: str, new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, **override_config) -> EnvironmentConfig:
        """Copy an existing environment configuration
        
        Args:
            env_name: Name of the environment to copy
            new_name: New name for the copied environment. If None, uses original name.
            new_version: New version for the copied environment. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            EnvironmentConfig: New environment configuration
        """
        original_config = self._environment_configs.get(env_name)
        if original_config is None:
            raise ValueError(f"Environment {env_name} not found")
        
        # Determine new name
        if new_name is None:
            new_name = env_name
        
        # Determine new version from version_manager
        if new_version is None:
            if new_name == env_name:
                # If copying with same name, get next version from version_manager
                new_version = await version_manager.generate_next_version("environment", new_name, "patch")
            else:
                # If copying with different name, get or generate version for new name
                new_version = await version_manager.get_or_generate_version("environment", new_name)
        
        # Create copy of config
        new_config_dict = original_config.model_dump()
        new_config_dict["name"] = new_name
        new_config_dict["version"] = new_version
        
        # Apply overrides
        if override_config:
            if "description" in override_config:
                new_config_dict["description"] = override_config.pop("description")
            if "version" in override_config:
                new_config_dict["version"] = override_config.pop("version")
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
        
        # Update actions env_name in copied actions
        if "actions" in new_config_dict and new_config_dict["actions"]:
            for action_name, action_config_dict in new_config_dict["actions"].items():
                if isinstance(action_config_dict, dict):
                    action_config_dict["env_name"] = new_name
        
        new_config = EnvironmentConfig(**new_config_dict)
        
        # Register new environment
        self._environment_configs[new_name] = new_config
        
        # Store in version history
        if new_name not in self._environment_history_versions:
            self._environment_history_versions[new_name] = {}
        self._environment_history_versions[new_name][new_version] = new_config
        
        # Register version record to version manager
        await version_manager.register_version(
            "environment", 
            new_name, 
            new_config.version,
            description=f"Copied from {env_name}@{original_config.version}"
        )
        
        # Register to embedding index
        await self._store(new_config)
        
        logger.info(f"| 📋 Copied environment {env_name}@{original_config.version} to {new_name}@{new_config.version}")
        return new_config
    
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
        
        # Cleanup instance if exists
        if env_config.instance and hasattr(env_config.instance, "cleanup"):
            try:
                if asyncio.iscoroutinefunction(env_config.instance.cleanup):
                    await env_config.instance.cleanup()
                else:
                    env_config.instance.cleanup()
            except Exception as e:
                logger.warning(f"| ⚠️ Error cleaning up environment {env_name} instance: {e}")
        
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
                    # Serialize all versions for this environment as a dict: {version_str: config_dict}
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for version_str, env_config in version_map.items():
                        # Serialize environment config (excluding non-serializable fields)
                        # - cls: Cannot serialize Type objects, code is saved if available
                        # - instance: Runtime state, should be recreated via build() on load
                        # - actions: Will be serialized separately to exclude function field
                        config_dict = env_config.model_dump(mode="json", exclude={"cls", "instance", "actions"})
                        
                        # Store code if available
                        if env_config.cls is not None:
                            if dynamic_manager.is_dynamic_class(env_config.cls):
                                code = dynamic_manager.get_class_source_code(env_config.cls)
                            else:
                                code = self._get_full_module_source(env_config.cls)
                            if code:
                                config_dict["code"] = code
                        
                        # Serialize actions (excluding function which is not serializable, similar to tool's instance)
                        actions_dict = {}
                        if env_config.actions:
                            for action_name, action_config in env_config.actions.items():
                                # Serialize ActionConfig excluding non-serializable fields
                                # - function: Callable, not serializable
                                # - args_schema: Type[BaseModel], will be serialized separately as schema info
                                # - function_calling, text: These are computed properties, exclude them
                                action_dict = action_config.model_dump(
                                    mode="json",
                                    exclude={"function", "args_schema", "function_calling", "text"}
                                )
                                
                                # Serialize args_schema separately if it exists
                                # Use object.__getattribute__ to bypass property and get the field value directly
                                try:
                                    # Check if args_schema field is set in the model
                                    if "args_schema" in action_config.model_fields:
                                        # Get the field value directly, bypassing the property
                                        stored_args_schema = object.__getattribute__(action_config, 'args_schema')
                                        # Only serialize if it's actually a Type[BaseModel], not None or property
                                        if stored_args_schema is not None and isinstance(stored_args_schema, type) and issubclass(stored_args_schema, BaseModel):
                                            try:
                                                args_schema_info = serialize_args_schema(stored_args_schema)
                                                if args_schema_info:
                                                    action_dict["args_schema"] = args_schema_info
                                            except Exception as e:
                                                logger.warning(f"| ⚠️ Failed to serialize args_schema for action {action_name}: {e}")
                                except (AttributeError, TypeError) as e:
                                    # Field not set or not accessible, skip
                                    pass
                                
                                actions_dict[action_name] = action_dict
                        
                        config_dict["actions"] = actions_dict
                        
                        # Use version string as key
                        versions_data[env_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if env_name in self._environment_configs:
                        current_config = self._environment_configs[env_name]
                        if current_config is not None:
                            current_version = current_config.version
                    else:
                        # Fallback: use the latest version from version history
                        if version_map:
                            # Get the latest version by comparing version strings
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
                            # Try to load class from code
                            cls = None
                            config_dict_copy = config_dict.copy()
                            
                            # Case 1: Load from code (for dynamically generated environments)
                            if "code" in config_dict and config_dict["code"]:
                                try:
                                    # Extract class name from config or code
                                    class_name = config_dict.get("config", {}).get("type")
                                    if not class_name:
                                        # Try to extract from code by parsing
                                        class_name = dynamic_manager.extract_class_name_from_code(config_dict["code"])
                                    
                                    if class_name:
                                        # Use context="environment" for automatic import injection
                                        cls = dynamic_manager.load_class(
                                            config_dict["code"], 
                                            class_name, 
                                            Environment,
                                            context="environment"
                                        )
                                        logger.debug(f"| ✅ Loaded environment class {class_name} from code for {env_name}")
                                    else:
                                        logger.warning(f"| ⚠️ Cannot determine class name from code for {env_name}")
                                except Exception as e:
                                    logger.warning(f"| ⚠️ Failed to load class from code for {env_name}: {e}")
                            
                            # Ensure version field is present
                            if "version" not in config_dict_copy:
                                config_dict_copy["version"] = version_str
                            
                            # Remove code from dict before creating EnvironmentConfig
                            config_dict_copy.pop("code", None)
                            
                            # Restore actions from saved data
                            actions = {}
                            if "actions" in config_dict_copy and config_dict_copy["actions"]:
                                for action_name, action_data in config_dict_copy["actions"].items():
                                    if isinstance(action_data, dict):
                                        # Restore args_schema from saved schema info (if present)
                                        args_schema = None
                                        args_schema_info = action_data.get("args_schema")
                                        if args_schema_info:
                                            try:
                                                args_schema = deserialize_args_schema(args_schema_info)
                                            except Exception as e:
                                                logger.warning(f"| ⚠️ Failed to restore args_schema for action {action_name}@{version_str}: {e}")
                                        
                                        # Remove args_schema from dict before creating ActionConfig
                                        action_data_copy = action_data.copy()
                                        action_data_copy.pop("args_schema", None)
                                        
                                        try:
                                            action_config = ActionConfig(**action_data_copy)
                                            
                                            # Set args_schema after creation to bypass validation
                                            if args_schema is not None:
                                                action_config.args_schema = args_schema
                                            
                                            actions[action_name] = action_config
                                        except Exception as e:
                                            logger.warning(f"| ⚠️ Failed to restore action {action_name} for {env_name}@{version_str}: {e}")
                            
                            config_dict_copy["actions"] = actions
                            
                            # Create EnvironmentConfig
                            env_config = EnvironmentConfig(**config_dict_copy)
                            
                            # Set cls after creation
                            if cls is not None:
                                env_config.cls = cls
                            
                            version_configs.append(env_config)
                            
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
                                def environment_factory():
                                    # Filter out None values from config
                                    filtered_config = {}
                                    if latest_config.config:
                                        filtered_config = {k: v for k, v in latest_config.config.items() if v is not None}
                                    if filtered_config:
                                        return latest_config.cls(**filtered_config)
                                    else:
                                        return latest_config.cls()
                                await self.build(latest_config, environment_factory)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load environment {env_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} environments with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load environments from {file_path}: {e}")
                return False
    
    def _get_full_module_source(self, cls: Type[Environment]) -> str:
        """Get the full source code of the module containing the class, including all imports.
        
        This is more reliable than inspect.getsource() which only gets the class definition.
        By reading the entire module file, we preserve all import statements and module-level code,
        ensuring the complete context is available when loading from JSON.
        
        Args:
            cls: The environment class
            
        Returns:
            Full module source code as string, or class source if file reading fails
        """
        try:
            # Get the module object
            module = inspect.getmodule(cls)
            if module is None:
                # Fallback to inspect.getsource if module is not available
                return inspect.getsource(cls)
            
            # Get the file path of the module
            file_path = inspect.getfile(module)
            
            # Read the entire file to preserve all imports and context
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (OSError, TypeError, IOError, AttributeError) as e:
            # Fallback to inspect.getsource if file reading fails
            logger.debug(f"| ⚠️ Failed to read module file for {cls.__name__}, falling back to inspect.getsource: {e}")
            try:
                return inspect.getsource(cls)
            except Exception:
                logger.warning(f"| ⚠️ Failed to get source code for {cls.__name__}")
                return ""
    
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
            version_history.current_version = version
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            def environment_factory():
                # Filter out None values from config
                filtered_config = {}
                if restored_config.config:
                    filtered_config = {k: v for k, v in restored_config.config.items() if v is not None}
                if filtered_config:
                    return restored_config.cls(**filtered_config)
                else:
                    return restored_config.cls()
            await self.build(restored_config, environment_factory)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🔄 Restored environment {env_name} to version {version}")
        return restored_config
    
    async def save_contract(self, env_names: Optional[List[str]] = None):
        """Save the contract for an environment"""
        contract = []
        if env_names is not None:
            for index, env_name in enumerate(env_names):
                env_info = await self.get_info(env_name)
                if env_info is None:
                    logger.warning(f"| ⚠️ Environment {env_name} not found, skipping")
                    continue
                # Get rules from config, or generate from instance if empty
                text = env_info.rules
                if not text and env_info.instance:
                    text = env_info.instance.rules
                elif not text and env_info.cls and env_info.config is not None:
                    # Try to create temporary instance to generate rules
                    try:
                        temp_instance = env_info.cls(**env_info.config)
                        text = temp_instance.rules
                    except Exception as e:
                        logger.debug(f"| ⚠️ Failed to generate rules for {env_name}: {e}")
                        text = ""
                contract.append(f"{index + 1:04d}: {text}")
        else:
            for index, env_name in enumerate(self._environment_configs.keys()):
                env_info = await self.get_info(env_name)
                if env_info is None:
                    logger.warning(f"| ⚠️ Environment {env_name} not found, skipping")
                    continue
                # Get rules from config, or generate from instance if empty
                text = env_info.rules
                if not text and env_info.instance:
                    text = env_info.instance.rules
                elif not text and env_info.cls and env_info.config is not None:
                    # Try to create temporary instance to generate rules
                    try:
                        temp_instance = env_info.cls(**env_info.config)
                        text = temp_instance.rules
                    except Exception as e:
                        logger.debug(f"| ⚠️ Failed to generate rules for {env_name}: {e}")
                        text = ""
                contract.append(f"{index + 1:04d}: {text}")
        contract_text = "\n".join(contract)
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
                        if asyncio.iscoroutinefunction(env_config.instance.cleanup):
                            await env_config.instance.cleanup()
                        else:
                            env_config.instance.cleanup()
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
