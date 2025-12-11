"""Environment Context Manager for managing environment lifecycle and resources."""

import os
import json
import atexit
import asyncio
import importlib
import pkgutil
import inflection
from datetime import datetime
from typing import Any, Dict, Callable, Optional, List, Union, Type
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import assemble_project_path
from src.utils.file_utils import file_lock
from src.environment.types import Environment, EnvironmentConfig, ActionConfig

class EnvironmentContextManager(BaseModel):
    """Global context manager for all environments."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the environments")
    save_path: str = Field(default=None, description="The path to save the environments")
    
    DEFAULT_DISCOVERY_PACKAGES: List[str] = [
        "src.environment",
    ]
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None, 
                 auto_discover: bool = True,
                 **kwargs):
        """Initialize the environment context manager.
        
        Args:
            base_dir: Base directory for storing environment data
            save_path: Path to save environment configurations
            auto_discover: Whether to automatically discover and register environments from packages
        """
        super().__init__(**kwargs)
        
        # Set up paths
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "environments"))
        os.makedirs(self.base_dir, exist_ok=True)
        
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "environments.json")
        
        self._environment_configs: Dict[str, EnvironmentConfig] = {}  # Store environment metadata
        self._cleanup_registered = False
        self.auto_discover = auto_discover
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
            
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
    
    async def initialize(self):
        """Initialize the environment context manager."""
        if self.auto_discover:
            await self.discover()
    
    async def _collect_environment_classes(self, packages: List[str]) -> List[Type[Environment]]:
        """Collect all Environment subclasses from packages.
        
        Args:
            packages: List of package names to scan
            
        Returns:
            List of Environment subclasses
        """
        environment_classes = []
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
                    
                    # Find all Environment subclasses in the module
                    found_classes = []
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (isinstance(obj, type) and 
                            issubclass(obj, Environment) and 
                            obj is not Environment):
                            found_classes.append(obj)
                    return found_classes
                except Exception as e:
                    logger.debug(f"| ⚠️ Failed to import module {module_name}: {e}")
                    return []
            
            # Import all modules concurrently
            import_tasks = [import_module(module_name) for module_name in module_names]
            results = await asyncio.gather(*import_tasks, return_exceptions=True)
            
            # Collect all environment classes
            for result in results:
                if isinstance(result, list):
                    for env_cls in result:
                        if env_cls not in environment_classes:
                            environment_classes.append(env_cls)
        
        return environment_classes
    
    async def _register_environment_class(self, env_cls: Type[Environment], override: bool = False):
        """Register an environment class.
        
        Args:
            env_cls: Environment class to register
            override: Whether to override existing registration
        """
        env_name = None
        try:
            # Get environment config from global config
            env_config_key = inflection.underscore(env_cls.__name__)
            env_config = config.get(f"{env_config_key}_environment", {})
            
            # Try to create temporary instance to get name and description
            try:
                temp_instance = env_cls(**env_config)
                env_name = temp_instance.name
                env_description = temp_instance.description
            except Exception:
                # If instantiation fails, try without config
                try:
                    temp_instance = env_cls()
                    env_name = temp_instance.name
                    env_description = temp_instance.description
                except Exception:
                    # If still fails, try to get from model_fields
                    if hasattr(env_cls, 'model_fields') and 'name' in env_cls.model_fields:
                        field_info = env_cls.model_fields['name']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            # Check if default is a string (not None and not undefined)
                            if isinstance(default_value, str) and default_value:
                                env_name = default_value
                    
                    if hasattr(env_cls, 'model_fields') and 'description' in env_cls.model_fields:
                        field_info = env_cls.model_fields['description']
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                            # Check if default is a string
                            if isinstance(default_value, str):
                                env_description = default_value
                            else:
                                env_description = ''
                    else:
                        env_description = ''
                    
                    if not env_name:
                        logger.warning(f"| ⚠️ Environment class {env_cls.__name__} has no name, skipping")
                        return
            
            if not env_name:
                logger.warning(f"| ⚠️ Environment class {env_cls.__name__} has empty name, skipping")
                return
            
            if env_name in self._environment_configs and not override:
                logger.debug(f"| ⚠️ Environment {env_name} already registered, skipping")
                return
            
            # Register the environment class
            await self.register(env_cls, override=override, **env_config)
            
            logger.debug(f"| 📝 Registered environment: {env_name} ({env_cls.__name__})")
            
        except Exception as e:
            env_name_str = env_name or env_cls.__name__
            logger.warning(f"| ⚠️ Failed to register environment class {env_cls.__name__} (name: {env_name_str}): {e}")
            import traceback
            logger.debug(f"| Traceback: {traceback.format_exc()}")
            raise
    
    async def discover(self, packages: Optional[List[str]] = None):
        """Discover and register all Environment subclasses from specified packages.
        
        Args:
            packages: List of package names to scan. Defaults to DEFAULT_DISCOVERY_PACKAGES.
        """
        packages = packages or self.DEFAULT_DISCOVERY_PACKAGES
        
        logger.info(f"| 🔍 Discovering environments from packages: {packages}")
        
        # Collect all Environment subclasses
        environment_classes = await self._collect_environment_classes(packages)
        
        # Register each environment class concurrently
        registration_tasks = [
            self._register_environment_class(env_cls) for env_cls in environment_classes
        ]
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Count successful registrations
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(environment_classes)} environments")
    
    async def build(self, 
              env_config: EnvironmentConfig,
              env_factory: Callable,
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
            instance = env_factory()
            
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
                env_config.rules = self._generate_rules(env_config)
            
            # Store metadata
            self._environment_configs[env_config.name] = env_config
            
            logger.info(f"| ✅ Environment {env_config.name} created and stored")
            return env_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create environment {env_config.name}: {e}")
            raise
    
    def _generate_rules(self, env_config: EnvironmentConfig) -> str:
        """Generate environment rules from environment config.
        
        Args:
            env_config: Environment configuration
            
        Returns:
            str: Generated environment rules
        """
        metadata = env_config.metadata if env_config.metadata else {}
        has_vision = metadata.get('has_vision', False)
        additional_rules = metadata.get('additional_rules', None)
        
        return self._generate_rules_from_metadata(
            env_config.name,
            env_config.description,
            env_config.actions,
            has_vision,
            additional_rules
        )
    
    def _generate_rules_from_metadata(self,
            env_name: str, 
            description: str, 
            actions: Dict[str, ActionConfig],
            has_vision: bool = False,
            additional_rules: Optional[Dict[str, str]] = None) -> str:
        """Generate environment rules from actions and metadata
        
        Args:
            env_name: Environment name
            description: Environment description
            actions: Dictionary of actions
            has_vision: Whether environment has vision capabilities
            additional_rules: Dictionary with custom rules for 'state', 'vision', 'interaction'
            
        Returns:
            str: Generated environment rules
        """
        # Start building the rules
        rules_parts = [f"<environment_{inflection.underscore(env_name)}>"]
        
        # Add state section
        rules_parts.append("<state>")
        if additional_rules and 'state' in additional_rules:
            rules_parts.append(additional_rules['state'])
        else:
            rules_parts.append(f"The environment state about {env_name}.")
        rules_parts.append("</state>")
        
        # Add vision section
        rules_parts.append("<vision>")
        if additional_rules and 'vision' in additional_rules:
            rules_parts.append(additional_rules['vision'])
        else:
            if has_vision:
                rules_parts.append("The environment vision information.")
            else:
                rules_parts.append("No vision available.")
        rules_parts.append("</vision>")
        
        # Add additional rules if provided (for backward compatibility)
        if additional_rules and 'additional_rules' in additional_rules:
            rules_parts.append("<additional_rules>")
            rules_parts.append(additional_rules['additional_rules'])
            rules_parts.append("</additional_rules>")
        
        # Add interaction section with actions
        rules_parts.append("<interaction>")
        
        if additional_rules and 'interaction' in additional_rules:
            # Use custom interaction rules
            rules_parts.append(additional_rules['interaction'])
        else:
            # Use default interaction rules
            rules_parts.append("Available actions:")
            
            # Sort actions by name for consistent output
            sorted_actions = sorted(actions.items(), key=lambda x: x[0])
            
            for i, (action_name, action_config) in enumerate(sorted_actions, 1):
                rules_parts.append(f"{i}. {action_name}: {action_config.description}")
            
            rules_parts.append("Input format: JSON string with action-specific parameters.")
            rules_parts.append("Example: {\"name\": \"action_name\", \"args\": {\"action-specific parameters\"}}")
        
        rules_parts.append("</interaction>")
        
        # Close the environment tag
        rules_parts.append(f"</environment_{inflection.underscore(env_name)}>")
        
        return "\n".join(rules_parts)
    
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
                        from src.environment.types import ActionConfig
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
            
            # Create EnvironmentConfig
            if env_instance is not None:
                # Registering an instance
                env_config = EnvironmentConfig(
                    name=env_name,
                    description=env_description,
                    rules="",  # Will be generated when needed
                    actions=actions,
                    cls=env_cls,
                    config={},
                    instance=env_instance,
                    metadata=metadata
                )
            else:
                # Registering a class - store config for lazy loading
                env_config = EnvironmentConfig(
                    name=env_name,
                    description=env_description,
                    rules="",  # Will be generated when needed
                    actions=actions,
                    cls=env_cls,
                    config=kwargs,
                    instance=None,  # Will be created on initialize
                    metadata=metadata
                )
            
            # Generate rules if we have enough information
            if env_config.instance or (env_config.actions and metadata):
                env_config.rules = self._generate_rules_from_metadata(
                    env_config.name,
                    env_config.description,
                    env_config.actions,
                    metadata.get('has_vision', False) if metadata else False,
                    metadata.get('additional_rules', None) if metadata else None
                )
            
            # Store environment config
            self._environment_configs[env_name] = env_config
            
            # Register version record to version manager
            await version_manager.register_version("environment", env_name, env_config.version)
            
            logger.debug(f"| 📝 Registered environment config: {env_name} v{env_config.version}")
            
            return env_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register environment: {e}")
            raise
        
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
        
    def list(self) -> List[str]:
        """Get list of registered environments
        
        Returns:
            List[str]: List of registered environment names
        """
        return [name for name in self._environment_configs.keys()]
    
    def get_info(self, env_name: str) -> Optional[EnvironmentConfig]:
        """Get environment configuration by name
        
        Args:
            env_name: Environment name
            
        Returns:
            EnvironmentConfig: Environment configuration or None if not found
        """
        return self._environment_configs.get(env_name)
    
    def get(self, env_name: str) -> Optional[Environment]:
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
        
        # Determine new version
        if new_version is None:
            # Auto-increment version
            try:
                version_parts = original_config.version.split(".")
                if len(version_parts) >= 3:
                    major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                    patch += 1
                    new_version = f"{major}.{minor}.{patch}"
                else:
                    new_version = f"{original_config.version}.1"
            except:
                new_version = f"{original_config.version}.updated"
        
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
                    from src.environment.types import ActionConfig
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
                metadata=metadata
            )
        else:
            updated_config = EnvironmentConfig(
                name=env_name,
                description=description or new_description,
                rules="",  # Will be generated below
                version=new_version,
                actions=actions,
                cls=env_cls,
                config=kwargs,
                instance=None,
                metadata=metadata
            )
        
        # Generate rules
        updated_config.rules = self._generate_rules_from_metadata(
            updated_config.name,
            updated_config.description,
            updated_config.actions,
            metadata.get('has_vision', False) if metadata else False,
            metadata.get('additional_rules', None) if metadata else None
        )
        
        # Update the environment config (replaces current version)
        self._environment_configs[env_name] = updated_config
        
        # Register new version record to version manager
        await version_manager.register_version(
            "environment", 
            env_name, 
            new_version,
            description=description or f"Updated from {original_config.version}"
        )
        
        logger.info(f"| 🔄 Updated environment {env_name} from v{original_config.version} to v{new_version}")
        return updated_config
    
    async def copy(self, env_name: str, new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, **override_config) -> EnvironmentConfig:
        """Copy an existing environment configuration
        
        Args:
            env_name: Name of the environment to copy
            new_name: New name for the copied environment. If None, uses original name.
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
        
        if new_name in self._environment_configs:
            raise ValueError(f"Environment '{new_name}' already exists. Use a different name.")
        
        # Determine new version
        if new_version is None:
            if new_name == env_name:
                # If copying with same name, increment version
                try:
                    version_parts = original_config.version.split(".")
                    if len(version_parts) >= 3:
                        major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                        patch += 1
                        new_version = f"{major}.{minor}.{patch}"
                    else:
                        new_version = f"{original_config.version}.1"
                except:
                    new_version = f"{original_config.version}.copy"
            else:
                # If copying with different name, keep original version
                new_version = original_config.version
        
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
        
        # Register version record to version manager
        await version_manager.register_version(
            "environment", 
            new_name, 
            new_config.version,
            description=f"Copied from {env_name}@{original_config.version}"
        )
        
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
        
        logger.info(f"| 🗑️ Unregistered environment {env_name}")
        return True
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all environment configurations to JSON
        
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
                    "environment_count": len(self._environment_configs),
                },
                "environments": {}
            }
            
            for env_name, env_config in self._environment_configs.items():
                try:
                    # Serialize environment config (excluding non-serializable cls and instance)
                    config_dict = env_config.model_dump(mode="json", exclude={"cls", "instance"})
                    
                    # Store class path if available
                    if env_config.cls is not None:
                        config_dict["cls_path"] = f"{env_config.cls.__module__}.{env_config.cls.__name__}"
                    
                    # Serialize actions (excluding function)
                    if "actions" in config_dict and config_dict["actions"]:
                        for action_name, action_config in config_dict["actions"].items():
                            if isinstance(action_config, dict):
                                action_config.pop("function", None)
                    
                    save_data["environments"][env_name] = config_dict
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize environment {env_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._environment_configs)} environments to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load environment configurations from JSON
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically initialize environments after loading
            
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
                        # Try to load class if cls_path is available
                        cls = None
                        if "cls_path" in env_data:
                            module_path, class_name = env_data["cls_path"].rsplit(".", 1)
                            try:
                                module = importlib.import_module(module_path)
                                cls = getattr(module, class_name)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load class {env_data['cls_path']} for {env_name}: {e}")
                        
                        # Remove cls_path from dict before creating EnvironmentConfig
                        env_data_copy = env_data.copy()
                        env_data_copy.pop("cls_path", None)
                        
                        # Create EnvironmentConfig
                        env_config = EnvironmentConfig(**env_data_copy)
                        if cls is not None:
                            env_config.cls = cls
                        
                        # Register environment config
                        self._environment_configs[env_name] = env_config
                        
                        # Register version to version manager
                        await version_manager.register_version("environment", env_name, env_config.version)
                        
                        # Initialize if requested
                        if auto_initialize and env_config.cls is not None:
                            def environment_factory():
                                if env_config.config:
                                    return env_config.cls(**env_config.config)
                                else:
                                    return env_config.cls()
                            await self.build(env_config, environment_factory)
                        
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load environment {env_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} environments from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load environments from {file_path}: {e}")
                return False
    
    def cleanup(self):
        """Cleanup all environment instances and resources."""
        try:
            # Clear instances and configs
            self._environment_configs.clear()
            logger.info("| 🧹 Environment context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during environment context manager cleanup: {e}")
