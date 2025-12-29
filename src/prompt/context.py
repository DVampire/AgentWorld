"""Prompt Context Manager for managing prompt lifecycle and resources with version management."""

import atexit
import os
import json
import inspect
from datetime import datetime
from typing import Any, Dict, Optional, List, Union, Type, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import assemble_project_path, gather_with_concurrency
from src.utils.file_utils import file_lock
from src.prompt.types import PromptConfig, Prompt
from src.registry import PROMPT
from src.message.types import Message
from src.optimizer.types import Variable
from src.dynamic import dynamic_manager

class PromptContextManager(BaseModel):
    """Global context manager for all prompts with version management."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the prompts")
    save_path: str = Field(default=None, description="The path to save the prompts")
    contract_path: str = Field(default=None, description="The path to save the prompt contract")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 contract_path: Optional[str] = None,
                 **kwargs):
        """Initialize the prompt context manager.
        
        Args:
            base_dir: Base directory for storing prompt data
            save_path: Path to save prompt configurations
            contract_path: Path to save prompt contract
        """
        super().__init__(**kwargs)
        
        # Set up paths
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "prompt"))
        os.makedirs(self.base_dir, exist_ok=True)
        
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "prompt.json")
        
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        
        logger.info(f"| 📁 Prompt context manager base directory: {self.base_dir} and save path: {self.save_path}")
        logger.info(f"| 📁 Prompt context manager contract path: {self.contract_path}")
        
        self._prompt_configs: Dict[str, PromptConfig] = {}  # Current active configs (latest version)
        # Prompt version history, e.g., {"prompt_name": {"1.0.0": PromptConfig, "1.0.1": PromptConfig}}
        self._prompt_history_versions: Dict[str, Dict[str, PromptConfig]] = {}
        
        self._cleanup_registered = False
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
    
    async def initialize(self, prompt_names: Optional[List[str]] = None):
        """Initialize the prompt context manager."""
        # Load prompts from PROMPT registry
        prompt_configs = {}
        registry_prompt_configs: Dict[str, PromptConfig] = await self._load_from_registry()
        prompt_configs.update(registry_prompt_configs)
        
        # Load prompts from JSON file
        code_prompt_configs: Dict[str, PromptConfig] = await self._load_from_code()
        
        # Merge code configs with registry configs, only override if code version is strictly greater
        for prompt_name, code_config in code_prompt_configs.items():
            if prompt_name in prompt_configs:
                registry_config = prompt_configs[prompt_name]
                # Compare versions: only override if code version is strictly greater
                if version_manager.compare_versions(code_config.version, registry_config.version) > 0:
                    logger.info(f"| 🔄 Overriding prompt {prompt_name} from registry (v{registry_config.version}) with code version (v{code_config.version})")
                    prompt_configs[prompt_name] = code_config
                else:
                    logger.info(f"| 📌 Keeping prompt {prompt_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater")
                    # If versions are equal, update the history with registry config (which has real class, not dynamic)
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        # Replace the code config in history with registry config to preserve real class reference
                        if prompt_name in self._prompt_history_versions:
                            self._prompt_history_versions[prompt_name][registry_config.version] = registry_config
            else:
                # New prompt from code, add it
                prompt_configs[prompt_name] = code_config
        
        # Filter prompts by names if provided
        # prompt_names are base names like "tool_calling", need to match both system_prompt and agent_message_prompt
        if prompt_names is not None:
            filtered_configs = {}
            for base_name in prompt_names:
                # Match system_prompt
                system_prompt_name = f"{base_name}_system_prompt"
                if system_prompt_name in prompt_configs:
                    filtered_configs[system_prompt_name] = prompt_configs[system_prompt_name]
                # Match agent_message_prompt
                agent_message_prompt_name = f"{base_name}_agent_message_prompt"
                if agent_message_prompt_name in prompt_configs:
                    filtered_configs[agent_message_prompt_name] = prompt_configs[agent_message_prompt_name]
            prompt_configs = filtered_configs
        
        # Store all prompts
        for prompt_name, prompt_config in prompt_configs.items():
            self._prompt_configs[prompt_name] = prompt_config
        
        # Build all prompts concurrently with a concurrency limit
        prompt_names = list(prompt_configs.keys())
        tasks = [
            self.build(prompt_configs[name]) for name in prompt_names
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for prompt_name, result in zip(prompt_names, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize prompt {prompt_name}: {result}")
                continue
            self._prompt_configs[prompt_name] = result
            logger.info(f"| 🔧 Prompt {prompt_name} initialized")
        
        # Save prompt configs to json file
        await self.save_to_json()
        # Save contract to file
        await self.save_contract(prompt_names=prompt_names)
        
        logger.info(f"| ✅ Prompts initialization completed")
    
    async def _load_from_registry(self):
        """Load prompts from PROMPT registry."""
        prompt_configs: Dict[str, PromptConfig] = {}
        
        async def register_prompt_class(prompt_cls: Type[Prompt]):
            """Register a prompt class.
            
            Args:
                prompt_cls: Prompt class to register
            """
            try:
                # Create prompt instance to get properties
                prompt_instance = prompt_cls()
                
                # Initialize prompt if it has an initialize method
                if hasattr(prompt_instance, "initialize"):
                    await prompt_instance.initialize()
                
                # Get prompt_config from instance
                prompt_config_dict = prompt_instance.prompt_config
                if prompt_config_dict is None:
                    raise ValueError(f"Prompt class {prompt_cls.__name__} must have 'prompt_config' field")
                
                # Use name from prompt_config_dict if available, otherwise use prompt_instance.name
                prompt_type = prompt_config_dict.get('type', prompt_instance.type)
                prompt_name = prompt_config_dict.get('name', prompt_instance.name)
                prompt_description = prompt_config_dict.get('description', prompt_instance.description)
                
                # Get or generate version from version_manager
                prompt_version = await version_manager.get_version("prompt", prompt_name)
                
                prompt_template = prompt_config_dict.get('template', '')
                prompt_variables = prompt_config_dict.get('variables', [])
                prompt_metadata = prompt_config_dict.get('metadata', {})
                
                # Get source code for the prompt class
                prompt_code = dynamic_manager.get_full_module_source(prompt_cls)
                
                # Create PromptConfig
                prompt_config = PromptConfig(
                    name=prompt_name,
                    type=prompt_type,
                    description=prompt_description,
                    version=prompt_version,
                    template=prompt_template,
                    variables=prompt_variables,
                    cls=prompt_cls,
                    instance=prompt_instance,  # Instance will be built when needed
                    config={},
                    metadata=prompt_metadata,
                    code=prompt_code,
                )
                
                # Store prompt config
                prompt_configs[prompt_name] = prompt_config
                
                # Store in version history (by version string)
                if prompt_name not in self._prompt_history_versions:
                    self._prompt_history_versions[prompt_name] = {}
                self._prompt_history_versions[prompt_name][prompt_version] = prompt_config
                
                logger.info(f"| 📝 Registered prompt: {prompt_name} ({prompt_cls.__name__})")
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register prompt class {prompt_cls.__name__}: {e}")
                raise
        
        # Get all registered prompt classes from PROMPT registry
        prompt_classes = list(PROMPT._module_dict.values())
        
        logger.info(f"| 🔍 Discovering {len(prompt_classes)} prompts from PROMPT registry")
        
        # Register each prompt class concurrently with a concurrency limit
        tasks = [
            register_prompt_class(prompt_cls) for prompt_cls in prompt_classes
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(prompt_classes)} prompts from PROMPT registry")
        
        return prompt_configs
    
    async def _load_from_code(self):
        """Load prompts from JSON file.
        
        JSON file content example:
        {
            "metadata": {
                "saved_at": str,  # "YYYY-MM-DD HH:MM:SS"
                "num_prompts": int,  # total prompt count
                "num_versions": int  # total version count
            },
            "prompts": {
                "prompt_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": {
                            "name": str,
                            "type": str,
                            "description": str,
                            "version": str,
                            "template": str,
                            "variables": list,
                            "metadata": dict
                        },
                        ...
                    }
                }
            }
        }
        """
        prompt_configs: Dict[str, PromptConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Prompt config file not found at {self.save_path}, skipping code-based loading")
            return prompt_configs
        
        # Load all prompt configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse prompt config JSON from {self.save_path}: {e}")
            return prompt_configs
        
        metadata = load_data.get("metadata", {})
        prompts_data = load_data.get("prompts", {})
        
        async def register_prompt_class(prompt_name: str, prompt_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, PromptConfig], Optional[PromptConfig]]]:
            """Load all versions for a single prompt from JSON."""
            try:
                current_version = prompt_data.get("current_version", "1.0.0")
                versions = prompt_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Prompt {prompt_name} has no versions")
                    return None
                
                version_map: Dict[str, PromptConfig] = {}
                current_config: Optional[PromptConfig] = None  # Active config for current_version
                
                for _, version_data in versions.items():
                    # Create PromptConfig using model_validate to handle cls and code
                    prompt_config = PromptConfig.model_validate(version_data)
                    version = prompt_config.version
                    version_map[version] = prompt_config
                    
                    if version == current_version:
                        current_config = prompt_config
                
                return prompt_name, version_map, current_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load prompt {prompt_name} from code JSON: {e}")
                return None
        
        # Launch loading of each prompt concurrently with a concurrency limit
        tasks = [
            register_prompt_class(prompt_name, prompt_data) for prompt_name, prompt_data in prompts_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            prompt_name, version_map, current_config = result
            if not version_map:
                continue
            
            # Store all versions in history (mapped by version string)
            self._prompt_history_versions[prompt_name] = version_map
            # Active config: the one corresponding to current_version
            if current_config is not None:
                prompt_configs[prompt_name] = current_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Prompt {prompt_name} current_version not found, using last available version")
                prompt_configs[prompt_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for prompt_config in version_map.values():
                await version_manager.register_version("prompt", prompt_name, prompt_config.version)
        
        logger.info(f"| 📂 Loaded {len(prompt_configs)} prompts from {self.save_path}")
        return prompt_configs
    
    async def register(self, prompt: Union[Prompt, Dict[str, Any]], *, override: bool = False, **kwargs: Any) -> PromptConfig:
        """Register a prompt or prompt template dictionary.
        
        Args:
            prompt: Prompt instance or template dictionary
            override: Whether to override existing registration
            **kwargs: Configuration for prompt initialization
            
        Returns:
            PromptConfig: Prompt configuration
        """
        try:
            if isinstance(prompt, Prompt):
                prompt_name = prompt.name
                prompt_type = prompt.type
                prompt_description = prompt.description
                # Get template and variables from prompt_config
                prompt_config_dict = prompt.prompt_config
                if prompt_config_dict is None:
                    raise ValueError(f"Prompt instance must have 'prompt_config' field")
                prompt_template = prompt_config_dict.get('template', '')
                prompt_variables = prompt_config_dict.get('variables', [])
                prompt_cls = type(prompt)
                prompt_instance = prompt
            elif isinstance(prompt, dict):
                prompt_name = prompt.get('name')
                prompt_type = prompt.get('type', 'prompt')
                prompt_description = prompt.get('description', '')
                prompt_template = prompt.get('template', '')
                prompt_variables = prompt.get('variables', [])
                prompt_cls = None
                prompt_instance = None
            else:
                raise TypeError(f"Expected Prompt instance or dict, got {type(prompt)!r}")
            
            if not prompt_name:
                raise ValueError("Prompt.name cannot be empty.")
            
            if prompt_name in self._prompt_configs and not override:
                raise ValueError(f"Prompt '{prompt_name}' already registered. Use override=True to replace it.")
            
            # Get or generate version from version_manager
            version = await version_manager.get_version("prompt", prompt_name)
            
            # Get source code if cls is available
            prompt_code = None
            if prompt_cls is not None:
                prompt_code = dynamic_manager.get_full_module_source(prompt_cls)
            
            # Create PromptConfig
            prompt_config = PromptConfig(
                name=prompt_name,
                type=prompt_type,
                description=prompt_description,
                version=version,
                template=prompt_template,
                variables=prompt_variables,
                cls=prompt_cls,
                instance=None,  # Instance will be built when needed
                config=kwargs if kwargs else {},
                metadata=prompt.get('metadata', {}) if isinstance(prompt, dict) else {},
                code=prompt_code
            )
            
            # Store metadata
            self._prompt_configs[prompt_name] = prompt_config
            
            # Store in version history (by version string)
            if prompt_name not in self._prompt_history_versions:
                self._prompt_history_versions[prompt_name] = {}
            self._prompt_history_versions[prompt_name][prompt_config.version] = prompt_config
            
            # Register version record to version manager
            await version_manager.register_version("prompt", prompt_name, prompt_config.version)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.debug(f"| 📝 Registered prompt: {prompt_name} v{prompt_config.version}")
            return prompt_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register prompt: {e}")
            raise
    
    async def build(self, prompt_config: PromptConfig) -> PromptConfig:
        """Create a prompt instance and store it.
        
        Args:
            prompt_config: Prompt configuration
            
        Returns:
            PromptConfig: Prompt configuration with instance
        """
        if prompt_config.name in self._prompt_configs:
            existing_config = self._prompt_configs[prompt_config.name]
            if existing_config.instance is not None:
                return existing_config
        
        # Create new prompt instance
        try:
            # cls should already be loaded (either from registry or from code in _load_from_code)
            if prompt_config.cls is None:
                raise ValueError(f"Cannot create prompt {prompt_config.name}: no class provided. Class should be loaded during initialization.")
            
            if prompt_config.instance is None:
                # Instantiate prompt instance
                prompt_instance = prompt_config.cls(**prompt_config.config) if prompt_config.config else prompt_config.cls()
                
                # Initialize prompt if it has an initialize method
                if hasattr(prompt_instance, "initialize"):
                    await prompt_instance.initialize()
            else:
                prompt_instance = prompt_config.instance
            
            prompt_config.instance = prompt_instance
            
            # Store prompt metadata
            self._prompt_configs[prompt_config.name] = prompt_config
            
            logger.info(f"| 🔧 Prompt {prompt_config.name} created and stored")
            
            return prompt_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create prompt {prompt_config.name}: {e}")
            raise
    
    async def update(self, prompt_name: str, prompt: Union[Prompt, Dict[str, Any]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> PromptConfig:
        """Update an existing prompt with new configuration and create a new version
        
        Args:
            prompt_name: Name of the prompt to update
            prompt: New prompt instance or template dictionary with updated content
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for prompt initialization
            
        Returns:
            PromptConfig: Updated prompt configuration
        """
        original_config = self._prompt_configs.get(prompt_name)
        if original_config is None:
            raise ValueError(f"Prompt {prompt_name} not found. Use register() to register a new prompt.")
        
        # Get new prompt info
        if isinstance(prompt, Prompt):
            new_description = prompt.description
            # Get template and variables from prompt_config
            prompt_config_dict = prompt.prompt_config
            if prompt_config_dict is None:
                raise ValueError(f"Prompt instance must have 'prompt_config' field")
            prompt_template = prompt_config_dict.get('template', original_config.template)
            prompt_variables = prompt_config_dict.get('variables', original_config.variables)
            prompt_cls = type(prompt)
            prompt_instance = prompt
        elif isinstance(prompt, dict):
            new_description = prompt.get('description', original_config.description)
            prompt_template = prompt.get('template', original_config.template)
            prompt_variables = prompt.get('variables', original_config.variables)
            prompt_cls = None
            prompt_instance = None
        else:
            raise TypeError(f"Expected Prompt instance or dict, got {type(prompt)!r}")
        
        # Determine new version from version_manager
        if new_version is None:
            # Get current version from version_manager and generate next patch version
            new_version = await version_manager.generate_next_version("prompt", prompt_name, "patch")
        
        # Get source code if cls is available
        prompt_code = None
        if prompt_cls is not None:
            prompt_code = dynamic_manager.get_full_module_source(prompt_cls)
        elif original_config.code is not None:
            prompt_code = original_config.code
        
        # Create updated config
        if prompt_instance is not None:
            updated_config = PromptConfig(
                name=prompt_name,
                type=original_config.type,
                description=description or new_description,
                version=new_version,
                template=prompt_template,
                variables=prompt_variables,
                cls=prompt_cls,
                config={},
                instance=None,  # Instance will be built when needed
                metadata=prompt.get('metadata', {}) if isinstance(prompt, dict) else original_config.metadata,
                code=prompt_code
            )
        else:
            updated_config = PromptConfig(
                name=prompt_name,
                type=original_config.type,
                description=description or new_description,
                version=new_version,
                template=prompt_template,
                variables=prompt_variables,
                cls=prompt_cls,
                config=kwargs,
                instance=None,
                metadata=prompt.get('metadata', {}) if isinstance(prompt, dict) else original_config.metadata,
                code=prompt_code
            )
        
        # Update the prompt config (replaces current version)
        self._prompt_configs[prompt_name] = updated_config
        
        # Store in version history
        if prompt_name not in self._prompt_history_versions:
            self._prompt_history_versions[prompt_name] = {}
        self._prompt_history_versions[prompt_name][updated_config.version] = updated_config
        
        # Register version record to version manager
        await version_manager.register_version(
            "prompt", 
            prompt_name, 
            new_version,
            description=description or f"Updated from {original_config.version}"
        )
        
        # Persist to JSON
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 📝 Updated prompt: {prompt_name} v{updated_config.version}")
        return updated_config
    
    async def copy(self, prompt_name: str, new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, **override_config) -> PromptConfig:
        """Copy an existing prompt configuration
        
        Args:
            prompt_name: Name of the prompt to copy
            new_name: New name for the copied prompt. If None, uses original name.
            new_version: New version for the copied prompt. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            PromptConfig: New prompt configuration
        """
        original_config = self._prompt_configs.get(prompt_name)
        if original_config is None:
            raise ValueError(f"Prompt {prompt_name} not found")
        
        # Determine new name
        if new_name is None:
            new_name = prompt_name
        
        # Determine new version from version_manager
        if new_version is None:
            if new_name == prompt_name:
                # If copying with same name, get next version from version_manager
                new_version = await version_manager.generate_next_version("prompt", new_name, "patch")
            else:
                # If copying with different name, get or generate version for new name
                new_version = await version_manager.get_or_generate_version("prompt", new_name)
        
        # Create copy of config using model_dump and model_validate to handle cls and code
        new_config_dict = original_config.model_dump()
        new_config_dict["name"] = new_name
        new_config_dict["version"] = new_version
        
        # Apply overrides
        if override_config:
            if "description" in override_config:
                new_config_dict["description"] = override_config.pop("description")
            if "metadata" in override_config:
                new_config_dict["metadata"].update(override_config.pop("metadata"))
            # Merge remaining overrides into config
            new_config_dict["config"].update(override_config)
        
        # Clear instance (will be created on demand)
        new_config_dict["instance"] = None
        
        # Use model_validate to ensure cls is loaded from code if needed
        new_config = PromptConfig.model_validate(new_config_dict)
        
        # Register new prompt
        self._prompt_configs[new_name] = new_config
        
        # Store in version history
        if new_name not in self._prompt_history_versions:
            self._prompt_history_versions[new_name] = {}
        self._prompt_history_versions[new_name][new_version] = new_config
        
        # Register version record to version manager
        await version_manager.register_version(
            "prompt", 
            new_name, 
            new_version,
            description=f"Copied from {prompt_name}@{original_config.version}"
        )
        
        # Persist to JSON
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 📋 Copied prompt {prompt_name}@{original_config.version} to {new_name}@{new_version}")
        return new_config
    
    async def unregister(self, prompt_name: str) -> bool:
        """Unregister a prompt
        
        Args:
            prompt_name: Name of the prompt to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if prompt_name not in self._prompt_configs:
            logger.warning(f"| ⚠️ Prompt {prompt_name} not found")
            return False
        
        prompt_config = self._prompt_configs[prompt_name]
        
        # Remove from configs
        del self._prompt_configs[prompt_name]
        
        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered prompt {prompt_name}@{prompt_config.version}")
        return True
    
    async def restore(self, prompt_name: str, version: str, auto_initialize: bool = True) -> Optional[PromptConfig]:
        """Restore a specific version of a prompt from history
        
        Args:
            prompt_name: Name of the prompt
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored prompt
            
        Returns:
            PromptConfig of the restored version, or None if not found
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if prompt_name in self._prompt_history_versions:
            version_config = self._prompt_history_versions[prompt_name].get(version)
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for prompt {prompt_name}")
            return None
        
        # Create a copy to avoid modifying the history
        # Use model_validate to ensure cls is loaded from code if needed
        restored_config = PromptConfig.model_validate(version_config.model_dump())
        
        # Set as current active config
        self._prompt_configs[prompt_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("prompt", prompt_name)
        if version_history:
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("prompt", prompt_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("prompt", prompt_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🔄 Restored prompt {prompt_name} to version {version}")
        return restored_config
    
    async def get(self, name: str) -> Optional[Prompt]:
        """Get prompt instance by name
        
        Args:
            name: Name of the prompt
            
        Returns:
            Prompt: Prompt instance or None if not found
        """
        prompt_config = self._prompt_configs.get(name)
        if prompt_config is None:
            return None
        return prompt_config.instance if prompt_config.instance is not None else None
    
    async def get_info(self, name: str) -> Optional[PromptConfig]:
        """Get a prompt configuration by name
        
        Args:
            name: Name of the prompt
        """
        return self._prompt_configs.get(name)
    
    async def list(self) -> List[str]:
        """Get list of registered prompts
        
        Returns:
            List[str]: List of prompt names
        """
        return [name for name in self._prompt_configs.keys()]
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all prompt configurations with version history to JSON.
        
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
            
            # Prepare save data - save all versions for each prompt
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_prompts": len(self._prompt_configs),
                    "num_versions": sum(len(versions) for versions in self._prompt_history_versions.values()),
                },
                "prompts": {}
            }
            
            for prompt_name, version_map in self._prompt_history_versions.items():
                try:
                    # Serialize all versions for this prompt as a dict: {version_str: config_dict}
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for version_str, prompt_config in version_map.items():
                        # Serialize prompt config using custom model_dump (which handles cls and code)
                        config_dict = prompt_config.model_dump()
                        
                        # Use version string as key
                        versions_data[prompt_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    current_version = None
                    if prompt_name in self._prompt_configs:
                        current_config = self._prompt_configs[prompt_name]
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
                    
                    save_data["prompts"][prompt_name] = {
                        "versions": versions_data,
                        "current_version": current_version
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize prompt {prompt_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._prompt_configs)} prompts with version history to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load prompt configurations with version history from JSON.
        
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
                logger.warning(f"| ⚠️ Prompt file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                prompts_data = load_data.get("prompts", {})
                loaded_count = 0
                
                for prompt_name, prompt_data in prompts_data.items():
                    try:
                        # Expected format: multiple versions stored as a dict {version_str: config_dict}
                        versions_data = prompt_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Prompt {prompt_name} has invalid format for 'versions' (expected dict), skipping")
                            continue
                        
                        current_version_str = prompt_data.get("current_version")
                        
                        # Load all versions
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for version_str, config_dict in versions_data.items():
                            # Ensure version field is present
                            if "version" not in config_dict:
                                config_dict["version"] = version_str
                            
                            # Create PromptConfig using model_validate to handle cls and code
                            prompt_config = PromptConfig.model_validate(config_dict)
                            
                            version_configs.append(prompt_config)
                            
                            # Track latest version
                            if latest_config is None or (
                                current_version_str and prompt_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    version_manager.compare_versions(prompt_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = prompt_config
                                latest_version = prompt_config.version
                        
                        # Store all versions in history (dict-based)
                        self._prompt_history_versions[prompt_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }
                        
                        # Only set latest version as active
                        if latest_config:
                            self._prompt_configs[prompt_name] = latest_config
                            
                            # Register all versions to version manager (only version records)
                            for prompt_config in version_configs:
                                await version_manager.register_version("prompt", prompt_name, prompt_config.version)
                            
                            # Create instance if requested (instance is not saved in JSON, must be created via build)
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load prompt {prompt_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} prompts with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load prompts from {file_path}: {e}")
                return False
    
    async def get_system_message(self, 
                          prompt_name: Optional[str] = None,
                          modules: Dict[str, Any] = None, 
                          reload: bool = False, 
                          **kwargs):
        """Get a system message using SystemPrompt.
        
        Args:
            prompt_name: Name of the prompt (e.g., "tool_calling"). 
                        If None, will try to infer from kwargs or use default.
            modules: Modules to render in the template
            reload: Whether to reload the prompt
            **kwargs: Additional arguments (may include prompt_name for backward compatibility)
        """
        prompt_name = f"{prompt_name}_system_prompt"
        prompt_instance = await self.get(prompt_name)
        return await prompt_instance.get_message(modules, reload, **kwargs)
    
    async def get_agent_message(self, 
                         prompt_name: Optional[str] = None,
                         modules: Dict[str, Any] = None, 
                         reload: bool = True, 
                         **kwargs):
        """Get an agent message using AgentMessagePrompt.
        
        Args:
            prompt_name: Name of the prompt (e.g., "tool_calling").
                        If None, will try to infer from kwargs or use default.
            modules: Modules to render in the template
            reload: Whether to reload the prompt
            **kwargs: Additional arguments (may include prompt_name for backward compatibility)
        """
        prompt_name = f"{prompt_name}_agent_message_prompt"
        prompt_instance = await self.get(prompt_name)
        return await prompt_instance.get_message(modules, reload, **kwargs)
    
    async def get_messages(
            self,
            prompt_name: Optional[str] = None,
            system_modules: Dict[str, Any] = None,
            agent_modules: Dict[str, Any] = None,
            **kwargs
        ) -> List[Message]:
        """Get a system and agent message using SystemPrompt and AgentMessagePrompt.
        
        Args:
            prompt_name (str): Name of the prompt (e.g., "tool_calling").
            system_modules(Dict[str, Any]): Modules to render in the system prompt
            agent_modules(Dict[str, Any]): Modules to render in the agent message prompt
            **kwargs(Any): Additional arguments (may include prompt_name for backward compatibility)
        """
        system_message = await self.get_system_message(prompt_name, 
                                                       system_modules, 
                                                       reload=False, 
                                                       **kwargs)
        agent_message = await self.get_agent_message(prompt_name, 
                                                     agent_modules, 
                                                     reload=True,
                                                     **kwargs)
        return [system_message, agent_message]
    
    async def get_variables(self, prompt_name: Optional[str] = None) -> List[Variable]:
        """Get top-level variables from system and agent prompts.
        
        Args:
            prompt_name (str): Name of the prompt (e.g., "tool_calling").
            
        Returns:
            List[Variable]: List of top-level variables (system prompt and agent message prompt)
        """
        system_prompt_name = f"{prompt_name}_system_prompt"
        agent_prompt_name = f"{prompt_name}_agent_message_prompt"
        system_prompt_instance = await self.get(system_prompt_name)
        agent_prompt_instance = await self.get(agent_prompt_name)
        variables: List[Variable] = []
        
        if system_prompt_instance is not None:
            variables.append(await system_prompt_instance.get_variable())
        if agent_prompt_instance is not None:
            variables.append(await agent_prompt_instance.get_variable())
        
        return variables
    
    async def get_trainable_variables(self, prompt_name: Optional[str] = None) -> List[Variable]:
        """Get top-level variables from system and agent prompts with filtered trainable sub-variables.
        
        Args:
            prompt_name (str): Name of the prompt (e.g., "tool_calling").
            
        Returns:
            List[Variable]: List of top-level variables (system prompt and agent message prompt)
                          with sub-variables filtered to only include require_grad=True
        """
        system_prompt_name = f"{prompt_name}_system_prompt"
        agent_prompt_name = f"{prompt_name}_agent_message_prompt"
        system_prompt_instance = await self.get(system_prompt_name)
        agent_prompt_instance = await self.get(agent_prompt_name)
        variables: List[Variable] = []
        
        if system_prompt_instance is not None:
            system_var = await system_prompt_instance.get_variable()
            # Filter to only include trainable sub-variables
            filtered_system_var = system_var.filter_trainable_sub_variables()
            variables.append(filtered_system_var)
        
        if agent_prompt_instance is not None:
            agent_var = await agent_prompt_instance.get_variable()
            # Filter to only include trainable sub-variables
            filtered_agent_var = agent_var.filter_trainable_sub_variables()
            variables.append(filtered_agent_var)
        
        return variables
    
    async def save_contract(self, prompt_names: Optional[List[str]] = None):
        """Save the contract for prompts
        
        Args:
            prompt_names: Optional list of prompt names to include in contract.
                         If None, includes all registered prompts.
        """
        contract = []
        if prompt_names is not None:
            # Filter to include both system and agent message prompts
            filtered_names = []
            for base_name in prompt_names:
                system_prompt_name = f"{base_name}_system_prompt"
                agent_prompt_name = f"{base_name}_agent_message_prompt"
                if system_prompt_name in self._prompt_configs:
                    filtered_names.append(system_prompt_name)
                if agent_prompt_name in self._prompt_configs:
                    filtered_names.append(agent_prompt_name)
            
            for index, prompt_name in enumerate(filtered_names):
                prompt_info = await self.get_info(prompt_name)
                if prompt_info:
                    # Format: name, type, description, template
                    contract_text = f"Prompt: {prompt_info.name}\nType: {prompt_info.type}\nDescription: {prompt_info.description}\nTemplate:\n{prompt_info.template}\n"
                    contract.append(f"{index + 1:04d}\n{contract_text}\n")
        else:
            for index, prompt_name in enumerate(self._prompt_configs.keys()):
                prompt_info = await self.get_info(prompt_name)
                if prompt_info:
                    # Format: name, type, description, template
                    contract_text = f"Prompt: {prompt_info.name}\nType: {prompt_info.type}\nDescription: {prompt_info.description}\nTemplate:\n{prompt_info.template}\n"
                    contract.append(f"{index + 1:04d}\n{contract_text}\n")
        
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} prompts contract to {self.contract_path}")
    
    async def load_contract(self) -> str:
        """Load the contract for prompts
        
        Returns:
            str: Contract text content
        """
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
    async def cleanup(self):
        """Cleanup all prompt instances and resources."""
        try:
            # Clear all prompt configs and version history
            self._prompt_configs.clear()
            self._prompt_history_versions.clear()
            logger.info("| 🧹 Prompt context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during prompt context manager cleanup: {e}")