"""Prompt Context Manager for managing prompt lifecycle and resources."""

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
from src.prompt.types import PromptConfig, Prompt


class PromptContextManager(BaseModel):
    """Global context manager for all prompts."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the prompts")
    save_path: str = Field(default=None, description="The path to save the prompts")
    
    DEFAULT_DISCOVERY_PACKAGES: List[str] = [
        "src.prompt.template",
    ]
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 auto_discover: bool = True, 
                 **kwargs):
        """Initialize the prompt context manager.
        
        Args:
            base_dir: Base directory for storing prompt data
            save_path: Path to save prompt configurations
            auto_discover: Whether to automatically discover and register prompts from packages
        """
        super().__init__(**kwargs)
        
        # Set up paths
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "prompts"))
        os.makedirs(self.base_dir, exist_ok=True)
        
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "prompts.json")
        
        self._prompt_configs: Dict[str, PromptConfig] = {}  # Store prompt metadata
        self._cleanup_registered = False
        self.auto_discover = auto_discover
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
    
    async def initialize(self):
        """Initialize the prompt context manager."""
        if self.auto_discover:
            await self.discover()
    
    async def _collect_prompt_templates(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Collect all PROMPT_TEMPLATES dictionaries from packages.
        
        Args:
            packages: List of package names to scan
            
        Returns:
            List of prompt template dictionaries
        """
        prompt_templates = []
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
                    
                    # Find PROMPT_TEMPLATES dictionary in the module
                    if hasattr(module, 'PROMPT_TEMPLATES'):
                        templates_dict = getattr(module, 'PROMPT_TEMPLATES')
                        if isinstance(templates_dict, dict):
                            # Convert dict to list of templates, ensuring each has a 'name' field
                            templates = []
                            for key, template_dict in templates_dict.items():
                                if isinstance(template_dict, dict):
                                    # Ensure template has 'name' field (use key if missing)
                                    template_with_name = template_dict.copy()
                                    if 'name' not in template_with_name:
                                        template_with_name['name'] = key
                                    templates.append(template_with_name)
                            return templates
                    return []
                except Exception as e:
                    logger.debug(f"| ⚠️ Failed to import module {module_name}: {e}")
                    return []
            
            # Import all modules concurrently
            import_tasks = [import_module(module_name) for module_name in module_names]
            results = await asyncio.gather(*import_tasks, return_exceptions=True)
            
            # Collect all prompt templates
            for result in results:
                if isinstance(result, list):
                    for template in result:
                        if template not in prompt_templates:
                            prompt_templates.append(template)
        
        return prompt_templates
    
    async def _register_prompt_template(self, template_dict: Dict[str, Any], override: bool = False):
        """Register a prompt template.
        
        Args:
            template_dict: Prompt template dictionary
            override: Whether to override existing registration
        """
        prompt_name = template_dict.get('name')
        if not prompt_name:
            logger.warning(f"| ⚠️ Prompt template has no name, skipping")
            return
        
        try:
            prompt_type = template_dict.get('type', 'prompt')
            prompt_description = template_dict.get('description', '')
            prompt_template = template_dict.get('template', '')
            prompt_variables = template_dict.get('variables', [])
            
            if prompt_name in self._prompt_configs and not override:
                logger.debug(f"| ⚠️ Prompt {prompt_name} already registered, skipping")
                return
            
            # Get or generate version from version_manager
            version = await version_manager.get_version("prompt", prompt_name)
            
            # Create PromptConfig
            prompt_config = PromptConfig(
                name=prompt_name,
                type=prompt_type,
                description=prompt_description,
                version=version,
                template=prompt_template,
                variables=prompt_variables,
                cls=None,  # Prompts are dictionaries, not classes
                instance=None,
                config={},
                metadata=template_dict.get('metadata', {})
            )
            
            # Store metadata
            self._prompt_configs[prompt_name] = prompt_config
            
            # Register version record to version manager
            await version_manager.register_version("prompt", prompt_name, prompt_config.version)
            
            logger.debug(f"| 📝 Registered prompt: {prompt_name} v{prompt_config.version}")
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to register prompt template {prompt_name}: {e}")
            import traceback
            logger.debug(f"| Traceback: {traceback.format_exc()}")
            raise
    
    async def discover(self, packages: Optional[List[str]] = None):
        """Discover and register all prompt templates from specified packages.
        
        Args:
            packages: List of package names to scan. Defaults to DEFAULT_DISCOVERY_PACKAGES.
        """
        packages = packages or self.DEFAULT_DISCOVERY_PACKAGES
        
        logger.info(f"| 🔍 Discovering prompts from packages: {packages}")
        
        # Collect all prompt templates
        prompt_templates = await self._collect_prompt_templates(packages)
        
        # Register each prompt template concurrently
        registration_tasks = [
            self._register_prompt_template(template_dict) for template_dict in prompt_templates
        ]
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Count successful registrations
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(prompt_templates)} prompts")
    
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
                prompt_template = prompt.template
                prompt_variables = prompt.variables
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
            
            # Create PromptConfig
            prompt_config = PromptConfig(
                name=prompt_name,
                type=prompt_type,
                description=prompt_description,
                version=version,
                template=prompt_template,
                variables=prompt_variables,
                cls=prompt_cls,
                instance=prompt_instance,
                config=kwargs if kwargs else {},
                metadata=prompt.get('metadata', {}) if isinstance(prompt, dict) else {}
            )
            
            # Store metadata
            self._prompt_configs[prompt_name] = prompt_config
            
            # Register version record to version manager
            await version_manager.register_version("prompt", prompt_name, prompt_config.version)
            
            logger.debug(f"| 📝 Registered prompt: {prompt_name} v{prompt_config.version}")
            return prompt_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register prompt: {e}")
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
            prompt_template = prompt.template
            prompt_variables = prompt.variables
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
                instance=prompt_instance,
                metadata=prompt.get('metadata', {}) if isinstance(prompt, dict) else original_config.metadata
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
                metadata=prompt.get('metadata', {}) if isinstance(prompt, dict) else original_config.metadata
            )
        
        # Store updated config
        self._prompt_configs[prompt_name] = updated_config
        
        # Register version record to version manager
        await version_manager.register_version("prompt", prompt_name, updated_config.version)
        
        logger.info(f"| 📝 Updated prompt: {prompt_name} v{updated_config.version}")
        return updated_config
    
    async def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt template dictionary by name
        
        Args:
            name: Name of the prompt
        """
        prompt_config = self._prompt_configs.get(name)
        if prompt_config:
            return {
                "name": prompt_config.name,
                "type": prompt_config.type,
                "description": prompt_config.description,
                "template": prompt_config.template,
                "variables": prompt_config.variables,
                "metadata": prompt_config.metadata
            }
        return None
    
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
        """Save all prompt configurations to JSON
        
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
                    "prompt_count": len(self._prompt_configs),
                },
                "prompts": {}
            }
            
            for prompt_name, prompt_config in self._prompt_configs.items():
                try:
                    # Serialize prompt config (excluding non-serializable cls and instance)
                    config_dict = prompt_config.model_dump(mode="json", exclude={"cls", "instance"})
                    
                    save_data["prompts"][prompt_name] = config_dict
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize prompt {prompt_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._prompt_configs)} prompts to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load prompt configurations from JSON
        
        Args:
            file_path: File path to load from
            
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
                        # Create PromptConfig
                        prompt_config = PromptConfig(**prompt_data)
                        
                        # Register prompt config
                        self._prompt_configs[prompt_name] = prompt_config
                        
                        # Register version to version manager
                        await version_manager.register_version("prompt", prompt_name, prompt_config.version)
                        
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load prompt {prompt_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} prompts from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load prompts from {file_path}: {e}")
                return False
    
    async def cleanup(self):
        """Cleanup all prompt instances and resources."""
        try:
            # Clear instances and configs
            self._prompt_configs.clear()
            logger.info("| 🧹 Prompt context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during prompt context manager cleanup: {e}")
