"""Tool Context Manager for managing tool lifecycle and resources with lazy loading."""


import importlib
import pkgutil
import os
import asyncio
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union
from datetime import datetime
import inflection
import json
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.utils import assemble_project_path
from src.tool.types import Tool, ToolConfig
from src.version import version_manager
from src.utils.file_utils import file_lock


class ToolContextManager(BaseModel):
    """Global context manager for all tools with lazy loading support."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    
    DEFAULT_DISCOVERY_PACKAGES: List[str] = [
        "src.tool.default_tools",
        "src.tool.workflow_tools",
        "src.tool.other_tools",
    ]
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 auto_discover: bool = True, 
                 model_name: str = "openrouter/text-embedding-3-large",
                 **kwargs):
        """Initialize the tool context manager.
        
        Args:
            auto_discover: Whether to automatically discover and register tools from packages
        """
        super().__init__(**kwargs)
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "tools"))
        os.makedirs(self.base_dir, exist_ok=True)
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "tools.json")
        logger.info(f"| 📁 Tool context manager base directory: {self.base_dir} and save path: {self.save_path}")
        
        self._tool_configs: Dict[str, ToolConfig] = {}  # Current active configs (latest version)
        self._tool_version_history: Dict[str, List[ToolConfig]] = {}  # All versions for each tool
        self._next_tool_id: int = 1
        self._cleanup_registered = False
        
        self.model_name = model_name
        self.auto_discover = auto_discover
        
    async def initialize(self):
        """Initialize the tool context manager."""
        
        # Initialize Faiss service for tool embedding
        # config.workdir is already a string after process_general
        base_dir = os.path.join(config.workdir, "tools")
        os.makedirs(base_dir, exist_ok=True)
        
        self._faiss_service = FaissService(
            base_dir=base_dir,
            model_name=self.model_name
        )
        
        # Register async cleanup on exit using asyncio-atexit
        # Only register if there's a running event loop, otherwise defer to first async call
        if not self._cleanup_registered:
            try:
                asyncio.get_running_loop()
                # There's a running loop, register now
                async_atexit_register(self.cleanup)
                self._cleanup_registered = True
            except RuntimeError:
                # No running loop, will register later when we have one
                pass
    
    def _ensure_cleanup_registered(self):
        """Ensure async cleanup is registered. Call this at the start of async methods."""
        if not self._cleanup_registered:
            try:
                asyncio.get_running_loop()
                # There's a running loop now, register cleanup
                async_atexit_register(self.cleanup)
                self._cleanup_registered = True
            except RuntimeError:
                # Still no running loop, skip for now
                pass
    
    async def _collect_tool_classes(self, packages: List[str]) -> List[Type[Tool]]:
        """Collect all Tool subclasses from packages.
        
        Args:
            packages: List of package names to scan
            
        Returns:
            List of Tool subclasses
        """
        tool_classes = []
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
                    
                    # Find all Tool subclasses in the module
                    found_classes = []
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (isinstance(obj, type) and 
                            issubclass(obj, Tool) and 
                            obj is not Tool):
                            found_classes.append(obj)
                    return found_classes
                except Exception as e:
                    logger.debug(f"| ⚠️ Failed to import module {module_name}: {e}")
                    return []
            
            # Import all modules concurrently
            import_tasks = [import_module(module_name) for module_name in module_names]
            results = await asyncio.gather(*import_tasks, return_exceptions=True)
            
            # Collect all tool classes
            for result in results:
                if isinstance(result, list):
                    for tool_cls in result:
                        if tool_cls not in tool_classes:
                            tool_classes.append(tool_cls)
        
        return tool_classes
    
    async def _register_tool_class(self, tool_cls: Type[Tool], override: bool = False):
        """Register a tool class synchronously.
        
        Args:
            tool_cls: Tool class to register
            override: Whether to override existing registration
        """
        tool_name = None
        try:
            # Get tool config from global config
            tool_config_key = inflection.underscore(tool_cls.__name__)
            tool_config = config.get(f"{tool_config_key}_tool", {})
            
            # Try to create temporary instance to get name and description
            try:
                temp_instance = tool_cls(**tool_config)
                tool_name = temp_instance.name
                tool_description = temp_instance.description
            except Exception:
                # If instantiation fails, try without config
                try:
                    temp_instance = tool_cls()
                    tool_name = temp_instance.name
                    tool_description = temp_instance.description
                except Exception:
                    # If still fails, try to get from class attributes or model_fields
                    tool_name = getattr(tool_cls, 'name', None)
                    tool_description = getattr(tool_cls, 'description', '')
                    
                    if not tool_name:
                        logger.warning(f"| ⚠️ Tool class {tool_cls.__name__} has no name, skipping")
                        return
            
            if not tool_name:
                logger.warning(f"| ⚠️ Tool class {tool_cls.__name__} has empty name, skipping")
                return
            
            if tool_name in self._tool_configs and not override:
                logger.debug(f"| ⚠️ Tool {tool_name} already registered, skipping")
                return
            
            # Create ToolConfig with auto-increment ID
            tool_config = ToolConfig(
                id=self._next_tool_id,
                name=tool_name,
                description=tool_description,
                enabled=True,
                version="1.0.0",
                cls=tool_cls,
                config={},
                instance=None,
                metadata={}
            )
            self._next_tool_id += 1
            
            # Store tool config
            self._tool_configs[tool_name] = tool_config
            
            # Store in version history
            if tool_name not in self._tool_version_history:
                self._tool_version_history[tool_name] = []
            self._tool_version_history[tool_name].append(tool_config)
            
            # Register version record to version manager
            await version_manager.register_version("tool", tool_name, tool_config.version)
            
            # Register to embedding index asynchronously
            await self._store(tool_config)
            
            logger.debug(f"| 📝 Registered tool: {tool_name} ({tool_cls.__name__})")
            
        except Exception as e:
            tool_name_str = tool_name or tool_cls.__name__
            logger.warning(f"| ⚠️ Failed to register tool class {tool_cls.__name__} (name: {tool_name_str}): {e}")
            import traceback
            logger.debug(f"| Traceback: {traceback.format_exc()}")
            raise
            
    def _ensure_tool_instance(self, tool: Union[Tool, Type[Tool]], **kwargs: Any) -> Tool:
        """Ensure we have a tool instance.
        
        Args:
            tool: Tool instance or class
            **kwargs: Configuration for tool initialization
            
        Returns:
            Tool instance
        """
        if isinstance(tool, Tool):
            if kwargs:
                raise ValueError("Extra keyword arguments are not allowed when registering tool instances.")
            return tool
        if isinstance(tool, type) and issubclass(tool, Tool):
            return tool(**kwargs)
        raise TypeError(f"Expected Tool instance or subclass, got {type(tool)!r}")
    
    async def _store(self, tool_config: ToolConfig):
        """Add tool information to the embedding index.
        
        Args:
            tool_config: Tool configuration
        """
        if self._faiss_service is None:
            return
            
        try:
            # Create comprehensive text representation
            tool_text = f"Tool: {tool_config.name}\nDescription: {tool_config.description}"
            
            # Add to FAISS index
            request = FaissAddRequest(
                texts=[tool_text],
                metadatas=[{
                    "name": tool_config.name,
                    "description": tool_config.description
                }]
            )
            
            await self._faiss_service.add_documents(request)
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add tool {tool_config.name} to FAISS index: {e}")
    
    async def build(self, tool_config: ToolConfig) -> ToolConfig:
        """Create a tool instance and store it.
        
        Args:
            tool_config: Tool configuration
            
        Returns:
            ToolConfig: Tool configuration with instance
        """
        if tool_config.name in self._tool_configs:
            existing_config = self._tool_configs[tool_config.name]
            # If instance already exists, return it
            if existing_config.instance is not None:
                return existing_config
            # Otherwise, update config and create instance
            existing_config.config = tool_config.config
            existing_config.cls = tool_config.cls
            # Preserve the existing ID
            tool_config.id = existing_config.id
        
        # Create new tool instance
        try:
            if tool_config.cls is None:
                raise ValueError(f"Cannot create tool {tool_config.name}: no class provided")
            
            tool_instance = tool_config.cls(**tool_config.config)
            tool_config.instance = tool_instance
            
            # Store tool metadata
            self._tool_configs[tool_config.name] = tool_config
            
            logger.debug(f"| 🔧 Tool {tool_config.name} created and stored")
            
            # Note: Tool is already added to embedding index during registration (_register_tool_class)
            # No need to call _store() again here
            
            return tool_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create tool {tool_config.name}: {e}")
            raise
    
    async def discover(self, packages: Optional[List[str]] = None):
        """Discover and register all Tool subclasses from specified packages.
        
        Args:
            packages: List of package names to scan. Defaults to DEFAULT_DISCOVERY_PACKAGES.
        """
        self._ensure_cleanup_registered()
        packages = packages or self.DEFAULT_DISCOVERY_PACKAGES
        
        logger.info(f"| 🔍 Discovering tools from packages: {packages}")
        
        # Collect all Tool subclasses
        tool_classes = await self._collect_tool_classes(packages)
        
        # Register each tool class concurrently
        registration_tasks = [
            self._register_tool_class(tool_cls) for tool_cls in tool_classes
        ]
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Count successful registrations
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(tool_classes)} tools")
    
    async def register(self, tool: Union[Tool, Type[Tool]], *, override: bool = False, **kwargs: Any) -> ToolConfig:
        """Register a tool class or instance.
        
        Args:
            tool: Tool class or instance
            override: Whether to override existing registration
            **kwargs: Configuration for tool initialization
            
        Returns:
            ToolConfig: Tool configuration
        """
        self._ensure_cleanup_registered()
        # Create temporary instance to get name and description
        try:
            temp_instance = self._ensure_tool_instance(tool, **kwargs)
            tool_name = temp_instance.name
            tool_description = temp_instance.description
            
            if not tool_name:
                raise ValueError("Tool.name cannot be empty.")
            
            if tool_name in self._tool_configs and not override:
                raise ValueError(f"Tool '{tool_name}' already registered. Use override=True to replace it.")
            
            # Determine if we're registering a class or instance
            if isinstance(tool, Tool):
                # Registering an instance
                tool_config = ToolConfig(
                    id=self._next_tool_id,
                    name=tool_name,
                    description=tool_description,
                    enabled=tool.enabled,
                    version=getattr(tool, 'version', '1.0.0'),
                    cls=type(tool),
                    config={},
                    instance=tool,
                    metadata=getattr(tool, 'metadata', {})
                )
            else:
                # Registering a class - store config for lazy loading
                tool_config = ToolConfig(
                    id=self._next_tool_id,
                    name=tool_name,
                    description=tool_description,
                    enabled=True,  # Default to enabled
                    version=kwargs.pop("version", "1.0.0"),  # Extract version if provided
                    cls=tool,
                    config=kwargs,
                    instance=None,  # Will be created on initialize
                    metadata={}
                )
            self._next_tool_id += 1
            
            # Store tool config (without instance - lazy loading)
            self._tool_configs[tool_name] = tool_config
            
            # Store in version history
            if tool_name not in self._tool_version_history:
                self._tool_version_history[tool_name] = []
            self._tool_version_history[tool_name].append(tool_config)
            
            # Register version record to version manager (only version info, no config)
            await version_manager.register_version("tool", tool_name, tool_config.version)
            
            # Register to embedding index
            await self._store(tool_config)
            
            logger.debug(f"| 📝 Registered tool config: {tool_name} v{tool_config.version}")
            
            return tool_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register tool: {e}")
            raise
    
    
    async def get(self, tool_name: str) -> Tool:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool: Tool instance or None if not found
        """
        tool_config = self._tool_configs.get(tool_name)
        if tool_config is None:
            return None
        return tool_config.instance if tool_config.instance is not None else None
    
    async def list(self, include_disabled: bool = False) -> List[str]:
        """Get list of registered tools
        
        Args:
            include_disabled: Whether to include disabled tools
            
        Returns:
            List[str]: List of tool names
        """
        if include_disabled:
            return list(self._tool_configs.keys())
        return [
            name for name, config in self._tool_configs.items()
            if config.enabled
        ]
    
    async def to_text(self, tool_name: str) -> str:
        """Convert tool information to string
        
        Args:
            tool_name: Tool name
            
        Returns:
            str: Tool information string
        """
        tool_config = self._tool_configs.get(tool_name)
        if not tool_config:
            raise ValueError(f"Tool {tool_name} not found")
        
        # Get instance or create temporary one for schema
        if tool_config.instance is not None:
            instance = tool_config.instance
        elif tool_config.cls is not None:
            try:
                instance = tool_config.cls(**tool_config.config)
            except Exception as e:
                return f"{tool_config.id:04d}. {tool_config.name}: Failed to create instance ({e})"
        else:
            return f"{tool_config.id:04d}. {tool_config.name}: No class available"
        
        tool_text = f"{tool_config.id:04d}. {tool_config.name}: {tool_config.description}\n"
        tool_text += instance.to_text()
        return tool_text
    
    async def to_function_call(self, tool_name: str) -> Dict[str, Any]:
        """Convert tool information to function call
        
        Args:
            tool_name: Tool name
            
        Returns:
            Dict[str, Any]: Function call
        """
        tool_config = await self.get(tool_name)
        if not tool_config:
            raise ValueError(f"Tool {tool_name} not found")
        
        return tool_config.instance.to_function_call()
    
    async def update(self, tool_name: str, tool: Union[Tool, Type[Tool]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> ToolConfig:
        """Update an existing tool with new configuration and create a new version
        
        Args:
            tool_name: Name of the tool to update
            tool: New tool class or instance with updated implementation
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for tool initialization
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        original_config = self._tool_configs.get(tool_name)
        if original_config is None:
            raise ValueError(f"Tool {tool_name} not found. Use register() to register a new tool.")
        
        # Create temporary instance to get new name and description
        temp_instance = self._ensure_tool_instance(tool, **kwargs)
        new_description = temp_instance.description
        
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
        
        # Create new tool config with updated content
        if isinstance(tool, Tool):
            # Updating with an instance
            updated_config = ToolConfig(
                id=original_config.id,  # Keep same ID
                name=tool_name,  # Keep same name
                description=new_description,
                enabled=tool.enabled,
                version=new_version,
                cls=type(tool),
                config={},
                instance=tool,
                metadata=getattr(tool, 'metadata', {})
            )
        else:
            # Updating with a class
            updated_config = ToolConfig(
                id=original_config.id,  # Keep same ID
                name=tool_name,  # Keep same name
                description=new_description,
                enabled=original_config.enabled,
                version=new_version,
                cls=tool,
                config=kwargs,
                instance=None,  # Will be created on initialize
                metadata={}
            )
        
        # Update the tool config (replaces current version)
        self._tool_configs[tool_name] = updated_config
        
        # Store in version history
        if tool_name not in self._tool_version_history:
            self._tool_version_history[tool_name] = []
        self._tool_version_history[tool_name].append(updated_config)
        
        # Register new version record to version manager
        await version_manager.register_version(
            "tool", 
            tool_name, 
            new_version,
            description=description or f"Updated from {original_config.version}"
        )
        
        # Update embedding index
        await self._store(updated_config)
        
        logger.info(f"| 🔄 Updated tool {tool_name} from v{original_config.version} to v{new_version}")
        return updated_config
    
    async def copy(self, tool_name: str, new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, **override_config) -> ToolConfig:
        """Copy an existing tool configuration
        
        Args:
            tool_name: Name of the tool to copy
            new_name: New name for the copied tool. If None, uses original name.
            new_version: New version for the copied tool. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            ToolConfig: New tool configuration
        """
        original_config = self._tool_configs.get(tool_name)
        if original_config is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        # Determine new name
        if new_name is None:
            new_name = tool_name
        
        # Determine new version
        if new_version is None:
            # Try to increment version
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
        
        # Create copy of config
        new_config_dict = original_config.model_dump()
        new_config_dict["name"] = new_name
        new_config_dict["version"] = new_version
        new_config_dict["id"] = self._next_tool_id
        self._next_tool_id += 1
        
        # Apply overrides
        if override_config:
            if "description" in override_config:
                new_config_dict["description"] = override_config.pop("description")
            if "enabled" in override_config:
                new_config_dict["enabled"] = override_config.pop("enabled")
            if "metadata" in override_config:
                new_config_dict["metadata"].update(override_config.pop("metadata"))
            # Merge remaining overrides into config
            new_config_dict["config"].update(override_config)
        
        # Clear instance (will be created on demand)
        new_config_dict["instance"] = None
        
        new_config = ToolConfig(**new_config_dict)
        
        # Register new tool
        self._tool_configs[new_name] = new_config
        
        # Store in version history
        if new_name not in self._tool_version_history:
            self._tool_version_history[new_name] = []
        self._tool_version_history[new_name].append(new_config)
        
        # Register version record to version manager
        await version_manager.register_version(
            "tool", 
            new_name, 
            new_version,
            description=f"Copied from {tool_name}@{original_config.version}"
        )
        
        # Register to embedding index
        await self._store(new_config)
        
        logger.info(f"| 📋 Copied tool {tool_name}@{original_config.version} to {new_name}@{new_version}")
        return new_config
    
    async def unregister(self, tool_name: str) -> bool:
        """Unregister a tool
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if tool_name not in self._tool_configs:
            logger.warning(f"| ⚠️ Tool {tool_name} not found")
            return False
        
        tool_config = self._tool_configs[tool_name]
        
        # Remove from configs
        del self._tool_configs[tool_name]
        
        # Note: We keep version history in version_manager, just remove from active configs
        logger.info(f"| 🗑️ Unregistered tool {tool_name}@{tool_config.version}")
        return True
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all tool configurations with version history to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare save data - save all versions for each tool
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "tool_count": len(self._tool_configs),
                    "total_versions": sum(len(versions) for versions in self._tool_version_history.values())
                },
                "tools": {}
            }
            
            for tool_name, version_list in self._tool_version_history.items():
                try:
                    # Serialize all versions for this tool
                    versions_data = []
                    for tool_config in version_list:
                        # Serialize tool config (excluding non-serializable cls and instance)
                        config_dict = tool_config.model_dump(mode="json", exclude={"cls", "instance"})
                        
                        # Store class path if available
                        if tool_config.cls is not None:
                            config_dict["cls_path"] = f"{tool_config.cls.__module__}.{tool_config.cls.__name__}"
                        
                        versions_data.append(config_dict)
                    
                    # Sort by version (newest first)
                    save_data["tools"][tool_name] = {
                        "versions": versions_data,
                        "current_version": self._tool_configs.get(tool_name).version if tool_name in self._tool_configs else None
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize tool {tool_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._tool_configs)} tools with version history to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load tool configurations with version history from JSON
        
        Only the latest version will be instantiated by default.
        
        Args:
            file_path: File path to load from
            auto_initialize: Whether to automatically initialize latest version tools after loading
            
        Returns:
            True if loaded successfully, False otherwise
        """
        
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Tool file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                tools_data = load_data.get("tools", {})
                loaded_count = 0
                
                for tool_name, tool_data in tools_data.items():
                    try:
                        # Handle both old format (single config) and new format (versions list)
                        if "versions" in tool_data:
                            # New format: multiple versions
                            versions_data = tool_data["versions"]
                            current_version_str = tool_data.get("current_version")
                        else:
                            # Old format: single config (backward compatibility)
                            versions_data = [tool_data]
                            current_version_str = None
                        
                        # Load all versions
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for config_dict in versions_data:
                            # Try to load class if cls_path is available
                            cls = None
                            if "cls_path" in config_dict:
                                module_path, class_name = config_dict["cls_path"].rsplit(".", 1)
                                try:
                                    module = importlib.import_module(module_path)
                                    cls = getattr(module, class_name)
                                except Exception as e:
                                    logger.warning(f"| ⚠️ Failed to load class {config_dict['cls_path']} for {tool_name}: {e}")
                            
                            # Remove cls_path from dict before creating ToolConfig
                            config_dict_copy = config_dict.copy()
                            config_dict_copy.pop("cls_path", None)
                            
                            # Create ToolConfig
                            tool_config = ToolConfig(**config_dict_copy)
                            if cls is not None:
                                tool_config.cls = cls
                            
                            version_configs.append(tool_config)
                            
                            # Track latest version
                            if latest_config is None or (
                                current_version_str and tool_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    self._compare_versions(tool_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = tool_config
                                latest_version = tool_config.version
                        
                        # Store all versions in history
                        self._tool_version_history[tool_name] = version_configs
                        
                        # Only set latest version as active
                        if latest_config:
                            self._tool_configs[tool_name] = latest_config
                            
                            # Register all versions to version manager (only version records)
                            for tool_config in version_configs:
                                await version_manager.register_version("tool", tool_name, tool_config.version)
                            
                            # Initialize latest version tool if requested
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load tool {tool_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} tools with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load tools from {file_path}: {e}")
                return False
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]
            
            # Pad with zeros to same length
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))
            
            for p1, p2 in zip(parts1, parts2):
                if p1 > p2:
                    return 1
                elif p1 < p2:
                    return -1
            return 0
        except:
            # Fallback: string comparison
            return 1 if v1 > v2 else (-1 if v1 < v2 else 0)
    
    async def restore_version(self, tool_name: str, version: str, auto_initialize: bool = True) -> Optional[ToolConfig]:
        """Restore a specific version of a tool from history
        
        Args:
            tool_name: Name of the tool
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored tool
            
        Returns:
            ToolConfig of the restored version, or None if not found
        """
        if tool_name not in self._tool_version_history:
            logger.warning(f"| ⚠️ Tool {tool_name} not found in version history")
            return None
        
        # Find the version in history
        version_config = None
        for config in self._tool_version_history[tool_name]:
            if config.version == version:
                version_config = config
                break
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for tool {tool_name}")
            return None
        
        # Create a copy to avoid modifying the history
        restored_config = ToolConfig(**version_config.model_dump())
        
        # Set as current active config
        self._tool_configs[tool_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("tool", tool_name)
        if version_history:
            version_history.current_version = version
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        logger.info(f"| 🔄 Restored tool {tool_name} to version {version}")
        return restored_config
    
    async def cleanup(self):
        """Cleanup all active tools."""
        try:
            # Clear all tool configs and version history
            self._tool_configs.clear()
            self._tool_version_history.clear()
                
            # Clean up Faiss service (async)
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            logger.info("| 🧹 Tool context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during tool context manager cleanup: {e}")
