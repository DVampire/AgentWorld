"""Tool Context Manager for managing tool lifecycle and resources with lazy loading."""


import importlib
import pkgutil
import os
import asyncio
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union
import inflection

from src.logger import logger
from src.config import config
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.utils import assemble_project_path
from src.tool.types import Tool, ToolConfig

class ToolContextManager:
    """Global context manager for all tools with lazy loading support."""
    
    DEFAULT_DISCOVERY_PACKAGES: List[str] = [
        "src.tool.default_tools",
        "src.tool.workflow_tools",
        "src.tool.other_tools",
    ]
    
    def __init__(self, auto_discover: bool = True, model_name: str = "openrouter/text-embedding-3-large"):
        """Initialize the tool context manager.
        
        Args:
            auto_discover: Whether to automatically discover and register tools from packages
        """
        self._tool_configs: Dict[str, ToolConfig] = {}
        self._next_tool_id: int = 1
        self._cleanup_registered = False
        
        self.model_name = model_name
        self.auto_discover = auto_discover
        
    async def initialize(self):
        """Initialize the tool context manager."""
        
        # Initialize Faiss service for tool embedding
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
                cls=tool_cls,
                config={},
                instance=None,
                metadata={}
            )
            self._next_tool_id += 1
            
            # Store tool config
            self._tool_configs[tool_name] = tool_config
            
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
                    cls=tool,
                    config=kwargs,
                    instance=None,  # Will be created on initialize
                    metadata={}
                )
            self._next_tool_id += 1
            
            # Store tool config (without instance - lazy loading)
            self._tool_configs[tool_name] = tool_config
            
            # Register to embedding index
            await self._store(tool_config)
            
            logger.debug(f"| 📝 Registered tool config: {tool_name}")
            
            return tool_config
            
        except Exception as e:
            logger.error(f"| ❌ Failed to register tool: {e}")
            raise
    
    
    async def get(self, tool_name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolConfig: Tool configuration or None if not found
        """
        return self._tool_configs.get(tool_name)
    
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
    
    async def cleanup(self):
        """Cleanup all active tools."""
        try:
            # Clear all tool configs
            self._tool_configs.clear()
                
            # Clean up Faiss service (async)
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            logger.info("| 🧹 Tool context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during tool context manager cleanup: {e}")
