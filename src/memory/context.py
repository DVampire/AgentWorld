"""Memory Context Manager for managing memory lifecycle and resources."""

import asyncio
import atexit
import importlib
import pkgutil
import inflection
import os
import json
import inspect
from datetime import datetime
from typing import Any, Dict, Optional, List, Union, Type
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import assemble_project_path
from src.utils.file_utils import file_lock
from src.memory.types import MemoryConfig, Memory


class MemoryContextManager(BaseModel):
    """Global context manager for all memory systems."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the memory systems")
    save_path: str = Field(default=None, description="The path to save the memory systems")
    
    DEFAULT_DISCOVERY_PACKAGES: List[str] = [
        "src.memory",
    ]
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 auto_discover: bool = True, 
                 **kwargs):
        """Initialize the memory context manager.
        
        Args:
            base_dir: Base directory for storing memory data
            save_path: Path to save memory configurations
            auto_discover: Whether to automatically discover and register memory systems from packages
        """
        super().__init__(**kwargs)
        
        # Set up paths
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "memory"))
        os.makedirs(self.base_dir, exist_ok=True)
        
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "memory.json")
        
        self._memory_configs: Dict[str, MemoryConfig] = {}  # Store memory metadata
        self._cleanup_registered = False
        self.auto_discover = auto_discover
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
    
    async def initialize(self):
        """Initialize the memory context manager."""
        if self.auto_discover:
            await self.discover()
            self.auto_discover = False  # Prevent duplicate discovery calls
    
    async def _collect_memory_classes(self, packages: List[str]) -> List[Type[Memory]]:
        """Collect all memory system classes from packages.
        
        Args:
            packages: List of package names to scan
            
        Returns:
            List of memory system classes
        """
        memory_classes = []
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
                    
                    # Find all Memory subclasses in the module (similar to tool discovery)
                    found_classes = []
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (isinstance(obj, type) and 
                            issubclass(obj, Memory) and 
                            obj is not Memory):
                            found_classes.append(obj)
                    return found_classes
                except Exception as e:
                    logger.debug(f"| ⚠️ Failed to import module {module_name}: {e}")
                    return []
            
            # Import all modules concurrently
            import_tasks = [import_module(module_name) for module_name in module_names]
            results = await asyncio.gather(*import_tasks, return_exceptions=True)
            
            # Collect all memory classes
            for result in results:
                if isinstance(result, list):
                    for cls in result:
                        if cls not in memory_classes:
                            memory_classes.append(cls)
        
        return memory_classes
    
    async def _register_memory_class(self, memory_cls: Type[Memory], override: bool = False):
        """Register a memory system class (similar to tool registration).
        
        Args:
            memory_cls: Memory system class (must inherit from Memory)
            override: Whether to override existing registration
        """
        memory_name = None
        try:
            # Get memory config from global config
            memory_config_key = inflection.underscore(memory_cls.__name__)
            memory_config_dict = config.get(memory_config_key, {})
            
            # Try to create temporary instance to get name and description
            try:
                temp_instance = memory_cls(**memory_config_dict)
                memory_name = temp_instance.name
                memory_description = temp_instance.description
            except Exception:
                # If instantiation fails, try without config
                try:
                    temp_instance = memory_cls()
                    memory_name = temp_instance.name
                    memory_description = temp_instance.description
                except Exception:
                    # If still fails, try to get from class attributes or use defaults
                    memory_name = getattr(memory_cls, 'name', None)
                    memory_description = getattr(memory_cls, 'description', '')
                    
                    if not memory_name:
                        # Use class name as fallback
                        memory_name = inflection.underscore(memory_cls.__name__)
                    if not memory_description:
                        memory_description = memory_cls.__doc__ or f"{memory_cls.__name__} memory system"
                        if memory_description:
                            memory_description = memory_description.strip().split('\n')[0]
            
            if not memory_name:
                logger.warning(f"| ⚠️ Memory class {memory_cls.__name__} has no name, skipping")
                return
            
            if memory_name in self._memory_configs and not override:
                logger.debug(f"| ⚠️ Memory {memory_name} already registered, skipping")
                return
            
            # Get or generate version from version_manager
            version = await version_manager.get_version("memory", memory_name)
            
            # Create MemoryConfig
            memory_config = MemoryConfig(
                name=memory_name,
                description=memory_description,
                version=version,
                cls=memory_cls,
                instance=None,
                config=memory_config_dict,
                metadata={}
            )
            
            # Store metadata
            self._memory_configs[memory_name] = memory_config
            
            # Register version record to version manager
            await version_manager.register_version("memory", memory_name, memory_config.version)
            
            logger.debug(f"| 🧠 Registered memory: {memory_name} v{memory_config.version}")
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to register memory class {memory_cls.__name__}: {e}")
            import traceback
            logger.debug(f"| Traceback: {traceback.format_exc()}")
            raise
    
    async def discover(self, packages: Optional[List[str]] = None):
        """Discover and register all memory systems from specified packages.
        
        Args:
            packages: List of package names to scan. Defaults to DEFAULT_DISCOVERY_PACKAGES.
        """
        packages = packages or self.DEFAULT_DISCOVERY_PACKAGES
        
        logger.info(f"| 🔍 Discovering memory systems from packages: {packages}")
        
        # Collect all memory classes
        memory_classes = await self._collect_memory_classes(packages)
        
        # Register each memory class concurrently
        registration_tasks = [
            self._register_memory_class(memory_cls) for memory_cls in memory_classes
        ]
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Count successful registrations
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(memory_classes)} memory systems")
    
    async def register(self, memory: Union[Memory, Type[Memory]], *, override: bool = False, **kwargs: Any) -> MemoryConfig:
        """Register a memory system or memory class (similar to tool registration).
        
        Args:
            memory: Memory instance or class
            override: Whether to override existing registration
            **kwargs: Configuration for memory initialization
            
        Returns:
            MemoryConfig: Memory configuration
        """
        # Get memory config from global config
        memory_config_key = None
        memory_config_dict = {}
        
        # Create temporary instance to get name and description
        try:
            temp_instance = self._ensure_memory_instance(memory, **kwargs)
            memory_name = temp_instance.name
            memory_description = temp_instance.description
            
            if not memory_name:
                raise ValueError("Memory.name cannot be empty.")
            
            if memory_name in self._memory_configs and not override:
                raise ValueError(f"Memory '{memory_name}' already registered. Use override=True to replace it.")
            
            # Get memory config from global config
            memory_config_key = inflection.underscore(type(temp_instance).__name__)
            memory_config_dict = config.get(memory_config_key, {})
            
        except Exception as e:
            logger.error(f"| ❌ Failed to create temporary memory instance: {e}")
            raise
        
        # Determine if we're registering a class or instance
        if isinstance(memory, Memory):
            # Registering an instance
            memory_config = MemoryConfig(
                name=memory_name,
                description=memory_description,
                version=kwargs.pop("version", "1.0.0"),
                cls=type(memory),
                config={},
                instance=memory,
                metadata={}
            )
        else:
            # Registering a class - store config for lazy loading
            memory_config = MemoryConfig(
                name=memory_name,
                description=memory_description,
                version=kwargs.pop("version", "1.0.0"),
                cls=memory,
                config=kwargs if kwargs else memory_config_dict,
                instance=None,  # Will be created on get()
                metadata={}
            )
        
        # Get or generate version from version_manager
        if memory_config.version == "1.0.0":
            memory_config.version = await version_manager.get_or_generate_version("memory", memory_name)
        
        # Store metadata
        self._memory_configs[memory_name] = memory_config
        
        # Register version record to version manager
        await version_manager.register_version("memory", memory_name, memory_config.version)
        
        logger.debug(f"| 🧠 Registered memory: {memory_name} v{memory_config.version}")
        return memory_config
    
    def _ensure_memory_instance(self, memory: Union[Memory, Type[Memory]], **kwargs) -> Memory:
        """Ensure we have a memory instance (similar to tool's _ensure_tool_instance).
        
        Args:
            memory: Memory instance or class
            **kwargs: Configuration for memory initialization
            
        Returns:
            Memory instance
        """
        if isinstance(memory, Memory):
            if kwargs:
                raise ValueError("Extra keyword arguments are not allowed when registering memory instances.")
            return memory
        if isinstance(memory, type) and issubclass(memory, Memory):
            # Get config from global config if available
            memory_config_key = inflection.underscore(memory.__name__)
            global_config = config.get(memory_config_key, {})
            # Merge global config with kwargs
            merged_config = {**global_config, **kwargs}
            return memory(**merged_config)
        raise TypeError(f"Expected Memory instance or subclass, got {type(memory)!r}")
    
    async def update(self, memory_name: str, memory: Union[Memory, Type[Memory]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> MemoryConfig:
        """Update an existing memory system with new configuration and create a new version (similar to tool update).
        
        Args:
            memory_name: Name of the memory system to update
            memory: New memory instance or class with updated content
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for memory initialization
            
        Returns:
            MemoryConfig: Updated memory configuration
        """
        original_config = self._memory_configs.get(memory_name)
        if original_config is None:
            raise ValueError(f"Memory {memory_name} not found. Use register() to register a new memory system.")
        
        # Create temporary instance to get new name and description
        temp_instance = self._ensure_memory_instance(memory, **kwargs)
        new_description = temp_instance.description
        
        # Determine new version from version_manager
        if new_version is None:
            # Get current version from version_manager and generate next patch version
            new_version = await version_manager.generate_next_version("memory", memory_name, "patch")
        
        # Create updated config
        if isinstance(memory, Memory):
            # Updating with an instance
            updated_config = MemoryConfig(
                name=memory_name,  # Keep same name
                description=new_description,
                version=new_version,
                cls=type(memory),
                config={},
                instance=memory,
                metadata={}
            )
        else:
            # Updating with a class
            updated_config = MemoryConfig(
                name=memory_name,  # Keep same name
                description=new_description,
                version=new_version,
                cls=memory,
                config=kwargs,
                instance=None,  # Will be created on get()
                metadata=original_config.metadata
            )
        
        # Store updated config
        self._memory_configs[memory_name] = updated_config
        
        # Register version record to version manager
        await version_manager.register_version(
            "memory", 
            memory_name, 
            new_version,
            description=description or f"Updated from {original_config.version}"
        )
        
        logger.info(f"| 🧠 Updated memory {memory_name} from v{original_config.version} to v{new_version}")
        return updated_config
    
    async def get(self, name: str) -> Optional[Type]:
        """Get a memory system class by name
        
        Args:
            name: Name of the memory system
        """
        memory_config = self._memory_configs.get(name)
        if memory_config:
            return memory_config.cls
        return None
    
    async def get_info(self, name: str) -> Optional[MemoryConfig]:
        """Get a memory configuration by name
        
        Args:
            name: Name of the memory system
        """
        return self._memory_configs.get(name)
    
    async def list(self) -> List[str]:
        """Get list of registered memory systems
        
        Returns:
            List[str]: List of memory system names
        """
        return [name for name in self._memory_configs.keys()]
    
    async def build(self, memory_config: MemoryConfig, memory_factory: Optional[callable] = None) -> Any:
        """Build a memory system instance from config.
        
        Args:
            memory_config: Memory configuration
            memory_factory: Optional factory function to create memory instance
            
        Returns:
            Memory system instance
        """
        if memory_config.instance is not None:
            return memory_config.instance
        
        if memory_factory:
            instance = memory_factory()
        else:
            # Use config to instantiate
            instance = memory_config.cls(**memory_config.config)
        
        memory_config.instance = instance
        return instance
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all memory configurations to JSON
        
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
                    "memory_count": len(self._memory_configs),
                },
                "memory_systems": {}
            }
            
            for memory_name, memory_config in self._memory_configs.items():
                try:
                    # Serialize memory config (excluding non-serializable cls and instance)
                    config_dict = memory_config.model_dump(mode="json", exclude={"cls", "instance"})
                    
                    save_data["memory_systems"][memory_name] = config_dict
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize memory {memory_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._memory_configs)} memory systems to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load memory configurations from JSON
        
        Args:
            file_path: File path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Memory file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                memory_data = load_data.get("memory_systems", {})
                loaded_count = 0
                
                for memory_name, memory_data_item in memory_data.items():
                    try:
                        # Create MemoryConfig (cls will be None, need to reload from discovery)
                        memory_config = MemoryConfig(**memory_data_item)
                        
                        # Register memory config
                        self._memory_configs[memory_name] = memory_config
                        
                        # Register version to version manager
                        await version_manager.register_version("memory", memory_name, memory_config.version)
                        
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load memory {memory_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} memory systems from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load memory systems from {file_path}: {e}")
                return False
    
    async def cleanup(self):
        """Cleanup all memory instances and resources."""
        try:
            # Clear instances and configs
            self._memory_configs.clear()
            logger.info("| 🧹 Memory context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during memory context manager cleanup: {e}")
            
    async def start_session(self, 
                            memory_name: str, 
                            session_id: str, 
                            agent_name: Optional[str] = None, 
                            task_id: Optional[str] = None, 
                            description: Optional[str] = None) -> str:
        """Start a memory session (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Session ID
            agent_name: Optional agent name
            task_id: Optional task ID
            description: Optional description
            
        Returns:
            Session ID
        """
        instance = self._get_memory_instance(memory_name)
        return await instance.start_session(session_id, agent_name, task_id, description)
    
    async def add_event(self, memory_name: str, step_number: int, event_type: Any, data: Any,
                       agent_name: str, task_id: Optional[str] = None, session_id: Optional[str] = None, **kwargs):
        """Add an event to memory (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            step_number: Step number
            event_type: Event type
            data: Event data
            agent_name: Agent name
            task_id: Optional task ID
            session_id: Optional session ID
            **kwargs: Additional arguments
        """
        instance = self._get_memory_instance(memory_name)
        return await instance.add_event(step_number, event_type, data, agent_name, task_id, session_id, **kwargs)
    
    async def end_session(self, memory_name: str, session_id: Optional[str] = None):
        """End a memory session (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
        """
        instance = self._get_memory_instance(memory_name)
        return await instance.end_session(session_id)
    
    async def get_session_info(self, memory_name: str, session_id: Optional[str] = None):
        """Get session info (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
            
        Returns:
            SessionInfo or None
        """
        instance = self._get_memory_instance(memory_name)
        return await instance.get_session_info(session_id)
    
    async def clear_session(self, memory_name: str, session_id: Optional[str] = None):
        """Clear a memory session (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
        """
        instance = self._get_memory_instance(memory_name)
        return await instance.clear_session(session_id)
    
    def _get_memory_instance(self, memory_name: str) -> Any:
        """Get memory instance from manager's instance storage.
        
        Args:
            memory_name: Name of the memory system
            
        Returns:
            Memory system instance
            
        Raises:
            ValueError: If memory system or instance not found
        """
        if not hasattr(self, '_memory_instances'):
            raise ValueError(f"Memory system '{memory_name}' instance not found. Please call MemoryManager.get() first to create an instance.")
        
        if memory_name not in self._memory_instances:
            raise ValueError(f"Memory system '{memory_name}' instance not found. Please call MemoryManager.get() first to create an instance.")
        
        return self._memory_instances[memory_name]
    
    def _set_memory_instance(self, memory_name: str, instance: Any):
        """Set memory instance for use by context manager methods.
        
        Args:
            memory_name: Name of the memory system
            instance: Memory system instance
        """
        if not hasattr(self, '_memory_instances'):
            self._memory_instances: Dict[str, Any] = {}
        self._memory_instances[memory_name] = instance
    
    async def get_state(self, memory_name: str, n: Optional[int] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory state (events, summaries, insights) for a memory system.
        
        Args:
            memory_name: Name of the memory system
            n: Number of items to retrieve. If None, returns all items.
            session_id: Optional session ID. If None, uses current session.
            
        Returns:
            Dictionary containing 'events', 'summaries', and 'insights'
        """
        instance = self._get_memory_instance(memory_name)
        
        # Get events, summaries, and insights from memory instance
        events = await instance.get_event(n=n, session_id=session_id)
        summaries = await instance.get_summary(n=n, session_id=session_id)
        insights = await instance.get_insight(n=n, session_id=session_id)
        
        return {
            "events": events,
            "summaries": summaries,
            "insights": insights
        }
