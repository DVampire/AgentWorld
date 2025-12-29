"""Memory Context Manager for managing memory lifecycle and resources."""

import atexit
import inflection
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, List, Union, Type, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import assemble_project_path
from src.utils.file_utils import file_lock
from src.memory.types import MemoryConfig, Memory
from src.dynamic import dynamic_manager
from src.registry import MEMORY_SYSTEM
from src.utils import gather_with_concurrency

class MemoryContextManager(BaseModel):
    """Global context manager for all memory systems."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the memory systems")
    save_path: str = Field(default=None, description="The path to save the memory systems")
    contract_path: str = Field(default=None, description="The path to save the memory contract")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 contract_path: Optional[str] = None,
                 **kwargs):
        """Initialize the memory context manager.
        
        Args:
            base_dir: Base directory for storing memory data
            save_path: Path to save memory configurations
            contract_path: Path to save memory contract
        """
        super().__init__(**kwargs)
        
        # Set up paths
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "memory"))
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| 📁 Memory context manager base directory: {self.base_dir}")
        
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "memory.json")
        logger.info(f"| 📁 Memory context manager save path: {self.save_path}")
        
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Memory context manager contract path: {self.contract_path}")
        
        self._memory_configs: Dict[str, MemoryConfig] = {}  # Current active configs (latest version)
        # Memory version history, e.g., {"memory_name": {"1.0.0": MemoryConfig, "1.0.1": MemoryConfig}}
        self._memory_history_versions: Dict[str, Dict[str, MemoryConfig]] = {}
        self._cleanup_registered = False
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
    
    async def initialize(self, memory_names: Optional[List[str]] = None):
        """Initialize the memory context manager."""
        # Load memory systems from MEMORY_SYSTEM registry
        memory_configs = {}
        registry_memory_configs: Dict[str, MemoryConfig] = await self._load_from_registry()
        memory_configs.update(registry_memory_configs)
        
        # Load memory systems from code (JSON file)
        code_memory_configs: Dict[str, MemoryConfig] = await self._load_from_code()
        
        # Merge code configs with registry configs, only override if code version is strictly greater
        for memory_name, code_config in code_memory_configs.items():
            if memory_name in memory_configs:
                registry_config = memory_configs[memory_name]
                # Compare versions: only override if code version is strictly greater
                if version_manager.compare_versions(code_config.version, registry_config.version) > 0:
                    logger.info(f"| 🔄 Overriding memory {memory_name} from registry (v{registry_config.version}) with code version (v{code_config.version})")
                    memory_configs[memory_name] = code_config
                else:
                    logger.info(f"| 📌 Keeping memory {memory_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater")
                    # If versions are equal, update the history with registry config (which has real class, not dynamic)
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        # Replace the code config in history with registry config to preserve real class reference
                        if memory_name in self._memory_history_versions:
                            self._memory_history_versions[memory_name][registry_config.version] = registry_config
            else:
                # New memory from code, add it
                memory_configs[memory_name] = code_config
        
        # Filter memory systems by names if provided
        if memory_names is not None:
            memory_configs = {name: memory_configs[name] for name in memory_names if name in memory_configs}
        
        # Build all memory systems concurrently with a concurrency limit
        memory_names_list = list(memory_configs.keys())
        tasks = [
            self.build(memory_configs[name]) for name in memory_names_list
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for memory_name, result in zip(memory_names_list, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize memory {memory_name}: {result}")
                continue
            self._memory_configs[memory_name] = result
            logger.info(f"| 🔧 Memory {memory_name} initialized")
        
        # Save memory configs to json file
        await self.save_to_json()
        # Save contract to file
        await self.save_contract(memory_names=memory_names_list)
        
        logger.info(f"| ✅ Memory systems initialization completed")
    
    async def _load_from_registry(self):
        """Load memory systems from MEMORY_SYSTEM registry."""
        
        memory_configs: Dict[str, MemoryConfig] = {}
        
        async def register_memory_class(memory_cls: Type[Memory]):
            """Register a memory class.
            
            Args:
                memory_cls: Memory class to register
            """
            try:
                # Get memory config from global config
                memory_config_key = inflection.underscore(memory_cls.__name__)
                memory_config_dict = config.get(memory_config_key, {})
                memory_require_grad = memory_config_dict.get("require_grad", False) if memory_config_dict and "require_grad" in memory_config_dict else False
                
                # Create temporary instance to get name and description
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
                            memory_description = memory_cls.__doc__ or ""
                
                # Get or generate version from version_manager
                memory_version = await version_manager.get_version("memory", memory_name)
                
                # Get full module source code
                memory_code = dynamic_manager.get_full_module_source(memory_cls)
                
                # Create memory config
                memory_config = MemoryConfig(
                    name=memory_name,
                    description=memory_description,
                    require_grad=memory_require_grad,
                    version=memory_version,
                    cls=memory_cls,
                    config=memory_config_dict,
                    instance=None,  # Instance will be built when needed
                    metadata={},
                    code=memory_code,
                )
                
                # Store memory config
                memory_configs[memory_name] = memory_config
                
                # Store in version history (by version string)
                if memory_name not in self._memory_history_versions:
                    self._memory_history_versions[memory_name] = {}
                self._memory_history_versions[memory_name][memory_version] = memory_config
                
                # Register version to version manager
                await version_manager.register_version("memory", memory_name, memory_version)
                
                logger.info(f"| 📝 Registered memory: {memory_name} ({memory_cls.__name__})")
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register memory class {memory_cls.__name__}: {e}")
                raise
        
        import src.memory  # noqa: F401
        
        # Get all registered memory classes from MEMORY_SYSTEM registry
        memory_classes = list(MEMORY_SYSTEM._module_dict.values())
        
        logger.info(f"| 🔍 Discovering {len(memory_classes)} memory systems from MEMORY_SYSTEM registry")
        
        # Register each memory class concurrently with a concurrency limit
        tasks = [
            register_memory_class(memory_cls) for memory_cls in memory_classes
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(memory_classes)} memory systems from MEMORY_SYSTEM registry")
        
        return memory_configs
    
    async def _load_from_code(self):
        """Load memory systems from JSON file.
        
        JSON file content example:
        {
            "metadata": {
                "saved_at": str,  # "YYYY-MM-DD HH:MM:SS"
                "num_memories": int,  # total memory count
                "num_versions": int  # total version count
            },
            "memory_systems": {
                "memory_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": {
                            "name": str,
                            "description": str,
                            "require_grad": bool,
                            "version": str,
                            "cls": Type[Memory],
                            "config": dict,
                            "metadata": dict,
                            "code": str
                        },
                        ...
                    }
                }
            }
        }
        """
        memory_configs: Dict[str, MemoryConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Memory config file not found at {self.save_path}, skipping code-based loading")
            return memory_configs
        
        # Load all memory configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse memory config JSON from {self.save_path}: {e}")
            return memory_configs
        
        metadata = load_data.get("metadata", {})
        memories_data = load_data.get("memory_systems", {})
        
        async def register_memory_class(memory_name: str, memory_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, MemoryConfig], Optional[MemoryConfig]]]:
            """Load all versions for a single memory from JSON."""
            try:
                current_version = memory_data.get("current_version", "1.0.0")
                versions = memory_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Memory {memory_name} has no versions")
                    return None
                
                version_map: Dict[str, MemoryConfig] = {}
                current_config: Optional[MemoryConfig] = None  # Active config for current_version
                
                for _, version_data in versions.items():
                    # Create MemoryConfig using model_validate to handle cls and code
                    memory_config = MemoryConfig.model_validate(version_data)
                    version = memory_config.version
                    version_map[version] = memory_config
                    
                    if version == current_version:
                        current_config = memory_config
                
                return memory_name, version_map, current_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load memory {memory_name} from code JSON: {e}")
                return None
        
        # Launch loading of each memory concurrently with a concurrency limit
        tasks = [
            register_memory_class(memory_name, memory_data) for memory_name, memory_data in memories_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            memory_name, version_map, current_config = result
            if not version_map:
                continue
            
            # Store all versions in history (mapped by version string)
            self._memory_history_versions[memory_name] = version_map
            # Active config: the one corresponding to current_version
            if current_config is not None:
                memory_configs[memory_name] = current_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Memory {memory_name} current_version not found, using last available version")
                memory_configs[memory_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for memory_config in version_map.values():
                await version_manager.register_version("memory", memory_name, memory_config.version)
        
        logger.info(f"| 📂 Loaded {len(memory_configs)} memory systems from {self.save_path}")
        return memory_configs
    
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
        
        # Get require_grad from instance or config
        if isinstance(memory, Memory):
            memory_require_grad = memory.require_grad
        else:
            memory_require_grad = memory_config_dict.get("require_grad", False)
        
        # Determine if we're registering a class or instance
        if isinstance(memory, Memory):
            # Registering an instance
            memory_config = MemoryConfig(
                name=memory_name,
                description=memory_description,
                require_grad=memory_require_grad,
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
                require_grad=memory_require_grad,
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
        
        # Get require_grad from kwargs or instance
        if isinstance(memory, Memory):
            memory_require_grad = memory.require_grad
        else:
            memory_require_grad = kwargs.get("require_grad", temp_instance.require_grad)
        
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
                require_grad=memory_require_grad,
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
                require_grad=memory_require_grad,
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
    
    async def get(self, name: str) -> Memory:
        """Get a memory system class by name
        
        Args:
            name: Name of the memory system
        """
        memory_config = self._memory_configs.get(name)
        if memory_config is None:
            return None
        return memory_config.instance if memory_config.instance is not None else None
    
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
    
    async def build(self, memory_config: MemoryConfig, memory_factory: Optional[callable] = None) -> MemoryConfig:
        """Build a memory system instance from config.
        
        Args:
            memory_config: Memory configuration
            memory_factory: Optional factory function to create memory instance
            
        Returns:
            MemoryConfig: Memory configuration with instance
        """
        if memory_config.name in self._memory_configs:
            existing_config = self._memory_configs[memory_config.name]
            if existing_config.instance is not None:
                return existing_config
        
        # Create new memory instance
        try:
            # cls should already be loaded (either from registry or from code in _load_from_code)
            if memory_config.cls is None:
                raise ValueError(f"Cannot create memory {memory_config.name}: no class provided. Class should be loaded during initialization.")
            
            # Instantiate memory instance
            memory_instance = memory_config.cls(**memory_config.config) if memory_config.config else memory_config.cls()
            
            # Initialize memory if it has an initialize method
            if hasattr(memory_instance, "initialize"):
                await memory_instance.initialize()
            
            memory_config.instance = memory_instance
            
            # Store memory metadata
            self._memory_configs[memory_config.name] = memory_config
            
            logger.info(f"| 🔧 Memory {memory_config.name} created and stored")
            
            return memory_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create memory {memory_config.name}: {e}")
            raise
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all memory configurations with version history to JSON.
        
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
            
            # Prepare save data - save all versions for each memory
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_memories": len(self._memory_configs),
                    "num_versions": sum(len(versions) for versions in self._memory_history_versions.values()),
                },
                "memory_systems": {}
            }
            
            for memory_name, version_map in self._memory_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, memory_config in version_map.items():
                        config_dict = memory_config.model_dump()
                        versions_data[memory_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if memory_name in self._memory_configs:
                        current_config = self._memory_configs[memory_name]
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
                    
                    save_data["memory_systems"][memory_name] = {
                        "versions": versions_data,
                        "current_version": current_version
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize memory {memory_name}: {e}")
                    continue
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved {len(self._memory_configs)} memory systems with version history to {file_path}")
            return str(file_path)
    
    async def save_contract(self, memory_names: Optional[List[str]] = None):
        """Save the contract for memory systems
        
        Args:
            memory_names: Optional list of memory names to include in contract. If None, includes all.
        """
        contract = []
        if memory_names is not None:
            for index, memory_name in enumerate(memory_names):
                memory_info = await self.get_info(memory_name)
                if memory_info:
                    text = f"Name: {memory_info.name}\nDescription: {memory_info.description}\nRequire Grad: {memory_info.require_grad}"
                    contract.append(f"{index + 1:04d}\n{text}\n")
        else:
            for index, memory_name in enumerate(self._memory_configs.keys()):
                memory_info = await self.get_info(memory_name)
                if memory_info:
                    text = f"Name: {memory_info.name}\nDescription: {memory_info.description}\nRequire Grad: {memory_info.require_grad}"
                    contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} memory systems contract to {self.contract_path}")
        
    async def load_contract(self) -> str:
        """Load the contract for memory systems
        
        Returns:
            str: Contract text
        """
        if not os.path.exists(self.contract_path):
            return ""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
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
    
    async def get_variables(self, memory_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get variables from memory systems, where each memory's code is used as the variable value.
        
        Args:
            memory_name (Optional[str]): Name of a specific memory system. If None, returns variables for all memory systems.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping memory names to Variable objects. Each Variable has:
                - name: memory name
                - type: "memory_code"
                - description: memory description
                - require_grad: memory's require_grad value
                - variables: memory's code (as string value)
        """
        # Lazy import to avoid circular dependency
        from src.optimizer.types import Variable
        
        variables: Dict[str, Variable] = {}
        
        if memory_name is not None:
            # Get specific memory
            memory_config = self._memory_configs.get(memory_name)
            if memory_config is None:
                logger.warning(f"| ⚠️ Memory {memory_name} not found")
                return variables
            
            memory_configs = {memory_name: memory_config}
        else:
            # Get all memory systems
            memory_configs = self._memory_configs
        
        for name, memory_config in memory_configs.items():
            # Get memory code from class
            memory_code = ""
            if memory_config.cls is not None:
                memory_code = dynamic_manager.get_full_module_source(memory_config.cls) or ""
            
            # Create Variable for this memory system
            variable = Variable(
                name=name,
                type="memory_code",
                description=memory_config.description or f"Code for memory system {name}",
                require_grad=memory_config.require_grad,
                template=None,
                variables=memory_code  # Store code as the variable value
            )
            variables[name] = variable
        
        return variables
    
    async def get_trainable_variables(self, memory_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get trainable variables from memory systems, filtering out memory systems with require_grad=False.
        
        Only returns variables for memory systems where require_grad=True.
        
        Args:
            memory_name (Optional[str]): Name of a specific memory system. If None, returns trainable variables for all memory systems.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping memory names to Variable objects for memory systems with require_grad=True.
                                Each Variable has:
                - name: memory name
                - type: "memory_code"
                - description: memory description
                - require_grad: True
                - variables: memory's code (as string value)
        """
        # Get all variables first
        all_variables = await self.get_variables(memory_name=memory_name)
        
        # Filter to only include variables with require_grad=True
        trainable_variables = {
            name: variable for name, variable in all_variables.items()
            if variable.require_grad is True
        }
        
        return trainable_variables
    
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
                            description: Optional[str] = None,
                            **kwargs) -> str:
        """Start a memory session (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Session ID
            agent_name: Optional agent name
            task_id: Optional task ID
            description: Optional description
            **kwargs: Additional arguments to pass to memory system's start_session
            
        Returns:
            Session ID
        """
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.start_session(session_id, agent_name, task_id, description, **kwargs)
    
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
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.add_event(step_number, event_type, data, agent_name, task_id, session_id, **kwargs)
    
    async def end_session(self, memory_name: str, session_id: Optional[str] = None):
        """End a memory session (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
        """
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.end_session(session_id)
    
    async def get_session_info(self, memory_name: str, session_id: Optional[str] = None):
        """Get session info (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
            
        Returns:
            SessionInfo or None
        """
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.get_session_info(session_id)
    
    async def clear_session(self, memory_name: str, session_id: Optional[str] = None):
        """Clear a memory session (delegates to memory system instance).
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
        """
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
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
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        
        # Get events, summaries, and insights from memory instance
        events = await instance.get_event(n=n, session_id=session_id)
        summaries = await instance.get_summary(n=n, session_id=session_id)
        insights = await instance.get_insight(n=n, session_id=session_id)
        
        return {
            "events": events,
            "summaries": summaries,
            "insights": insights
        }
