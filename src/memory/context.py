"""Memory Context Manager for managing memory lifecycle and resources with lazy loading."""
import os
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union, Tuple, TYPE_CHECKING
from datetime import datetime
import inflection
import json
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from src.logger import logger
from src.config import config
from src.version import version_manager
from src.utils import (assemble_project_path, 
                       gather_with_concurrency,
                       file_lock
                       )
from src.memory.types import MemoryConfig, Memory
from src.dynamic import dynamic_manager
from src.registry import MEMORY_SYSTEM

class MemoryContextManager(BaseModel):
    """Global context manager for all memory systems with lazy loading support."""
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
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "memory"))
        logger.info(f"| 📁 Memory context manager base directory: {self.base_dir}.")    
        os.makedirs(self.base_dir, exist_ok=True)
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "memory.json")
        logger.info(f"| 📁 Memory context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Memory context manager contract path: {self.contract_path}.")

        self._memory_configs: Dict[str, MemoryConfig] = {}  # Current active configs (latest version)
        # Memory version history, e.g., {"memory_name": {"1.0.0": MemoryConfig, "1.0.1": MemoryConfig}}
        self._memory_history_versions: Dict[str, Dict[str, MemoryConfig]] = {}
        
        self._cleanup_registered = False
    
    async def initialize(self, memory_names: Optional[List[str]] = None):
        """Initialize the memory context manager."""
        # Register memory-related symbols for auto-injection in dynamic code
        dynamic_manager.register_symbol("MEMORY_SYSTEM", MEMORY_SYSTEM)
        dynamic_manager.register_symbol("Memory", Memory)
        
        # Register memory context provider for automatic import injection
        def memory_context_provider():
            """Provide memory-related imports for dynamic memory classes."""
            return {
                "MEMORY_SYSTEM": MEMORY_SYSTEM,
                "Memory": Memory,
            }
        dynamic_manager.register_context_provider("memory", memory_context_provider)
        
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
        
        # Register cleanup callback
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True
        
        logger.info(f"| ✅ Memory systems initialization completed")
    
    async def _load_from_registry(self):
        """Load memory systems from MEMORY_SYSTEM registry."""
        
        memory_configs: Dict[str, MemoryConfig] = {}
        
        async def register_memory_class(memory_cls: Type[Memory]):
            """Register a memory class synchronously.
            
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
                    instance=None,
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
    
    async def register(self, 
                       memory: Union[Memory, Type[Memory]],
                       memory_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> MemoryConfig:
        """Register a memory class or instance.
        
        This will:
        - Create (or reuse) a memory instance
        - Create a `MemoryConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager`
        
        Args:
            memory: Memory instance or class
            memory_config_dict: Configuration dict for memory initialization (required when memory is a class)
            override: Whether to override existing registration
            version: Optional version string
            
        Returns:
            MemoryConfig: Memory configuration
        """
        
        try:
            # Handle both instance and class cases
            if isinstance(memory, Memory):
                # Registering an instance
                memory_instance = memory
                memory_cls = type(memory)
                if memory_config_dict:
                    raise ValueError("Extra keyword arguments are not allowed when registering memory instances.")
                memory_config_dict = {}
            else:
                # Registering a class
                memory_cls = memory
                if memory_config_dict is None:
                    # Fallback to global config by class name
                    memory_config_key = inflection.underscore(memory_cls.__name__)
                    memory_config_dict = config.get(memory_config_key, {})
                
                # Instantiate memory immediately (register is a runtime operation)
                try:
                    memory_instance = memory_cls(**memory_config_dict)
                except Exception as e:
                    logger.error(f"| ❌ Failed to create memory instance for {memory_cls.__name__}: {e}")
                    raise ValueError(f"Failed to instantiate memory {memory_cls.__name__} with provided config: {e}")
            
            memory_name = memory_instance.name
            memory_description = memory_instance.description
            memory_metadata = getattr(memory_instance, 'metadata', {})
            # Get require_grad from memory_config_dict if provided, otherwise from memory_instance
            memory_require_grad = memory_config_dict.get("require_grad", memory_instance.require_grad) if memory_config_dict and "require_grad" in memory_config_dict else memory_instance.require_grad
            
            if not memory_name:
                raise ValueError("Memory.name cannot be empty.")
            
            if memory_name in self._memory_configs and not override:
                raise ValueError(f"Memory '{memory_name}' already registered. Use override=True to replace it.")
            
            # Get or generate version from version_manager
            if version is None:
                memory_version = await version_manager.get_version("memory", memory_name)
            else:
                memory_version = version
                
            # Get memory code
            memory_code = dynamic_manager.get_full_module_source(memory_cls)
            if not memory_code:
                logger.warning(f"| ⚠️ Memory {memory_name} source code cannot be extracted")
            
            # --- Build MemoryConfig ---
            memory_config = MemoryConfig(
                name=memory_name,
                description=memory_description,
                require_grad=memory_require_grad,
                version=memory_version,
                cls=memory_cls,
                config=memory_config_dict or {},
                instance=memory_instance if isinstance(memory, Memory) else None,
                metadata=memory_metadata,
                code=memory_code,
            )
            
            # --- Persist current config and history ---
            self._memory_configs[memory_name] = memory_config
            
            # Store in dict-based history (for quick lookup by version)
            if memory_name not in self._memory_history_versions:
                self._memory_history_versions[memory_name] = {}
            self._memory_history_versions[memory_name][memory_config.version] = memory_config
            
            # Register version in version manager
            await version_manager.register_version("memory", memory_name, memory_config.version)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📝 Registered memory config: {memory_name}: {memory_config.version}")
            return memory_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to register memory: {e}")
            raise
    
    async def update(self, 
                     memory_name: str,
                     memory: Union[Memory, Type[Memory]],
                     memory_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> MemoryConfig:
        """Update an existing memory system with new configuration and create a new version
        
        Args:
            memory_name: Name of the memory system to update
            memory: New memory instance or class with updated implementation
            memory_config_dict: Configuration dict for memory initialization
                   If None, will try to get from global config
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            MemoryConfig: Updated memory configuration
        """
        try:
            # Handle both instance and class cases
            if isinstance(memory, Memory):
                # Updating with an instance
                memory_instance = memory
                memory_cls = type(memory)
                if memory_config_dict:
                    raise ValueError("Extra keyword arguments are not allowed when updating with memory instances.")
                memory_config_dict = {}
            else:
                # Updating with a class
                memory_cls = memory
                if memory_config_dict is None:
                    # Fallback to global config by class name
                    memory_config_key = inflection.underscore(memory_cls.__name__)
                    memory_config_dict = config.get(memory_config_key, {})
                
                # Instantiate memory immediately (update is a runtime operation)
                try:
                    memory_instance = memory_cls(**memory_config_dict)
                except Exception as e:
                    logger.error(f"| ❌ Failed to create memory instance for {memory_cls.__name__}: {e}")
                    raise ValueError(f"Failed to instantiate memory {memory_cls.__name__} with provided config: {e}")
            
            # Check if memory exists
            original_config = self._memory_configs.get(memory_name)
            if original_config is None:
                raise ValueError(f"Memory {memory_name} not found. Use register() to register a new memory system.")
            
            memory_description = memory_instance.description
            memory_metadata = getattr(memory_instance, 'metadata', {})
            # Get require_grad from memory_config_dict if provided, otherwise from memory_instance
            memory_require_grad = memory_config_dict.get("require_grad", memory_instance.require_grad) if memory_config_dict and "require_grad" in memory_config_dict else memory_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                # Get current version from version_manager and generate next patch version
                new_version = await version_manager.generate_next_version("memory", memory_name, "patch")
            
            # Get memory code
            memory_code = dynamic_manager.get_full_module_source(memory_cls)
            if not memory_code:
                logger.warning(f"| ⚠️ Memory {memory_name} source code cannot be extracted")
            
            # --- Build MemoryConfig ---
            updated_config = MemoryConfig(
                name=memory_name,  # Keep same name
                description=memory_description,
                require_grad=memory_require_grad,
                version=new_version,
                cls=memory_cls,
                config=memory_config_dict or {},
                instance=memory_instance if isinstance(memory, Memory) else None,
                metadata=memory_metadata,
                code=memory_code,
            )
            
            # Update the memory config (replaces current version)
            self._memory_configs[memory_name] = updated_config
            
            # Store in version history
            if memory_name not in self._memory_history_versions:
                self._memory_history_versions[memory_name] = {}
            self._memory_history_versions[memory_name][updated_config.version] = updated_config
            
            # Register new version record to version manager
            await version_manager.register_version(
                "memory", 
                memory_name, 
                new_version,
                description=description or f"Updated from {original_config.version}"
            )
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 🔄 Updated memory {memory_name} from v{original_config.version} to v{new_version}")
            return updated_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to update memory: {e}")
            raise
    
    async def copy(self, 
                  memory_name: str,
                  new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, 
                  new_config: Optional[Dict[str, Any]] = None) -> MemoryConfig:
        """Copy an existing memory configuration
        
        Args:
            memory_name: Name of the memory system to copy
            new_name: New name for the copied memory. If None, uses original name.
            new_version: New version for the copied memory. If None, increments version.
            new_config: New configuration dict for the copied memory. If None, uses original config.
            
        Returns:
            MemoryConfig: New memory configuration
        """
        try:
            original_config = self._memory_configs.get(memory_name)
            if original_config is None:
                raise ValueError(f"Memory {memory_name} not found")
            
            if original_config.cls is None:
                raise ValueError(f"Cannot copy memory {memory_name}: no class provided")
            
            # Determine new name
            if new_name is None:
                new_name = memory_name
            
            # Prepare config dict (merge original config with new config)
            memory_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # Merge new config into original config
                memory_config_dict.update(new_config)
            
            # Instantiate memory instance (copy is a runtime operation)
            try:
                memory_instance = original_config.cls(**memory_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create memory instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate memory {original_config.cls.__name__} with provided config: {e}")
            
            # Apply name override if provided (after instantiation)
            if new_name != memory_name:
                memory_instance.name = new_name
            
            memory_description = memory_instance.description
            memory_metadata = getattr(memory_instance, 'metadata', {})
            memory_require_grad = memory_config_dict.get("require_grad", memory_instance.require_grad) if memory_config_dict and "require_grad" in memory_config_dict else memory_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                if new_name == memory_name:
                    # If copying with same name, get next version from version_manager
                    new_version = await version_manager.generate_next_version("memory", new_name, "patch")
                else:
                    # If copying with different name, get or generate version for new name
                    new_version = await version_manager.get_version("memory", new_name)
            
            # Get memory code
            memory_code = dynamic_manager.get_full_module_source(original_config.cls)
            if not memory_code:
                logger.warning(f"| ⚠️ Memory {new_name} source code cannot be extracted")
            
            # --- Build MemoryConfig ---
            new_memory_config = MemoryConfig(
                name=new_name,
                description=memory_description,
                require_grad=memory_require_grad,
                version=new_version,
                cls=original_config.cls,
                config=memory_config_dict,
                instance=memory_instance,
                metadata=memory_metadata,
                code=memory_code,
            )
            
            # Register new memory
            self._memory_configs[new_name] = new_memory_config
            
            # Store in version history
            if new_name not in self._memory_history_versions:
                self._memory_history_versions[new_name] = {}
            self._memory_history_versions[new_name][new_version] = new_memory_config
            
            # Register version record to version manager
            await version_manager.register_version(
                "memory", 
                new_name, 
                new_version,
                description=f"Copied from {memory_name}@{original_config.version}"
            )
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📋 Copied memory {memory_name}@{original_config.version} to {new_name}@{new_version}")
            return new_memory_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to copy memory: {e}")
            raise
    
    async def unregister(self, memory_name: str) -> bool:
        """Unregister a memory system
        
        Args:
            memory_name: Name of the memory system to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        if memory_name not in self._memory_configs:
            logger.warning(f"| ⚠️ Memory {memory_name} not found")
            return False
        
        memory_config = self._memory_configs[memory_name]
        
        # Remove from configs
        del self._memory_configs[memory_name]

        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered memory {memory_name}@{memory_config.version}")
        return True
    
    async def get(self, memory_name: str) -> Memory:
        """Get memory configuration by name
        
        Args:
            memory_name: Memory name
            
        Returns:
            Memory: Memory instance or None if not found
        """
        memory_config = self._memory_configs.get(memory_name)
        if memory_config is None:
            return None
        return memory_config.instance if memory_config.instance is not None else None
    
    async def get_info(self, memory_name: str) -> Optional[MemoryConfig]:
        """Get memory info by name
        
        Args:
            memory_name: Memory name
            
        Returns:
            MemoryConfig: Memory info or None if not found
        """
        return self._memory_configs.get(memory_name)
    
    async def list(self) -> List[str]:
        """Get list of registered memory systems
        
        Returns:
            List[str]: List of memory system names
        """
        return [name for name in self._memory_configs.keys()]
    
    async def build(self, memory_config: MemoryConfig) -> MemoryConfig:
        """Create a memory instance and store it.
        
        Args:
            memory_config: Memory configuration
            
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
        """Save the contract for a memory system"""
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
    
    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """Load memory configurations with version history from JSON.
        
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
                logger.warning(f"| ⚠️ Memory file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                memories_data = load_data.get("memory_systems", {})
                loaded_count = 0
                
                for memory_name, memory_data in memories_data.items():
                    try:
                        # Expected format: multiple versions stored as a dict {version_str: config_dict}
                        versions_data = memory_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Memory {memory_name} has invalid format for 'versions' (expected dict), skipping")
                            continue
                        
                        current_version_str = memory_data.get("current_version")
                        
                        # Load all versions
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for version_str, config_dict in versions_data.items():
                            # Ensure version field is present
                            if "version" not in config_dict:
                                config_dict["version"] = version_str
                            
                            try:
                                memory_config = MemoryConfig.model_validate(config_dict)
                                version_configs.append(memory_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load memory config for {memory_name}@{version_str}: {e}")
                                continue
                            
                            # Track latest version
                            if latest_config is None or (
                                current_version_str and memory_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    version_manager.compare_versions(memory_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = memory_config
                                latest_version = memory_config.version
                        
                        # Store all versions in history (dict-based)
                        self._memory_history_versions[memory_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }
                        
                        # Only set latest version as active
                        if latest_config:
                            self._memory_configs[memory_name] = latest_config
                            
                            # Register all versions to version manager (only version records)
                            for memory_config in version_configs:
                                await version_manager.register_version("memory", memory_name, memory_config.version)
                            
                            # Create instance if requested (instance is not saved in JSON, must be created via build)
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load memory {memory_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} memory systems with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load memory systems from {file_path}: {e}")
                return False
    
    async def restore(self, memory_name: str, version: str, auto_initialize: bool = True) -> Optional[MemoryConfig]:
        """Restore a specific version of a memory system from history
        
        Args:
            memory_name: Name of the memory system
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored memory
            
        Returns:
            MemoryConfig of the restored version, or None if not found
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if memory_name in self._memory_history_versions:
            version_config = self._memory_history_versions[memory_name].get(version)
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for memory {memory_name}")
            return None
        
        # Create a copy to avoid modifying the history
        restored_config = MemoryConfig(**version_config.model_dump())
        
        # Set as current active config
        self._memory_configs[memory_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("memory", memory_name)
        if version_history:
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("memory", memory_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("memory", memory_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored memory {memory_name} to version {version}")
        return restored_config
    
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
            # Get memory code
            memory_code = memory_config.code or ""
            
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
    
    async def set_variables(self, memory_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> MemoryConfig:
        """Set variable values in a memory system and create a new version.
        
        Args:
            memory_name: Name of the memory system to update
            variable_updates: Dictionary mapping variable names to new values.
                For memory systems, this is typically {"name": "memory_name", "variables": "memory code"}
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            MemoryConfig: Updated memory configuration
        """
        original_config = self._memory_configs.get(memory_name)
        if original_config is None:
            raise ValueError(f"Memory {memory_name} not found. Use register() to register a new memory system.")
        
        # For memory systems, variable_updates format is {"name": "memory_name", "variables": "memory code"}
        # Extract the new code from "variables" field
        if "variables" not in variable_updates:
            raise ValueError(f"variable_updates must contain 'variables' field with memory code, got: {list(variable_updates.keys())}")
        
        new_code = variable_updates["variables"]
        if not isinstance(new_code, str):
            raise ValueError(f"Memory code must be a string, got {type(new_code)}")
        
        # Load memory class from code
        class_name = dynamic_manager.extract_class_name_from_code(new_code)
        if not class_name:
            raise ValueError(f"Cannot extract class name from code")
        
        try:
            memory_cls = dynamic_manager.load_class(
                new_code,
                class_name=class_name,
                base_class=Memory,
                context="memory"
            )
        except Exception as e:
            logger.error(f"| ❌ Failed to load memory class from code: {e}")
            raise ValueError(f"Failed to load memory class from code: {e}")
        
        # Use update() function to handle version management and persistence
        update_description = description or f"Updated code for {memory_name}"
        return await self.update(
            memory_name=memory_name,
            memory=memory_cls,
            memory_config_dict=original_config.config,
            new_version=new_version,
            description=update_description
        )
    
    async def cleanup(self):
        """Cleanup all active memory systems."""
        try:
            # Clear all memory configs and version history
            self._memory_configs.clear()
            self._memory_history_versions.clear()
                
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
    
    async def add_event(self, 
                        memory_name: str,
                        step_number: int, 
                        event_type: Any, 
                        data: Any,
                        agent_name: str, 
                        task_id: Optional[str] = None, 
                        session_id: Optional[str] = None,
                        **kwargs):
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
