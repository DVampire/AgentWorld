"""Memory Manager

Manager implementation for the Memory Context Protocol.
"""
import os
from typing import Any, Dict, List, Optional, Union, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.utils import assemble_project_path
from src.logger import logger
from src.memory.types import MemoryConfig, Memory
from src.memory.context import MemoryContextManager


class MemoryManager(BaseModel):
    """Memory Manager for managing memory system registration and lifecycle"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the memory systems")
    save_path: str = Field(default=None, description="The path to save the memory systems")
    
    def __init__(self, **kwargs):
        """Initialize the Memory Manager."""
        super().__init__(**kwargs)
        self._registered_memories: Dict[str, MemoryConfig] = {}  # memory_name -> MemoryConfig
        self._memory_instances: Dict[str, Any] = {}  # memory_name -> Memory instance (for __call__)
    
    async def initialize(self, memory_names: Optional[List[str]] = None):
        """Initialize memory systems by discovering and registering them (similar to tool).
        
        Args:
            memory_names: List of memory system names to initialize. If None, initialize all discovered memory systems.
        """
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "memory"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "memory.json")
        logger.info(f"| 📁 Memory Manager base directory: {self.base_dir} and save path: {self.save_path}")
        
        # Initialize memory context manager
        self.memory_context_manager = MemoryContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            auto_discover=True
        )
        await self.memory_context_manager.initialize()
        
        # Auto-discover memory systems if needed
        if self.memory_context_manager.auto_discover:
            await self.memory_context_manager.discover()
            self.memory_context_manager.auto_discover = False
        
        memories_to_init = memory_names if memory_names is not None else list(self.memory_context_manager._memory_configs.keys())
        
        logger.info(f"| 🧠 Registering {len(memories_to_init)} memory systems...")
        
        # Sync registered_memories and merge global config (similar to tool)
        for memory_name in memories_to_init:
            # Get memory config from context manager (discover() registers memories here)
            memory_config = self.memory_context_manager._memory_configs.get(memory_name)
            if memory_config is None:
                # Also check registered_memories for manually registered memories
                memory_config = self._registered_memories.get(memory_name)
                if memory_config is None:
                    logger.warning(f"| ⚠️ Memory {memory_name} not found in registered configs")
                    continue
            
            # Get memory config from global config if available (similar to tool)
            global_config = config.get(memory_name, {})
            if global_config:
                # Merge with existing config
                memory_config.config = {**memory_config.config, **global_config}
            
            # Sync to registered_memories for consistency
            self._registered_memories[memory_name] = memory_config
        
        logger.info(f"| ✅ Memory systems initialization completed ({len(memories_to_init)}/{len(self.memory_context_manager._memory_configs)} memory systems registered)")
    
    async def register(self, memory: Union[Memory, Type[Memory]], *, override: bool = False, **kwargs: Any) -> MemoryConfig:
        """Register a memory system or memory class asynchronously.
        
        Args:
            memory: Memory instance or class to register
            override: Whether to override existing registration
            **kwargs: Configuration for memory initialization
            
        Returns:
            MemoryConfig: Memory configuration
        """
        memory_config = await self.memory_context_manager.register(memory, override=override, **kwargs)
        self._registered_memories[memory_config.name] = memory_config
        return memory_config
    
    async def update(self, memory_name: str, memory: Union[Memory, Type[Memory]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> MemoryConfig:
        """Update an existing memory system with new configuration and create a new version
        
        Args:
            memory_name: Name of the memory system to update
            memory: New memory instance or class with updated content
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for memory initialization
            
        Returns:
            MemoryConfig: Updated memory configuration
        """
        memory_config = await self.memory_context_manager.update(memory_name, memory, new_version, description, **kwargs)
        self._registered_memories[memory_config.name] = memory_config
        return memory_config
    
    async def get_info(self, memory_name: str) -> Optional[MemoryConfig]:
        """Get memory configuration by name
        
        Args:
            memory_name: Memory system name
            
        Returns:
            MemoryConfig: Memory configuration or None if not found
        """
        return await self.memory_context_manager.get_info(memory_name)
    
    async def list(self) -> List[str]:
        """List all registered memory systems
        
        Returns:
            List[str]: List of memory system names
        """
        return await self.memory_context_manager.list()
    
    async def get_class(self, memory_name: str) -> Optional[Type]:
        """Get memory system class by name
        
        Args:
            memory_name: Memory system name
            
        Returns:
            Type: Memory system class or None if not found
        """
        return await self.memory_context_manager.get(memory_name)
    
    async def get(self, memory_name: str) -> Any:
        """Get memory system instance by name (similar to tcp.get()).
        
        Note: Unlike tools, memory systems create a new instance each time since each agent
        needs its own memory system instance to manage its own sessions.
        The instance is also stored for use with __call__ method.
        
        Args:
            memory_name: Memory system name
            
        Returns:
            Memory system instance (new instance each time)
        """
        # Get memory config from context manager (similar to tool)
        memory_config = self.memory_context_manager._memory_configs.get(memory_name)
        if memory_config is None:
            # Also check registered_memories for manually registered memories
            memory_config = self._registered_memories.get(memory_name)
            if memory_config is None:
                available = await self.list()
                raise ValueError(f"Memory system '{memory_name}' not found. Available: {available}")
        
        # Get memory config from global config if available (similar to tool)
        global_config = config.get(memory_name, {})
        if global_config:
            # Merge with existing config
            config_dict = {**memory_config.config.copy(), **global_config}
        else:
            config_dict = memory_config.config.copy()
        
        # Create new instance (each agent needs its own memory system instance)
        logger.debug(f"| 🔧 Creating memory instance: {memory_name}")
        instance = memory_config.cls(**config_dict)
        logger.debug(f"| ✅ Memory instance created: {memory_name}")
        
        # Store instance for context manager methods (overwrites previous instance for this memory_name)
        self._memory_instances[memory_name] = instance
        # Also store in context manager for its methods
        self.memory_context_manager._set_memory_instance(memory_name, instance)
        logger.debug(f"| ✅ Memory instance stored: {memory_name}")
        
        return instance
    
    async def build(self, memory_config: Dict[str, Any]) -> Any:
        """Build a memory system instance from config.
        
        Args:
            memory_config: Memory configuration dictionary (e.g., {"type": "general_memory_system", "model_name": "gpt-4.1"})
            
        Returns:
            Memory system instance
        """
        memory_type = memory_config.get("type", "general_memory_system")
        memory_cls = await self.get_class(memory_type)
        
        if memory_cls is None:
            available = await self.list()
            raise ValueError(f"Memory system '{memory_type}' not found. Available: {available}")
        
        # Remove 'type' from config as it's not a parameter
        config_dict = {k: v for k, v in memory_config.items() if k != "type"}
        
        # Get memory config from global config if available (similar to tool)
        global_config = config.get(memory_type, {})
        if global_config:
            # Merge with provided config (provided config takes precedence)
            config_dict = {**global_config, **config_dict}
        
        # Create instance directly
        instance = memory_cls(**config_dict)
        return instance
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all memory configurations to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        return await self.memory_context_manager.save_to_json(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load memory configurations from JSON
        
        Args:
            file_path: File path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        success = await self.memory_context_manager.load_from_json(file_path)
        if success:
            # Sync registered_memories
            memory_names = await self.memory_context_manager.list()
            for memory_name in memory_names:
                memory_config = await self.memory_context_manager.get_info(memory_name)
                if memory_config:
                    self._registered_memories[memory_name] = memory_config
        return success
    
    async def cleanup(self):
        """Cleanup all memory systems using context manager."""
        if hasattr(self, 'memory_context_manager'):
            await self.memory_context_manager.cleanup()
            
    async def start_session(self, memory_name: str, session_id: str, agent_name: Optional[str] = None,
                           task_id: Optional[str] = None, description: Optional[str] = None) -> str:
        """Start a memory session.
        
        Args:
            memory_name: Name of the memory system
            session_id: Session ID
            agent_name: Optional agent name
            task_id: Optional task ID
            description: Optional description
            
        Returns:
            Session ID
        """
        return await self.memory_context_manager.start_session(memory_name, session_id, agent_name, task_id, description)
    
    async def add_event(self, memory_name: str, step_number: int, event_type: Any, data: Any,
                       agent_name: str, task_id: Optional[str] = None, session_id: Optional[str] = None, **kwargs):
        """Add an event to memory.
        
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
        return await self.memory_context_manager.add_event(memory_name, step_number, event_type, data, agent_name, task_id, session_id, **kwargs)
    
    async def end_session(self, memory_name: str, session_id: Optional[str] = None):
        """End a memory session.
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
        """
        return await self.memory_context_manager.end_session(memory_name, session_id)
    
    async def get_session_info(self, memory_name: str, session_id: Optional[str] = None):
        """Get session info.
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
            
        Returns:
            SessionInfo or None
        """
        return await self.memory_context_manager.get_session_info(memory_name, session_id)
    
    async def clear_session(self, memory_name: str, session_id: Optional[str] = None):
        """Clear a memory session.
        
        Args:
            memory_name: Name of the memory system
            session_id: Optional session ID
        """
        return await self.memory_context_manager.clear_session(memory_name, session_id)
    
    async def get_state(self, name: str, n: Optional[int] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory state (events, summaries, insights) for a memory system.
        
        Args:
            name: Memory system name
            n: Number of items to retrieve. If None, returns all items.
            session_id: Optional session ID. If None, uses current session.
            
        Returns:
            Dictionary containing 'events', 'summaries', and 'insights'
        """
        return await self.memory_context_manager.get_state(name, n, session_id)
    
    async def get_variables(self, memory_name: Optional[str] = None) -> List[Any]:
        """Get variables from memory systems, where each memory's code is used as the variable value.
        
        Args:
            memory_name (Optional[str]): Name of a specific memory system. If None, returns variables for all memory systems.
            
        Returns:
            List[Variable]: List of Variable objects, one for each memory system.
        """
        return await self.memory_context_manager.get_variables(memory_name=memory_name)
    
    async def get_trainable_variables(self, memory_name: Optional[str] = None) -> List[Any]:
        """Get trainable variables from memory systems, filtering out memory systems with require_grad=False.
        
        Args:
            memory_name (Optional[str]): Name of a specific memory system. If None, returns trainable variables for all memory systems.
            
        Returns:
            List[Variable]: List of Variable objects for memory systems with require_grad=True.
        """
        return await self.memory_context_manager.get_trainable_variables(memory_name=memory_name)


# Global Memory Manager instance
memory_manager = MemoryManager()
