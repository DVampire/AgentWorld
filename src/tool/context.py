"""Tool Context Manager for managing tool lifecycle and resources with lazy loading."""
import os
import asyncio
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
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.utils import (assemble_project_path, 
                       gather_with_concurrency,
                       file_lock
                       )
from src.tool.types import Tool, ToolConfig, ToolResponse
from src.session import SessionContext
from src.version import version_manager
from src.dynamic import dynamic_manager
from src.registry import TOOL

class ToolContextManager(BaseModel):
    """Global context manager for all tools with lazy loading support."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    contract_path: str = Field(default=None, description="The path to save the tool contract")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 contract_path: Optional[str] = None,
                 model_name: str = "openrouter/gemini-3-flash-preview",
                 embedding_model_name: str = "openrouter/text-embedding-3-large",
                 default_timeout: Optional[float] = 1800.0,
                 **kwargs):
        """Initialize the tool context manager.
        
        Args:
            base_dir: Base directory for storing tool data
            save_path: Path to save tool configurations
            model_name: The model to use for the tools
            embedding_model_name: The model to use for the tool embeddings
            default_timeout: Default timeout in seconds for tool calls (None means no timeout, default 1800s = 30 minutes)
        """
        super().__init__(**kwargs)
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "tool"))
        logger.info(f"| 📁 Tool context manager base directory: {self.base_dir}.")    
        os.makedirs(self.base_dir, exist_ok=True)
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "tool.json")
        logger.info(f"| 📁 Tool context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Tool context manager contract path: {self.contract_path}.")

        self._tool_configs: Dict[str, ToolConfig] = {}  # Current active configs (latest version)
        # Tool version history, e.g., {"tool_name": {"1.0.0": ToolConfig, "1.0.1": ToolConfig}}
        self._tool_history_versions: Dict[str, Dict[str, ToolConfig]] = {}
        
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.default_timeout = default_timeout
        
        self._cleanup_registered = False
        self._faiss_service = None
        self._variables_lock = asyncio.Lock()  # Lock for get/set trainable variables
        
    async def initialize(self, tool_names: Optional[List[str]] = None):
        """Initialize the tool context manager."""
        
        # Register tool-related symbols for auto-injection in dynamic code
        dynamic_manager.register_symbol("TOOL", TOOL)
        dynamic_manager.register_symbol("Tool", Tool)
        dynamic_manager.register_symbol("ToolResponse", ToolResponse)
        
        # Register tool context provider for automatic import injection
        def tool_context_provider():
            """Provide tool-related imports for dynamic tool classes."""
            return {
                "TOOL": TOOL,
                "Tool": Tool,
                "ToolResponse": ToolResponse,
            }
        dynamic_manager.register_context_provider("tool", tool_context_provider)
        
        # Initialize Faiss service for tool embedding
        self._faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name
        )
        
        # Load tools from TOOL registry
        tool_configs = {}
        registry_tool_configs: Dict[str, ToolConfig] = await self._load_from_registry()
        tool_configs.update(registry_tool_configs)
        
        # Load tools from code
        code_tool_configs: Dict[str, ToolConfig] = await self._load_from_code()
        
        # Merge code configs with registry configs, only override if code version is strictly greater
        for tool_name, code_config in code_tool_configs.items():
            if tool_name in tool_configs:
                registry_config = tool_configs[tool_name]
                # Compare versions: only override if code version is strictly greater
                if version_manager.compare_versions(code_config.version, registry_config.version) > 0:
                    logger.info(f"| 🔄 Overriding tool {tool_name} from registry (v{registry_config.version}) with code version (v{code_config.version})")
                    tool_configs[tool_name] = code_config
                else:
                    logger.info(f"| 📌 Keeping tool {tool_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater")
                    # If versions are equal, update the history with registry config (which has real class, not dynamic)
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        # Replace the code config in history with registry config to preserve real class reference
                        if tool_name in self._tool_history_versions:
                            self._tool_history_versions[tool_name][registry_config.version] = registry_config
            else:
                # New tool from code, add it
                tool_configs[tool_name] = code_config
        
        # Filter tools by names if provided
        if tool_names is not None:
            tool_configs = {name: tool_configs[name] for name in tool_names}
        
        # Build all tools concurrently with a concurrency limit
        tool_names = list(tool_configs.keys())
        tasks = [
            self.build(tool_configs[name]) for name in tool_names
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for tool_name, result in zip(tool_names, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize tool {tool_name}: {result}")
                continue
            self._tool_configs[tool_name] = result
            logger.info(f"| 🔧 Tool {tool_name} initialized")
        
        # Save tool configs to json file
        await self.save_to_json()
        # Save contract to file
        await self.save_contract(tool_names=tool_names)
        
        # Register cleanup callback
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True
        
        logger.info(f"| ✅ Tools initialization completed")
        
    async def _load_from_registry(self):
        """Load tools from TOOL registry."""
        
        tool_configs: Dict[str, ToolConfig] = {}
        
        async def register_tool_class(tool_cls: Type[Tool]):
            """Register a tool class synchronously.
            
            Args:
                tool_cls: Tool class to register
            """
            try:
                # Get tool config from global config
                tool_config_key = inflection.underscore(tool_cls.__name__)
                tool_config_dict = config.get(tool_config_key, {})
                tool_require_grad = tool_config_dict.get("require_grad", False) if tool_config_dict and "require_grad" in tool_config_dict else False
                
                # Get tool properties from tool class
                tool_name = tool_cls.model_fields['name'].default
                tool_description = tool_cls.model_fields['description'].default
                tool_metadata = tool_cls.model_fields['metadata'].default
                
                # Get or generate version from version_manager
                tool_version = await version_manager.get_version("tool", tool_name)
                
                # Get full module source code
                tool_code = dynamic_manager.get_full_module_source(tool_cls)
                
                tool_parameters = dynamic_manager.get_parameters(tool_cls)
                tool_function_calling = dynamic_manager.build_function_calling(tool_name, tool_description, tool_parameters)
                tool_text = dynamic_manager.build_text_representation(tool_name, tool_description, tool_parameters)
                tool_args_schema = dynamic_manager.build_args_schema(tool_name, tool_parameters)
                
                # Create tool config (ToolConfig.id is auto-incremented internally if needed)
                tool_config = ToolConfig(
                    name=tool_name,
                    description=tool_description,
                    version=tool_version,
                    cls=tool_cls,
                    config=tool_config_dict,
                    instance=None,
                    function_calling=tool_function_calling,
                    text=tool_text,
                    args_schema=tool_args_schema,
                    metadata=tool_metadata,
                    require_grad=tool_require_grad,
                    code=tool_code,
                )
                
                # Store tool config
                tool_configs[tool_name] = tool_config
                
                # Store in version history (by version string)
                if tool_name not in self._tool_history_versions:
                    self._tool_history_versions[tool_name] = {}
                self._tool_history_versions[tool_name][tool_version] = tool_config
                
                # Register version to version manager
                await version_manager.register_version("tool", tool_name, tool_version)
                
                logger.info(f"| 📝 Registered tool: {tool_name} ({tool_cls.__name__})")
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register tool class {tool_cls.__name__}: {e}")
                raise
            
        import src.tool  # noqa: F401
        
        # Get all registered tool classes from TOOL registry
        tool_classes = list(TOOL._module_dict.values())
        
        logger.info(f"| 🔍 Discovering {len(tool_classes)} tools from TOOL registry")
        
        # Register each tool class concurrently with a concurrency limit
        tasks = [
            register_tool_class(tool_cls) for tool_cls in tool_classes
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"| ✅ Discovered and registered {success_count}/{len(tool_classes)} tools from TOOL registry")
        
        return tool_configs
    
    async def _load_from_code(self):
        """Load tools from code files.
        
        JSON file content example:
        {
            "metadata": {
                "saved_at": str,  # "YYYY-MM-DD HH:MM:SS"
                "num_tools": int,  # total tool count
                "num_versions": int  # total version count
            },
            "tools": {
                "tool_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": {
                            "name": str,
                            "description": str,
                            "metadata": dict,
                            "require_grad": bool,
                            "version": str,
                            "cls": Type[Tool],
                            "config": dict,
                            "instance": Tool, # will be built when needed
                            "function_calling": dict, 
                            "text": str, 
                            "args_schema": BaseModel,
                            "code": str
                        },
                        ...
                    }
                }
            }
        }
        """
        
        tool_configs: Dict[str, ToolConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Tool config file not found at {self.save_path}, skipping code-based loading")
            return tool_configs
        
        # Load all tool configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse tool config JSON from {self.save_path}: {e}")
            return tool_configs
        
        metadata = load_data.get("metadata", {})
        tools_data = load_data.get("tools", {})

        async def register_tool_class(tool_name: str, tool_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, ToolConfig], Optional[ToolConfig]]]:
            """Load all versions for a single tool from JSON."""
            try:
                current_version = tool_data.get("current_version", "1.0.0")
                versions = tool_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Tool {tool_name} has no versions")
                    return None
                
                version_map: Dict[str, ToolConfig] = {}
                current_tool_config: Optional[ToolConfig] = None
                
                for _, version_data in versions.items():
                    tool_config = ToolConfig.model_validate(version_data)
                    version = tool_config.version
                    version_map[version] = tool_config
                    
                    if version == current_version:
                        current_tool_config = tool_config
                
                return tool_name, version_map, current_tool_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load tool {tool_name} from code JSON: {e}")
                return None

        # Launch loading of each tool concurrently with a concurrency limit
        tasks = [
            register_tool_class(tool_name, tool_data) for tool_name, tool_data in tools_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            tool_name, version_map, current_tool_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._tool_history_versions[tool_name] = version_map
            # Active config: the one corresponding to current_version
            if current_tool_config is not None:
                tool_configs[tool_name] = current_tool_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Tool {tool_name} current_version not found, using last available version")
                tool_configs[tool_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for tool_config in version_map.values():
                await version_manager.register_version("tool", tool_name, tool_config.version)
            
        logger.info(f"| 📂 Loaded {len(tool_configs)} tools from {self.save_path}")
        return tool_configs
    
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
            if existing_config.instance is not None:
                return existing_config
        
        # Create new tool instance
        try:
            # cls should already be loaded (either from registry or from code in _load_from_code)
            if tool_config.cls is None:
                raise ValueError(f"Cannot create tool {tool_config.name}: no class provided. Class should be loaded during initialization.")
            
            # Instantiate tool instance
            tool_instance = tool_config.cls(**tool_config.config) if tool_config.config else tool_config.cls()
            
            # Initialize tool if it has an initialize method
            if hasattr(tool_instance, "initialize"):
                await tool_instance.initialize()
            
            tool_config.instance = tool_instance
            
            # Store tool metadata
            self._tool_configs[tool_config.name] = tool_config
            
            logger.info(f"| 🔧 Tool {tool_config.name} created and stored")
            
            return tool_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create tool {tool_config.name}: {e}")
            raise
    
    async def register(self, 
                       tool_cls: Type[Tool],
                       tool_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None,
                       code: Optional[str] = None) -> ToolConfig:
        """Register a tool class or instance.
        
        This will:
        - Create (or reuse) a tool instance
        - Create a `ToolConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager` and FAISS index
        - Persist the tool source code (if available / provided)
        """
        
        try:
            if tool_config_dict is None:
                # Fallback to global config by class name
                tool_config_key = inflection.underscore(tool_cls.__name__)
                tool_config_dict = config.get(tool_config_key, {})
            
            # Instantiate tool immediately (register is a runtime operation)
            try:
                tool_instance = tool_cls(**tool_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create tool instance for {tool_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate tool {tool_cls.__name__} with provided config: {e}")
            
            tool_name = tool_instance.name
            tool_description = tool_instance.description
            tool_metadata = tool_instance.metadata
            # Get require_grad from tool_config_dict if provided, otherwise from tool_instance
            tool_require_grad = tool_config_dict.get("require_grad", tool_instance.require_grad) if tool_config_dict and "require_grad" in tool_config_dict else tool_instance.require_grad
            
            # Get or generate version from version_manager
            if version is None:
                tool_version = await version_manager.get_version("tool", tool_name)
            else:
                tool_version = version
                
            # Get tool code (prefer explicit code if provided)
            tool_code = code if code is not None else dynamic_manager.get_source_code(tool_cls)
            if not tool_code:
                logger.warning(f"| ⚠️ Tool {tool_name} is dynamic but source code cannot be extracted (and no code was provided)")
            
            # Get tool parameters
            tool_parameters = dynamic_manager.get_parameters(tool_cls)
            tool_function_calling = dynamic_manager.build_function_calling(tool_name, tool_description, tool_parameters)
            tool_text = dynamic_manager.build_text_representation(tool_name, tool_description, tool_parameters)
            tool_args_schema = dynamic_manager.build_args_schema(tool_name, tool_parameters)
            
            # --- Build ToolConfig ---
            tool_config = ToolConfig(
                name=tool_name,
                description=tool_description,
                metadata=tool_metadata,
                require_grad=tool_require_grad,
                version=tool_version,
                cls=tool_cls,
                config=tool_config_dict or {},
                instance=tool_instance,
                function_calling=tool_function_calling,
                text=tool_text,
                args_schema=tool_args_schema,
                code=tool_code,
            )
            
            # --- Persist current config and history ---
            self._tool_configs[tool_name] = tool_config
            
            # Store in dict-based history (for quick lookup by version)
            if tool_name not in self._tool_history_versions:
                self._tool_history_versions[tool_name] = {}
            self._tool_history_versions[tool_name][tool_config.version] = tool_config
            
            # Register version in version manager
            await version_manager.register_version("tool", tool_name, tool_config.version)
            
            # Add to FAISS index
            await self._store(tool_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📝 Registered tool config: {tool_name}: {tool_config.version}")
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
    
    async def get_info(self, tool_name: str) -> Optional[ToolConfig]:
        """Get tool info by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolConfig: Tool info or None if not found
        """
        return self._tool_configs.get(tool_name)
    
    async def list(self) -> List[str]:
        """Get list of registered tools
        
        Returns:
            List[str]: List of tool names
        """
        return [name for name in self._tool_configs.keys()]
    
    async def update(self, 
                     tool_cls: Type[Tool],
                     tool_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None,
                     code: Optional[str] = None) -> ToolConfig:
        """Update an existing tool with new configuration and create a new version
        
        Args:
            tool_cls: New tool class with updated implementation
            tool_config_dict: Configuration dict for tool initialization
                   If None, will try to get from global config
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            code: Optional source code string. If provided, uses this instead of extracting from tool_cls.
                  This is useful when tool_cls is dynamically created from code string.
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        try:
            if tool_config_dict is None:
                # Fallback to global config by class name
                tool_config_key = inflection.underscore(tool_cls.__name__)
                tool_config_dict = config.get(tool_config_key, {})
            
            # Instantiate tool immediately (update is a runtime operation)
            try:
                tool_instance = tool_cls(**tool_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create tool instance for {tool_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate tool {tool_cls.__name__} with provided config: {e}")
            
            tool_name = tool_instance.name
            
            # Check if tool exists
            original_config = self._tool_configs.get(tool_name)
            if original_config is None:
                raise ValueError(f"Tool {tool_name} not found. Use register() to register a new tool.")
            
            tool_description = tool_instance.description
            tool_metadata = tool_instance.metadata
            # Get require_grad from tool_config_dict if provided, otherwise from tool_instance
            tool_require_grad = tool_config_dict.get("require_grad", tool_instance.require_grad) if tool_config_dict and "require_grad" in tool_config_dict else tool_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                # Get current version from version_manager and generate next patch version
                new_version = await version_manager.generate_next_version("tool", tool_name, "patch")
            
            # Get tool code - use provided code if available (for dynamically created classes)
            if code is not None:
                tool_code = code
            else:
                tool_code = dynamic_manager.get_source_code(tool_cls)
                if not tool_code:
                    logger.warning(f"| ⚠️ Tool {tool_name} is dynamic but source code cannot be extracted")
            
            # Get tool parameters and build properties using dynamic_manager methods
            tool_parameters = dynamic_manager.get_parameters(tool_cls)
            tool_function_calling = dynamic_manager.build_function_calling(tool_name, tool_description, tool_parameters)
            tool_text = dynamic_manager.build_text_representation(tool_name, tool_description, tool_parameters)
            tool_args_schema = dynamic_manager.build_args_schema(tool_name, tool_parameters)
            
            # --- Build ToolConfig ---
            updated_config = ToolConfig(
                name=tool_name,  # Keep same name
                description=tool_description,
                metadata=tool_metadata,
                require_grad=tool_require_grad,
                version=new_version,
                cls=tool_cls,
                config=tool_config_dict or {},
                instance=tool_instance,
                function_calling=tool_function_calling,
                text=tool_text,
                args_schema=tool_args_schema,
                code=tool_code,
            )
            
            # Update the tool config (replaces current version)
            self._tool_configs[tool_name] = updated_config
            
            # Store in version history
            if tool_name not in self._tool_history_versions:
                self._tool_history_versions[tool_name] = {}
            self._tool_history_versions[tool_name][updated_config.version] = updated_config
            
            # Register new version record to version manager
            await version_manager.register_version(
                "tool", 
                tool_name, 
                new_version,
                description=description or f"Updated from {original_config.version}"
            )
            
            # Update embedding index
            await self._store(updated_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 🔄 Updated tool {tool_name} from v{original_config.version} to v{new_version}")
            return updated_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to update tool: {e}")
            raise
    
    async def copy(self, 
                  tool_name: str,
                  new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, 
                  new_config: Optional[Dict[str, Any]] = None) -> ToolConfig:
        """Copy an existing tool configuration
        
        Args:
            tool_name: Name of the tool to copy
            new_name: New name for the copied tool. If None, uses original name.
            new_version: New version for the copied tool. If None, increments version.
            new_config: New configuration dict for the copied tool. If None, uses original config.
            
        Returns:
            ToolConfig: New tool configuration
        """
        try:
            original_config = self._tool_configs.get(tool_name)
            if original_config is None:
                raise ValueError(f"Tool {tool_name} not found")
            
            if original_config.cls is None:
                raise ValueError(f"Cannot copy tool {tool_name}: no class provided")
            
            # Determine new name
            if new_name is None:
                new_name = tool_name
            
            # Prepare config dict (merge original config with new config)
            tool_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # Merge new config into original config
                tool_config_dict.update(new_config)
            
            # Instantiate tool instance (copy is a runtime operation)
            try:
                tool_instance = original_config.cls(**tool_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create tool instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate tool {original_config.cls.__name__} with provided config: {e}")
            
            # Apply name override if provided (after instantiation)
            if new_name != tool_name:
                tool_instance.name = new_name
            
            tool_description = tool_instance.description
            tool_metadata = tool_instance.metadata
            tool_require_grad = tool_config_dict.get("require_grad", tool_instance.require_grad) if tool_config_dict and "require_grad" in tool_config_dict else tool_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                if new_name == tool_name:
                    # If copying with same name, get next version from version_manager
                    new_version = await version_manager.generate_next_version("tool", new_name, "patch")
                else:
                    # If copying with different name, get or generate version for new name
                    new_version = await version_manager.get_version("tool", new_name)
            
            # Get tool code
            tool_code = dynamic_manager.get_source_code(original_config.cls)
            if not tool_code:
                logger.warning(f"| ⚠️ Tool {new_name} is dynamic but source code cannot be extracted")
            
            # Get tool parameters and build properties using dynamic_manager methods
            tool_parameters = dynamic_manager.get_parameters(original_config.cls)
            tool_function_calling = dynamic_manager.build_function_calling(new_name, tool_description, tool_parameters)
            tool_text = dynamic_manager.build_text_representation(new_name, tool_description, tool_parameters)
            tool_args_schema = dynamic_manager.build_args_schema(new_name, tool_parameters)
            
            # --- Build ToolConfig ---
            new_config = ToolConfig(
                name=new_name,
                description=tool_description,
                metadata=tool_metadata,
                require_grad=tool_require_grad,
                version=new_version,
                cls=original_config.cls,
                config=tool_config_dict,
                instance=tool_instance,
                function_calling=tool_function_calling,
                text=tool_text,
                args_schema=tool_args_schema,
                code=tool_code,
            )
            
            # Register new tool
            self._tool_configs[new_name] = new_config
            
            # Store in version history
            if new_name not in self._tool_history_versions:
                self._tool_history_versions[new_name] = {}
            self._tool_history_versions[new_name][new_version] = new_config
            
            # Register version record to version manager
            await version_manager.register_version(
                "tool", 
                new_name, 
                new_version,
                description=f"Copied from {tool_name}@{original_config.version}"
            )
            
            # Register to embedding index
            await self._store(new_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📋 Copied tool {tool_name}@{original_config.version} to {new_name}@{new_version}")
            return new_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to copy tool: {e}")
            raise
    
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

        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered tool {tool_name}@{tool_config.version}")
        return True
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all tool configurations with version history to JSON.
        
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
            
            # Prepare save data - save all versions for each tool
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_tools": len(self._tool_configs),
                    "num_versions": sum(len(versions) for versions in self._tool_history_versions.values()),
                },
                "tools": {}
            }
            
            for tool_name, version_map in self._tool_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, tool_config in version_map.items():
                        config_dict = tool_config.model_dump()
                        versions_data[tool_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if tool_name in self._tool_configs:
                        current_config = self._tool_configs[tool_name]
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
                    
                    save_data["tools"][tool_name] = {
                        "versions": versions_data,
                        "current_version": current_version
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
        """Load tool configurations with version history from JSON.
        
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
                logger.warning(f"| ⚠️ Tool file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                tools_data = load_data.get("tools", {})
                loaded_count = 0
                
                for tool_name, tool_data in tools_data.items():
                    try:
                        # Expected format: multiple versions stored as a dict {version_str: config_dict}
                        versions_data = tool_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Tool {tool_name} has invalid format for 'versions' (expected dict), skipping")
                            continue
                        
                        current_version_str = tool_data.get("current_version")
                        
                        # Load all versions
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for version_str, config_dict in versions_data.items():
                            # Ensure version field is present
                            if "version" not in config_dict:
                                config_dict["version"] = version_str
                            
                            try:
                                tool_config = ToolConfig.model_validate(config_dict)
                                version_configs.append(tool_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load tool config for {tool_name}@{version_str}: {e}")
                                continue
                            
                            # Track latest version
                            if latest_config is None or (
                                current_version_str and tool_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    version_manager.compare_versions(tool_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = tool_config
                                latest_version = tool_config.version
                        
                        # Store all versions in history (dict-based)
                        self._tool_history_versions[tool_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }
                        
                        # Only set latest version as active
                        if latest_config:
                            self._tool_configs[tool_name] = latest_config
                            
                            # Register all versions to version manager (only version records)
                            for tool_config in version_configs:
                                await version_manager.register_version("tool", tool_name, tool_config.version)
                            
                            # Create instance if requested (instance is not saved in JSON, must be created via build)
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
    
    async def restore(self, tool_name: str, version: str, auto_initialize: bool = True) -> Optional[ToolConfig]:
        """Restore a specific version of a tool from history
        
        Args:
            tool_name: Name of the tool
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored tool
            
        Returns:
            ToolConfig of the restored version, or None if not found
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if tool_name in self._tool_history_versions:
            version_config = self._tool_history_versions[tool_name].get(version)
        
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
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("tool", tool_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("tool", tool_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored tool {tool_name} to version {version}")
        return restored_config
    
    async def save_contract(self, tool_names: Optional[List[str]] = None):
        """Save the contract for a tool"""
        contract = []
        if tool_names is not None:
            for index, tool_name in enumerate(tool_names):
                tool_info = await self.get_info(tool_name)
                text = tool_info.text
                contract.append(f"{index + 1:04d}\n{text}\n")
        else:
            for index, tool_name in enumerate(self._tool_configs.keys()):
                tool_info = await self.get_info(tool_name)
                text = tool_info.text
                contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} tools contract to {self.contract_path}")
        
    async def load_contract(self) -> str:
        """Load the contract for a tool"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve similar tools using FAISS similarity search.
        
        Args:
            query: Query string to search for
            k: Number of results to return (default: 4)
            
        Returns:
            List of dictionaries containing tool information with similarity scores
        """
        if self._faiss_service is None:
            logger.warning("| ⚠️ FAISS service not initialized, cannot retrieve tools")
            return []
        
        try:
            from src.environment.faiss.types import FaissSearchRequest
            
            request = FaissSearchRequest(
                query=query,
                k=k,
                fetch_k=k * 5  # Fetch more candidates before filtering
            )
            
            result = await self._faiss_service.search_similar(request)
            
            if not result.success:
                logger.warning(f"| ⚠️ FAISS search failed: {result.message}")
                return []
            
            # Extract documents and scores from result
            documents = []
            if result.extra and "documents" in result.extra:
                docs = result.extra["documents"]
                scores = result.extra.get("scores", [])
                
                for doc, score in zip(docs, scores):
                    # Extract tool name from metadata
                    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                    tool_name = metadata.get("name", "")
                    
                    # Get tool config if available
                    tool_config = None
                    if tool_name and tool_name in self._tool_configs:
                        tool_config = self._tool_configs[tool_name]
                    
                    documents.append({
                        "name": tool_name,
                        "description": metadata.get("description", ""),
                        "score": float(score),
                        "content": doc.get("page_content", "") if isinstance(doc, dict) else str(doc),
                        "config": tool_config.model_dump() if tool_config else None
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"| ❌ Error retrieving tools: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup all active tools."""
        try:
            # Clear all tool configs and version history
            self._tool_configs.clear()
            self._tool_history_versions.clear()
                
            # Clean up Faiss service (async)
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            logger.info("| 🧹 Tool context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during tool context manager cleanup: {e}")
            
    async def get_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get variables from tools, where each tool's code is used as the variable value.
        
        Args:
            tool_name (Optional[str]): Name of a specific tool. If None, returns variables for all tools.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping tool names to Variable objects. Each Variable has:
                - name: tool name
                - type: "tool_code"
                - description: tool description
                - require_grad: tool's require_grad value
                - variables: tool's code (as string value)
        """
        # Lazy import to avoid circular dependency
        from src.optimizer.types import Variable
        
        variables: Dict[str, Variable] = {}
        
        if tool_name is not None:
            # Get specific tool
            tool_config = await self.get_info(tool_name)
            if tool_config is None:
                logger.warning(f"| ⚠️ Tool {tool_name} not found")
                return variables
            
            tool_configs = {tool_name: tool_config}
        else:
            # Get all tools
            tool_configs = self._tool_configs
        
        for name, tool_config in tool_configs.items():
            # Get tool code
            tool_code = tool_config.code or ""
            
            # Create Variable for this tool
            variable = Variable(
                name=name,
                type="tool_code",
                description=tool_config.description or f"Code for tool {name}",
                require_grad=tool_config.require_grad,
                template=None,
                variables=tool_code  # Store code as the variable value
            )
            variables[name] = variable
        
        return variables
    
    async def get_trainable_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get trainable variables from tools, filtering out tools with require_grad=False.
        
        Only returns variables for tools where require_grad=True.
        
        Args:
            tool_name (Optional[str]): Name of a specific tool. If None, returns trainable variables for all tools.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping tool names to Variable objects for tools with require_grad=True.
                                Each Variable has:
                - name: tool name
                - type: "tool_code"
                - description: tool description
                - require_grad: True
                - variables: tool's code (as string value)
        """
        async with self._variables_lock:
            # Get all variables first
            all_variables = await self.get_variables(tool_name=tool_name)
            
            # Filter to only include variables with require_grad=True
            trainable_variables = {
                name: variable for name, variable in all_variables.items()
                if variable.require_grad is True
            }
            
            return trainable_variables
    
    async def set_variables(self, tool_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> ToolConfig:
        """Set variable values in a tool and create a new version.
        
        Args:
            tool_name: Name of the tool to update
            variable_updates: Dictionary mapping variable names to new values.
                For tools, this is typically {"code": new_code_string}
                - example:
                {
                    "name": "tool_name",
                    "variables": "tool code"
                }
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        async with self._variables_lock:
            original_config = self._tool_configs.get(tool_name)
            if original_config is None:
                raise ValueError(f"Tool {tool_name} not found. Use register() to register a new tool.")
            
            # For tools, variable_updates format is {"name": "tool_name", "variables": "tool code"}
            # Extract the new code from "variables" field
            if "variables" not in variable_updates:
                raise ValueError(f"variable_updates must contain 'variables' field with tool code, got: {list(variable_updates.keys())}")
            
            new_code = variable_updates["variables"]
            if not isinstance(new_code, str):
                raise ValueError(f"Tool code must be a string, got {type(new_code)}")
            
            # Load tool class from code
            class_name = dynamic_manager.extract_class_name_from_code(new_code)
            if not class_name:
                raise ValueError(f"Cannot extract class name from code")
            
            try:
                tool_cls = dynamic_manager.load_class(
                    new_code,
                    class_name=class_name,
                    base_class=Tool,
                    context="tool"
                )
            except Exception as e:
                logger.error(f"| ❌ Failed to load tool class from code: {e}")
                raise ValueError(f"Failed to load tool class from code: {e}")
            
            # Use update() function to handle version management and persistence
            # Pass the code directly to avoid re-extracting from dynamically created class
            update_description = description or f"Updated code for {tool_name}"
            return await self.update(
                tool_cls=tool_cls,
                tool_config_dict=original_config.config,
                new_version=new_version,
                description=update_description,
                code=new_code  # Pass code directly since tool_cls is dynamically created
            )
    
    async def __call__(self,
                       name: str,
                       input: Dict[str, Any], 
                       timeout: Optional[float] = None,
                       ctx: SessionContext = None,
                       **kwargs
                       ) -> ToolResponse:
        """Call a tool by name with optional timeout
        
        Args:
            name: Tool name
            input: Input for the tool
            timeout: Timeout in seconds for this specific call (overrides default_timeout if provided)
            
        Returns:
            ToolResponse: Tool result
        """
        
        if ctx is None:
            ctx = SessionContext()
        
        tool_info = await self.get_info(name)
        
        if tool_info is None:
            error_msg = f"Tool '{name}' is not registered. Available tools: {list(self._tool_configs.keys())}"
            logger.error(f"| ❌ {error_msg}")
            return ToolResponse(success=False, message=error_msg)
        
        version = tool_info.version
        tool_instance = tool_info.instance
        logger.info(f"| ✅ Using tool {name}@{version}")
        
        # Use provided timeout, or fall back to default_timeout
        effective_timeout = timeout if timeout is not None else self.default_timeout
        
        # Other tool args
        tool_kwargs = dict(ctx=ctx)
        
        # If timeout is None (no timeout), call tool directly
        if effective_timeout is None:
            return await tool_instance(**input, **tool_kwargs)
        
        # Otherwise, use asyncio.wait_for to enforce timeout
        try:
            return await asyncio.wait_for(tool_instance(**input, **tool_kwargs), timeout=effective_timeout)
        except asyncio.TimeoutError:
            error_msg = f"Tool '{name}' execution timed out after {effective_timeout} seconds"
            logger.error(f"| ⏱️ {error_msg}")
            return ToolResponse(
                success=False,
                message=error_msg,
                extra=None
            )
