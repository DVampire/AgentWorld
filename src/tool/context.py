"""Tool Context Manager for managing tool lifecycle and resources with lazy loading."""
import importlib
import os
import ast
import asyncio
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union, Tuple
from datetime import datetime
import inflection
import inspect
import json
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.utils import assemble_project_path, gather_with_concurrency
from src.utils.serialization import serialize_args_schema, deserialize_args_schema
from src.tool.types import Tool, ToolConfig, ToolResponse
from src.version import version_manager
from src.dynamic import dynamic_manager
from src.utils.file_utils import file_lock
from src.registry import TOOL

class ToolContextManager(BaseModel):
    """Global context manager for all tools with lazy loading support."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 model_name: str = "openrouter/gpt-4.1",
                 embedding_model_name: str = "openrouter/text-embedding-3-large",
                 **kwargs):
        """Initialize the tool context manager.
        
        Args:
            base_dir: Base directory for storing tool data
            save_path: Path to save tool configurations
            model_name: The model to use for the tools
            embedding_model_name: The model to use for the tool embeddings
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
        # Tool version history, e.g., {"tool_name": {"1.0.0": ToolConfig, "1.0.1": ToolConfig}}
        self._tool_history_versions: Dict[str, Dict[str, ToolConfig]] = {}
        
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        
        self._cleanup_registered = False
        self._faiss_service = None
        
    async def initialize(self):
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
        tool_configs.update(code_tool_configs)
        
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
                
                # Set type to tool class name, will be used for building tool instance
                tool_cls_name = tool_cls.__name__
                if 'type' not in tool_config_dict:
                    tool_config_dict['type'] = tool_cls_name
                
                # Get tool properties from tool class
                tool_name = tool_cls.model_fields['name'].default
                tool_description = tool_cls.model_fields['description'].default
                tool_enabled = tool_cls.model_fields['enabled'].default
                
                # Get or generate version from version_manager
                tool_version = await version_manager.get_version("tool", tool_name)
                
                # Get full module source code (including imports)
                # This ensures all imports are preserved when saving/loading from JSON
                tool_code = self._get_full_module_source(tool_cls)
                
                # Create tool config (ToolConfig.id is auto-incremented internally if needed)
                tool_config = ToolConfig(
                    name=tool_name,
                    description=tool_description,
                    enabled=tool_enabled,
                    version=tool_version,
                    cls=tool_cls,
                    config=tool_config_dict,
                    instance=None,
                    function_calling=None,
                    text=None,
                    args_schema=None,
                    metadata={},
                    code=tool_code,
                )
                
                # Store tool config
                tool_configs[tool_name] = tool_config
                
                # Store in version history (by version string)
                if tool_name not in self._tool_history_versions:
                    self._tool_history_versions[tool_name] = {}
                self._tool_history_versions[tool_name][tool_version] = tool_config
                
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
                            "enabled": bool,
                            "version": str,
                            "cls": Type[Tool], # will be loaded from code
                            "config": dict,
                            "instance": Tool, # will be built when needed
                            "metadata": dict,
                            "function_calling": dict, # will be built when needed
                            "text": str, # will be built when needed
                            "args_schema": BaseModel, # will be built when needed
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
                current_config: Optional[ToolConfig] = None  # Active config for current_version
                
                for version_str, version_data in versions.items():
                    name = version_data.get("name", "")
                    description = version_data.get("description", "")
                    enabled = version_data.get("enabled", True)
                    version = version_data.get("version", version_str)
                    
                    code = version_data.get("code", None)
                    config = version_data.get("config", {})
                    
                    if code:
                        # Dynamic class: load class from source code
                        # Extract class name from config
                        class_name = config.get("type")
                        if not class_name:
                            # Try to extract from code
                            class_name = dynamic_manager.extract_class_name_from_code(code)
                        
                        if class_name:
                            try:
                                # Use context="tool" for automatic import injection
                                # The full module source code includes all imports, so they should be available
                                cls = dynamic_manager.load_class(
                                    code, 
                                    class_name=class_name,
                                    base_class=Tool,
                                    context="tool"
                                )
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load dynamic class for {tool_name}@{version}: {e}")
                                cls = None
                        else:
                            logger.warning(f"| ⚠️ Cannot determine class name from code for {tool_name}@{version}")
                            cls = None
                    else:
                        cls = None
                    instance = version_data.get("instance", None)
                    metadata = version_data.get("metadata", {})
                    function_calling = version_data.get("function_calling", None)
                    text = version_data.get("text", None)
                    
                    # Restore args_schema from saved schema info (if present)
                    args_schema = None
                    args_schema_info = version_data.get("args_schema")
                    if args_schema_info:
                        try:
                            args_schema = deserialize_args_schema(args_schema_info)
                        except Exception as e:
                            logger.warning(f"| ⚠️ Failed to restore args_schema for {tool_name}@{version}: {e}")
                    
                    # Create ToolConfig (args_schema will be set after creation to avoid validation errors)
                    tool_config = ToolConfig(
                        name=name,
                        description=description,
                        enabled=enabled,
                        version=version,
                        cls=cls,
                        config=config,
                        instance=instance,
                        metadata=metadata,
                        function_calling=function_calling,
                        text=text,
                        code=code,
                    )
                    
                    # Set args_schema after creation to bypass validation
                    if args_schema is not None:
                        tool_config.args_schema = args_schema
                    version_map[version] = tool_config
                    
                    if version == current_version:
                        current_config = tool_config
                
                return tool_name, version_map, current_config
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
            tool_name, version_map, current_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._tool_history_versions[tool_name] = version_map
            # Active config: the one corresponding to current_version
            if current_config is not None:
                tool_configs[tool_name] = current_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Tool {tool_name} current_version not found, using last available version")
                tool_configs[tool_name] = list(version_map.values())[-1]
            
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
            
            tool_instance = tool_config.cls(**tool_config.config) if tool_config.config else tool_config.cls()
            tool_config.instance = tool_instance
            
            # Lazy compute function_calling, text, and args_schema if not already set
            if tool_config.function_calling is None or tool_config.text is None or tool_config.args_schema is None:
                try:
                    tool_config.function_calling = tool_instance.function_calling
                    tool_config.text = tool_instance.text
                    tool_config.args_schema = tool_instance.args_schema
                except Exception as e:
                    logger.debug(f"| ⚠️ Failed to get properties from tool instance {tool_config.name}: {e}")
            
            # Store tool metadata
            self._tool_configs[tool_config.name] = tool_config
            
            logger.info(f"| 🔧 Tool {tool_config.name} created and stored")
            
            return tool_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create tool {tool_config.name}: {e}")
            raise
    
    async def register(self, 
                       tool: Union[Tool, Type[Tool]],
                       tool_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> ToolConfig:
        """Register a tool class or instance.
        
        This will:
        - Create (or reuse) a tool instance
        - Create a `ToolConfig`
        - Store it as the current config and append to version history
        - Register the version in `version_manager` and FAISS index
        """
        
        try:
            # --- Normalize to a common representation ---
            if isinstance(tool, Tool):
                # Instance already created
                tool_instance = tool
                tool_cls = type(tool_instance)
                tool_name = tool_instance.name
                tool_description = tool_instance.description
                tool_enabled = tool_instance.enabled
                function_calling = tool_instance.function_calling
                text = tool_instance.text
                args_schema = tool_instance.args_schema
                metadata = getattr(tool_instance, "metadata", {})
                # Instances don't need an init config
                tool_config_dict = {}
                # Version: use explicit one or instance attribute, otherwise default
                tool_version = version or getattr(tool_instance, "version", "1.0.0")
            else:
                # Tool is a class, we need to build an instance
                tool_cls = tool
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
                tool_enabled = tool_instance.enabled
                function_calling = tool_instance.function_calling
                text = tool_instance.text
                args_schema = tool_instance.args_schema
                metadata = getattr(tool_instance, "metadata", {})
                
                # Version: ask version_manager if not provided
                if version is None:
                    tool_version = await version_manager.get_version("tool", tool_name)
                else:
                    tool_version = version
            
            # --- Common validations ---
            if not tool_name:
                raise ValueError("Tool.name cannot be empty.")
            
            if tool_name in self._tool_configs and not override:
                raise ValueError(f"Tool '{tool_name}' already registered. Use override=True to replace it.")
            
            # --- Dynamic code handling ---
            tool_code = None
            if dynamic_manager.is_dynamic_class(tool_cls):
                tool_code = dynamic_manager.get_class_source_code(tool_cls)
                if not tool_code:
                    logger.warning(f"| ⚠️ Tool {tool_name} is dynamic but source code cannot be extracted")
            
            # --- Build ToolConfig ---
            tool_config = ToolConfig(
                name=tool_name,
                description=tool_description,
                enabled=tool_enabled,
                version=tool_version,
                cls=tool_cls,
                config=tool_config_dict or {},
                instance=tool_instance,
                metadata=metadata,
                function_calling=function_calling,
                text=text,
                args_schema=args_schema,
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
    
    async def update(self, 
                     tool_name: str, tool: Union[Tool, Type[Tool]], 
                     tool_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> ToolConfig:
        """Update an existing tool with new configuration and create a new version
        
        Args:
            tool_name: Name of the tool to update
            tool: New tool class or instance with updated implementation
            tool_config_dict: Configuration dict for tool initialization (required when tool is a class)
                   If None and tool is a class, will try to get from global config
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        original_config = self._tool_configs.get(tool_name)
        if original_config is None:
            raise ValueError(f"Tool {tool_name} not found. Use register() to register a new tool.")
        
        # Determine new version from version_manager
        if new_version is None:
            # Get current version from version_manager and generate next patch version
            new_version = await version_manager.generate_next_version("tool", tool_name, "patch")
        
        # Create new tool config with updated content
        if isinstance(tool, Tool):
            # Check if this is a dynamically generated class that needs code saved
            tool_code = None
            tool_cls = type(tool)
            if dynamic_manager.is_dynamic_class(tool_cls):
                tool_code = dynamic_manager.get_class_source_code(tool_cls)
                if not tool_code:
                    logger.warning(f"| ⚠️ Tool {tool_name} is dynamic but source code cannot be extracted")
            
            # Updating with an instance - get properties directly from instance
            updated_config = ToolConfig(
                name=tool_name,  # Keep same name
                description=tool.description,
                enabled=tool.enabled,
                version=new_version,
                cls=tool_cls,
                config={},  # Instance already created, no config needed
                instance=tool,  # Use the provided instance
                metadata=getattr(tool, 'metadata', {}),
                function_calling=tool.function_calling,
                text=tool.text,
                args_schema=tool.args_schema,
                code=tool_code,
            )
        else:
            # Updating with a class - need tool_config_dict for instantiation
            if tool_config_dict is None:
                # Try to get config from global config
                tool_config_key = inflection.underscore(tool.__name__)
                tool_config_dict = config.get(tool_config_key, {})
            
            # Create instance immediately since update may be called at runtime
            try:
                tool_instance = tool(**tool_config_dict)
                tool_description = tool_instance.description
                function_calling = tool_instance.function_calling
                text = tool_instance.text
                args_schema = tool_instance.args_schema
                tool_enabled = tool_instance.enabled
            except Exception as e:
                logger.error(f"| ❌ Failed to create tool instance for {tool.__name__}: {e}")
                raise ValueError(f"Failed to instantiate tool {tool.__name__} with provided config: {e}")
            
            # Prepare config dict with 'type' field for TOOL.build() (for future rebuilds)
            # tool_config_dict is already set above
            if 'type' not in tool_config_dict:
                tool_config_dict['type'] = tool.__name__
            
            # Check if this is a dynamically generated class that needs code saved
            tool_code = None
            if dynamic_manager.is_dynamic_class(tool):
                tool_code = dynamic_manager.get_class_source_code(tool)
                if not tool_code:
                    logger.warning(f"| ⚠️ Tool {tool_name} is dynamic but source code cannot be extracted")
            
            updated_config = ToolConfig(
                name=tool_name,  # Keep same name
                description=tool_description,
                enabled=tool_enabled,
                version=new_version,
                cls=tool,
                config=tool_config_dict,
                instance=tool_instance,  # Instance is ready for immediate use
                metadata={},
                function_calling=function_calling,
                text=text,
                args_schema=args_schema,
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
        
        # Determine new version from version_manager
        if new_version is None:
            if new_name == tool_name:
                # If copying with same name, get next version from version_manager
                new_version = await version_manager.generate_next_version("tool", new_name, "patch")
            else:
                # If copying with different name, get or generate version for new name
                new_version = await version_manager.get_or_generate_version("tool", new_name)
        
        # Create copy of config
        new_config_dict = original_config.model_dump()
        new_config_dict["name"] = new_name
        new_config_dict["version"] = new_version
        
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
        
        # Ensure 'type' field exists in config for TOOL.build()
        # Get cls from original_config since model_dump() may not include it properly
        if original_config.cls and "config" in new_config_dict:
            if 'type' not in new_config_dict["config"]:
                new_config_dict["config"]['type'] = original_config.cls.__name__
        
        # Clear instance (will be created on demand)
        new_config_dict["instance"] = None
        
        new_config = ToolConfig(**new_config_dict)
        
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

        # Persist to JSON after unregister
        await self.save_to_json()
        
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
                    # Serialize all versions for this tool as a dict: {version_str: config_dict}
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for version_str, tool_config in version_map.items():
                        # Serialize tool config (excluding non-serializable fields)
                        # - cls: Cannot serialize Type objects, code is saved if available
                        # - instance: Runtime state, should be recreated via build() on load
                        # - args_schema: Will be serialized separately as schema info
                        # Other fields like name, description, version, config, function_calling, text, code are saved
                        config_dict = tool_config.model_dump(mode="json", exclude={"cls", "instance", "args_schema"})
                        
                        # Serialize args_schema (BaseModel type) as schema info
                        if tool_config.args_schema is not None:
                            args_schema_info = serialize_args_schema(tool_config.args_schema)
                            if args_schema_info:
                                config_dict["args_schema"] = args_schema_info
                        
                        # Use version string as key
                        versions_data[tool_config.version] = config_dict
                    
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
                            # Try to load class - priority: code > TOOL registry
                            cls = None
                            config_dict_copy = config_dict.copy()
                            
                            # Case 1: Load from code (for dynamically generated tools)
                            if "code" in config_dict and config_dict["code"]:
                                try:
                                    # Extract class name from config or code
                                    class_name = config_dict.get("config", {}).get("type")
                                    if not class_name:
                                        # Try to extract from code by parsing
                                        tree = ast.parse(config_dict["code"])
                                        for node in ast.walk(tree):
                                            if isinstance(node, ast.ClassDef):
                                                class_name = node.name
                                                break
                                    
                                    if class_name:
                                        # Use context="tool" for automatic import injection
                                        cls = dynamic_manager.load_class(
                                            config_dict["code"], 
                                            class_name, 
                                            Tool,
                                            context="tool"
                                        )
                                        logger.debug(f"| ✅ Loaded tool class {class_name} from code for {tool_name}")
                                    else:
                                        logger.warning(f"| ⚠️ Cannot determine class name from code for {tool_name}")
                                except Exception as e:
                                    logger.warning(f"| ⚠️ Failed to load class from code for {tool_name}: {e}")
                            
                            # Ensure version field is present
                            if "version" not in config_dict_copy:
                                config_dict_copy["version"] = version_str
                            
                            # Restore args_schema from saved schema info
                            args_schema = None
                            if "args_schema" in config_dict_copy:
                                args_schema_info = config_dict_copy.pop("args_schema")
                                try:
                                    args_schema = deserialize_args_schema(args_schema_info)
                                except Exception as e:
                                    logger.warning(f"| ⚠️ Failed to restore args_schema for {tool_name}@{version_str}: {e}")
                            
                            # Remove cls and args_schema from dict before creating ToolConfig
                            # They will be set after creation to avoid validation errors
                            config_dict_copy.pop("cls", None)
                            
                            # Create ToolConfig
                            tool_config = ToolConfig(**config_dict_copy)
                            
                            # Set cls and args_schema after creation
                            if cls is not None:
                                tool_config.cls = cls
                            if args_schema is not None:
                                tool_config.args_schema = args_schema
                            
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
    
    def _get_full_module_source(self, cls: Type[Tool]) -> str:
        """Get the full source code of the module containing the class, including all imports.
        
        This is more reliable than inspect.getsource() which only gets the class definition.
        By reading the entire module file, we preserve all import statements and module-level code,
        ensuring the complete context is available when loading from JSON.
        
        Args:
            cls: The tool class
            
        Returns:
            Full module source code as string, or class source if file reading fails
        """
        try:
            # Get the module object
            module = inspect.getmodule(cls)
            if module is None:
                # Fallback to inspect.getsource if module is not available
                return inspect.getsource(cls)
            
            # Get the file path of the module
            file_path = inspect.getfile(module)
            
            # Read the entire file to preserve all imports and context
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (OSError, TypeError, IOError, AttributeError) as e:
            # Fallback to inspect.getsource if file reading fails
            logger.debug(f"| ⚠️ Failed to read module file for {cls.__name__}, falling back to inspect.getsource: {e}")
            try:
                return inspect.getsource(cls)
            except Exception:
                logger.warning(f"| ⚠️ Failed to get source code for {cls.__name__}")
                return ""
    
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
            version_history.current_version = version
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored tool {tool_name} to version {version}")
        return restored_config
    
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
            
    async def __call__(self, name: str, input: Dict[str, Any]) -> ToolResponse:
        """Call a tool by name
        
        Args:
            name: Tool name
            input: Input for the tool
            
        Returns:
            ToolResponse: Tool result
        """
        tool = await self.get(name)
        if tool is None:
            raise ValueError(f"Tool {name} not found")
        return await tool(**input)
