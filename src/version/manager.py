"""Version Manager

Unified version management system for agents, environments, and tools.
Supports version tracking, evolution, and history management.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import json
import os

from src.logger import logger
from src.config import config
from src.utils import assemble_project_path
from src.utils.file_utils import file_lock

T = TypeVar('T', bound=BaseModel)


class VersionStatus(str, Enum):
    """Version status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VersionInfo(BaseModel):
    """Version information"""
    version: str = Field(description="Version string (e.g., '1.0.0', '2.1.3')")
    status: VersionStatus = Field(default=VersionStatus.ACTIVE, description="Version status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    description: Optional[str] = Field(default=None, description="Version description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Version metadata")


class ComponentVersionHistory(BaseModel):
    """Version history for a component (only version records, no configs)"""
    name: str = Field(description="Name of the component")
    component_type: str = Field(description="Type of component (tool, environment, agent)")
    current_version: str = Field(description="Current active version")
    versions: Dict[str, VersionInfo] = Field(default_factory=dict, description="Version history records")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Component metadata")
    
    def add_version(self, version: str, description: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> VersionInfo:
        """Add a new version record
        
        Args:
            version: Version string
            description: Version description
            metadata: Version metadata
            
        Returns:
            VersionInfo: Created version info
        """
        if version in self.versions:
            logger.warning(f"| ⚠️ Version {version} already exists for {self.name}, updating...")
            version_info = self.versions[version]
            version_info.updated_at = datetime.now()
            if description:
                version_info.description = description
            if metadata:
                version_info.metadata.update(metadata)
        else:
            version_info = VersionInfo(
                version=version,
                description=description,
                metadata=metadata or {}
            )
            self.versions[version] = version_info
        
        self.current_version = version
        
        logger.debug(f"| ✅ Added version record {version} for {self.name}")
        return version_info
    
    def list_versions(self) -> List[str]:
        """List all available versions
        
        Returns:
            List of version strings
        """
        return list(self.versions.keys())
    
    def deprecate_version(self, version: str):
        """Deprecate a version
        
        Args:
            version: Version string to deprecate
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found for {self.name}")
        
        if version == self.current_version:
            raise ValueError(f"Cannot deprecate current version {version}")
        
        self.versions[version].status = VersionStatus.DEPRECATED
        logger.info(f"| 📝 Deprecated version {version} for {self.name}")
    
    def archive_version(self, version: str):
        """Archive a version
        
        Args:
            version: Version string to archive
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found for {self.name}")
        
        self.versions[version].status = VersionStatus.ARCHIVED
        logger.info(f"| 📦 Archived version {version} for {self.name}")


class VersionManager(BaseModel):
    """Unified version manager for all components - only manages version records"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the version histories")
    save_path: str = Field(default=None, description="The path to save version histories")
    
    def __init__(self, **kwargs):
        """Initialize version manager"""
        super().__init__(**kwargs)
        
        # Storage: component_type -> name -> ComponentVersionHistory
        self._version_histories: Dict[str, Dict[str, ComponentVersionHistory]] = {
            "tool": {},
            "environment": {},
            "agent": {}
        }

    async def initialize(self):
        """Initialize version manager (for backward compatibility)"""
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "version"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "versions.json")
        logger.info(f"| 📁 Version manager base directory: {self.base_dir} and save path: {self.save_path}")
    
    async def register_version(self, component_type: str, name: str, version: str,
                        description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ComponentVersionHistory:
        """Register a version record (only version info, no config)
        
        Args:
            component_type: Type of component (tool, environment, agent)
            name: Component name
            version: Version string
            description: Version description
            metadata: Version metadata
            
        Returns:
            ComponentVersionHistory: Version history for the component
        """
        if component_type not in self._version_histories:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if name not in self._version_histories[component_type]:
            version_history = ComponentVersionHistory(
                name=name,
                component_type=component_type,
                current_version=version
            )
            self._version_histories[component_type][name] = version_history
        else:
            version_history = self._version_histories[component_type][name]
        
        version_history.add_version(version, description, metadata)
        
        return version_history
    
    async def list(self) -> Dict[str, Dict[str, List[str]]]:
        """List all versions for all components
        
        Returns:
            Dictionary mapping component_type -> component_name -> list of versions
        """
        result = {}
        for component_type, histories in self._version_histories.items():
            result[component_type] = {}
            for name, version_history in histories.items():
                result[component_type][name] = version_history.list_versions()
        return result
    
    async def get_version_history(self, component_type: str, name: str) -> Optional[ComponentVersionHistory]:
        """Get version history for a component
        
        Args:
            component_type: Type of component
            name: Component name
            
        Returns:
            ComponentVersionHistory or None if not found
        """
        if component_type not in self._version_histories:
            return None
        
        return self._version_histories[component_type].get(name)
    
    async def get_current_version(self, component_type: str, name: str) -> Optional[str]:
        """Get current version for a component
        
        Args:
            component_type: Type of component
            name: Component name
            
        Returns:
            Current version string or None if not found
        """
        version_history = await self.get_version_history(component_type, name)
        if version_history is None:
            return None
        return version_history.current_version
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all version histories to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Serialize all version histories
            save_data = {
                "component_type": {},
                "metadata": {
                    "saved_at": datetime.now().isoformat()
                }
            }
            
            for component_type, histories in self._version_histories.items():
                save_data["component_type"][component_type] = {}
                for name, version_history in histories.items():
                    # Convert to dict
                    history_dict = version_history.model_dump(mode="json")
                    save_data["component_type"][component_type][name] = history_dict
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Saved version histories to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load version histories from JSON
        
        Args:
            file_path: File path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Version file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                # Clear existing histories
                for component_type in self._version_histories:
                    self._version_histories[component_type].clear()
                
                # Load histories
                component_types = load_data.get("component_type", {})
                for component_type, histories in component_types.items():
                    if component_type not in self._version_histories:
                        logger.warning(f"| ⚠️ Unknown component type: {component_type}")
                        continue
                    
                    for name, history_dict in histories.items():
                        try:
                            # Reconstruct ComponentVersionHistory
                            version_history = ComponentVersionHistory(**history_dict)
                            self._version_histories[component_type][name] = version_history
                        except Exception as e:
                            logger.error(f"| ❌ Failed to load version history for {name}: {e}")
                            continue
                
                logger.info(f"| 📂 Loaded version histories from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load version data from {file_path}: {e}")
                return False


# Global version manager instance
version_manager = VersionManager()
