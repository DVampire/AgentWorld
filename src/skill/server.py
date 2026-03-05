"""SCP Server — Skill Context Protocol.

Server implementation that mirrors the TCP (Tool Context Protocol) pattern,
providing a unified interface for skill discovery, loading, registration,
update, and execution.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.skill.context import SkillContextManager
from src.skill.types import SkillConfig, SkillResponse
from src.session import SessionContext
from src.utils import assemble_project_path


class SCPServer(BaseModel):
    """SCP Server for managing skill registration and context generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="Base directory for skill data")
    save_path: str = Field(default=None, description="Path to persist skill configs")
    contract_path: str = Field(default=None, description="Path to save skill contract")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skill_context_manager: Optional[SkillContextManager] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, skill_names: Optional[List[str]] = None):
        """Initialize skills by scanning default (and custom) skill directories.

        Args:
            skill_names: If provided, only these skills are loaded.
        """
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "skill"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "skill.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(
            f"| 📁 SCP Server base directory: {self.base_dir} "
            f"with save path: {self.save_path} and contract path: {self.contract_path}"
        )

        self.skill_context_manager = SkillContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
        )
        await self.skill_context_manager.initialize(skill_names=skill_names)

        logger.info("| ✅ Skills initialization completed")

    async def cleanup(self):
        """Release all skills."""
        if self.skill_context_manager is not None:
            await self.skill_context_manager.cleanup()

    # ------------------------------------------------------------------
    # Register / Update / Unregister / Copy / Restore
    # ------------------------------------------------------------------

    async def register(
        self,
        skill_dir: str,
        override: bool = False,
        version: Optional[str] = None,
    ) -> SkillConfig:
        """Register a skill from a directory containing SKILL.md.

        Args:
            skill_dir: Path to the skill directory.
            override: If True, overwrite an existing skill with the same name.
            version: Explicit version string.

        Returns:
            The registered SkillConfig.
        """
        return await self.skill_context_manager.register(
            skill_dir=skill_dir,
            override=override,
            version=version,
        )

    async def update(
        self,
        name: str,
        skill_dir: Optional[str] = None,
        new_version: Optional[str] = None,
        description: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillConfig:
        """Update an existing skill and create a new version.

        Args:
            name: Skill name.
            skill_dir: If provided, re-parse this directory.
            new_version: Explicit new version string.
            description: Override description.
            content: Override SKILL.md body content.
            metadata: Override metadata dict.

        Returns:
            Updated SkillConfig.
        """
        return await self.skill_context_manager.update(
            name=name,
            skill_dir=skill_dir,
            new_version=new_version,
            description=description,
            content=content,
            metadata=metadata,
        )

    async def unregister(self, name: str) -> bool:
        """Remove a skill.

        Args:
            name: Skill name.

        Returns:
            True if removed, False if not found.
        """
        return await self.skill_context_manager.unregister(name)

    async def copy(
        self,
        name: str,
        new_name: Optional[str] = None,
        new_version: Optional[str] = None,
        new_skill_dir: Optional[str] = None,
    ) -> SkillConfig:
        """Copy an existing skill, optionally under a new name.

        Args:
            name: Source skill name.
            new_name: Name for the copy.
            new_version: Version for the copy.
            new_skill_dir: If provided, physically copies the skill directory.

        Returns:
            New SkillConfig.
        """
        return await self.skill_context_manager.copy(
            name=name,
            new_name=new_name,
            new_version=new_version,
            new_skill_dir=new_skill_dir,
        )

    async def restore(self, name: str, version: str) -> Optional[SkillConfig]:
        """Restore a specific version of a skill from history.

        Args:
            name: Skill name.
            version: Version string to restore.

        Returns:
            Restored SkillConfig, or None if not found.
        """
        return await self.skill_context_manager.restore(name, version)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    async def get(self, skill_name: str) -> Optional[SkillConfig]:
        """Get a loaded skill by name."""
        return await self.skill_context_manager.get(skill_name)

    async def get_info(self, skill_name: str) -> Optional[SkillConfig]:
        """Get skill configuration by name."""
        return await self.skill_context_manager.get_info(skill_name)

    async def list(self) -> List[str]:
        """List all loaded skill names."""
        return await self.skill_context_manager.list()

    # ------------------------------------------------------------------
    # Context & Contract
    # ------------------------------------------------------------------

    async def get_context(self, skill_names: Optional[List[str]] = None) -> str:
        """Build the skill context string for prompt injection."""
        return await self.skill_context_manager.get_context(skill_names=skill_names)

    async def get_contract(self) -> str:
        """Load the persisted contract text."""
        return await self.skill_context_manager.load_contract()

    # ------------------------------------------------------------------
    # Skill execution
    # ------------------------------------------------------------------

    async def __call__(
        self,
        name: str,
        input: Dict[str, Any],
        model_name: Optional[str] = None,
        ctx: SessionContext = None,
        **kwargs,
    ) -> SkillResponse:
        """Execute a skill by name.

        Args:
            name: Skill name.
            input: User-provided arguments.
            model_name: LLM model override.
            ctx: Session context.
        """
        return await self.skill_context_manager(
            name=name,
            input=input,
            model_name=model_name,
            ctx=ctx,
            **kwargs,
        )


# Global SCP server instance
scp = SCPServer()
