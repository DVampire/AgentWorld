"""SCP Server — Skill Context Protocol.

Server implementation that mirrors the TCP (Tool Context Protocol) pattern,
providing a unified interface for skill discovery, loading, and context injection.
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
        """Build the skill context string for prompt injection.

        This is the primary method used by the agent to obtain skill instructions.
        """
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

        Reads the skill's SKILL.md, asks an LLM to interpret its instructions
        with the given input, and returns the result.

        Args:
            name: Skill name.
            input: User-provided arguments for the skill.
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


# Global SCP server instance (mirrors the global `tcp` pattern)
scp = SCPServer()
