"""Skill type definitions for the Skill Context Protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SkillExtra(BaseModel):
    """Extra data attached to a skill response."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    file_path: Optional[str] = Field(default=None, description="Related file path")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary extra data")


class SkillResponse(BaseModel):
    """Response returned by a skill operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Human-readable result message")
    extra: Optional[SkillExtra] = Field(default=None, description="Extra data")


class SkillConfig(BaseModel):
    """Configuration for a loaded skill, parsed from SKILL.md and its directory."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="Skill name from YAML frontmatter")
    description: str = Field(description="Skill description from YAML frontmatter")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional YAML frontmatter fields")
    require_grad: bool = Field(default=False, description="Whether the skill is trainable")
    version: str = Field(default="1.0.0", description="Version of the skill")

    skill_dir: str = Field(description="Absolute path to the skill directory")
    content: str = Field(default="", description="Full markdown body of SKILL.md (after frontmatter)")
    scripts: List[str] = Field(default_factory=list, description="Paths to scripts under scripts/")
    resources: List[str] = Field(default_factory=list, description="Paths to files under resources/")
    reference_files: List[str] = Field(default_factory=list, description="Paths to extra markdown files")

    text: Optional[str] = Field(default=None, description="Pre-built text representation for prompt injection")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "require_grad": self.require_grad,
            "version": self.version,
            "skill_dir": self.skill_dir,
            "content": self.content,
            "scripts": self.scripts,
            "resources": self.resources,
            "reference_files": self.reference_files,
            "text": self.text,
        }


__all__ = [
    "SkillConfig",
    "SkillResponse",
    "SkillExtra",
]
