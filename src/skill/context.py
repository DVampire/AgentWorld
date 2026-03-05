"""Skill Context Manager for loading, managing, and serving skills."""

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.skill.types import SkillConfig, SkillResponse, SkillExtra
from src.model import model_manager
from src.message.types import SystemMessage, HumanMessage
from src.session import SessionContext
from src.utils import assemble_project_path


# Path to built-in default skills shipped with the project
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "default_skills"


class SkillContextManager(BaseModel):
    """Manages the lifecycle of skills: discovery, loading, and context generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="Base directory for skill runtime data")
    save_path: str = Field(default=None, description="Path to persist loaded skill configs")
    contract_path: str = Field(default=None, description="Path to save the skill contract")

    _skill_configs: Dict[str, SkillConfig] = {}

    def __init__(
        self,
        base_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        contract_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "skill"))
        os.makedirs(self.base_dir, exist_ok=True)

        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "skill.json")

        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")

        self._skill_configs: Dict[str, SkillConfig] = {}

        logger.info(f"| 📁 Skill context manager base directory: {self.base_dir}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self, skill_names: Optional[List[str]] = None):
        """Discover and load skills.

        Args:
            skill_names: If provided, only load these skills. Otherwise load all discovered skills.
        """
        discovered: Dict[str, SkillConfig] = {}

        # 1. Load from built-in default_skills directory
        default_configs = await self._load_from_directory(DEFAULT_SKILLS_DIR)
        discovered.update(default_configs)

        # 2. (Future) Load from user / project custom skill directories

        # 3. Filter by name if requested
        if skill_names is not None:
            filtered: Dict[str, SkillConfig] = {}
            for name in skill_names:
                if name in discovered:
                    filtered[name] = discovered[name]
                else:
                    logger.warning(f"| ⚠️ Requested skill '{name}' not found in discovered skills")
            discovered = filtered

        # 4. Build text representations and store
        for name, skill_config in discovered.items():
            skill_config.text = self._build_text_representation(skill_config)
            self._skill_configs[name] = skill_config
            logger.info(f"| 🎯 Skill '{name}' loaded from {skill_config.skill_dir}")

        # 5. Persist
        await self.save_to_json()
        await self.save_contract()

        logger.info(f"| ✅ Skills initialization completed — {len(self._skill_configs)} skill(s) loaded")

    # ------------------------------------------------------------------
    # Directory scanning & SKILL.md parsing
    # ------------------------------------------------------------------

    async def _load_from_directory(self, root_dir: Path) -> Dict[str, SkillConfig]:
        """Scan *root_dir* for sub-directories that contain a SKILL.md file."""
        configs: Dict[str, SkillConfig] = {}

        if not root_dir.exists():
            logger.info(f"| 📂 Skill directory does not exist, skipping: {root_dir}")
            return configs

        for child in sorted(root_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                skill_config = self._parse_skill_dir(child)
                configs[skill_config.name] = skill_config
            except Exception as e:
                logger.error(f"| ❌ Failed to parse skill at {child}: {e}")

        return configs

    def _parse_skill_dir(self, skill_dir: Path) -> SkillConfig:
        """Parse a single skill directory into a SkillConfig."""
        skill_md = skill_dir / "SKILL.md"
        raw = skill_md.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        frontmatter, body = self._parse_frontmatter(raw)

        name = frontmatter.get("name", skill_dir.name)
        description = frontmatter.get("description", "")
        metadata = {k: v for k, v in frontmatter.items() if k not in ("name", "description")}

        # Discover scripts/
        scripts_dir = skill_dir / "scripts"
        scripts: List[str] = []
        if scripts_dir.is_dir():
            scripts = [str(p) for p in sorted(scripts_dir.rglob("*")) if p.is_file()]

        # Discover resources/
        resources_dir = skill_dir / "resources"
        resources: List[str] = []
        if resources_dir.is_dir():
            resources = [str(p) for p in sorted(resources_dir.rglob("*")) if p.is_file()]

        # Discover extra markdown files (e.g. examples.md, reference.md)
        reference_files: List[str] = []
        for md_file in sorted(skill_dir.glob("*.md")):
            if md_file.name == "SKILL.md":
                continue
            reference_files.append(str(md_file))

        return SkillConfig(
            name=name,
            description=description,
            metadata=metadata,
            skill_dir=str(skill_dir),
            content=body.strip(),
            scripts=scripts,
            resources=resources,
            reference_files=reference_files,
        )

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
        """Split YAML frontmatter (between --- delimiters) from the markdown body.

        Returns (frontmatter_dict, body_string). Uses a simple regex-based parser
        so we don't force a PyYAML dependency on every user.
        """
        pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = pattern.match(text)

        if not match:
            return {}, text

        yaml_block = match.group(1)
        body = text[match.end():]

        # Minimal key: value parser (handles single-line string values)
        frontmatter: Dict[str, Any] = {}
        for line in yaml_block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                frontmatter[key.strip()] = value.strip()

        return frontmatter, body

    # ------------------------------------------------------------------
    # Text representation (for prompt injection)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text_representation(skill_config: SkillConfig) -> str:
        """Build a *concise* summary for prompt injection.

        Only includes name, description, and file listings with absolute paths
        so the agent can read them on demand via tools (bash, python_interpreter).
        The full SKILL.md body is NOT included to save context window tokens.
        """
        parts = [
            f"Skill: {skill_config.name}",
            f"Description: {skill_config.description}",
            f"Skill Directory: {skill_config.skill_dir}",
            f"SKILL.md: {os.path.join(skill_config.skill_dir, 'SKILL.md')}",
        ]

        if skill_config.scripts:
            parts.append("Scripts:")
            for s in skill_config.scripts:
                parts.append(f"  - {s}")

        if skill_config.resources:
            parts.append("Resources:")
            for r in skill_config.resources:
                parts.append(f"  - {r}")

        if skill_config.reference_files:
            parts.append("References:")
            for r in skill_config.reference_files:
                parts.append(f"  - {r}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    async def get(self, skill_name: str) -> Optional[SkillConfig]:
        """Get a loaded skill config by name."""
        return self._skill_configs.get(skill_name)

    async def get_info(self, skill_name: str) -> Optional[SkillConfig]:
        """Alias for get() — returns skill info."""
        return self._skill_configs.get(skill_name)

    async def list(self) -> List[str]:
        """Return names of all loaded skills."""
        return list(self._skill_configs.keys())

    # ------------------------------------------------------------------
    # Context generation (for agent prompt)
    # ------------------------------------------------------------------

    async def get_context(self, skill_names: Optional[List[str]] = None) -> str:
        """Build the full skill context string for prompt injection.

        Args:
            skill_names: Subset of skills to include. If None, includes all loaded skills.
        """
        if not self._skill_configs:
            return ""

        targets = skill_names if skill_names else list(self._skill_configs.keys())
        parts: List[str] = []

        for name in targets:
            cfg = self._skill_configs.get(name)
            if cfg is None:
                continue
            parts.append(f"<skill name=\"{cfg.name}\">\n{cfg.text}\n</skill>")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Contract (persistent text summary)
    # ------------------------------------------------------------------

    async def save_contract(self, skill_names: Optional[List[str]] = None):
        """Save a human-readable contract file listing all loaded skills."""
        targets = skill_names if skill_names else list(self._skill_configs.keys())
        lines: List[str] = []
        for idx, name in enumerate(targets):
            cfg = self._skill_configs.get(name)
            if cfg is None:
                continue
            lines.append(f"{idx + 1:04d}\n{cfg.text}\n")

        contract_text = "---\n".join(lines)
        os.makedirs(os.path.dirname(self.contract_path), exist_ok=True)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(lines)} skill(s) contract to {self.contract_path}")

    async def load_contract(self) -> str:
        """Load the contract text from disk."""
        if not os.path.exists(self.contract_path):
            return ""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Persistence (JSON)
    # ------------------------------------------------------------------

    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Persist all loaded skill configs to a JSON file."""
        file_path = file_path or self.save_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        save_data = {
            "metadata": {
                "num_skills": len(self._skill_configs),
            },
            "skills": {
                name: cfg.model_dump() for name, cfg in self._skill_configs.items()
            },
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

        logger.info(f"| 💾 Saved {len(self._skill_configs)} skill(s) to {file_path}")
        return file_path

    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load skill configs from a JSON file."""
        file_path = file_path or self.save_path
        if not os.path.exists(file_path):
            logger.warning(f"| ⚠️ Skill file not found: {file_path}")
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)

            skills_data = load_data.get("skills", {})
            for name, data in skills_data.items():
                self._skill_configs[name] = SkillConfig(**data)

            logger.info(f"| 📂 Loaded {len(self._skill_configs)} skill(s) from {file_path}")
            return True
        except Exception as e:
            logger.error(f"| ❌ Failed to load skills from {file_path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Skill execution (__call__)
    # ------------------------------------------------------------------

    async def __call__(
        self,
        name: str,
        input: Dict[str, Any],
        model_name: Optional[str] = None,
        ctx: SessionContext = None,
        **kwargs,
    ) -> SkillResponse:
        """Execute a skill: read SKILL.md → ask LLM to interpret instructions → return result.

        Args:
            name: Skill name.
            input: User-provided arguments for the skill.
            model_name: LLM model to use for instruction interpretation. Falls back to config default.
            ctx: Session context.
        """
        if ctx is None:
            ctx = SessionContext()

        skill_config = self._skill_configs.get(name)
        if skill_config is None:
            return SkillResponse(
                success=False,
                message=f"Skill '{name}' not found. Available skills: {list(self._skill_configs.keys())}",
            )

        logger.info(f"| 🎯 Executing skill '{name}' with input: {input}")

        # Build the LLM prompt from SKILL.md content + user input
        system_content = (
            "You are a skill execution engine. You are given a skill's full instructions "
            "(from its SKILL.md) and user-provided arguments. Your job is to:\n"
            "1. Read and understand the skill instructions.\n"
            "2. Follow the workflow described in the skill.\n"
            "3. Apply the user arguments to generate the appropriate output.\n"
            "4. Return ONLY the final result that the skill should produce.\n\n"
            f"Skill directory: {skill_config.skill_dir}\n"
        )

        if skill_config.scripts:
            system_content += f"Available scripts: {', '.join(skill_config.scripts)}\n"
        if skill_config.resources:
            system_content += f"Available resources: {', '.join(skill_config.resources)}\n"

        user_content = (
            f"## Skill Instructions (from SKILL.md)\n\n"
            f"{skill_config.content}\n\n"
            f"## User Arguments\n\n"
            f"```json\n{json.dumps(input, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Execute this skill and return the result."
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

        effective_model = model_name or getattr(config, "model_name", "openrouter/gemini-3-flash-preview")

        try:
            llm_response = await model_manager(
                model=effective_model,
                messages=messages,
            )

            result_text = llm_response.message
            logger.info(f"| ✅ Skill '{name}' executed successfully")

            return SkillResponse(
                success=True,
                message=result_text,
                extra=SkillExtra(
                    data={
                        "skill_name": name,
                        "input": input,
                        "skill_dir": skill_config.skill_dir,
                    }
                ),
            )

        except Exception as e:
            logger.error(f"| ❌ Skill '{name}' execution failed: {e}")
            return SkillResponse(
                success=False,
                message=f"Skill execution failed: {e}",
                extra=SkillExtra(data={"skill_name": name, "error": str(e)}),
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Release all loaded skills."""
        self._skill_configs.clear()
        logger.info("| 🧹 Skill context manager cleaned up")
