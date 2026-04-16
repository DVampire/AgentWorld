"""Skill Context Manager for loading, managing, and serving skills."""

import os
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.skill.types import SkillConfig, SkillResponse, SkillExtra
from src.model import model_manager
from src.message.types import SystemMessage, HumanMessage
from src.session import SessionContext
from src.utils import assemble_project_path, file_lock
from src.version import version_manager


DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "default_skills"


class SkillContextManager(BaseModel):
    """Manages the lifecycle of skills: discovery, loading, registration, update, and execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="Base directory for skill runtime data")
    save_path: str = Field(default=None, description="Path to persist loaded skill configs")
    contract_path: str = Field(default=None, description="Path to save the skill contract")

    _skill_configs: Dict[str, SkillConfig] = {}
    _skill_history_versions: Dict[str, Dict[str, SkillConfig]] = {}

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
        self._skill_history_versions: Dict[str, Dict[str, SkillConfig]] = {}

        logger.info(f"| 📁 Skill context manager base directory: {self.base_dir}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self, skill_names: Optional[List[str]] = None):
        """Discover and load skills from default and persisted sources.

        Args:
            skill_names: If provided, only load these skills.
        """
        discovered: Dict[str, SkillConfig] = {}

        # 1. Load from built-in default_skills directory
        default_configs = await self._load_from_directory(DEFAULT_SKILLS_DIR)
        discovered.update(default_configs)

        # 2. Load previously persisted skills from JSON (may contain user-registered skills)
        persisted_configs = await self._load_from_json()
        for name, persisted_cfg in persisted_configs.items():
            if name in discovered:
                existing = discovered[name]
                if version_manager.compare_versions(persisted_cfg.version, existing.version) > 0:
                    logger.info(f"| 🔄 Overriding skill '{name}' from directory (v{existing.version}) with persisted (v{persisted_cfg.version})")
                    discovered[name] = persisted_cfg
            else:
                discovered[name] = persisted_cfg

        # 3. Filter by name if requested
        if skill_names is not None:
            filtered: Dict[str, SkillConfig] = {}
            for name in skill_names:
                if name in discovered:
                    filtered[name] = discovered[name]
                else:
                    logger.warning(f"| ⚠️ Requested skill '{name}' not found in discovered skills")
            discovered = filtered

        # 4. Build text representations, register versions, and store
        for name, skill_config in discovered.items():
            skill_config.text = self._build_text_representation(skill_config)
            self._skill_configs[name] = skill_config

            if name not in self._skill_history_versions:
                self._skill_history_versions[name] = {}
            self._skill_history_versions[name][skill_config.version] = skill_config

            await version_manager.register_version("skill", name, skill_config.version)
            logger.info(f"| 🎯 Skill '{name}' v{skill_config.version} loaded from {skill_config.skill_dir}")

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

        frontmatter, body = self._parse_frontmatter(raw)

        name = frontmatter.get("name", skill_dir.name)
        description = frontmatter.get("description", "")
        version = frontmatter.get("version", "1.0.0")
        require_grad = str(frontmatter.get("require_grad", "false")).lower() == "true"
        metadata = {k: v for k, v in frontmatter.items() if k not in ("name", "description", "version", "require_grad")}

        scripts_dir = skill_dir / "scripts"
        scripts: List[str] = []
        if scripts_dir.is_dir():
            scripts = [str(p) for p in sorted(scripts_dir.rglob("*")) if p.is_file()]

        resources_dir = skill_dir / "resources"
        resources: List[str] = []
        if resources_dir.is_dir():
            resources = [str(p) for p in sorted(resources_dir.rglob("*")) if p.is_file()]

        reference_files: List[str] = []
        for md_file in sorted(skill_dir.glob("*.md")):
            if md_file.name == "SKILL.md":
                continue
            reference_files.append(str(md_file))

        return SkillConfig(
            name=name,
            description=description,
            metadata=metadata,
            require_grad=require_grad,
            version=version,
            skill_dir=str(skill_dir),
            content=body.strip(),
            scripts=scripts,
            resources=resources,
            reference_files=reference_files,
        )

    async def _load_from_json(self) -> Dict[str, SkillConfig]:
        """Load previously persisted skill configs (with version history) from JSON."""
        configs: Dict[str, SkillConfig] = {}

        if not os.path.exists(self.save_path):
            return configs

        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse skill config JSON: {e}")
            return configs

        skills_data = load_data.get("skills", {})
        for skill_name, skill_data in skills_data.items():
            try:
                current_version = skill_data.get("current_version", "1.0.0")
                versions = skill_data.get("versions", {})

                if not versions:
                    continue

                version_map: Dict[str, SkillConfig] = {}
                current_cfg: Optional[SkillConfig] = None

                for ver_str, ver_data in versions.items():
                    cfg = SkillConfig(**ver_data)
                    version_map[ver_str] = cfg
                    if ver_str == current_version:
                        current_cfg = cfg

                if skill_name not in self._skill_history_versions:
                    self._skill_history_versions[skill_name] = {}
                self._skill_history_versions[skill_name].update(version_map)

                if current_cfg is not None:
                    configs[skill_name] = current_cfg
                elif version_map:
                    configs[skill_name] = list(version_map.values())[-1]

                for cfg in version_map.values():
                    await version_manager.register_version("skill", skill_name, cfg.version)

            except Exception as e:
                logger.error(f"| ❌ Failed to load skill '{skill_name}' from JSON: {e}")

        logger.info(f"| 📂 Loaded {len(configs)} skill(s) from {self.save_path}")
        return configs

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
        """Split YAML frontmatter (between --- delimiters) from the markdown body."""
        pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = pattern.match(text)

        if not match:
            return {}, text

        yaml_block = match.group(1)
        body = text[match.end():]

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
        """Build a concise summary for prompt injection (no full SKILL.md body)."""
        parts = [
            f"Skill: {skill_config.name}",
            f"Description: {skill_config.description}",
            f"Version: {skill_config.version}",
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
            version: Explicit version string. If None, reads from frontmatter or auto-generates.

        Returns:
            The registered SkillConfig.
        """
        skill_dir_path = Path(skill_dir)
        if not (skill_dir_path / "SKILL.md").exists():
            raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")

        skill_config = self._parse_skill_dir(skill_dir_path)

        if version is not None:
            skill_config.version = version
        else:
            existing_version = await version_manager.get_version("skill", skill_config.name)
            if existing_version and skill_config.version == "1.0.0":
                skill_config.version = existing_version

        if skill_config.name in self._skill_configs and not override:
            raise ValueError(
                f"Skill '{skill_config.name}' already registered. Use override=True or update()."
            )

        skill_config.text = self._build_text_representation(skill_config)
        self._skill_configs[skill_config.name] = skill_config

        if skill_config.name not in self._skill_history_versions:
            self._skill_history_versions[skill_config.name] = {}
        self._skill_history_versions[skill_config.name][skill_config.version] = skill_config

        await version_manager.register_version("skill", skill_config.name, skill_config.version)

        await self.save_to_json()
        await self.save_contract()

        logger.info(f"| 📝 Registered skill: {skill_config.name} v{skill_config.version}")
        return skill_config

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

        You can either point to a new skill_dir (re-parse SKILL.md) or update
        individual fields (description, content, metadata) in-place.

        Args:
            name: Name of the skill to update.
            skill_dir: If provided, re-parse this directory as the new skill source.
            new_version: Explicit new version. If None, auto-increments patch.
            description: Override description text.
            content: Override SKILL.md body content.
            metadata: Override metadata dict.

        Returns:
            Updated SkillConfig.
        """
        original = self._skill_configs.get(name)
        if original is None:
            raise ValueError(f"Skill '{name}' not found. Use register() first.")

        if skill_dir is not None:
            updated = self._parse_skill_dir(Path(skill_dir))
        else:
            updated = SkillConfig(**original.model_dump())

        if description is not None:
            updated.description = description
        if content is not None:
            updated.content = content
        if metadata is not None:
            updated.metadata = metadata

        if new_version is None:
            new_version = await version_manager.generate_next_version("skill", name, "patch")
        updated.version = new_version

        updated.text = self._build_text_representation(updated)
        self._skill_configs[name] = updated

        if name not in self._skill_history_versions:
            self._skill_history_versions[name] = {}
        self._skill_history_versions[name][new_version] = updated

        await version_manager.register_version(
            "skill", name, new_version,
            description=description or f"Updated from {original.version}",
        )

        await self.save_to_json()
        await self.save_contract()

        logger.info(f"| 🔄 Updated skill '{name}' from v{original.version} to v{new_version}")
        return updated

    async def unregister(self, name: str) -> bool:
        """Remove a skill from the active set.

        Args:
            name: Skill name to unregister.

        Returns:
            True if removed, False if not found.
        """
        if name not in self._skill_configs:
            logger.warning(f"| ⚠️ Skill '{name}' not found")
            return False

        version = self._skill_configs[name].version
        del self._skill_configs[name]

        await self.save_to_json()
        await self.save_contract()

        logger.info(f"| 🗑️ Unregistered skill '{name}' v{version}")
        return True

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
            new_name: Name for the copy. If None, keeps the original name.
            new_version: Version for the copy. If None, auto-generates.
            new_skill_dir: If provided, physically copies the skill directory.

        Returns:
            The new SkillConfig.
        """
        original = self._skill_configs.get(name)
        if original is None:
            raise ValueError(f"Skill '{name}' not found")

        if new_name is None:
            new_name = name

        copied = SkillConfig(**original.model_dump())
        copied.name = new_name

        if new_skill_dir is not None:
            dest = Path(new_skill_dir)
            if not dest.exists():
                shutil.copytree(original.skill_dir, str(dest))
            copied.skill_dir = str(dest)

        if new_version is None:
            if new_name == name:
                new_version = await version_manager.generate_next_version("skill", new_name, "patch")
            else:
                new_version = await version_manager.get_version("skill", new_name)
        copied.version = new_version

        copied.text = self._build_text_representation(copied)
        self._skill_configs[new_name] = copied

        if new_name not in self._skill_history_versions:
            self._skill_history_versions[new_name] = {}
        self._skill_history_versions[new_name][new_version] = copied

        await version_manager.register_version(
            "skill", new_name, new_version,
            description=f"Copied from {name}@{original.version}",
        )

        await self.save_to_json()
        await self.save_contract()

        logger.info(f"| 📋 Copied skill '{name}' v{original.version} -> '{new_name}' v{new_version}")
        return copied

    async def restore(self, name: str, version: str) -> Optional[SkillConfig]:
        """Restore a specific version of a skill from history.

        Args:
            name: Skill name.
            version: Version string to restore.

        Returns:
            Restored SkillConfig, or None if version not found.
        """
        version_map = self._skill_history_versions.get(name, {})
        target = version_map.get(version)
        if target is None:
            logger.warning(f"| ⚠️ Version {version} not found for skill '{name}'")
            return None

        restored = SkillConfig(**target.model_dump())
        restored.text = self._build_text_representation(restored)
        self._skill_configs[name] = restored

        version_history = await version_manager.get_version_history("skill", name)
        if version_history:
            if version not in version_history.versions:
                await version_manager.register_version("skill", name, version)
            version_history.current_version = version
        else:
            await version_manager.register_version("skill", name, version)

        await self.save_to_json()

        logger.info(f"| 🔄 Restored skill '{name}' to v{version}")
        return restored

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    async def get(self, skill_name: str) -> Optional[SkillConfig]:
        """Get a loaded skill config by name."""
        return self._skill_configs.get(skill_name)

    async def get_info(self, skill_name: str) -> Optional[SkillConfig]:
        """Alias for get()."""
        return self._skill_configs.get(skill_name)

    async def list(self) -> List[str]:
        """Return names of all loaded skills."""
        return list(self._skill_configs.keys())

    # ------------------------------------------------------------------
    # Context generation (for agent prompt)
    # ------------------------------------------------------------------

    async def get_context(self, skill_names: Optional[List[str]] = None) -> str:
        """Build the full skill context string for prompt injection."""
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
    # Persistence (JSON) — with version history
    # ------------------------------------------------------------------

    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Persist all loaded skill configs with version history to JSON."""
        file_path = file_path or self.save_path

        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            save_data: Dict[str, Any] = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_skills": len(self._skill_configs),
                    "num_versions": sum(len(v) for v in self._skill_history_versions.values()),
                },
                "skills": {},
            }

            for skill_name, version_map in self._skill_history_versions.items():
                versions_data: Dict[str, Any] = {}
                for ver, cfg in version_map.items():
                    versions_data[ver] = cfg.model_dump()

                current_version = None
                if skill_name in self._skill_configs:
                    current_version = self._skill_configs[skill_name].version
                if current_version is None and version_map:
                    latest = None
                    for v in version_map:
                        if latest is None or version_manager.compare_versions(v, latest) > 0:
                            latest = v
                    current_version = latest

                save_data["skills"][skill_name] = {
                    "current_version": current_version,
                    "versions": versions_data,
                }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.info(f"| 💾 Saved {len(self._skill_configs)} skill(s) with version history to {file_path}")
            return file_path

    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load skill configs with version history from JSON."""
        file_path = file_path or self.save_path

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Skill file not found: {file_path}")
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)

                skills_data = load_data.get("skills", {})
                loaded = 0

                for skill_name, skill_data in skills_data.items():
                    try:
                        versions = skill_data.get("versions", {})
                        if not isinstance(versions, dict):
                            continue

                        current_version_str = skill_data.get("current_version")

                        version_configs: Dict[str, SkillConfig] = {}
                        latest_cfg: Optional[SkillConfig] = None

                        for ver_str, ver_data in versions.items():
                            cfg = SkillConfig(**ver_data)
                            version_configs[ver_str] = cfg

                            if current_version_str and cfg.version == current_version_str:
                                latest_cfg = cfg
                            elif latest_cfg is None:
                                latest_cfg = cfg

                        self._skill_history_versions[skill_name] = version_configs

                        if latest_cfg:
                            self._skill_configs[skill_name] = latest_cfg
                            for cfg in version_configs.values():
                                await version_manager.register_version("skill", skill_name, cfg.version)
                            loaded += 1

                    except Exception as e:
                        logger.error(f"| ❌ Failed to load skill '{skill_name}': {e}")

                logger.info(f"| 📂 Loaded {loaded} skill(s) with version history from {file_path}")
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
        """Execute a skill: read SKILL.md -> ask LLM to interpret instructions -> return result."""
        if ctx is None:
            ctx = SessionContext()

        skill_config = self._skill_configs.get(name)
        if skill_config is None:
            return SkillResponse(
                success=False,
                message=f"Skill '{name}' not found. Available skills: {list(self._skill_configs.keys())}",
            )

        logger.info(f"| 🎯 Executing skill '{name}' v{skill_config.version} with input: {input}")

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
                        "version": skill_config.version,
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
        self._skill_history_versions.clear()
        logger.info("| 🧹 Skill context manager cleaned up")
