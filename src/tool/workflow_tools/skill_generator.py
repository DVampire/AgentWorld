"""Skill Generator Tool - A workflow tool for creating skills and registering them to SCP.

File generation is split across parallel Generator classes so that each LLM call
handles a focused, reasonably-sized task instead of generating all files at once.
"""

import asyncio
import os
import json
import ast
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.model import model_manager
from src.utils import dedent
from src.message.types import HumanMessage, SystemMessage
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.skill.server import scp
from src.registry import TOOL


# ======================================================================
# Pydantic models
# ======================================================================

_SKILL_GENERATOR_DESCRIPTION = """Skill generator tool that creates complete skill packages and registers them in SCP.
This tool will:
1. Analyze task requirements and design a skill specification
2. Create a build plan (plan.md) listing all files to generate
3. Build from plan — generate all files concurrently, updating plan status in real-time
4. Validate the generated skill for structural correctness
5. Write files to disk, register in SCP, and run test cases

Args:
- task (str): Description of what the skill should do.
- skill_name (Optional[str]): Explicit skill name (kebab-case). If not provided, will be generated from task.
- description (Optional[str]): Explicit skill description. If not provided, will be generated from task.

Example: {"name": "skill_generator", "args": {"task": "Create a skill that translates text between languages", "skill_name": "text-translator", "description": "Translate text between multiple languages using templates and locale data."}}.
"""


class SkillFileSpec(BaseModel):
    """Specification for a single file to be included in the skill."""
    filename: str = Field(description="Relative path within the skill directory")
    purpose: str = Field(description="Brief description of what this file does")


class SkillSpecification(BaseModel):
    """Skill specification extracted from task analysis."""
    skill_name: str = Field(description="Name of the skill (kebab-case)")
    description: str = Field(description="Concise description of what the skill does")
    scripts: List[SkillFileSpec] = Field(default_factory=list, description="Scripts under scripts/")
    resources: List[SkillFileSpec] = Field(default_factory=list, description="Resources under resources/")
    include_examples: bool = Field(default=True, description="Whether to generate examples.md")
    include_reference: bool = Field(default=True, description="Whether to generate reference.md")
    implementation_plan: str = Field(description="High-level plan for the skill workflow")


class SkillFileContent(BaseModel):
    """A single generated file with its content."""
    filename: str = Field(description="Relative path within the skill directory")
    content: str = Field(description="Full file content")


class SkillEvaluation(BaseModel):
    """Validation result for a generated skill."""
    is_valid: bool = Field(description="Whether the skill passes validation")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Brief explanation")


class SkillTestCase(BaseModel):
    """A single test case stored in resources/test_cases.json."""
    name: str = Field(description="Short name for the test case")
    input: Dict[str, Any] = Field(description="Input arguments to pass to the skill")
    expected_behavior: str = Field(description="What success looks like")


class SkillFilePlan(BaseModel):
    """Plan entry for a single file to be generated."""
    filename: str = Field(description="Relative path within the skill directory")
    purpose: str = Field(description="What this file does")
    generator: str = Field(description="Generator class responsible for this file")
    status: str = Field(default="pending", description="pending | generating | completed | failed")


class SkillBuildPlan(BaseModel):
    """Complete build plan for a skill generation session."""
    skill_name: str
    description: str
    implementation_plan: str
    files: List[SkillFilePlan] = Field(default_factory=list)
    log: List[str] = Field(default_factory=list)
    created_at: str = Field(default="")


# ======================================================================
# Parallel file generators
# ======================================================================

_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*\n?", re.MULTILINE)
_FENCE_TAIL_PATTERN = re.compile(r"\n?```\s*$")


def _prefixed(prefix: str, filename: str) -> str:
    """Ensure *filename* has exactly one leading *prefix* (e.g. 'scripts/')."""
    return filename if filename.startswith(prefix) else f"{prefix}{filename}"


class _BaseFileGenerator:
    """Shared LLM helper and JSON parser for all generators."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(self, spec: SkillSpecification) -> List[SkillFileContent]:
        raise NotImplementedError

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await model_manager(model=self.model_name, messages=messages)
        return response.message.strip()

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove optional markdown code fences wrapping the entire response."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = _FENCE_PATTERN.sub("", cleaned, count=1)
            cleaned = _FENCE_TAIL_PATTERN.sub("", cleaned)
        return cleaned.strip()

    def _parse_file_list(self, raw: str, generator_name: str) -> List[SkillFileContent]:
        """Parse a JSON array of {filename, content} objects from LLM output."""
        cleaned = self._strip_fences(raw)
        try:
            data = json.loads(cleaned)
            items = data.get("files", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                items = [items]
            return [SkillFileContent(**item) for item in items]
        except Exception as e:
            logger.error(f"{generator_name}: failed to parse JSON output: {e}")
            return []


class SkillMdGenerator(_BaseFileGenerator):
    """Generates SKILL.md with YAML frontmatter and full instructions."""

    async def generate(self, spec: SkillSpecification) -> List[SkillFileContent]:
        scripts_ref = "\n".join(f"  - {_prefixed('scripts/', s.filename)}: {s.purpose}" for s in spec.scripts) or "  (none)"
        resources_ref = "\n".join(f"  - {_prefixed('resources/', r.filename)}: {r.purpose}" for r in spec.resources) or "  (none)"

        content = await self._call_llm(
            system_prompt=dedent("""You are an expert at writing SKILL.md files for AI agent skills.
            The file must start with YAML frontmatter between --- delimiters containing
            "name" and "description" fields, followed by comprehensive markdown instructions:
            quick start, step-by-step workflow, input/output tables, and references to
            scripts and resources."""),
            user_prompt=dedent(f"""Generate a SKILL.md for the following skill:

            Skill Name: {spec.skill_name}
            Description: {spec.description}
            Implementation Plan: {spec.implementation_plan}

            Scripts the skill will have:
            {scripts_ref}

            Resources the skill will have:
            {resources_ref}

            IMPORTANT:
            - Start with YAML frontmatter: name: {spec.skill_name} and description
            - Include sections: Quick Start, Instructions (step-by-step), Workflow,
              Examples, Configuration, Utility Scripts, Resources
            - Reference scripts and resources by their paths
            - Return ONLY the markdown content, no extra wrapping
            """),
        )

        return [SkillFileContent(filename="SKILL.md", content=self._strip_fences(content))]


_EXAMPLE_SCRIPT = '''\
"""Example skill script — demonstrates the expected class-based pattern."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"


class ExampleProcessor:
    """Core logic class that loads data from resources/ and exposes methods."""

    def __init__(self, resource_path: Optional[Path] = None):
        self._resource_path = resource_path or RESOURCES_DIR / "config.json"
        self._data = self._load_resource()

    def _load_resource(self) -> dict:
        try:
            with open(self._resource_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def run(self, input_text: str) -> str:
        """Process input and return result."""
        return f"Processed: {input_text}"


def main():
    parser = argparse.ArgumentParser(description="Example skill script")
    parser.add_argument("--input", required=True, help="Input text")
    args = parser.parse_args()

    processor = ExampleProcessor()
    print(processor.run(args.input))


if __name__ == "__main__":
    main()
'''


class ScriptsGenerator(_BaseFileGenerator):
    """Generates Python scripts under scripts/."""

    async def generate(self, spec: SkillSpecification) -> List[SkillFileContent]:
        if not spec.scripts:
            return []

        scripts_desc = "\n".join(f"- {s.filename}: {s.purpose}" for s in spec.scripts)
        resources_ref = "\n".join(f"- {_prefixed('resources/', r.filename)}: {r.purpose}" for r in spec.resources) or "(none)"

        raw = await self._call_llm(
            system_prompt=dedent("""You are an expert Python developer. Generate complete, production-ready
            Python scripts for an AI agent skill. Each script must be syntactically valid,
            include proper imports, argument parsing (argparse), and clear docstrings.
            Each script MUST use a class-based pattern: a core logic class that loads
            data from resources/ and exposes methods, plus a main() with argparse for CLI."""),
            user_prompt=dedent(f"""Generate Python scripts for the skill "{spec.skill_name}":

            Description: {spec.description}
            Implementation Plan: {spec.implementation_plan}

            Scripts to generate:
            {scripts_desc}

            Available resources the scripts can load:
            {resources_ref}

            Follow this reference pattern for each script:

            ```python
            {_EXAMPLE_SCRIPT}
            ```

            Key conventions:
            - RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
            - One main class per script with resource loading in __init__
            - A main() function with argparse for CLI usage
            - if __name__ == "__main__": main()

            Return a JSON array of objects with "filename" and "content" keys.
            Each filename should be prefixed with "scripts/" (e.g. "scripts/main.py").
            Return ONLY the JSON array, no markdown fences.
            """),
        )

        return self._parse_file_list(raw, "ScriptsGenerator")


class ResourcesGenerator(_BaseFileGenerator):
    """Generates resource / data files under resources/, including test_cases.json."""

    async def generate(self, spec: SkillSpecification) -> List[SkillFileContent]:
        resource_specs = list(spec.resources)
        resource_specs.append(SkillFileSpec(
            filename="test_cases.json",
            purpose="Structured test cases for automated skill testing. "
                    "Array of objects with name, input (dict), and expected_behavior (string).",
        ))

        resources_desc = "\n".join(f"- {r.filename}: {r.purpose}" for r in resource_specs)

        raw = await self._call_llm(
            system_prompt=dedent("""You are an expert at creating data and configuration files for AI agent skills.
            Generate valid JSON resource files. Each file must be well-structured and
            immediately usable by the skill's scripts."""),
            user_prompt=dedent(f"""Generate resource files for the skill "{spec.skill_name}":

            Description: {spec.description}
            Implementation Plan: {spec.implementation_plan}

            Resources to generate:
            {resources_desc}

            IMPORTANT for test_cases.json:
            - Must be a JSON array of objects
            - Each object: {{"name": "...", "input": {{...}}, "expected_behavior": "..."}}
            - Generate 2-3 realistic, self-contained test cases
            - "input" should be a dict of arguments the skill can accept

            Return a JSON array of objects with "filename" and "content" keys.
            Each filename should be prefixed with "resources/" (e.g. "resources/config.json").
            "content" must be a string containing valid JSON.
            Return ONLY the JSON array, no markdown fences.
            """),
        )

        return self._parse_file_list(raw, "ResourcesGenerator")


class DocsGenerator(_BaseFileGenerator):
    """Generates examples.md and/or reference.md."""

    async def generate(self, spec: SkillSpecification) -> List[SkillFileContent]:
        docs_to_gen = []
        if spec.include_examples:
            docs_to_gen.append("examples.md")
        if spec.include_reference:
            docs_to_gen.append("reference.md")
        if not docs_to_gen:
            return []

        scripts_ref = "\n".join(f"- {_prefixed('scripts/', s.filename)}: {s.purpose}" for s in spec.scripts) or "(none)"
        resources_ref = "\n".join(f"- {_prefixed('resources/', r.filename)}: {r.purpose}" for r in spec.resources) or "(none)"

        raw = await self._call_llm(
            system_prompt=dedent("""You are an expert technical writer. Generate documentation files
            for an AI agent skill. examples.md should have concrete, varied usage examples.
            reference.md should have API/CLI reference with parameter tables and exit codes."""),
            user_prompt=dedent(f"""Generate documentation for the skill "{spec.skill_name}":

            Description: {spec.description}
            Implementation Plan: {spec.implementation_plan}

            Scripts: {scripts_ref}
            Resources: {resources_ref}

            Files to generate: {json.dumps(docs_to_gen)}

            Return a JSON array of objects with "filename" and "content" keys.
            Return ONLY the JSON array, no markdown fences.
            """),
        )

        return self._parse_file_list(raw, "DocsGenerator")


# ======================================================================
# Main tool
# ======================================================================

_TEST_TIMEOUT = 120.0


@TOOL.register_module(force=True)
class SkillGeneratorTool(Tool):
    """A workflow tool that generates complete skill packages and registers them in SCP."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "skill_generator"
    description: str = _SKILL_GENERATOR_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")

    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for skill generation.",
    )
    base_dir: str = Field(
        default="workdir/skill_generator",
        description="Root directory where generated skill directories are stored.",
    )

    def __init__(
        self,
        base_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        require_grad: bool = False,
        **kwargs,
    ):
        super().__init__(require_grad=require_grad, **kwargs)

        from src.utils import assemble_project_path

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        elif hasattr(self, "base_dir"):
            self.base_dir = assemble_project_path(self.base_dir)
        else:
            self.base_dir = assemble_project_path("workdir/skill_generator")

        os.makedirs(self.base_dir, exist_ok=True)

        if model_name is not None:
            self.model_name = model_name

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        skill_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResponse:
        """Execute the skill generation workflow.

        Args:
            task: Description of what the skill should do.
            skill_name: Explicit skill name (kebab-case).
            description: Explicit skill description.
        """
        try:
            logger.info(f"🎯 Starting skill generation for task: {task}")

            # Step 1 — Analyze task
            logger.info("📋 Step 1: Analyzing task requirements...")
            spec = await self._analyze_task(task, skill_name, description)
            logger.info(f"| ✅ Skill specification: {spec.skill_name}")

            # Step 2 — Create build plan
            logger.info("📝 Step 2: Creating build plan...")
            plan = self._create_build_plan(spec)
            skill_dir = os.path.join(self.base_dir, spec.skill_name)
            plan_path = self._write_plan(plan, skill_dir)
            logger.info(f"| ✅ Build plan written to {plan_path} ({len(plan.files)} file(s) planned)")

            # Step 3 — Build from plan (concurrent file generation)
            logger.info("🔨 Step 3: Building from plan (concurrent generation)...")
            files = await self._build_from_plan(spec, plan, skill_dir)
            logger.info(f"| ✅ Generated {len(files)} file(s)")

            # Step 4 — Validate
            logger.info("✅ Step 4: Evaluating generated skill...")
            evaluation = self._evaluate_skill(spec, files)

            if not evaluation.is_valid:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                plan.log.append(f"[{now}] ❌ Evaluation failed: {evaluation.reasoning}")
                self._write_plan(plan, skill_dir)
                logger.warning(f"| ⚠️ Skill evaluation failed: {evaluation.reasoning}")
                return ToolResponse(
                    success=False,
                    message=f"Skill generation failed evaluation: {evaluation.reasoning}. Errors: {', '.join(evaluation.errors)}",
                    extra=ToolExtra(data={
                        "skill_name": spec.skill_name,
                        "action": "evaluation_failed",
                        "evaluation": evaluation.model_dump(),
                        "specification": spec.model_dump(),
                        "plan": plan.model_dump(),
                    }),
                )

            if evaluation.warnings:
                for w in evaluation.warnings:
                    logger.warning(f"| ⚠️ {w}")

            # Step 5 — Write generated files to disk
            logger.info("📝 Step 5: Writing skill files to disk...")
            skill_dir = self._write_skill_directory(spec, files)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plan.log.append(f"[{now}] All files written to {skill_dir}")
            self._write_plan(plan, skill_dir)
            logger.info(f"| ✅ Skill directory created at {skill_dir}")

            # Step 6 — Register in SCP
            logger.info("📝 Step 6: Registering skill in SCP...")
            try:
                skill_config = await self._register_skill(skill_dir)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                plan.log.append(f"[{now}] Registered in SCP: {skill_config.name} v{skill_config.version}")
                self._write_plan(plan, skill_dir)
                logger.info(f"| ✅ Registered skill: {skill_config.name} v{skill_config.version}")
            except Exception as e:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                plan.log.append(f"[{now}] ❌ Registration failed: {e}")
                self._write_plan(plan, skill_dir)
                logger.error(f"| ❌ Skill registration failed: {e}")
                return ToolResponse(
                    success=False,
                    message=f"Skill generation succeeded but registration failed: {e}",
                    extra=ToolExtra(file_path=skill_dir, data={
                        "skill_name": spec.skill_name,
                        "skill_dir": skill_dir,
                        "action": "registration_failed",
                        "error": str(e),
                        "specification": spec.model_dump(),
                        "evaluation": evaluation.model_dump(),
                        "plan": plan.model_dump(),
                    }),
                )

            # Step 7 — Test using test_cases.json from the generated skill
            logger.info("🧪 Step 7: Running test cases from resources/test_cases.json...")
            test_passed, test_details = await self._test_skill(skill_config.name, skill_dir)

            if not test_passed:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                plan.log.append(f"[{now}] ❌ Tests failed, skill unregistered: {test_details}")
                self._write_plan(plan, skill_dir)
                logger.warning(f"| ⚠️ Skill testing failed, unregistering: {test_details}")
                await scp.unregister(skill_config.name)
                return ToolResponse(
                    success=False,
                    message=f"Skill '{skill_config.name}' failed runtime testing and was discarded. Details: {test_details}",
                    extra=ToolExtra(file_path=skill_dir, data={
                        "skill_name": skill_config.name,
                        "skill_dir": skill_dir,
                        "action": "test_failed_and_discarded",
                        "test_details": test_details,
                        "specification": spec.model_dump(),
                        "evaluation": evaluation.model_dump(),
                        "plan": plan.model_dump(),
                    }),
                )

            logger.info(f"| ✅ Skill '{skill_config.name}' passed all tests")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plan.log.append(f"[{now}] ✅ All tests passed — skill ready")
            self._write_plan(plan, skill_dir)

            file_list = "\n".join(f"  - {f.filename}" for f in files)
            return ToolResponse(
                success=True,
                message=(
                    f"Successfully generated, tested, and registered skill '{skill_config.name}' "
                    f"(v{skill_config.version}).\n"
                    f"Description: {spec.description}\n"
                    f"Directory: {skill_dir}\n"
                    f"Plan: {plan_path}\n"
                    f"Files:\n{file_list}"
                ),
                extra=ToolExtra(file_path=skill_dir, data={
                    "skill_name": skill_config.name,
                    "skill_version": skill_config.version,
                    "skill_dir": skill_dir,
                    "action": "created_tested_and_registered",
                    "specification": spec.model_dump(),
                    "evaluation": evaluation.model_dump(),
                    "plan": plan.model_dump(),
                    "test_details": test_details,
                    "files": [f.filename for f in files],
                }),
            )

        except Exception as e:
            logger.error(f"❌ Error in skill generation: {e}")
            return ToolResponse(success=False, message=f"Error during skill generation: {e}")

    # ------------------------------------------------------------------
    # Step 1: Task analysis
    # ------------------------------------------------------------------

    async def _analyze_task(
        self,
        task: str,
        skill_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SkillSpecification:
        """Use LLM to analyze the task and produce a SkillSpecification."""

        system_prompt = dedent("""You are an expert at designing AI agent skills.
        A skill is a self-contained package with a SKILL.md instruction file,
        optional Python scripts (under scripts/), optional resource/data files
        (under resources/), and optional reference documentation.
        Analyze the given task and design a complete skill specification.""")

        user_prompt = dedent(f"""Design a skill specification for the following task:

        Task: {task}

        {"Skill name (if specified): " + skill_name if skill_name else ""}
        {"Skill description (if specified): " + description if description else ""}

        Requirements:
        1. skill_name: A clear, descriptive name in kebab-case (e.g. "text-translator")
        2. description: A concise description of what the skill does and when an agent should use it
        3. scripts: List of Python utility scripts needed (filename under scripts/, plus purpose)
        4. resources: List of data/config files needed (filename under resources/, plus purpose)
        5. include_examples: Whether an examples.md would be useful
        6. include_reference: Whether a reference.md would be useful
        7. implementation_plan: High-level plan describing the skill workflow
        """)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await model_manager(
            model=self.model_name,
            messages=messages,
            response_format=SkillSpecification,
        )

        if response.extra and hasattr(response.extra, "parsed_model") and response.extra.parsed_model:
            spec = response.extra.parsed_model
        else:
            try:
                spec_dict = json.loads(response.message.strip())
                spec = SkillSpecification(**spec_dict)
            except Exception as e:
                logger.warning(f"Failed to parse specification, using defaults: {e}")
                spec = SkillSpecification(
                    skill_name=skill_name or "generated-skill",
                    description=description or task,
                    implementation_plan="Generate skill based on task requirements",
                )

        if skill_name:
            spec.skill_name = skill_name
        if description:
            spec.description = description

        return spec

    # ------------------------------------------------------------------
    # Step 2: Build plan
    # ------------------------------------------------------------------

    def _create_build_plan(self, spec: SkillSpecification) -> SkillBuildPlan:
        """Derive a build plan from the skill specification."""
        files: List[SkillFilePlan] = []

        files.append(SkillFilePlan(
            filename="SKILL.md",
            purpose="Main skill instruction file with YAML frontmatter and workflow docs",
            generator="SkillMdGenerator",
        ))

        for s in spec.scripts:
            files.append(SkillFilePlan(
                filename=_prefixed("scripts/", s.filename),
                purpose=s.purpose,
                generator="ScriptsGenerator",
            ))

        for r in spec.resources:
            files.append(SkillFilePlan(
                filename=_prefixed("resources/", r.filename),
                purpose=r.purpose,
                generator="ResourcesGenerator",
            ))

        has_test_cases = any(fp.filename == "resources/test_cases.json" for fp in files)
        if not has_test_cases:
            files.append(SkillFilePlan(
                filename="resources/test_cases.json",
                purpose="Structured test cases for automated skill testing",
                generator="ResourcesGenerator",
            ))

        if spec.include_examples:
            files.append(SkillFilePlan(
                filename="examples.md",
                purpose="Concrete usage examples",
                generator="DocsGenerator",
            ))
        if spec.include_reference:
            files.append(SkillFilePlan(
                filename="reference.md",
                purpose="API/CLI reference documentation",
                generator="DocsGenerator",
            ))

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plan = SkillBuildPlan(
            skill_name=spec.skill_name,
            description=spec.description,
            implementation_plan=spec.implementation_plan,
            files=files,
            created_at=now,
        )
        plan.log.append(f"[{now}] Plan created — {len(files)} file(s) to generate")
        return plan

    @staticmethod
    def _render_plan_md(plan: SkillBuildPlan) -> str:
        """Render the build plan as a human-readable markdown document."""
        _STATUS_ICON = {
            "pending": "⏳",
            "generating": "🔄",
            "completed": "✅",
            "failed": "❌",
        }

        lines = [
            f"# Build Plan: {plan.skill_name}",
            "",
            f"> {plan.description}",
            "",
            "## Implementation Plan",
            "",
            plan.implementation_plan,
            "",
            "## Files",
            "",
        ]

        for i, fp in enumerate(plan.files, 1):
            icon = _STATUS_ICON.get(fp.status, "❓")
            checkbox = "x" if fp.status == "completed" else " "
            lines.append(
                f"- [{checkbox}] {icon} `{fp.filename}` — {fp.purpose}  "
                f"*({fp.generator})*"
            )

        lines.extend(["", "## Build Log", ""])
        for entry in plan.log:
            lines.append(f"- {entry}")
        lines.append("")

        return "\n".join(lines)

    def _write_plan(self, plan: SkillBuildPlan, skill_dir: str) -> str:
        """Write (or overwrite) plan.md inside the skill directory."""
        os.makedirs(skill_dir, exist_ok=True)
        plan_path = os.path.join(skill_dir, "plan.md")
        with open(plan_path, "w", encoding="utf-8") as fh:
            fh.write(self._render_plan_md(plan))
        return plan_path

    # ------------------------------------------------------------------
    # Step 3: Build from plan (concurrent file generation)
    # ------------------------------------------------------------------

    async def _build_from_plan(
        self,
        spec: SkillSpecification,
        plan: SkillBuildPlan,
        skill_dir: str,
    ) -> List[SkillFileContent]:
        """Run all generators concurrently, updating plan.md as each completes."""

        generators: List[_BaseFileGenerator] = [
            SkillMdGenerator(self.model_name),
            ScriptsGenerator(self.model_name),
            ResourcesGenerator(self.model_name),
            DocsGenerator(self.model_name),
        ]

        plan_lock = asyncio.Lock()

        async def _run_gen(gen: _BaseFileGenerator) -> List[SkillFileContent]:
            gen_name = type(gen).__name__
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            async with plan_lock:
                for fp in plan.files:
                    if fp.generator == gen_name and fp.status == "pending":
                        fp.status = "generating"
                plan.log.append(f"[{now}] {gen_name} started")
                self._write_plan(plan, skill_dir)

            try:
                files = await gen.generate(spec)
                generated_names = {f.filename for f in files}
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                async with plan_lock:
                    for fp in plan.files:
                        if fp.generator == gen_name:
                            if fp.filename in generated_names:
                                fp.status = "completed"
                            elif fp.status == "generating":
                                fp.status = "failed"
                    plan.log.append(f"[{now}] {gen_name} completed — {len(files)} file(s)")
                    self._write_plan(plan, skill_dir)

                logger.info(f"| ✅ {gen_name} produced {len(files)} file(s)")
                return files

            except Exception as e:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                async with plan_lock:
                    for fp in plan.files:
                        if fp.generator == gen_name and fp.status == "generating":
                            fp.status = "failed"
                    plan.log.append(f"[{now}] {gen_name} FAILED — {e}")
                    self._write_plan(plan, skill_dir)
                logger.error(f"| ❌ {gen_name} failed: {e}")
                raise

        results = await asyncio.gather(
            *[_run_gen(g) for g in generators],
            return_exceptions=True,
        )

        all_files: List[SkillFileContent] = []
        for gen, result in zip(generators, results):
            if isinstance(result, Exception):
                continue
            all_files.extend(result)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plan.log.append(f"[{now}] Build finished — {len(all_files)} total file(s)")
        self._write_plan(plan, skill_dir)

        return all_files

    # ------------------------------------------------------------------
    # Step 4: Validation
    # ------------------------------------------------------------------

    def _evaluate_skill(
        self, spec: SkillSpecification, files: List[SkillFileContent]
    ) -> SkillEvaluation:
        """Validate generated files without calling LLM."""
        errors: List[str] = []
        warnings: List[str] = []

        filenames = {f.filename for f in files}

        if "SKILL.md" not in filenames:
            errors.append("Missing required file: SKILL.md")
        else:
            skill_md = next(f for f in files if f.filename == "SKILL.md")
            if "---" not in skill_md.content:
                errors.append("SKILL.md is missing YAML frontmatter (--- delimiters)")
            else:
                if "name:" not in skill_md.content:
                    errors.append("SKILL.md frontmatter missing 'name' field")
                if "description:" not in skill_md.content:
                    errors.append("SKILL.md frontmatter missing 'description' field")

        for f in files:
            if f.filename.endswith(".py"):
                try:
                    ast.parse(f.content)
                except SyntaxError as e:
                    errors.append(f"Syntax error in {f.filename}: {e}")

        for f in files:
            if f.filename.endswith(".json"):
                try:
                    json.loads(f.content)
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON in {f.filename}: {e}")

        for s in spec.scripts:
            fname = _prefixed("scripts/", s.filename)
            if fname not in filenames:
                warnings.append(f"Specified script not generated: {fname}")
        for r in spec.resources:
            fname = _prefixed("resources/", r.filename)
            if fname not in filenames:
                warnings.append(f"Specified resource not generated: {fname}")

        if "resources/test_cases.json" not in filenames:
            warnings.append("resources/test_cases.json not generated; runtime tests will be skipped")

        if errors:
            return SkillEvaluation(
                is_valid=False, errors=errors, warnings=warnings,
                reasoning=f"Validation failed with {len(errors)} error(s)",
            )

        return SkillEvaluation(
            is_valid=True, errors=errors, warnings=warnings,
            reasoning="Skill passed all validation checks",
        )

    # ------------------------------------------------------------------
    # Step 5: Write to disk
    # ------------------------------------------------------------------

    def _write_skill_directory(
        self, spec: SkillSpecification, files: List[SkillFileContent]
    ) -> str:
        """Write all generated files into base_dir/<skill_name>/."""
        skill_dir = os.path.join(self.base_dir, spec.skill_name)
        os.makedirs(skill_dir, exist_ok=True)

        for f in files:
            file_path = os.path.join(skill_dir, f.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(f.content)
            logger.info(f"| 📄 Wrote {f.filename}")

        return skill_dir

    # ------------------------------------------------------------------
    # Step 6: Register in SCP
    # ------------------------------------------------------------------

    async def _register_skill(self, skill_dir: str):
        """Register the skill directory in SCP."""
        return await scp.register(skill_dir=skill_dir, override=True)

    # ------------------------------------------------------------------
    # Step 7: Test from resources/test_cases.json
    # ------------------------------------------------------------------

    @staticmethod
    def _load_test_cases(skill_dir: str) -> List[SkillTestCase]:
        """Read test cases from the generated resources/test_cases.json."""
        path = os.path.join(skill_dir, "resources", "test_cases.json")
        if not os.path.exists(path):
            logger.warning(f"| ⚠️ test_cases.json not found at {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, list):
                return [SkillTestCase(**tc) for tc in raw]
            if isinstance(raw, dict) and "test_cases" in raw:
                return [SkillTestCase(**tc) for tc in raw["test_cases"]]

            logger.warning("test_cases.json has unexpected structure")
            return []
        except Exception as e:
            logger.error(f"Failed to load test_cases.json: {e}")
            return []

    async def _run_single_test(self, skill_name: str, tc: SkillTestCase) -> str:
        """Run one test case with a timeout, returning a PASS/FAIL/ERROR string."""
        try:
            logger.info(f"| 🧪 Running test '{tc.name}' with input: {tc.input}")
            response = await asyncio.wait_for(
                scp(name=skill_name, input=tc.input),
                timeout=_TEST_TIMEOUT,
            )
            if response.success:
                logger.info(f"| ✅ Test '{tc.name}' passed")
                return f"PASS: {tc.name}"
            logger.warning(f"| ❌ Test '{tc.name}' failed: {response.message}")
            return f"FAIL: {tc.name} — {response.message}"
        except asyncio.TimeoutError:
            logger.error(f"| ⏱️ Test '{tc.name}' timed out after {_TEST_TIMEOUT}s")
            return f"TIMEOUT: {tc.name}"
        except Exception as e:
            logger.error(f"| ❌ Test '{tc.name}' raised exception: {e}")
            return f"ERROR: {tc.name} — {str(e)}"

    async def _test_skill(self, skill_name: str, skill_dir: str) -> Tuple[bool, str]:
        """Load test cases from disk and run them via SCP in parallel.

        Returns:
            (passed, details)
        """
        test_cases = self._load_test_cases(skill_dir)
        if not test_cases:
            return True, "No test cases found; skipping runtime tests."

        results = await asyncio.gather(
            *[self._run_single_test(skill_name, tc) for tc in test_cases]
        )

        all_passed = all(r.startswith("PASS") for r in results)
        details = "; ".join(results)
        return all_passed, details
