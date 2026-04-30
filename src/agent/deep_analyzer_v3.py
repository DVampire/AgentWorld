from __future__ import annotations

import asyncio
import os
import re
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse, ThinkOutput
from src.logger import logger
from src.model import model_manager
from src.message import (
    HumanMessage,
    ContentPartText,
    ContentPartImage, ImageURL,
    ContentPartAudio, AudioURL,
    ContentPartVideo, VideoURL,
    ContentPartPdf, PdfURL,
)
from src.session import SessionContext
from src.registry import AGENT
from src.prompt import prompt_manager
from src.utils import fetch_url, make_file_url, dedent, parse_tool_args, PlanFile, make_plan_path
from src.tool.types import Tool, ToolResponse, ToolExtra

# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------

class FileTypeInfo(BaseModel):
    file: str = Field(description="Exact file path or URL as provided")
    file_type: str = Field(description="One of: text, pdf, image, audio, video")


class SynthOutput(BaseModel):
    per_model_summaries: str = Field(
        description=(
            "One line per model: '[model-name]: <2-3 sentence summary of that model's key conclusion>'. "
            "Preserve exact claims, numbers, and qualifiers."
        )
    )
    combined_reasoning: str = Field(
        description=(
            "Integration across all models: shared conclusions stated clearly, "
            "disagreements quoted specifically. Do not flatten differences into false consensus."
        )
    )


class EvalOutput(BaseModel):
    candidate_answer: Optional[str] = Field(
        default=None,
        description="The best answer from the synthesis; None if no clear answer emerged."
    )
    found_answer: bool = Field(
        description="True when a clear, consistent answer exists with no unresolved conflicts."
    )
    has_conflict: bool = Field(
        description="True if models produced meaningfully contradictory conclusions."
    )
    conflict_description: Optional[str] = Field(
        default=None,
        description="Exact conflict quote + what evidence would resolve it; None if no conflict."
    )



# ---------------------------------------------------------------------------
# File-type constants
# ---------------------------------------------------------------------------

_YOUTUBE_PATTERNS = [
    r"youtube\.com/watch\?v=",
    r"youtu\.be/",
    r"youtube\.com/embed/",
    r"youtube\.com/v/",
]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
_PDF_EXTS = {".pdf"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".m4b", ".m4p"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# BaseParallelTool — shared file handling and synth logic
# ---------------------------------------------------------------------------
class BaseParallelTool(Tool):
    """Base tool for running analysis tasks in parallel across multiple files using multiple language models."""

    name: str = "base_parallel_tool"
    description: str = "Runs analysis tasks in parallel across multiple files using multiple language models."
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradient computation")
    model_name: str = Field(default="", description="Model name for synthesis")
    workdir: str = Field(default="", description="Working directory for file downloads")
    timeout: int = Field(default=30, description="Maximum seconds to wait for each parallel execution")

    def __init__(self,
                 model_name: str,
                 workdir: str,
                 require_grad: bool = False,
                 **kwargs):
        super().__init__(model_name=model_name, workdir=workdir, require_grad=require_grad, **kwargs)

    async def _run_files_parallel(self, 
                                  task: str, 
                                  files: Optional[List[str]],
                                  prompt_name: str,
                                  data: Dict[str, Any],
                                  model_names: List[str])-> Optional[str]:
        if files:
            file_type_infos = [FileTypeInfo(file=f, file_type=_infer_file_type(f)) for f in files]
            file_results = await asyncio.gather(
                *[self._run_on_file(task, file_type_info, prompt_name, data, model_names) for file_type_info in file_type_infos],
                return_exceptions=True,
            )
            raw_parts: List[str] = []
            for fi, result in zip(file_type_infos, file_results):
                if isinstance(result, Exception):
                    logger.warning(f"| ❌ {self.__class__.__name__} failed ({fi.file}): {result}")
                elif result:
                    raw_parts.append(f"### {fi.file_type} — {fi.file}\n\n{result}")
            return "\n\n---\n\n".join(raw_parts)
        else:
            return await self._run_on_file(task, 
                                           file_type_info=None, 
                                           prompt_name=prompt_name, 
                                           data=data, 
                                           model_names=model_names)

    async def _run_on_file(self,
                           task: str,
                           file_type_info: Optional[FileTypeInfo],
                           prompt_name: str,
                           data: Dict[str, Any],
                           model_names: List[str]) -> str:
        
        messages = await prompt_manager.get_messages(
            prompt_name=prompt_name,
            agent_modules={
                "task": task,
                "data": data,
                "file_type": file_type_info.file_type if file_type_info else "",
            },
        )

        if file_type_info is not None:
            content_part = await self._build_content_part(file_type_info)
            if content_part is None:
                logger.warning(f"| ⚠️ {self.__class__.__name__} skipped ({file_type_info.file}): failed to load")
                return ""
            last = messages[-1]
            text_content = last.content if isinstance(last.content, str) else str(last.content)
            messages = messages[:-1] + [HumanMessage(content=[
                ContentPartText(text=text_content),
                content_part,
            ])]
        
        return await _run_models_parallel(messages, model_names)

    async def _synth(self,
                     task: str,
                     prompt_name: str,
                     data: Dict[str, Any]) -> Optional[str]:
        try:
            messages = await prompt_manager.get_messages(
                prompt_name=prompt_name,
                agent_modules={
                    "task": task,
                    "data": data,
                },
            )
            resp = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=SynthOutput,
            )
            if resp and resp.extra and hasattr(resp.extra, "parsed_model") and resp.extra.parsed_model:
                synth: SynthOutput = resp.extra.parsed_model
                return f"{synth.per_model_summaries}\n\n{synth.combined_reasoning}"
        except Exception as exc:
            logger.warning(f"| ⚠️ {self.__class__.__name__} synth failed: {exc}")
        return None

    async def _build_content_part(self, fi: FileTypeInfo) -> Optional[Any]:
        file, file_type = fi.file, fi.file_type

        if file_type == "image":
            local = await _download_file(file, "image", self.workdir) if _is_url(file) else file
            if not local:
                return None
            return ContentPartImage(image_url=ImageURL(url=make_file_url(file_path=local), detail="high"))

        if file_type == "pdf":
            local = await _download_file(file, "pdf", self.workdir) if _is_url(file) else file
            if not local:
                return None
            return ContentPartPdf(pdf_url=PdfURL(url=make_file_url(file_path=local)))

        if file_type == "audio":
            local = await _download_file(file, "audio", self.workdir) if _is_url(file) else file
            if not local:
                return None
            return ContentPartAudio(audio_url=AudioURL(url=make_file_url(file_path=local)))

        if file_type == "video":
            if _is_url(file):
                url = file if _is_youtube_url(file) else (
                    make_file_url(file_path=p) if (p := await _download_file(file, "video", self.workdir)) else None
                )
            else:
                url = make_file_url(file_path=file)
            if not url:
                return None
            return ContentPartVideo(video_url=VideoURL(url=url))

        try:
            if _is_url(file):
                result = await fetch_url(file)
                content = result.get("markdown") or result.get("text", "") if result else ""
            else:
                with open(file, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            if not content:
                return None
            return ContentPartText(text=f"File: {file}\n\n{content}")
        except Exception as exc:
            logger.error(f"| ❌ Failed to read text file ({file}): {exc}")
            return None
        
    async def __call__(self, **kwargs) -> ToolResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool: AnalyzeTool
# ---------------------------------------------------------------------------

class AnalyzeTool(BaseParallelTool):
    
    name: str = "analyze_tool"
    description: str = "Analyzes files and produces a synthesis of the results."
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradient computation")
    
    timeout: int = Field(default=30, description="Maximum seconds to wait for execution")

    async def __call__(self, **kwargs) -> ToolResponse:
        try:
            task = kwargs.get("task", "")
            files = kwargs.get("files", [])
            prompt_name = "deep_analyzer_v3_analyze"
            data = kwargs.get("data", {})
            model_names = kwargs.get("model_names", [self.model_name])

            results = await self._run_files_parallel(task, files, prompt_name, data, model_names)
            if not results:
                return ToolResponse(success=False, message="No output produced.")
            summary = await self._synth(task, prompt_name="deep_analyzer_v3_synth", data=results)
            if not summary:
                return ToolResponse(success=False, message="Synthesis produced no output.")
            return ToolResponse(success=True, message=summary, extra=ToolExtra(data={"summary": summary}))
        except Exception as exc:
            logger.error(f"| ❌ AnalyzeTool error: {exc}", exc_info=True)
            return ToolResponse(success=False, message=str(exc), extra=ToolExtra())


# ---------------------------------------------------------------------------
# Tool: EvalTool
# ---------------------------------------------------------------------------

class EvalTool(Tool):

    name: str = "eval_tool"
    description: str = "Reads the latest synthesis and judges whether the task is answered and whether models conflict."
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradient computation")
    model_name: str = Field(default="", description="Model name for evaluation")
    timeout: int = Field(default=30, description="Maximum seconds to wait for execution")

    def __init__(self,
                 model_name: str,
                 require_grad: bool = False,
                 **kwargs):
        super().__init__(model_name=model_name, require_grad=require_grad, **kwargs)

    async def __call__(self, **kwargs) -> ToolResponse:
        try:
            task = kwargs.get("task", "")
            data = kwargs.get("data", {})
            prompt_name = "deep_analyzer_v3_eval"

            messages = await prompt_manager.get_messages(
                prompt_name=prompt_name,
                agent_modules={
                    "task": task,
                    "data": data,
                },
            )
            resp = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=EvalOutput,
            )
            if resp and resp.extra and hasattr(resp.extra, "parsed_model") and resp.extra.parsed_model:
                eval_output: EvalOutput = resp.extra.parsed_model
                lines = [
                    f"- **Found Answer:** {eval_output.found_answer}",
                    f"- **Has Conflict:** {eval_output.has_conflict}",
                    f"- **Candidate Answer:** {eval_output.candidate_answer or 'N/A'}",
                ]
                if eval_output.conflict_description:
                    lines.append(f"\n**Conflict:**\n{eval_output.conflict_description}")
                return ToolResponse(
                    success=True,
                    message="\n".join(lines),
                    extra=ToolExtra(parsed_model=eval_output),
                )
        except Exception as exc:
            logger.warning(f"| ❌ Eval failed: {exc}")
        fallback = EvalOutput(found_answer=False, has_conflict=False)
        return ToolResponse(
            success=False,
            message="- **Found Answer:** False\n- **Has Conflict:** False\n- **Candidate Answer:** N/A",
            extra=ToolExtra(parsed_model=fallback),
        )


# ---------------------------------------------------------------------------
# Tool: VerifyTool
# ---------------------------------------------------------------------------

class VerifyTool(BaseParallelTool):
    name: str = "verify_tool"
    description: str = "Adversarially challenges the candidate answer with multiple LLMs. Call only after eval returns found_answer=true."
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradient computation")

    timeout: int = Field(default=30, description="Maximum seconds to wait for each parallel execution")

    def __init__(self,
                 model_name: str,
                 workdir: str,
                 require_grad: bool = False,
                 **kwargs):
        super().__init__(model_name=model_name, workdir=workdir, require_grad=require_grad, **kwargs)

    async def __call__(self, **kwargs) -> ToolResponse:
        try:
            task = kwargs.get("task", "")
            files = kwargs.get("files", [])
            prompt_name = "deep_analyzer_v3_verify"
            data = kwargs.get("data", {})
            model_names = kwargs.get("model_names", [self.model_name])

            results = await self._run_files_parallel(task, files, prompt_name, data, model_names)
            if not results:
                return ToolResponse(success=False, message="No output produced.")
            summary = await self._synth(task, prompt_name="deep_analyzer_v3_synth", data=results)
            if not summary:
                return ToolResponse(success=False, message="Synthesis produced no output.")
            return ToolResponse(success=True, message=summary, extra=ToolExtra(data={"summary": summary}))
        except Exception as exc:
            logger.error(f"| ❌ VerifyTool error: {exc}", exc_info=True)
            return ToolResponse(success=False, message=str(exc), extra=ToolExtra())

# ---------------------------------------------------------------------------
# Tool: FinishTool
# ---------------------------------------------------------------------------

class FinishTool(Tool):
    name: str = "finish_tool"
    description: str = "Terminates and returns the final answer."
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradient computation")

    async def __call__(self, reasoning: str, answer: str) -> ToolResponse:
        message = f"**Reasoning:**\n{reasoning}\n\n**Answer:**\n{answer}"
        return ToolResponse(success=True, message=message, extra=ToolExtra(data={"answer": answer, "reasoning": reasoning}))


# ---------------------------------------------------------------------------
# AnalysisStepEntry — per-step execution record for DeepAnalyzerV3
# ---------------------------------------------------------------------------

class AnalysisStepEntry:
    """One step entry for DeepAnalyzerV3's ExecutionHistory."""

    def __init__(self, step_number: int, thinking: str, evaluation: str,
                 memory: str, next_goal: str, action_name: str, action_result: str) -> None:
        self.step_number = step_number
        self.thinking = thinking
        self.evaluation = evaluation
        self.memory = memory
        self.next_goal = next_goal
        self.action_name = action_name
        self.action_result = action_result
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def render(self) -> List[str]:
        return [
            f"### Step {self.step_number} — {self.timestamp}\n",
            f"> **Evaluation:** {self.evaluation}",
            f"> **Memory:** {self.memory}",
            f"> **Next Goal:** {self.next_goal}",
            f"\n**Thinking:**\n{self.thinking}",
            f"\n**Action:** `{self.action_name}`",
            f"\n**Result:**\n{self.action_result}",
            "\n---",
        ]


# ---------------------------------------------------------------------------
# AnalysisPlanFile — thin wrapper over generic PlanFile for DeepAnalyzerV3
# ---------------------------------------------------------------------------

class AnalysisPlanFile(PlanFile):
    """PlanFile specialised for DeepAnalyzerV3Agent.

    Adds initialize_plan / update_plan (accept List[str]) and
    add_step (records an AnalysisStepEntry).
    """

    def initialize_plan(self, steps: List[str]) -> None:
        self.todo_list.set_steps(steps)
        self.flow_chart.set_steps(steps)

    def update_plan(self, steps: List[str]) -> None:
        self.todo_list.set_steps(steps)
        self.flow_chart.set_steps(steps)

    def add_step(self, step_number: int, thinking: str, evaluation: str,
                 memory: str, next_goal: str, action_name: str, action_result: str) -> None:
        self.exec_history.add_entry(AnalysisStepEntry(
            step_number=step_number, thinking=thinking, evaluation=evaluation,
            memory=memory, next_goal=next_goal, action_name=action_name, action_result=action_result,
        ))
        self.todo_list.complete_step(result=action_result if action_result else "")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _run_models_parallel(messages: List, model_names: List[str]) -> str:
    if not model_names:
        logger.warning("| ⚠️ No active models to run")
        return ""
    results = await asyncio.gather(
        *[_call_model(m, messages) for m in model_names],
        return_exceptions=True,
    )
    parts: List[str] = []
    for model, r in zip(model_names, results):
        if isinstance(r, Exception):
            logger.warning(f"| ❌ {model} failed: {r}")
        elif r:
            parts.append(f"**[{model}]**\n{r}")
            logger.info(f"| ✅ {model} done ({len(r)} chars)")
    return "\n\n".join(parts)


async def _call_model(model: str, messages: list) -> str:
    resp = await model_manager(model=model, messages=messages)
    return resp.message.strip() if resp and resp.message else ""


async def _download_file(url: str, file_type: str, workdir: str) -> Optional[str]:
    try:
        downloads_dir = os.path.join(workdir, "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or "." not in filename:
            default_exts = {"video": ".mp4", "audio": ".mp3", "image": ".jpg", "pdf": ".pdf"}
            filename = f"{file_type}_{hash(url) % 100000}{default_exts.get(file_type, '.bin')}"
        local_path = os.path.join(downloads_dir, filename)
        logger.info(f"| 📥 Downloading {file_type} from {url}")
        await asyncio.to_thread(urllib.request.urlretrieve, url, local_path)
        return local_path
    except Exception as exc:
        logger.error(f"| ❌ Download failed for {url}: {exc}")
        return None


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _is_youtube_url(url: str) -> bool:
    return any(re.search(p, url, re.IGNORECASE) for p in _YOUTUBE_PATTERNS)


def _infer_file_type(path: str) -> str:
    if _is_url(path):
        if _is_youtube_url(path):
            return "video"
        _, ext = os.path.splitext(urlparse(path).path.lower())
    else:
        _, ext = os.path.splitext(path.lower())
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _PDF_EXTS:
        return "pdf"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    return "text"


def _validate_file(path: str, max_file_size: int) -> bool:
    if _is_url(path):
        return True
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return False
    if os.path.getsize(path) > max_file_size:
        logger.warning(f"File too large: {path}")
        return False
    return True


# ---------------------------------------------------------------------------
# Tool contract — injected as available_tools in the prompt
# ---------------------------------------------------------------------------

_TOOL_CONTRACT = """
plan — Initialize or update the analysis plan. Call this FIRST (step 1) to set up the todo list and flowchart, or after eval when a replan is needed.
  args:
    steps (str, required): JSON array of step descriptions, e.g. ["Analyze X", "Verify Y"].
    reasoning (str, required): Why this plan / what changed since the last plan.

analyze — Run multiple LLMs in parallel to analyze the task (with or without files), then synthesize results.
  args: {} (no args needed)

eval — Read the latest synthesis and judge whether the task is answered and whether models conflict.
  args: {} (no args needed)

verify — Adversarially challenge the candidate answer with multiple LLMs. Call only after eval returns found_answer=true.
  args: {} (no args needed)

finish — Terminate and return the final answer.
  args:
    reasoning (str, required): The reasoning behind the final answer.
    answer (str, required): The final answer to return.
"""


# ---------------------------------------------------------------------------
# DeepAnalyzerV3Agent — ThinkOutput loop, mirrors ToolCallingAgent
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "ThinkOutput-driven multi-round analysis agent supporting text, image, PDF, audio, and video. "
    "The LLM selects the next internal tool (analyze / eval / verify / finish) at each step; "
    "analyze and verify run multiple LLMs in parallel and synthesize results; "
    "eval judges found_answer and has_conflict from the latest synthesis."
)


@AGENT.register_module(force=True)
class DeepAnalyzerV3Agent(Agent):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="deep_analyzer_v3_agent")
    description: str = Field(default=_DESCRIPTION)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = "deep_analyzer_v3",
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        max_rounds: int = 3,
        max_steps: int = 10,
        max_file_size: int = 10 * 1024 * 1024,
        summary_model_name: Optional[str] = None,
        general_analyze_models: Optional[List[str]] = None,
        llm_analyze_models: Optional[List[str]] = None,
        advanced_analyze_models: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs.setdefault("use_memory", False)
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            require_grad=require_grad,
            max_steps=max_steps,
            use_todo=False,
            **kwargs,
        )
        self.max_rounds = max_rounds
        self.max_file_size = max_file_size
        self.summary_model_name = summary_model_name
        self.general_analyze_models: List[str] = general_analyze_models or []
        self.llm_analyze_models: List[str] = llm_analyze_models or []
        self.advanced_analyze_models: List[str] = advanced_analyze_models or []

        self.analyze_tool = AnalyzeTool(model_name=self.model_name, workdir=workdir)
        self.eval_tool = EvalTool(model_name=self.model_name)
        self.verify_tool = VerifyTool(model_name=self.model_name, workdir=workdir)
        self.finish_tool = FinishTool()

    # ------------------------------------------------------------------
    # Override _get_agent_context — inject PlanFile content as agent history
    # ------------------------------------------------------------------

    async def _get_agent_context(self, task: str, step_number: int = 0, ctx: SessionContext = None, **kwargs) -> Dict[str, Any]:
        task_tag = f"<task>{task}</task>"
        step_info = dedent(f"""
            <step_info>
            Step {step_number + 1} of {self.max_steps} max possible steps
            Current date and time: {datetime.now().isoformat()}
            </step_info>
        """)
        current_plan = kwargs.get("plan")
        plan_content = current_plan.render() if current_plan else ""
        if plan_content:
            agent_history = f"<agent_history>\n{plan_content}\n</agent_history>"
        else:
            agent_history = "<agent_history>[No steps recorded yet. Call `plan` first.]</agent_history>"
        agent_context = dedent(f"""
            <agent_context>
            {task_tag}
            {step_info}
            {agent_history}
            <todo>[Todo is disabled.]</todo>
            </agent_context>
        """)
        return {"agent_context": agent_context, "active_sop": ""}

    # ------------------------------------------------------------------
    # Override _get_tool_context — inject the 4 analysis tools
    # ------------------------------------------------------------------

    async def _get_tool_context(self, ctx: SessionContext, **kwargs) -> Dict[str, Any]:  # noqa: ARG002  # pylint: disable=unused-argument
        tool_context = dedent(f"""
            <tool_context>
            <available_tools>
            {_TOOL_CONTRACT}
            </available_tools>
            </tool_context>
        """)
        return {"tool_context": tool_context}

    # ------------------------------------------------------------------
    # One step: think + execute actions
    # ------------------------------------------------------------------

    async def _think_and_action(
        self,
        task: str,
        messages: List,
        step_number: int,
        valid_files: List[str],
        has_heavy_media: bool,
        history: List[str],
        candidate_answer: Optional[str],
        plan: PlanFile,
    ) -> Dict[str, Any]:
        done = False
        final_answer: Optional[str] = None
        final_reasoning: Optional[str] = None

        try:
            resp = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ThinkOutput,
                caller="v3/think",
            )
            think: Optional[ThinkOutput] = resp.extra.parsed_model if resp and resp.extra else None
        except Exception as exc:
            logger.error(f"| ❌ ThinkOutput failed: {exc}")
            think = None

        if think is None:
            logger.warning("| ⚠️ ThinkOutput failed, stopping")
            return {"done": False, "final_answer": None, "final_reasoning": None,
                    "history": history, "candidate_answer": candidate_answer}

        logger.info(f"| 💭 Thinking: {think.thinking}")
        logger.info(f"| 🎯 Next Goal: {think.next_goal}")
        logger.info(f"| 🔧 Actions: {[a.name for a in think.actions]}")

        for i, action in enumerate(think.actions):
            action_name = action.name
            action_args = parse_tool_args(action.args) if action.args else {}
            logger.info(f"| 📝 Action {i+1}/{len(think.actions)}: {action_name}")

            action_result = ""

            if action_name == "plan":
                import json as _json
                raw_steps = action_args.get("steps", "[]")
                reasoning = action_args.get("reasoning", "")
                try:
                    steps: List[str] = _json.loads(raw_steps) if isinstance(raw_steps, str) else raw_steps
                except Exception:
                    steps = [raw_steps] if raw_steps else []
                if plan.final_result.is_set or not plan.exec_history._entries:
                    plan.initialize_plan(steps)
                    action_result = f"Plan initialized with {len(steps)} step(s): {steps}"
                    logger.info(f"| 📋 Plan initialized: {steps}")
                else:
                    plan.update_plan(steps)
                    action_result = f"Plan updated ({len(steps)} step(s)): {steps}. Reason: {reasoning}"
                    logger.info(f"| 📋 Plan updated: {steps}")

            elif action_name == "finish":
                final_reasoning = action_args.get("reasoning", "")
                final_answer = action_args.get("answer", candidate_answer or "")
                tool_resp = await self.finish_tool(reasoning=final_reasoning, answer=final_answer)
                action_result = tool_resp.message
                done = True
                logger.info(f"| ✅ finish — answer: {str(final_answer)}")

            elif action_name == "analyze":
                active_models = self._build_model_list(has_heavy_media, False)
                logger.info(f"| 🤖 Analyze models ({len(active_models)}): {active_models}")
                tool_resp = await self.analyze_tool(
                    task=task, files=valid_files, data={"history": history}, model_names=active_models,
                )
                if tool_resp.success:
                    history.append(tool_resp.message)
                    action_result = tool_resp.message
                    logger.info("| ✅ Analyze done")
                else:
                    action_result = tool_resp.message
                    logger.warning(f"| ❌ Analyze failed: {tool_resp.message}")

            elif action_name == "eval":
                if not history:
                    action_result = "Skipped — no analysis available."
                    logger.warning("| ⚠️ Eval skipped — no history")
                else:
                    tool_resp = await self.eval_tool(task=task, data={"history": history})
                    action_result = tool_resp.message
                    if tool_resp.success and tool_resp.extra and tool_resp.extra.parsed_model:
                        ev: EvalOutput = tool_resp.extra.parsed_model
                        candidate_answer = ev.candidate_answer
                        logger.info(f"| {'✅' if ev.found_answer else '❌'} Eval — found_answer={ev.found_answer} has_conflict={ev.has_conflict}")
                    else:
                        logger.warning("| ❌ Eval failed")

            elif action_name == "verify":
                if not candidate_answer:
                    action_result = "Skipped — no candidate answer."
                    logger.warning("| ⚠️ Verify skipped — no candidate answer")
                else:
                    active_models = self._build_model_list(has_heavy_media, False)
                    logger.info(f"| 🤖 Verify models ({len(active_models)}): {active_models}")
                    tool_resp = await self.verify_tool(
                        task=task, files=valid_files,
                        data={"history": history, "candidate_answer": candidate_answer},
                        model_names=active_models,
                    )
                    if tool_resp.success:
                        history.append(tool_resp.message)
                        action_result = tool_resp.message
                        logger.info("| ✅ Verify done")
                    else:
                        action_result = tool_resp.message
                        logger.warning(f"| ❌ Verify failed: {tool_resp.message}")

            else:
                action_result = f"Unknown tool: {action_name}"
                logger.warning(f"| ⚠️ Unknown action: {action_name}")

            plan.add_step(
                step_number=step_number,
                thinking=think.thinking,
                evaluation=think.evaluation_previous_goal,
                memory=think.memory,
                next_goal=think.next_goal,
                action_name=action_name,
                action_result=action_result,
            )
            await plan.save()

            if done:
                break

        return {
            "done": done,
            "final_answer": final_answer,
            "final_reasoning": final_reasoning,
            "history": history,
            "candidate_answer": candidate_answer,
        }

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        logger.info(f"| 🔍 DeepAnalyzerV3Agent starting: {task}")

        ctx = kwargs.get("ctx") or SessionContext()
        logger.info(f"| 🆔 Session: {ctx.id}")

        valid_files = [f for f in (files or []) if _validate_file(f, self.max_file_size)]
        has_heavy_media = any(_infer_file_type(f) in {"pdf", "audio", "video"} for f in valid_files)
        for f in valid_files:
            logger.info(f"| 📋 {f}: {_infer_file_type(f)}")

        plan_path = make_plan_path(
            workdir=os.path.join(self.workdir, "agent", self.name),
            session_id=ctx.id,
            suffix="analysis",
        )
        plan = AnalysisPlanFile(path=plan_path, task=task)

        history: List[str] = []
        candidate_answer: Optional[str] = None
        step_number = 0
        done = False
        final_answer: Optional[str] = None
        final_reasoning: Optional[str] = None

        try:
            while step_number < self.max_steps:
                logger.info(f"| 🔄 Step {step_number + 1}/{self.max_steps}")
                messages = await self._get_messages(task, ctx=ctx, plan=plan, step_number=step_number)

                result = await self._think_and_action(
                    task=task,
                    messages=messages,
                    step_number=step_number,
                    valid_files=valid_files,
                    has_heavy_media=has_heavy_media,
                    history=history,
                    candidate_answer=candidate_answer,
                    plan=plan,
                )

                history = result["history"]
                candidate_answer = result["candidate_answer"]
                done = result["done"]
                if result["final_answer"]:
                    final_answer = result["final_answer"]
                    final_reasoning = result["final_reasoning"]

                step_number += 1
                if done:
                    break

            if step_number >= self.max_steps and not done:
                logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping")
                final_answer = candidate_answer or "No answer found."

            answer_found = done or candidate_answer is not None

            if final_answer:
                message = f"**Reasoning:**\n{final_reasoning}\n\n**Answer:**\n{final_answer}" if final_reasoning else final_answer
            elif history:
                message = "\n\n---\n\n".join(history)
            else:
                message = f"No answer found after {step_number} step(s)."

            plan.finalize(answer=final_answer or candidate_answer or message, success=answer_found, reasoning=final_reasoning)
            await plan.save()

            logger.info(f"| ✅ DeepAnalyzerV3Agent finished — {step_number} step(s), answer_found={answer_found}")
            logger.info(f"| 📄 Analysis plan: {plan_path}")

            return AgentResponse(
                success=answer_found,
                message=message,
                extra=AgentExtra(data={
                    "task": task,
                    "session_id": ctx.id,
                    "steps": step_number,
                    "answer_found": answer_found,
                    "answer": final_answer or candidate_answer,
                    "plan_path": plan_path,
                }),
            )

        except Exception as exc:
            logger.error(f"| ❌ DeepAnalyzerV3Agent error: {exc}", exc_info=True)
            return AgentResponse(success=False, message=f"Error during analysis: {exc}")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_model_list(self, has_heavy_media: bool, use_advanced: bool) -> List[str]:
        if has_heavy_media:
            models = list(self.general_analyze_models)
        else:
            models = list(self.general_analyze_models) + list(self.llm_analyze_models)
        if use_advanced:
            models += list(self.advanced_analyze_models)
        return models
