"""Deep Analyzer Agent — multi-step file and task analysis as a standalone agent.

Responsibility boundary
-----------------------
The DeepAnalyzerAgent is a self-contained analysis agent that:

1. **File classification**: determines whether each attached file is text, PDF,
   image, audio, or video.
2. **Multi-step analysis loop**: processes each file according to its type
   (chunk-based for text/PDF, direct multimodal for image/audio/video) and
   accumulates per-round summaries.
3. **Answer synthesis**: merges all summaries and returns a final answer.
4. **Session management**: maintains an ``AnalysisSession`` per invocation keyed
   by ``ctx.id``.

It is structured like ``DeepResearcherAgent`` (own session state, registered
with the AGENT registry) but owns its own execution loop for file-based tasks
instead of a web-search loop.

All prompt text lives in ``src/prompt/template/deep_analyzer.py`` and is
accessed via ``prompt_manager.get_messages()``, keeping the agent code free of
inline prompt strings.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext
from src.tool.default_tools.mdify import MdifyTool
from src.tool.workflow_tools.reporter import Report
from src.message import (
    HumanMessage, SystemMessage,
    ContentPartText,
    ContentPartImage, ImageURL,
    ContentPartAudio, AudioURL,
    ContentPartVideo, VideoURL,
    ContentPartPdf, PdfURL,
)
from src.utils import (
    assemble_project_path,
    fetch_url,
    generate_unique_id,
    get_file_info,
    make_file_url,
)


# ---------------------------------------------------------------------------
# Shared structured-output schemas (mirror those in DeepAnalyzerTool)
# ---------------------------------------------------------------------------

class FileTypeInfo(BaseModel):
    """File type information for a single file."""
    file: str = Field(description="The exact file path or URL as provided — do NOT shorten or modify it")
    file_type: str = Field(description="File type: 'text', 'pdf', 'image', 'audio', or 'video'")


class FileTypeClassification(BaseModel):
    """Classification of multiple files by type."""
    files: List[FileTypeInfo] = Field(description="List of files with their types")


class AnalysisSummary(BaseModel):
    """Result of analyzing a piece of content."""
    summary: str = Field(description="Summary of findings (2-3 sentences)")
    found_answer: bool = Field(description="Whether the answer to the task was found")
    answer: Optional[str] = Field(default=None, description="The answer if found_answer is True")

# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

@dataclass
class AnalysisRound:
    """Record of one analysis round."""
    number: int
    file: str
    file_type: str
    summary: str
    found_answer: bool
    answer: Optional[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class AnalysisSession:
    """Tracks the state of one analysis invocation (analogous to ``ResearchSession``).

    Stores round history and wraps the ``Report`` that accumulates content and
    is eventually saved to a Markdown file.
    """

    def __init__(self, session_id: str, task: str, report: Report) -> None:
        self.session_id = session_id
        self.task = task
        self.report = report
        self.rounds: List[AnalysisRound] = []
        self.summaries: List[AnalysisSummary] = []
        self.final_answer: Optional[str] = None
        self.answer_found: bool = False

    def add_round(self, round_: AnalysisRound) -> None:
        self.rounds.append(round_)

    def add_summary(self, summary: AnalysisSummary) -> None:
        self.summaries.append(summary)

    def finalize(self, answer: Optional[str], answer_found: bool) -> None:
        self.final_answer = answer
        self.answer_found = answer_found

    def summaries_text(self) -> str:
        """Bulleted plain-text list of all accumulated summaries."""
        if not self.summaries:
            return "(no summaries yet)"
        return "\n".join(f"- {s.summary}" for s in self.summaries)


# ---------------------------------------------------------------------------
# DeepAnalyzerAgent
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
_TEXT_EXTS = {
    ".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
    ".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h",
    ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz",
}


@AGENT.register_module(force=True)
class DeepAnalyzerAgent(Agent):
    """A self-contained deep analysis agent.

    Classifies attached files, analyzes them with the appropriate strategy
    (chunk-based for text/PDF, direct multimodal for image/audio/video), and
    synthesizes all findings into a final answer.  Each call to ``__call__``
    is independent; session state is keyed by ``ctx.id``.

    Prompts are managed via ``prompt_manager`` using the ``deep_analyzer_*``
    prompt names defined in ``src/prompt/template/deep_analyzer.py``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="deep_analyzer_agent")
    description: str = Field(
        default=(
            "Deep analysis agent that performs multi-step analysis of tasks with files. "
            "Classifies files by type, analyzes text/PDF in chunks, and handles "
            "image/audio/video directly via multimodal LLM calls."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    # Active analysis sessions keyed by session id
    _analysis_sessions: Dict[str, AnalysisSession] = {}

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        file_model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        # Analyzer-specific config
        max_rounds: int = 3,
        max_steps: int = 3,
        chunk_size: int = 500,
        max_file_size: int = 10 * 1024 * 1024,
        **kwargs,
    ):
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name or "openrouter/gemini-3-flash-preview",
            prompt_name=prompt_name,
            memory_name=memory_name,
            require_grad=require_grad,
            **kwargs,
        )
        self.max_rounds = max_rounds
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.max_file_size = max_file_size
        self.file_model_name = file_model_name or "openrouter/gemini-3-flash-preview-plugins"
        self._analysis_sessions = {}

    async def initialize(self) -> None:
        await super().initialize()
        os.makedirs(self.workdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _create_session(self, session_id: str, task: str, title: str) -> AnalysisSession:
        file_path = os.path.join(self.workdir, f"{session_id}.md")
        report = Report(
            title=title,
            model_name=self.model_name,
            report_file_path=file_path,
        )
        session = AnalysisSession(session_id=session_id, task=task, report=report)
        self._analysis_sessions[session_id] = session
        return session

    def remove_session(self, session_id: str) -> None:
        self._analysis_sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Main call — full analysis loop
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        """Run the deep analysis loop for *task*.

        Args:
            task: The analysis question or task.
            files: Optional list of file paths or URLs to analyze.
            ctx: Optional session context; a new one is created if not provided.

        Returns:
            AgentResponse containing success status, message, and extra data.
        """
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()

        logger.info(f"| 🔍 DeepAnalyzerAgent starting: {task[:120]}")
        if files:
            logger.info(f"| 📂 Attached files: {files}")

        session = self._create_session(ctx.id, task, "Deep Analysis Report")

        try:
            # Add initial task section to report
            task_content = f"## Analysis Task\n\n{task}\n\n"
            if files:
                task_content += "## Files\n\n"
                for f in files:
                    task_content += f"- {f if self._is_url(f) else os.path.basename(f)}\n"
                task_content += "\n"
            await session.report.add_item(content=task_content)

            # Validate files
            valid_files: List[str] = []
            if files:
                for f in files:
                    if await self._validate_file(f):
                        valid_files.append(f)
                    else:
                        logger.warning(f"| Skipping invalid file: {f}")

            # No files → analyse task directly
            if not valid_files:
                logger.info("| 📝 No valid files, analysing task directly")
                await self._analyze_task_only(task, session)
            else:
                # Classify files
                logger.info(f"| 🔍 Classifying {len(valid_files)} file(s)")
                classifications = await self._classify_files(valid_files)

                classification_content = "## File Classifications\n\n"
                for fi in classifications:
                    label = fi.file if self._is_url(fi.file) else os.path.basename(fi.file)
                    logger.info(f"| 📋 {label}: {fi.file_type}")
                    classification_content += f"- **{label}**: {fi.file_type}\n"
                classification_content += "\n"
                await session.report.add_item(content=classification_content)

                # Main analysis loop
                for round_num in range(1, self.max_rounds + 1):
                    logger.info(f"| 🔄 Analysis round {round_num}/{self.max_rounds}")
                    round_content = f"## Round {round_num}\n\n"
                    round_summaries: List[AnalysisSummary] = []

                    for fi in classifications:
                        label = fi.file if self._is_url(fi.file) else os.path.basename(fi.file)
                        round_content += f"### {fi.file_type} — {label}\n\n"
                        logger.info(f"| 📄 Processing {fi.file_type}: {label}")

                        if fi.file_type == "text":
                            await self._analyze_text_file(task, fi.file, round_summaries)
                        elif fi.file_type == "pdf":
                            await self._analyze_pdf_file(task, fi.file, round_summaries)
                        elif fi.file_type == "image":
                            await self._analyze_image_file(task, fi.file, round_summaries)
                        elif fi.file_type == "audio":
                            await self._analyze_audio_file(task, fi.file, round_summaries)
                        elif fi.file_type == "video":
                            await self._analyze_video_file(task, fi.file, round_summaries)

                        for s in round_summaries:
                            round_content += f"- {s.summary}\n"
                            if s.found_answer:
                                round_content += f"  **Answer**: {s.answer}\n"

                        # Early-exit if answer found
                        if any(s.found_answer for s in round_summaries):
                            break

                    # Synthesize round
                    session.summaries.extend(round_summaries)

                    # Short-circuit: if any per-file analysis already found the answer,
                    # skip synthesis and use that answer directly.
                    direct_hit = next((s for s in round_summaries if s.found_answer and s.answer), None)
                    if direct_hit:
                        round_content += f"\n### Round {round_num} Synthesis\n\n{direct_hit.summary}\n\n"
                        round_content += f"**Answer Found**: {direct_hit.answer}\n\n"
                        await session.report.add_item(content=round_content)
                        logger.info(f"| ✅ Answer found after round {round_num}")
                        session.finalize(answer=direct_hit.answer, answer_found=True)
                        break

                    round_synthesis = await self._generate_summary(task, session.summaries)
                    round_content += f"\n### Round {round_num} Synthesis\n\n{round_synthesis.summary}\n\n"
                    if round_synthesis.found_answer:
                        round_content += f"**Answer Found**: {round_synthesis.answer}\n\n"
                    await session.report.add_item(content=round_content)

                    if round_synthesis.found_answer:
                        logger.info(f"| ✅ Answer found after round {round_num}")
                        session.finalize(answer=round_synthesis.answer, answer_found=True)
                        break
                else:
                    # All rounds done — final synthesis
                    final_synthesis = await self._generate_summary(task, session.summaries)
                    session.finalize(
                        answer=final_synthesis.answer if final_synthesis.found_answer else None,
                        answer_found=final_synthesis.found_answer,
                    )
                    final_content = f"## Final Synthesis\n\n{final_synthesis.summary}\n\n"
                    if final_synthesis.found_answer:
                        final_content += f"**Answer**: {final_synthesis.answer}\n\n"
                    await session.report.add_item(content=final_content)

            # ------------------------------------------------------------------
            # Finalize report
            # ------------------------------------------------------------------
            final_report_content = await session.report.complete()
            file_path = session.report.report_file_path

            message = f"Analysis complete. Answer found: {'Yes' if session.answer_found else 'No'}."
            if session.answer_found and session.final_answer:
                message += f"\n\nAnswer: {session.final_answer}"
            if file_path:
                message += f"\n\nReport saved to: {file_path}"

            logger.info(
                f"| ✅ DeepAnalyzerAgent finished. "
                f"Rounds: {len(session.rounds)}, answer_found={session.answer_found}"
            )

            return AgentResponse(
                success=session.answer_found,
                message=message,
                extra=AgentExtra(
                    file_path=file_path,
                    data={
                        "task": task,
                        "answer_found": session.answer_found,
                        "answer": session.final_answer,
                        "file_path": file_path,
                    },
                ),
            )

        except Exception as exc:
            logger.error(f"| ❌ DeepAnalyzerAgent error: {exc}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Error during deep analysis: {exc}",
            )
        finally:
            self.remove_session(ctx.id)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _is_url(self, path: str) -> bool:
        return path.startswith(("http://", "https://"))

    def _is_youtube_url(self, url: str) -> bool:
        return any(re.search(p, url, re.IGNORECASE) for p in _YOUTUBE_PATTERNS)

    def _classify_by_extension(self, path: str) -> str:
        """Return file type string based on path extension."""
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

    def _url_type(self, url: str) -> str:
        url_lower = url.lower()
        for ext in _IMAGE_EXTS:
            if url_lower.endswith(ext):
                return "image"
        if url_lower.endswith(".pdf"):
            return "pdf"
        for ext in _AUDIO_EXTS:
            if url_lower.endswith(ext):
                return "audio"
        if self._is_youtube_url(url):
            return "video"
        for ext in _VIDEO_EXTS:
            if url_lower.endswith(ext):
                return "video"
        return "text"

    async def _validate_file(self, path: str) -> bool:
        if self._is_url(path):
            return True
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return False
        size = os.path.getsize(path)
        if size > self.max_file_size:
            logger.warning(f"File too large ({size} bytes): {path}")
        return True

    async def _download_file(self, url: str, file_type: str = "file") -> Optional[str]:
        """Download a URL to workdir/downloads and return the local path."""
        try:
            downloads_dir = os.path.join(self.workdir, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename or "." not in filename:
                default_exts = {"video": ".mp4", "audio": ".mp3", "image": ".jpg"}
                filename = f"{file_type}_{hash(url) % 100000}{default_exts.get(file_type, '.bin')}"
            local_path = os.path.join(downloads_dir, filename)
            logger.info(f"| 📥 Downloading {file_type} from {url}")
            await asyncio.to_thread(urllib.request.urlretrieve, url, local_path)
            return local_path
        except Exception as exc:
            logger.error(f"| ❌ Download failed for {url}: {exc}")
            return None

    # ------------------------------------------------------------------
    # LLM helpers — all prompts from prompt_manager
    # ------------------------------------------------------------------

    async def _classify_files(self, files: List[str]) -> List[FileTypeInfo]:
        """Classify files by type; URL types are determined by extension/pattern."""
        url_results: List[FileTypeInfo] = []
        path_files: List[str] = []

        for f in files:
            if self._is_url(f):
                url_results.append(FileTypeInfo(file=f, file_type=self._url_type(f)))
            else:
                path_files.append(f)

        if not path_files:
            return url_results

        file_list = "\n".join(f"- {p}" for p in path_files)
        try:
            messages = await prompt_manager.get_messages(
                prompt_name="deep_analyzer_classify",
                agent_modules={"file_list": file_list},
            )
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=FileTypeClassification,
            )
            if response.extra and response.extra.parsed_model:
                llm_files = response.extra.parsed_model.files
                # Re-map LLM results back to the original absolute paths so that
                # the LLM cannot accidentally return a relative or shortened path.
                # Match by index (same order) or fall back to basename matching.
                basename_to_original = {os.path.basename(p): p for p in path_files}
                corrected: List[FileTypeInfo] = []
                for i, fi in enumerate(llm_files):
                    if i < len(path_files):
                        original = path_files[i]
                    else:
                        original = basename_to_original.get(os.path.basename(fi.file), fi.file)
                    corrected.append(FileTypeInfo(file=original, file_type=fi.file_type))
                return url_results + corrected
        except Exception as exc:
            logger.warning(f"| File classification failed: {exc}, using extension fallback")

        # Fallback
        return url_results + [
            FileTypeInfo(file=p, file_type=self._classify_by_extension(p))
            for p in path_files
        ]

    async def _analyze_markdown_chunk(
        self, task: str, chunk_text: str, chunk_num: int, start_line: int, end_line: int
    ) -> AnalysisSummary:
        """Analyse a chunk of markdown text via prompt_manager."""
        try:
            messages = await prompt_manager.get_messages(
                prompt_name="deep_analyzer_chunk",
                agent_modules={
                    "task": task,
                    "start_line": str(start_line),
                    "end_line": str(end_line),
                    "chunk_text": chunk_text,
                },
            )
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=AnalysisSummary,
            )
            if response.extra and response.extra.parsed_model:
                return response.extra.parsed_model
            return AnalysisSummary(
                summary=response.message.strip() if response else "No response",
                found_answer=False,
            )
        except Exception as exc:
            logger.error(f"| ❌ Chunk analysis error: {exc}")
            return AnalysisSummary(summary=f"Error: {exc}", found_answer=False)

    async def _analyze_task_only(self, task: str, session: AnalysisSession) -> None:
        """Multi-round analysis when no files are provided."""
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"| 🔄 Task-only round {round_num}/{self.max_rounds}")
            previous = "\n".join(f"- {s.summary}" for s in session.summaries)
            try:
                messages = await prompt_manager.get_messages(
                    prompt_name="deep_analyzer_task",
                    agent_modules={
                        "task": task,
                        "round_number": str(round_num),
                        "max_rounds": str(self.max_rounds),
                        "previous_summaries": previous,
                    },
                )
                response = await model_manager(
                    model=self.model_name,
                    messages=messages,
                    response_format=AnalysisSummary,
                )
                if response.extra and response.extra.parsed_model:
                    summary = response.extra.parsed_model
                else:
                    summary = AnalysisSummary(
                        summary=response.message.strip() if response else "No response",
                        found_answer=False,
                    )
            except Exception as exc:
                logger.error(f"| ❌ Task-only round error: {exc}")
                summary = AnalysisSummary(summary=f"Error: {exc}", found_answer=False)

            session.add_summary(summary)
            round_content = f"## Round {round_num}\n\n{summary.summary}\n\n"
            if summary.found_answer:
                round_content += f"**Answer**: {summary.answer}\n\n"
            await session.report.add_item(content=round_content)

            if summary.found_answer:
                session.finalize(answer=summary.answer, answer_found=True)
                logger.info(f"| ✅ Answer found in task-only round {round_num}")
                return

        # If no round found an answer, do final synthesis
        final = await self._generate_summary(task, session.summaries)
        session.finalize(answer=final.answer if final.found_answer else None, answer_found=final.found_answer)

    async def _direct_analysis(
        self,
        task: str,
        file_type: str,
        multimodal_content: Any,
        model_name: Optional[str] = None,
    ) -> AnalysisSummary:
        """Run a direct multimodal analysis using prompt_manager for text parts."""
        model = model_name or self.model_name
        try:
            messages = await prompt_manager.get_messages(
                prompt_name="deep_analyzer_direct",
                agent_modules={"task": task, "file_type": file_type},
            )
            # Append multimodal content to the last (human) message
            last = messages[-1]
            text_content = last.content if isinstance(last.content, str) else str(last.content)
            messages[-1] = HumanMessage(content=[
                ContentPartText(text=text_content),
                multimodal_content,
            ])
            response = await model_manager(
                model=model,
                messages=messages,
                response_format=AnalysisSummary,
            )
            if response.extra and response.extra.parsed_model:
                r = response.extra.parsed_model
                return AnalysisSummary(summary=r.summary, found_answer=r.found_answer, answer=r.answer)
            return AnalysisSummary(
                summary=response.message.strip() if response else "No response",
                found_answer=False,
            )
        except Exception as exc:
            logger.warning(f"| ⚠️ Direct {file_type} analysis failed: {exc}")
            return AnalysisSummary(summary=f"Direct analysis failed: {exc}", found_answer=False)

    # ------------------------------------------------------------------
    # File-type specific analysis methods
    # ------------------------------------------------------------------

    async def _analyze_text_file(
        self, task: str, file: str, summaries: List[AnalysisSummary]
    ) -> None:
        """Analyse a text file (or URL) in markdown chunks."""
        try:
            local_path = file
            if self._is_url(file):
                local_path = await self._download_file(file, "text")
                if not local_path:
                    summaries.append(AnalysisSummary(
                        summary=f"Failed to download: {file}", found_answer=False))
                    return

            mdify = MdifyTool(base_dir=self.workdir)
            _, ext = os.path.splitext(local_path.lower())
            if ext == ".md":
                saved_path = local_path
            else:
                resp = await mdify(file_path=local_path, output_format="markdown")
                if resp.extra and resp.extra.file_path:
                    saved_path = resp.extra.file_path
                else:
                    summaries.append(AnalysisSummary(
                        summary=f"Failed to convert to markdown: {local_path}", found_answer=False))
                    return

            with open(saved_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()

            total = len(lines)
            total_chunks = (total + self.chunk_size - 1) // self.chunk_size

            for chunk_num in range(1, total_chunks + 1):
                start = (chunk_num - 1) * self.chunk_size
                end = min(start + self.chunk_size, total)
                chunk_text = "".join(lines[start:end])
                summary = await self._analyze_markdown_chunk(task, chunk_text, chunk_num, start + 1, end)
                summaries.append(summary)
                if summary.found_answer:
                    logger.info(f"| ✅ Answer found in text chunk {chunk_num}")
                    return

        except Exception as exc:
            logger.error(f"| ❌ Text file analysis error: {exc}")
            summaries.append(AnalysisSummary(summary=f"Error: {exc}", found_answer=False))

    async def _analyze_pdf_file(
        self, task: str, file: str, summaries: List[AnalysisSummary]
    ) -> None:
        """Analyse a PDF file: first try direct LLM, then chunk-based fallback."""
        # Step 1: Direct PDF analysis
        pdf_url_value = file if self._is_url(file) else make_file_url(file_path=file)
        pdf_content = ContentPartPdf(pdf_url=PdfURL(url=pdf_url_value))
        summary = await self._direct_analysis(task, "PDF document", pdf_content, self.file_model_name)
        if summary.found_answer:
            summaries.append(summary)
            return
        if summary.summary and not summary.summary.startswith("Direct analysis failed"):
            summaries.append(summary)

        # Step 2: Chunk-based fallback
        try:
            if self._is_url(file):
                doc_result = await fetch_url(file)
                if not doc_result or not doc_result.markdown:
                    logger.warning(f"| Failed to fetch PDF URL: {file}")
                    return
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, dir=self.workdir
                ) as tmp:
                    tmp.write(doc_result.markdown)
                    saved_path = tmp.name
            else:
                mdify = MdifyTool(base_dir=self.workdir)
                resp = await mdify(file_path=file, output_format="markdown")
                if resp.extra and resp.extra.file_path:
                    saved_path = resp.extra.file_path
                else:
                    logger.warning(f"| Failed to convert PDF: {file}")
                    return

            with open(saved_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()

            total = len(lines)
            total_chunks = (total + self.chunk_size - 1) // self.chunk_size
            for chunk_num in range(1, total_chunks + 1):
                start = (chunk_num - 1) * self.chunk_size
                end = min(start + self.chunk_size, total)
                chunk_text = "".join(lines[start:end])
                s = await self._analyze_markdown_chunk(task, chunk_text, chunk_num, start + 1, end)
                summaries.append(s)
                if s.found_answer:
                    return

        except Exception as exc:
            logger.error(f"| ❌ PDF chunk analysis error: {exc}")
            summaries.append(AnalysisSummary(summary=f"Error: {exc}", found_answer=False))

    async def _analyze_image_file(
        self, task: str, file: str, summaries: List[AnalysisSummary]
    ) -> None:
        """Analyse an image file: direct multimodal analysis, with multi-step fallback."""
        local_path = file
        if self._is_url(file):
            local_path = await self._download_file(file, "image")
            if not local_path:
                summaries.append(AnalysisSummary(
                    summary=f"Failed to download image: {file}", found_answer=False))
                return

        image_url_value = make_file_url(file_path=local_path)

        # Step 1: Direct analysis
        image_content = ContentPartImage(image_url=ImageURL(url=image_url_value, detail="high"))
        summary = await self._direct_analysis(task, "image", image_content)
        if summary.found_answer:
            summaries.append(summary)
            return
        if summary.summary and not summary.summary.startswith("Direct analysis failed"):
            summaries.append(summary)

        # Step 2: Multi-step analysis
        for step_num in range(1, self.max_steps + 1):
            logger.info(f"| 🖼️ Image analysis step {step_num}/{self.max_steps}")
            image_content = ContentPartImage(image_url=ImageURL(url=image_url_value, detail="high"))
            s = await self._direct_analysis(task, "image", image_content)
            summaries.append(s)
            if s.found_answer:
                logger.info(f"| ✅ Answer found in image step {step_num}")
                return

    async def _analyze_audio_file(
        self, task: str, file: str, summaries: List[AnalysisSummary]
    ) -> None:
        """Analyse an audio file: direct multimodal analysis."""
        local_path = file
        if self._is_url(file):
            local_path = await self._download_file(file, "audio")
            if not local_path:
                summaries.append(AnalysisSummary(
                    summary=f"Failed to download audio: {file}", found_answer=False))
                return

        audio_url_value = make_file_url(file_path=local_path)
        audio_content = ContentPartAudio(audio_url=AudioURL(url=audio_url_value))
        summary = await self._direct_analysis(task, "audio", audio_content, self.file_model_name)
        summaries.append(summary)

    async def _analyze_video_file(
        self, task: str, file: str, summaries: List[AnalysisSummary]
    ) -> None:
        """Analyse a video file: direct multimodal analysis, chunk-based fallback."""
        is_url = self._is_url(file)
        local_path = file

        if is_url:
            if self._is_youtube_url(file):
                video_url_value = file
            else:
                local_path = await self._download_file(file, "video")
                if not local_path:
                    summaries.append(AnalysisSummary(
                        summary=f"Failed to download video: {file}", found_answer=False))
                    return
                video_url_value = make_file_url(file_path=local_path)
        else:
            if not os.path.exists(file):
                logger.warning(f"Video file not found: {file}")
                return
            video_url_value = make_file_url(file_path=file)

        # Step 1: Direct analysis
        video_content = ContentPartVideo(video_url=VideoURL(url=video_url_value))
        summary = await self._direct_analysis(task, "video", video_content, self.file_model_name)
        if summary.found_answer:
            summaries.append(summary)
            return
        if summary.summary and not summary.summary.startswith("Direct analysis failed"):
            summaries.append(summary)

        # Step 2: Chunk-based fallback
        try:
            if is_url and self._is_youtube_url(file):
                doc_result = await fetch_url(file)
                if not doc_result or not doc_result.markdown:
                    return
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, dir=self.workdir
                ) as tmp:
                    tmp.write(doc_result.markdown)
                    saved_path = tmp.name
            else:
                mdify = MdifyTool(base_dir=self.workdir)
                resp = await mdify(file_path=local_path, output_format="markdown")
                if resp.extra and resp.extra.file_path:
                    saved_path = resp.extra.file_path
                else:
                    return

            with open(saved_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
            total = len(lines)
            total_chunks = (total + self.chunk_size - 1) // self.chunk_size
            for chunk_num in range(1, total_chunks + 1):
                start = (chunk_num - 1) * self.chunk_size
                end = min(start + self.chunk_size, total)
                chunk_text = "".join(lines[start:end])
                s = await self._analyze_markdown_chunk(task, chunk_text, chunk_num, start + 1, end)
                summaries.append(s)
                if s.found_answer:
                    return

        except Exception as exc:
            logger.error(f"| ❌ Video chunk analysis error: {exc}")
            summaries.append(AnalysisSummary(summary=f"Error: {exc}", found_answer=False))

    async def _generate_summary(
        self, task: str, summaries: List[AnalysisSummary]
    ) -> AnalysisSummary:
        """Synthesize all summaries into a single answer via prompt_manager."""
        if not summaries:
            return AnalysisSummary(summary="No analysis summaries available.", found_answer=False)
        # Include answer when already found so the synthesis LLM can see it
        def _fmt(s: AnalysisSummary) -> str:
            line = f"- {s.summary}"
            if s.found_answer and s.answer:
                line += f" [Answer: {s.answer}]"
            return line
        summaries_text = "\n".join(_fmt(s) for s in summaries)
        try:
            messages = await prompt_manager.get_messages(
                prompt_name="deep_analyzer_summarize",
                agent_modules={"task": task, "summaries_text": summaries_text},
            )
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=AnalysisSummary,
            )
            if response.extra and response.extra.parsed_model:
                return response.extra.parsed_model
            return AnalysisSummary(
                summary=response.message.strip() if response else summaries_text,
                found_answer=False,
            )
        except Exception as exc:
            logger.error(f"| ❌ Synthesis error: {exc}")
            return AnalysisSummary(summary=f"Error during synthesis: {exc}", found_answer=False)

