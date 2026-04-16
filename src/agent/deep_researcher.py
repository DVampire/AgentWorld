"""Deep Researcher Agent — multi-round web research as a standalone agent.

Responsibility boundary
-----------------------
The DeepResearcherAgent is a self-contained research agent that:

1. **Multi-round search loop**: iteratively generates search queries, executes
   searches (via WebSearcherTool or LLM-based search models), evaluates whether
   the answer is complete, and repeats up to ``max_rounds``.
2. **Research session management**: maintains a ``ResearchSession`` per
   invocation that tracks round history, builds a ``Report``, and saves the
   final Markdown report to disk.

It is structured like ``PlanningAgent`` (own session state, registered with the
AGENT registry) but owns its full execution loop rather than delegating it to
an external bus.

All prompt text lives in ``src/prompt/template/deep_researcher.py`` and is
accessed via ``prompt_manager.get_messages()``, keeping the agent code free of
inline prompt strings.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from importlib_metadata import files
from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext
from src.tool.default_tools.web_searcher import WebSearcherTool
from src.tool.workflow_tools.reporter import Report
from src.utils import assemble_project_path, generate_unique_id, make_file_url


# ---------------------------------------------------------------------------
# LLM structured-output schemas
# ---------------------------------------------------------------------------

class ResearchSummary(BaseModel):
    """Final summary produced from the completed report."""
    summary: str = Field(description="Comprehensive summary of the research findings")
    found_answer: bool = Field(description="Whether a complete answer was found to the research task")
    answer: Optional[str] = Field(default=None, description="The answer if found_answer is True")


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

@dataclass
class ResearchRound:
    """Record of one research round."""
    number: int
    query: str
    summary: str
    found_answer: bool
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class ResearchSession:
    """Tracks the state of one research invocation (analogous to ``PlanFile``).

    Stores round history and wraps the ``Report`` that accumulates content and
    is eventually saved to a Markdown file.
    """

    def __init__(self, session_id: str, task: str, report: Report) -> None:
        self.session_id = session_id
        self.task = task
        self.report = report
        self.rounds: List[ResearchRound] = []
        self.final_summary: Optional[str] = None
        self.answer_found: bool = False

    def add_round(self, round_: ResearchRound) -> None:
        self.rounds.append(round_)

    def finalize(self, summary: str, answer_found: bool) -> None:
        self.final_summary = summary
        self.answer_found = answer_found

    def execution_log_text(self) -> str:
        """Plain-text log of all completed rounds (for query generation context)."""
        if not self.rounds:
            return "(no rounds completed yet)"
        lines: List[str] = []
        for r in self.rounds:
            lines.append(f"=== Round {r.number} — {r.timestamp} ===")
            lines.append(f"Query: {r.query}")
            lines.append(f"Summary: {r.summary}")
            lines.append(f"Answer Found: {'Yes' if r.found_answer else 'No'}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DeepResearcherAgent
# ---------------------------------------------------------------------------

@AGENT.register_module(force=True)
class DeepResearcherAgent(Agent):
    """A self-contained deep research agent.

    Runs a configurable number of search rounds, evaluates completeness after
    each round, and produces a structured Markdown report.  Each call to
    ``__call__`` is independent; session state is keyed by ``ctx.id``.

    Prompts are managed via ``prompt_manager`` using the ``deep_researcher_query``,
    ``deep_researcher_eval``, and ``deep_researcher_summary`` prompt names defined
    in ``src/prompt/template/deep_researcher.py``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="deep_researcher_agent")
    description: str = Field(
        default=(
            "Deep research agent that performs multi-round web search and content analysis. "
            "Generates search queries, searches the web, evaluates completeness, and produces "
            "a structured Markdown report."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    # Active research sessions keyed by session id
    _research_sessions: Dict[str, ResearchSession] = {}

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        # Research-specific config
        max_rounds: int = 3,
        num_results: int = 5,
        use_llm_search: bool = True,
        search_llm_models: Optional[List[str]] = None,
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
        self.num_results = num_results
        self.search_llm_models: List[str] = search_llm_models or [
            "openrouter/o3-deep-research",
            "openrouter/sonar-deep-research",
        ]
        self.use_llm_search = use_llm_search and len(self.search_llm_models) > 0
        self._web_searcher = WebSearcherTool(model_name=self.model_name)
        self._research_sessions = {}

    async def initialize(self) -> None:
        await super().initialize()
        os.makedirs(self.workdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _create_session(self, session_id: str, task: str, title: str) -> ResearchSession:
        file_path = os.path.join(self.workdir, f"{session_id}.md")
        report = Report(
            title=title,
            model_name=self.model_name,
            report_file_path=file_path,
        )
        session = ResearchSession(session_id=session_id, task=task, report=report)
        self._research_sessions[session_id] = session
        return session

    def remove_session(self, session_id: str) -> None:
        self._research_sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Main call — full research loop
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        """Run the deep research loop for task.

        Args:
            task: The research question or topic to investigate.
            files: Optional list of file paths to provide additional context for the research.

        Returns:
            AgentResponse containing success status, message, and extra data including report file path.
        """
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()

        title: str = "Research Report"
        image: str = None
        filter_year: int = kwargs.get("filter_year", __import__("datetime").datetime.now().year)

        if files:
            file = files[0]
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image = file

        logger.info(f"| 🔍 DeepResearcherAgent starting: {task}")

        session = self._create_session(ctx.id, task, title)

        try:
            # Add initial task section to report
            task_content = f"## Research Task\n\n{task}\n\n"
            if image:
                task_content += f"## Image\n\n{image}\n\n"
            await session.report.add_item(content=task_content)

            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"| 📋 Round {round_num}/{self.max_rounds}")

                query = await self._generate_search_query(task, round_num, image, session, filter_year)
                logger.info(f"| ✅ Query for round {round_num}: {query}")

                search_results = await self._parallel_search(task, query, filter_year)

                if not search_results:
                    logger.warning(f"| ❌ All searches failed in round {round_num}")
                    empty = (
                        f"## Round {round_num}\n\n"
                        f"### Search Query\n\n{query}\n\n"
                        f"### Search Results\n\nNo search results found.\n\n"
                    )
                    await session.report.add_item(content=empty)
                    continue

                merged = self._merge_search_results(search_results)
                logger.info(f"| ✅ Merged {len(search_results)} result(s)")

                round_summary = await self._generate_summary(task, merged)

                round_content = (
                    f"## Round {round_num}\n\n"
                    f"### Search Query\n\n{query}\n\n"
                    f"### Search Results\n\n{merged}\n\n"
                    f"### Summary\n\n{round_summary.summary}\n\n"
                    f"- **Answer Found**: {'Yes' if round_summary.found_answer else 'No'}\n\n"
                )
                await session.report.add_item(content=round_content)

                session.add_round(ResearchRound(
                    number=round_num,
                    query=query,
                    summary=merged,
                    found_answer=round_summary.found_answer,
                ))

                if round_summary.found_answer:
                    logger.info(f"| ✅ Answer found in round {round_num}")
                    break

            # ------------------------------------------------------------------
            # Finalize report
            # ------------------------------------------------------------------
            answer_found = any(r.found_answer for r in session.rounds)
            final_report_content = await session.report.complete()

            final_summary = await self._generate_summary(task, final_report_content)
            session.finalize(summary=final_summary.summary, answer_found=answer_found or final_summary.found_answer)

            file_path = session.report.report_file_path
            message = f"Deep research summary: {final_summary.summary}"
            if file_path:
                message += f"\n\nReport saved to: {file_path}"

            logger.info(f"| ✅ DeepResearcherAgent finished after {len(session.rounds)} round(s)")

            return AgentResponse(
                success=answer_found,
                message=message,
                extra=AgentExtra(
                    file_path=file_path,
                    data={
                        "task": task,
                        "rounds": len(session.rounds),
                        "history": [
                            {
                                "round_number": r.number,
                                "query": r.query,
                                "summary": r.summary,
                                "found_answer": r.found_answer,
                            }
                            for r in session.rounds
                        ],
                        "file_path": file_path,
                        "answer_found": answer_found,
                    },
                ),
            )

        except Exception as exc:
            logger.error(f"| ❌ DeepResearcherAgent error: {exc}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Error during deep research: {exc}",
            )
        finally:
            self.remove_session(ctx.id)

    # ------------------------------------------------------------------
    # Internal helpers — all LLM calls go through prompt_manager
    # ------------------------------------------------------------------

    async def _generate_search_query(
        self,
        task: str,
        round_num: int,
        image: Optional[str],
        session: ResearchSession,
        filter_year: Optional[int] = None,
    ) -> str:
        """Generate an optimised search query for the current round via prompt_manager."""
        messages = await prompt_manager.get_messages(
            prompt_name="deep_researcher_query",
            agent_modules={
                "task": task,
                "image": image or "",
                "round_number": str(round_num),
                "previous_context": session.execution_log_text(),
                "filter_year": str(filter_year) if filter_year else "",
            },
        )

        # If an image is provided, attach it to the last (human) message
        if image and image.lower().endswith((".jpg", ".jpeg", ".png")):
            from src.message.types import HumanMessage
            last = messages[-1]
            text_content = last.content if isinstance(last.content, str) else str(last.content)
            messages[-1] = HumanMessage(content=[
                {"type": "text", "text": text_content},
                {"type": "image_url", "image_url": {"url": make_file_url(
                    file_path=assemble_project_path(image)
                )}},
            ])

        response = await model_manager(model=self.model_name, messages=messages)
        return response.message.strip()

    async def _parallel_search(
        self,
        task: str,
        query: str,
        filter_year: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Execute searches in parallel and return successful results."""
        search_tasks = []

        if self.use_llm_search:
            for model_name in self.search_llm_models:
                def _make_llm_task(model: str):
                    async def _task():
                        try:
                            summary = await self._llm_search(model, task, query)
                            return {"source": model, "summary": summary, "success": True}
                        except Exception as exc:
                            logger.warning(f"| LLM search with {model} failed: {exc}")
                            return {"source": model, "summary": None, "success": False, "error": str(exc)}
                    return _task

                search_tasks.append(_make_llm_task(model_name)())
        else:
            async def _web_task():
                try:
                    resp = await self._web_searcher(
                        query=query,
                        num_results=self.num_results,
                        filter_year=filter_year,
                    )
                    if resp.success:
                        return {"source": "web_searcher", "summary": resp.message.strip(), "success": True}
                    return {"source": "web_searcher", "summary": None, "success": False, "error": resp.message}
                except Exception as exc:
                    logger.warning(f"| Web searcher failed: {exc}")
                    return {"source": "web_searcher", "summary": None, "success": False, "error": str(exc)}

            search_tasks.append(_web_task())

        raw = await asyncio.gather(*search_tasks, return_exceptions=True)

        results = []
        for r in raw:
            if isinstance(r, Exception):
                logger.error(f"| Search task raised: {r}")
                continue
            if r.get("success") and r.get("summary"):
                results.append(r)
            else:
                logger.warning(f"| Search from {r.get('source', '?')} failed: {r.get('error', 'unknown')}")
        return results

    async def _llm_search(self, model_name: str, task: str, query: str) -> str:
        """Use an LLM model to perform web research and return a summary.

        This call uses the model directly (no prompt template) because it is
        dispatched to a specialised search-capable model, not the agent's own
        model, and the prompt is intentionally minimal.
        """
        from src.message.types import HumanMessage
        prompt = (
            f"You are an expert web researcher. Research the following task and query, "
            f"then provide a comprehensive summary with citations where possible.\n\n"
            f"Research Task: {task}\n"
            f"Search Query: {query}\n\n"
            f"Return your research findings as a comprehensive summary."
        )
        logger.info(f"| Using LLM {model_name} to search the web.")
        response = await model_manager(model=model_name, messages=[HumanMessage(content=prompt)])
        logger.info(f"| LLM {model_name} response: {response.message.strip()[:200]}...")
        if response and response.message.strip():
            return response.message.strip()
        raise ValueError(f"LLM {model_name} returned empty response")

    def _merge_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Merge multiple search results into a single text block."""
        if not search_results:
            return "No search results available."
        if len(search_results) == 1:
            return search_results[0]["summary"]
        parts = []
        for i, r in enumerate(search_results, 1):
            source = r.get("source", f"Source {i}")
            parts.append(f"## {source}\n\n{r.get('summary', '')}\n")
        return "\n".join(parts)

    async def _generate_summary(
        self,
        task: str,
        content: str,
    ) -> ResearchSummary:
        """Generate a summary from the given content (per-round or final report)."""
        messages = await prompt_manager.get_messages(
            prompt_name="deep_researcher_summary",
            agent_modules={
                "task": task,
                "report_content": content,
            },
        )
        try:
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ResearchSummary,
            )
            if (
                response
                and response.extra
                and hasattr(response.extra, "parsed_model")
                and response.extra.parsed_model
            ):
                return response.extra.parsed_model
            return ResearchSummary(
                summary=response.message.strip() if response else "",
                found_answer=False,
            )

        except Exception as exc:
            logger.warning(f"| Summary generation failed: {exc}")
            return ResearchSummary(summary="", found_answer=False)
