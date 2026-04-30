"""Deep Researcher V3 Agent — ThinkOutput-driven multi-round research.

Five-tool pipeline driven by LLM ThinkOutput decisions:

  plan   — Initialize or update the research plan (todo list + flowchart).
            Called first (step 1); optionally again after eval if a replan is needed.
  query  — Generate an optimized search query for the current round.
            The LLM provides angle/focus guidance via ThinkOutput args.
  search — Execute concurrent API search (FirecrawlSearch) + LLM web searches,
            or multimodal_search (GoogleLensSearch) when an image is present.
            Produces a labeled report per source.
  eval   — Synthesize all reports, detect source conflicts, assess completeness.
            Returns ResearchSummary (found_answer, has_conflict, answer, reasoning).
  finish — Terminate and return the final answer.

ThinkOutput loop mirrors DeepAnalyzerV3Agent:
  The controller LLM selects the next tool each step via ThinkOutput JSON.
  History is maintained in AnalysisPlanFile (same generic PlanFile machinery).
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse, ThinkOutput
from src.logger import logger
from src.model import model_manager
from src.message.types import HumanMessage
from src.message import ContentPartText, ContentPartImage, ImageURL
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext
from src.tool.default_tools.search.firecrawl_search import FirecrawlSearch
from src.tool.default_tools.search.google_lens_search import GoogleLensSearch
from src.utils import (
    assemble_project_path, make_file_url, fetch_url, dedent,
    parse_tool_args, PlanFile, make_plan_path,
)
from src.tool.types import Tool, ToolResponse, ToolExtra


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class ResearchSummary(BaseModel):
    """Evaluation result for a research round."""
    reasoning: str = Field(
        description=(
            "Key conclusions and critical logic. Capture what the sources found and why "
            "the answer is correct or the conflict/gap that remains. "
            "For multiple-choice tasks, must include explicit analysis of every option."
        )
    )
    found_answer: bool = Field(description="Whether a complete answer was found")
    answer: Optional[str] = Field(
        default=None,
        description="The final answer if found_answer is True; concise and directly usable",
    )
    has_conflict: bool = Field(
        default=False,
        description="Whether sources contradicted each other on task-relevant points",
    )


# ---------------------------------------------------------------------------
# SearchLogger — saves each round's search results as markdown files
# ---------------------------------------------------------------------------

class SearchLogger:
    """Saves api_search and llm_search results per round as markdown files."""

    def __init__(self, base_dir: str, session_id: str):
        self.log_dir = os.path.join(base_dir, "search", session_id)
        os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def _model_slug(model: str) -> str:
        return re.sub(r'[^\w\-.]', '_', model)[:50]

    def _write(self, path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def save_api_search(self, round_num: int, query: str, content: str) -> None:
        path = os.path.join(self.log_dir, f"round_{round_num}_api_search.md")
        self._write(path, f"# Round {round_num} — API Search\n\n**Query:** `{query}`\n\n---\n\n{content}\n")

    def save_llm_search(self, round_num: int, model: str, content: str) -> None:
        slug = self._model_slug(model)
        path = os.path.join(self.log_dir, f"round_{round_num}_llm_{slug}.md")
        self._write(path, f"# Round {round_num} — LLM Search\n\n**Model:** `{model}`\n\n---\n\n{content}\n")


# ---------------------------------------------------------------------------
# Research session state
# ---------------------------------------------------------------------------

class ResearchRound(BaseModel):
    """Record of one research round."""
    number: int
    query: str
    reasoning: str
    found_answer: bool
    answer: Optional[str] = None
    has_conflict: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class ResearchSession:
    """Tracks the state of one research invocation."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.rounds: List[ResearchRound] = []

    def add_round(self, round_: ResearchRound) -> None:
        self.rounds.append(round_)

    def execution_log_text(self) -> str:
        """Plain-text log of all completed rounds."""
        if not self.rounds:
            return ""
        lines: List[str] = []
        for r in self.rounds:
            status = (
                "CONFLICT — not resolved" if r.has_conflict
                else ("Answered" if r.found_answer else "Incomplete")
            )
            lines.append(f"=== Round {r.number} [{status}] — {r.timestamp} ===")
            lines.append(f"Query: {r.query}")
            lines.append(f"Reasoning: {r.reasoning}")
            if r.answer:
                lines.append(f"Answer: {r.answer}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ResearchStepEntry — per-step execution record
# ---------------------------------------------------------------------------

class ResearchStepEntry:
    """One step entry for DeepResearcherV3's PlanFile."""

    def __init__(self, step_number: int, thinking: str, evaluation: str,
                 memory: str, next_goal: str, action_name: str, action_result: str) -> None:
        self.step_number = step_number
        self.thinking = thinking
        self.evaluation = evaluation
        self.memory = memory
        self.next_goal = next_goal
        self.action_name = action_name
        self.action_result = action_result
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
# ResearchPlanFile — thin wrapper over PlanFile for DeepResearcherV3
# ---------------------------------------------------------------------------

class ResearchPlanFile(PlanFile):
    """PlanFile specialised for DeepResearcherV3Agent."""

    def initialize_plan(self, steps: List[str]) -> None:
        self.todo_list.set_steps(steps)
        self.flow_chart.set_steps(steps)

    def update_plan(self, steps: List[str]) -> None:
        self.todo_list.set_steps(steps)
        self.flow_chart.set_steps(steps)

    def add_step(self, step_number: int, thinking: str, evaluation: str,
                 memory: str, next_goal: str, action_name: str, action_result: str) -> None:
        self.exec_history.add_entry(ResearchStepEntry(
            step_number=step_number, thinking=thinking, evaluation=evaluation,
            memory=memory, next_goal=next_goal, action_name=action_name,
            action_result=action_result,
        ))
        self.todo_list.complete_step(result=action_result if action_result else "")


# ---------------------------------------------------------------------------
# Tool: QueryTool — generate an optimized search query
# ---------------------------------------------------------------------------

class QueryTool(Tool):
    name: str = "query_tool"
    description: str = "Generates an optimized search query for the current research round."
    metadata: Dict[str, Any] = Field(default={})
    require_grad: bool = Field(default=False)
    model_name: str = Field(default="")

    def __init__(self, model_name: str, require_grad: bool = False, **kwargs):
        super().__init__(model_name=model_name, require_grad=require_grad, **kwargs)

    async def __call__(self, **kwargs) -> ToolResponse:
        task: str = kwargs.get("task", "")
        image: str = kwargs.get("image", "")
        filter_year: Optional[int] = kwargs.get("filter_year")
        session: ResearchSession = kwargs.get("session")
        guidance: str = kwargs.get("guidance", "")

        messages = await prompt_manager.get_messages(
            prompt_name="deep_researcher_v3_query",
            agent_modules={
                "task": task,
                "image": image,
                "filter_year": str(filter_year) if filter_year else "",
                "previous_context": session.execution_log_text() if session else "",
                "guidance": guidance,
            },
        )

        if image and image.lower().endswith((".jpg", ".jpeg", ".png")):
            last = messages[-1]
            text_content = last.content if isinstance(last.content, str) else str(last.content)
            messages[-1] = HumanMessage(content=[
                ContentPartText(text=text_content),
                ContentPartImage(image_url=ImageURL(
                    url=make_file_url(file_path=assemble_project_path(image)),
                    detail="high",
                )),
            ])

        resp = await model_manager(model=self.model_name, messages=messages, caller="v3/query")
        query = resp.message.strip() if resp and resp.message else ""
        return ToolResponse(
            success=bool(query),
            message=query,
            extra=ToolExtra(data={"query": query}),
        )


# ---------------------------------------------------------------------------
# Tool: SearchTool — concurrent API + LLM search (or Google Lens for images)
# ---------------------------------------------------------------------------

class SearchTool(Tool):
    name: str = "search_tool"
    description: str = "Executes concurrent web searches and returns labeled reports."
    metadata: Dict[str, Any] = Field(default={})
    require_grad: bool = Field(default=False)
    model_name: str = Field(default="")
    workdir: str = Field(default="")
    summary_model_name: str = Field(default="")
    num_results: int = Field(default=10)
    fetch_timeout: float = Field(default=20.0)
    llm_search_models: List[str] = Field(default_factory=list)

    def __init__(self, model_name: str, workdir: str, summary_model_name: str,
                 num_results: int = 10, fetch_timeout: float = 20.0,
                 llm_search_models: Optional[List[str]] = None,
                 require_grad: bool = False, **kwargs):
        super().__init__(
            model_name=model_name, workdir=workdir, summary_model_name=summary_model_name,
            num_results=num_results, fetch_timeout=fetch_timeout,
            llm_search_models=llm_search_models or [],
            require_grad=require_grad, **kwargs
        )
        self._firecrawl_search = FirecrawlSearch()
        self._google_lens_search = GoogleLensSearch()

    async def __call__(self, **kwargs) -> ToolResponse:
        task: str = kwargs.get("task", "")
        query: str = kwargs.get("query", "")
        image: str = kwargs.get("image", "")
        filter_year: Optional[int] = kwargs.get("filter_year")
        round_num: int = kwargs.get("round_num", 1)
        slog: Optional[SearchLogger] = kwargs.get("slog")

        reports: List[str] = []

        if image:
            result = await self._multimodal_search(query, filter_year, round_num, slog, image)
            if not isinstance(result, Exception) and result[0]:
                text, _, _, label = result
                reports.append(f"**[{label}]**\n{text}")
        else:
            api_coro = self._api_search(query, filter_year, round_num, slog)
            llm_coros = [self._llm_search(m, task, query) for m in self.llm_search_models]
            gathered = await asyncio.gather(api_coro, *llm_coros, return_exceptions=True)

            api_result = gathered[0]
            if not isinstance(api_result, Exception):
                api_text, _, _, api_label = api_result
                if api_text:
                    reports.append(f"**[{api_label}]**\n{api_text}")
            else:
                logger.warning(f"| ❌ API search failed: {api_result}")

            for m, r in zip(self.llm_search_models, gathered[1:]):
                if isinstance(r, Exception):
                    logger.warning(f"| ❌ LLM search {m} failed: {r}")
                elif r:
                    if slog:
                        slog.save_llm_search(round_num, m, r)
                    reports.append(f"**[{m}]**\n{r}")

        if not reports:
            return ToolResponse(success=False, message="No search results found.")

        combined = "\n\n---\n\n".join(reports)
        return ToolResponse(
            success=True,
            message=combined,
            extra=ToolExtra(data={"reports": reports, "query": query}),
        )

    async def _multimodal_search(self, query: str, filter_year: Optional[int],
                                  round_num: int, slog: Optional[SearchLogger],
                                  image: str) -> tuple:
        label = "google_lens_search"
        try:
            resp = await self._google_lens_search(
                query=query, image=image,
                num_results=self.num_results, filter_year=filter_year,
            )
            search_items = (
                resp.extra.data.get("search_items", [])
                if resp.success and resp.extra and resp.extra.data else []
            )
            logger.info(f"| ✅ {self._google_lens_search.name} returned {len(search_items)} results")
        except Exception as e:
            logger.error(f"| ❌ {self._google_lens_search.name} failed: {e}")
            search_items = []

        if not search_items:
            return "", [], [], label

        all_results, _ = await self._fetch_and_summarize(search_items, query)
        synthesized = await self._synthesize(query, all_results)
        round_text = self._build_round_summary(synthesized, all_results)
        if slog:
            slog.save_api_search(round_num, query, round_text)
        return round_text, [], [], label

    async def _api_search(self, query: str, filter_year: Optional[int],
                           round_num: int, slog: Optional[SearchLogger]) -> tuple:
        label = "firecrawl_search"
        try:
            resp = await self._firecrawl_search(
                query=query, num_results=self.num_results, filter_year=filter_year,
            )
            search_items = (
                resp.extra.data.get("search_items", [])
                if resp.success and resp.extra and resp.extra.data else []
            )
            logger.info(f"| ✅ {self._firecrawl_search.name} returned {len(search_items)} results")
        except Exception as e:
            logger.error(f"| ❌ {self._firecrawl_search.name} failed: {e}")
            search_items = []

        if not search_items:
            return "", [], [], label

        all_results, _ = await self._fetch_and_summarize(search_items, query)
        synthesized = await self._synthesize(query, all_results)
        round_text = self._build_round_summary(synthesized, all_results)
        if slog:
            slog.save_api_search(round_num, query, round_text)
        return round_text, [], [], label

    async def _llm_search(self, model: str, task: str, query: str) -> str:
        messages = await prompt_manager.get_messages(
            prompt_name="deep_researcher_v3_llm_search",
            agent_modules={"task": task, "query": query},
        )
        logger.info(f"| 🔍 LLM search using {model}")
        resp = await model_manager(model=model, messages=messages, caller="v3/llm_search")
        if not resp or not resp.success:
            raise ValueError(f"Model call failed: {resp.message if resp else 'no response'}")
        text = resp.message.strip() if resp.message else ""
        if not text:
            raise ValueError("Empty response")
        logger.info(f"| ✅ LLM search {model} done ({len(text)} chars)")
        return text

    async def _fetch_and_summarize(self, items: list, query: str) -> tuple:
        if not items:
            return [], []

        async def _process_one(item) -> tuple:
            url = getattr(item, "url", "") or item.get("url", "")
            title = getattr(item, "title", "") or item.get("title", "")
            snippet = getattr(item, "description", "") or item.get("description", "")

            prefetched = getattr(item, "content", None) or (
                item.get("content") if isinstance(item, dict) else None
            )
            content = None
            if prefetched:
                content = prefetched
            else:
                try:
                    resp = await asyncio.wait_for(
                        fetch_url(url=url, timeout=self.fetch_timeout),
                        timeout=self.fetch_timeout,
                    )
                    if resp and resp.get("markdown"):
                        content = resp["markdown"]
                except Exception:
                    pass

            summary = snippet
            if content:
                try:
                    page_messages = await prompt_manager.get_messages(
                        prompt_name="deep_researcher_v3_page_summary",
                        agent_modules={"query": query, "title": title, "content": content},
                    )
                    llm_resp = await model_manager(
                        model=self.summary_model_name, messages=page_messages, caller="v3/page_summary",
                    )
                    if llm_resp and llm_resp.message and llm_resp.message.strip():
                        summary = llm_resp.message.strip()
                except Exception as e:
                    logger.warning(f"| ❌ Summary LLM failed for {url}: {e}")

            return {"title": title, "url": url, "summary": summary}, {}

        tasks = [_process_one(item) for item in items]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        processed, logs = [], []
        for r in raw_results:
            if not isinstance(r, Exception):
                result, log_entry = r
                processed.append(result)
                logs.append(log_entry)
        return processed, logs

    async def _synthesize(self, query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        sources = "\n\n".join(
            f"[{i}] {r.get('title', 'Untitled')} ({r.get('url', '')})\n{r.get('summary', '')}"
            for i, r in enumerate(results, 1) if r.get("summary")
        )
        if not sources:
            return ""
        try:
            messages = await prompt_manager.get_messages(
                prompt_name="deep_researcher_v3_synthesis",
                agent_modules={
                    "query": query,
                    "num_sources": str(len(results)),
                    "sources": sources,
                },
            )
            resp = await model_manager(
                model=self.summary_model_name, messages=messages, caller="v3/synthesis",
            )
            if resp and resp.message and resp.message.strip():
                return resp.message.strip()
        except Exception as e:
            logger.warning(f"| ❌ Synthesis LLM failed: {e}")
        return "\n\n".join(r.get("summary", "") for r in results if r.get("summary"))

    def _build_round_summary(self, synthesized: str, results: List[Dict[str, Any]]) -> str:
        parts = []
        if synthesized:
            parts.append(synthesized)
            parts.append("")
        parts.append("## References")
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            parts.append(f"{i}. [{title}]({url})")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool: EvalTool — evaluate search reports for completeness + conflicts
# ---------------------------------------------------------------------------

class EvalTool(Tool):
    name: str = "eval_tool"
    description: str = "Evaluates search reports for task completeness and source conflicts."
    metadata: Dict[str, Any] = Field(default={})
    require_grad: bool = Field(default=False)
    model_name: str = Field(default="")

    def __init__(self, model_name: str, require_grad: bool = False, **kwargs):
        super().__init__(model_name=model_name, require_grad=require_grad, **kwargs)

    async def __call__(self, **kwargs) -> ToolResponse:
        task: str = kwargs.get("task", "")
        content: str = kwargs.get("content", "")
        session: ResearchSession = kwargs.get("session")

        messages = await prompt_manager.get_messages(
            prompt_name="deep_researcher_v3_eval",
            agent_modules={
                "task": task,
                "previous": session.execution_log_text() if session else "",
                "content": content,
            },
        )
        try:
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ResearchSummary,
                caller="v3/eval",
            )
            if (
                response and response.extra
                and hasattr(response.extra, "parsed_model")
                and response.extra.parsed_model
            ):
                result: ResearchSummary = response.extra.parsed_model
                lines = [
                    f"- **Found Answer:** {result.found_answer}",
                    f"- **Has Conflict:** {result.has_conflict}",
                    f"- **Answer:** {result.answer or 'N/A'}",
                    f"\n**Reasoning:**\n{result.reasoning}",
                ]
                return ToolResponse(
                    success=True,
                    message="\n".join(lines),
                    extra=ToolExtra(parsed_model=result),
                )
            raw = response.message if response else ""
            fallback = ResearchSummary(reasoning=raw.strip() if raw else "", found_answer=False)
            return ToolResponse(
                success=False,
                message="- **Found Answer:** False\n- **Has Conflict:** False\n- **Answer:** N/A",
                extra=ToolExtra(parsed_model=fallback),
            )
        except Exception as exc:
            logger.warning(f"| Evaluation failed: {exc}")
            fallback = ResearchSummary(reasoning="", found_answer=False)
            return ToolResponse(
                success=False,
                message="- **Found Answer:** False\n- **Has Conflict:** False\n- **Answer:** N/A",
                extra=ToolExtra(parsed_model=fallback),
            )


# ---------------------------------------------------------------------------
# Tool: FinishTool — terminate and return final answer
# ---------------------------------------------------------------------------

class FinishTool(Tool):
    name: str = "finish_tool"
    description: str = "Terminates and returns the final answer."
    metadata: Dict[str, Any] = Field(default={})
    require_grad: bool = Field(default=False)

    async def __call__(self, reasoning: str, answer: str) -> ToolResponse:
        message = f"**Reasoning:**\n{reasoning}\n\n**Answer:**\n{answer}"
        return ToolResponse(
            success=True,
            message=message,
            extra=ToolExtra(data={"answer": answer, "reasoning": reasoning}),
        )


# ---------------------------------------------------------------------------
# Tool contract — injected as available_tools in the prompt
# ---------------------------------------------------------------------------

_TOOL_CONTRACT = """
plan — Initialize or update the research plan. Call this FIRST (step 1), or after eval when a replan is needed.
  args:
    steps (str, required): JSON array of step descriptions, e.g. ["Query: find X", "Search", "Eval"].
    reasoning (str, required): Why this plan / what changed since the last plan.

query — Generate an optimized search query for the current round.
  args:
    guidance (str, required): Angle or focus for this round's query (what to target and why).

search — Execute concurrent API search (FirecrawlSearch) + LLM web searches, or Google Lens for image tasks.
  args: {} (no args needed)

eval — Evaluate the latest search results for task completeness and source conflicts.
  args: {} (no args needed)

finish — Terminate and return the final answer.
  args:
    reasoning (str, required): The reasoning behind the final answer.
    answer (str, required): The final answer to return.
"""


# ---------------------------------------------------------------------------
# DeepResearcherV3Agent — ThinkOutput loop, 5-tool research pipeline
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "ThinkOutput-driven multi-round web research agent supporting pure-text and multimodal "
    "image+text tasks. Five internal tools: plan / query / search / eval / finish. "
    "The LLM selects the next tool each step; search runs concurrent API + LLM web searches "
    "and synthesizes labeled reports; eval detects conflicts and judges completeness."
)


@AGENT.register_module(force=True)
class DeepResearcherV3Agent(Agent):
    """Multi-round ThinkOutput-driven research agent.

    Five-tool pipeline:
      plan → query → search → eval → [repeat or finish]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="deep_researcher_v3_agent")
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
        prompt_name: Optional[str] = "deep_researcher_v3",
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        max_rounds: int = 3,
        max_steps: int = 10,
        num_results: int = 10,
        summary_model_name: Optional[str] = None,
        llm_search_models: Optional[List[str]] = None,
        fetch_timeout: float = 20.0,
        enable_search_log: bool = True,
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
        self.num_results = num_results
        self.summary_model_name = summary_model_name or model_name or ""
        self.llm_search_models: List[str] = llm_search_models or []
        self.fetch_timeout = fetch_timeout
        self.enable_search_log = enable_search_log

        self.query_tool = QueryTool(model_name=self.model_name)
        self.search_tool = SearchTool(
            model_name=self.model_name,
            workdir=workdir,
            summary_model_name=self.summary_model_name,
            num_results=num_results,
            fetch_timeout=fetch_timeout,
            llm_search_models=self.llm_search_models,
        )
        self.eval_tool = EvalTool(model_name=self.model_name)
        self.finish_tool = FinishTool()

    # ------------------------------------------------------------------
    # Override _get_agent_context — inject PlanFile content as agent history
    # ------------------------------------------------------------------

    async def _get_agent_context(self, task: str, step_number: int = 0,
                                  ctx: SessionContext = None, **kwargs) -> Dict[str, Any]:  # noqa: ARG002
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
    # Override _get_tool_context — inject the 5 research tools
    # ------------------------------------------------------------------

    async def _get_tool_context(self, ctx: SessionContext, **kwargs) -> Dict[str, Any]:  # noqa: ARG002
        tool_context = dedent(f"""
            <tool_context>
            <available_tools>
            {_TOOL_CONTRACT}
            </available_tools>
            </tool_context>
        """)
        return {"tool_context": tool_context}

    # ------------------------------------------------------------------
    # One step: think + execute action
    # ------------------------------------------------------------------

    async def _think_and_action(
        self,
        task: str,
        messages: List,
        step_number: int,
        image: Optional[str],
        filter_year: Optional[int],
        session: ResearchSession,
        latest_search_content: Optional[str],
        current_query: Optional[str],
        round_num: int,
        slog: Optional[SearchLogger],
        plan: ResearchPlanFile,
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
            return {
                "done": False, "final_answer": None, "final_reasoning": None,
                "session": session, "latest_search_content": latest_search_content,
                "current_query": current_query, "round_num": round_num,
            }

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

            elif action_name == "query":
                guidance = action_args.get("guidance", "")
                tool_resp = await self.query_tool(
                    task=task,
                    image=image or "",
                    filter_year=filter_year,
                    session=session,
                    guidance=guidance,
                )
                current_query = tool_resp.message if tool_resp.success else current_query
                action_result = f"Query generated: {current_query}"
                logger.info(f"| ✅ Query: {current_query}")

            elif action_name == "search":
                if not current_query:
                    action_result = "Skipped — no query available. Call `query` first."
                    logger.warning("| ⚠️ Search skipped — no query")
                else:
                    tool_resp = await self.search_tool(
                        task=task,
                        query=current_query,
                        image=image or "",
                        filter_year=filter_year,
                        round_num=round_num,
                        slog=slog,
                    )
                    if tool_resp.success:
                        latest_search_content = tool_resp.message
                        action_result = tool_resp.message
                        round_num += 1
                        logger.info("| ✅ Search done")
                    else:
                        action_result = tool_resp.message
                        logger.warning(f"| ❌ Search failed: {tool_resp.message}")

            elif action_name == "eval":
                if not latest_search_content:
                    action_result = "Skipped — no search results available."
                    logger.warning("| ⚠️ Eval skipped — no search content")
                else:
                    tool_resp = await self.eval_tool(
                        task=task,
                        content=latest_search_content,
                        session=session,
                    )
                    action_result = tool_resp.message
                    if tool_resp.success and tool_resp.extra and tool_resp.extra.parsed_model:
                        ev: ResearchSummary = tool_resp.extra.parsed_model
                        session.add_round(ResearchRound(
                            number=len(session.rounds) + 1,
                            query=current_query or "",
                            reasoning=ev.reasoning or latest_search_content or "",
                            found_answer=ev.found_answer,
                            answer=ev.answer,
                            has_conflict=ev.has_conflict,
                        ))
                        logger.info(
                            f"| {'✅' if ev.found_answer else '❌'} Eval — "
                            f"found_answer={ev.found_answer} has_conflict={ev.has_conflict}"
                        )
                    else:
                        logger.warning("| ❌ Eval failed")

            elif action_name == "finish":
                final_reasoning = action_args.get("reasoning", "")
                final_answer = action_args.get("answer", "")
                tool_resp = await self.finish_tool(reasoning=final_reasoning, answer=final_answer)
                action_result = tool_resp.message
                done = True
                logger.info(f"| ✅ finish — answer: {str(final_answer)}")

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
            "session": session,
            "latest_search_content": latest_search_content,
            "current_query": current_query,
            "round_num": round_num,
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
        logger.info(f"| 🔍 DeepResearcherV3Agent starting: {task}")

        ctx = kwargs.get("ctx") or SessionContext()
        logger.info(f"| 🆔 Session: {ctx.id}")

        filter_year: int = kwargs.get("filter_year", datetime.now().year)

        image: Optional[str] = None
        if files:
            f = files[0]
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image = f

        slog: Optional[SearchLogger] = None
        if self.enable_search_log:
            session_id = kwargs.get("session_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
            slog = SearchLogger(base_dir=self.workdir, session_id=session_id)

        plan_path = make_plan_path(
            workdir=os.path.join(self.workdir, "agent", self.name),
            session_id=ctx.id,
            suffix="research",
        )
        plan = ResearchPlanFile(path=plan_path, task=task)

        session = ResearchSession(task=task)
        latest_search_content: Optional[str] = None
        current_query: Optional[str] = None
        round_num: int = 1
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
                    image=image,
                    filter_year=filter_year,
                    session=session,
                    latest_search_content=latest_search_content,
                    current_query=current_query,
                    round_num=round_num,
                    slog=slog,
                    plan=plan,
                )

                session = result["session"]
                latest_search_content = result["latest_search_content"]
                current_query = result["current_query"]
                round_num = result["round_num"]
                done = result["done"]
                if result["final_answer"]:
                    final_answer = result["final_answer"]
                    final_reasoning = result["final_reasoning"]

                step_number += 1
                if done:
                    break

            if step_number >= self.max_steps and not done:
                logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping")
                last_round = session.rounds[-1] if session.rounds else None
                final_answer = (last_round.answer if last_round and last_round.answer
                                else "No answer found.")

            answer_found = done or any(r.found_answer for r in session.rounds)

            if final_answer:
                message = (
                    f"**Reasoning:**\n{final_reasoning}\n\n**Answer:**\n{final_answer}"
                    if final_reasoning else final_answer
                )
            elif session.rounds:
                last = session.rounds[-1]
                parts = []
                if last.reasoning:
                    parts.append("**Reasoning:**\n" + last.reasoning)
                if last.answer:
                    parts.append(f"**Answer:** {last.answer}")
                message = "\n\n".join(parts) if parts else last.reasoning or "No answer found."
            else:
                message = f"No answer found after {step_number} step(s)."

            plan.finalize(answer=final_answer or message, success=answer_found, reasoning=final_reasoning)
            await plan.save()

            logger.info(
                f"| ✅ DeepResearcherV3Agent finished — "
                f"{step_number} step(s), answer_found={answer_found}"
            )
            logger.info(f"| 📄 Research plan: {plan_path}")

            return AgentResponse(
                success=answer_found,
                message=message,
                extra=AgentExtra(data={
                    "task": task,
                    "session_id": ctx.id,
                    "steps": step_number,
                    "rounds": len(session.rounds),
                    "answer_found": answer_found,
                    "answer": final_answer,
                    "plan_path": plan_path,
                    "history": [
                        {
                            "round_number": r.number,
                            "query": r.query,
                            "reasoning": r.reasoning,
                            "found_answer": r.found_answer,
                            "has_conflict": r.has_conflict,
                            "answer": r.answer,
                        }
                        for r in session.rounds
                    ],
                }),
            )

        except Exception as exc:
            logger.error(f"| ❌ DeepResearcherV3Agent error: {exc}", exc_info=True)
            return AgentResponse(success=False, message=f"Error during research: {exc}")
