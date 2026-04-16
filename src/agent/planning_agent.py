"""Planning agent — pure LLM reasoning + plan.md management.

Responsibility boundary
-----------------------
The PlanningAgent has exactly **two** responsibilities:

1. **LLM reasoning**: given a task context (original task, available agents,
   execution history), produce a ``PlanDecision`` — the structured answer to
   "what should we do next?".
2. **plan.md management**: maintain a ``plan.md`` file in ``workdir/<session_id>/``
   that records every round's decisions, dispatches, results, and analysis.

It does **NOT**:
- Import or call the AgentBus.
- Dispatch sub-agents.
- Run a multi-round loop.

All dispatching, result collection, and loop control is the bus's job.
The bus calls this agent once per round via agent manager (``agent_manager(name="planning", ...)``)
and reads the returned ``PlanDecision`` to decide what to do next.

Call contract
-------------
The bus passes a dict-serialised context as ``task`` (the string).
The planner returns ``AgentResponse`` with ``extra.data["decision"]``
containing the serialised ``PlanDecision``.

To feed results back, the bus calls the planner again with an updated context
string that includes the previous execution history.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext


# ---------------------------------------------------------------------------
# LLM structured-output schemas  (three focused schemas, one per step)
# ---------------------------------------------------------------------------

class SubTaskDispatch(BaseModel):
    """One sub-task to dispatch to a named agent."""

    agent_name: str = Field(
        description="Exact name of the agent to call (must match an available agent)."
    )
    task: str = Field(
        description="The sub-task description to send to this agent."
    )
    files: List[str] = Field(
        default_factory=list,
        description="Optional file paths to attach.",
    )


class EvaluationResult(BaseModel):
    """Step 1 — decide whether the task is fully complete."""

    reasoning: str = Field(
        description=(
            "Step-by-step evaluation: what has been accomplished, "
            "what is still missing, and the final verdict."
        )
    )
    is_done: bool = Field(
        description=(
            "True only if the entire original task is fully and correctly complete "
            "based on the execution history. False if any step failed or is missing."
        )
    )


class CompletionDecision(BaseModel):
    """Step 2a — extract a concise final answer (used when is_done=True)."""

    reasoning: str = Field(
        description="Brief justification for the chosen final answer."
    )
    final_result: str = Field(
        description=(
            "Concise final answer. "
            "Must be a number OR as few words as possible OR a comma-separated list of numbers/strings. "
            "No units unless required. No articles, abbreviations, or trailing punctuation unless required. "
            "If the answer cannot be determined, output exactly: Unable to determine"
        )
    )


class ContinuationDecision(BaseModel):
    """Step 2b — plan the next round of dispatches (used when is_done=False)."""

    reasoning: str = Field(
        description="Why these agents and tasks are needed in the next round."
    )
    plan_update: str = Field(
        description="Updated one-line description of the overall plan."
    )
    dispatches: List[SubTaskDispatch] = Field(
        description=(
            "Sub-tasks to dispatch in this round. All listed agents run concurrently. "
            "Must be non-empty."
        )
    )


# ---------------------------------------------------------------------------
# plan.md data model
# ---------------------------------------------------------------------------

@dataclass
class PlanRound:
    """Execution record for one planning round."""

    number: int
    goal: str
    agents: List[str]
    delivery_mode: str              # "UNICAST" | "BROADCAST"
    subtasks: Dict[str, str]        # agent_name → task text
    results: Dict[str, Any]         # agent_name → {success, result, error}
    analysis: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class PlanFile:
    """Manages the ``plan.md`` file for a single planning session.

    Structure mirrors Cursor plan files::

        ---
        name: <title>
        overview: "<task description>"
        todos:
          - id: step-1-agent_name
            content: "agent_name: subtask description"
            status: completed | pending
        isProject: false
        ---

        # <title>

        ## Execution Flow
        ```mermaid
        graph LR
          subgraph execution [Execution Flow]
            s1[Step 1: agent] --> s2[Step 2: agent]
          end
        ```

        ## Execution Log
        ### Round N — <timestamp>
        ...

        ## Final Result
        ...
    """

    def __init__(self, path: str, task: str, session_id: str) -> None:
        self.path = path
        self.full_task = task
        self.task_title = task
        self.session_id = session_id
        self.status = "running"
        self.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.rounds: List[PlanRound] = []
        self.final_result: Optional[str] = None

    # -- mutation ----------------------------------------------------------

    def add_round(self, round_: PlanRound) -> None:
        self.rounds.append(round_)

    def update_last_analysis(self, analysis: str) -> None:
        if self.rounds and analysis:
            self.rounds[-1].analysis = analysis

    def finalize(self, result: str, success: bool) -> None:
        self.status = "done" if success else "failed"
        self.final_result = result

    # -- persistence -------------------------------------------------------

    async def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        content = self._render()
        await asyncio.to_thread(self._write_sync, content)

    def _write_sync(self, content: str) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # -- context for LLM (plain text, no mermaid) --------------------------

    def execution_log_text(self) -> str:
        """Plain-text execution log passed to the LLM as context."""
        if not self.rounds:
            return "(no rounds completed yet)"
        lines: List[str] = []
        for r in self.rounds:
            lines.append(f"=== Round {r.number} — {r.timestamp} ===")
            lines.append(f"Goal: {r.goal}")
            lines.append(f"Dispatched ({r.delivery_mode}): {', '.join(r.agents)}")
            for agent in r.agents:
                lines.append(f"  {agent} subtask: {r.subtasks.get(agent, '')}")
            lines.append("Results:")
            for agent, res in r.results.items():
                ok = res.get("success", False)
                text = str(res.get("result") or res.get("error") or "")
                lines.append(f"  {'OK' if ok else 'FAIL'} {agent}: {text}")
            if r.analysis:
                lines.append(f"Analysis: {r.analysis}")
            lines.append("")
        return "\n".join(lines)

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _node_id(name: str) -> str:
        return name.replace("-", "_").replace(".", "_").replace(" ", "_")

    def _build_todos(self) -> List[Dict[str, str]]:
        """Build todo items from execution rounds for YAML frontmatter."""
        todos: List[Dict[str, str]] = []
        task_index = 0
        for r in self.rounds:
            for a in r.agents:
                task_index += 1
                st = r.subtasks.get(a, "")
                res = r.results.get(a, {})
                ok = res.get("success")
                if ok is True:
                    status = "completed"
                elif ok is False:
                    status = "failed"
                else:
                    status = "pending"
                todo_id = f"step-{task_index}-{self._node_id(a)}"
                todos.append({
                    "id": todo_id,
                    "content": f"{a}: {st}",
                    "status": status,
                })
        return todos

    @staticmethod
    def _yaml_escape(s: str) -> str:
        """Escape a string for use as a YAML double-quoted value."""
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')

    def _render_mermaid(self) -> List[str]:
        lines = ["```mermaid", "graph LR"]
        agent_rounds = [r for r in self.rounds if r.agents]

        if not agent_rounds:
            title = self.task_title.replace('"', "'")
            lines.append("  subgraph plan [Plan]")
            lines.append(f'    start(["{title}"])')
            if self.status in ("done", "failed"):
                lines.append(f'    start --> finish(["{self.status}"])')
            lines.append("  end")
            lines.append("```")
            return lines

        lines.append("  subgraph execution [Execution Flow]")
        task_index = 0
        prev_id = None
        for r in agent_rounds:
            for a in r.agents:
                task_index += 1
                a_id = f"s{task_index}"
                label = f"Step {task_index}: {a}"
                lines.append(f"    {a_id}[{label}]")
                if prev_id:
                    lines.append(f"    {prev_id} --> {a_id}")
                prev_id = a_id

        if self.status == "done" and prev_id:
            lines.append("    finish([Done])")
            lines.append(f"    {prev_id} --> finish")
        elif self.status == "failed" and prev_id:
            lines.append("    finish([Failed])")
            lines.append(f"    {prev_id} --> finish")

        lines.append("  end")
        lines.append("```")
        return lines

    def _render_round(self, r: PlanRound) -> List[str]:
        lines: List[str] = [f"### Round {r.number} — {r.timestamp}", ""]
        lines.append(f"> {r.goal}")
        lines.append("")
        if r.agents:
            lines.append(f"**Dispatched ({r.delivery_mode}):** {', '.join(f'`{a}`' for a in r.agents)}")
            lines.append("")
            for a in r.agents:
                st = r.subtasks.get(a, "")
                res = r.results.get(a, {})
                ok = res.get("success")
                if ok is True:
                    lines.append(f"- [x] **`{a}`**: {st}")
                    result_text = str(res.get("result") or "")
                    if result_text:
                        lines.append(f"  - Result: {result_text}")
                elif ok is False:
                    err = str(res.get("error") or "")
                    lines.append(f"- [x] ~~**`{a}`**: {st}~~ ❌")
                    if err:
                        lines.append(f"  - Error: {err}")
                else:
                    lines.append(f"- [ ] **`{a}`**: {st}")
            lines.append("")
        if r.analysis:
            lines += ["**Analysis:**", f"> {r.analysis}", ""]
        lines += ["---", ""]
        return lines

    def _render(self) -> str:
        todos = self._build_todos()
        esc = self._yaml_escape

        # --- YAML frontmatter (Cursor plan format) ---
        lines: List[str] = [
            "---",
            f"name: {self.task_title}",
            f'overview: "{esc(self.full_task)}"',
        ]
        if todos:
            lines.append("todos:")
            for t in todos:
                lines.append(f'  - id: {t["id"]}')
                lines.append(f'    content: "{esc(t["content"])}"')
                lines.append(f'    status: {t["status"]}')
        else:
            lines.append("todos: []")
        lines.append("isProject: false")
        lines.append("---")
        lines.append("")

        # --- Title ---
        lines.append(f"# {self.task_title}")
        lines.append("")

        # --- Execution Flow (Mermaid) ---
        lines.append("## Execution Flow")
        lines.append("")
        lines += self._render_mermaid()
        lines += ["", ""]

        # --- Execution Log ---
        lines += ["## Execution Log", ""]
        if not self.rounds:
            lines += ["*(planning in progress...)*", ""]
        else:
            for r in self.rounds:
                lines += self._render_round(r)

        # --- Final Result ---
        if self.final_result is not None:
            tag = "Completed" if self.status == "done" else "Failed"
            lines += [f"## Final Result — {tag}", "", self.final_result, ""]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

@AGENT.register_module(force=True)
class PlanningAgent(Agent):
    """Pure LLM planning agent.

    One LLM call per invocation.  Returns a ``PlanDecision`` to the caller
    (the AgentBus), which owns the multi-round loop and all dispatching.

    Also maintains a ``plan.md`` file that records every round.  The bus
    feeds results back by calling this agent again with an updated context,
    and the agent appends the new round to plan.md before returning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="planning_agent")
    description: str = Field(
        default=(
            "Decomposes tasks and decides which sub-agents to call next. "
            "Returns a PlanDecision; the AgentBus drives the loop."
        ),
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    # The PlanFile is stored per-session.  The bus creates it on the first
    # call and passes it back via kwargs on subsequent calls.
    _plan_files: Dict[str, PlanFile] = {}
    _agent_contract: str = None

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
        max_rounds: int = 20,
        **kwargs,
    ):
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name or "planning",
            memory_name=memory_name,
            require_grad=require_grad,
            **kwargs,
        )
        self.max_rounds = max_rounds
        self._plan_files: Dict[str, PlanFile] = {}
        self._agent_contract = None

    async def initialize(self) -> None:
        await super().initialize()


    # ------------------------------------------------------------------
    # Set and get the contract (called by the bus via kwargs)
    # ------------------------------------------------------------------

    async def _get_contract_context(self, agent_names: Optional[List[str]] = None) -> str:
        """Get the contract context string for prompt injection."""

        if self._agent_contract is None:
            from src.agent.server import agent_manager
            
            agent_names = agent_names or await agent_manager.list()
            agent_names = [agent_name for agent_name in agent_names if agent_name not in ["planning_agent"]]

            await agent_manager.set_contract(agent_names)
            self._agent_contract = await agent_manager.get_contract()
        
        return self._agent_contract

    # ------------------------------------------------------------------
    # plan.md lifecycle (called by the bus via kwargs)
    # ------------------------------------------------------------------

    async def _get_plan_file(self, session_id: str, task: str) -> PlanFile:
        """Return the existing PlanFile for a session, or create one."""
        if session_id not in self._plan_files:
            plan_path = os.path.join(self.workdir, f"{session_id}.plan.md")
            self._plan_files[session_id] = PlanFile(
                path=plan_path,
                task=task,
                session_id=session_id,
            )
        return self._plan_files[session_id]

    async def _remove_plan_file(self, session_id: str) -> None:
        """Clean up in-memory plan state for a completed session."""
        self._plan_files.pop(session_id, None)

    # ------------------------------------------------------------------
    # Main call — one LLM round
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        """Execute one planning round.

        Args:
            task: The original task description (same on every round).
            files: Optional list of file paths for additional context (same on every round).

        Returns:
            AgentResponse with extra.data["decision"] = PlanDecision dict.
        """
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()

        round_number = kwargs.get("round_number", 1)
        execution_history = kwargs.get("execution_history", "")
        round_results = kwargs.get("round_results", {})
        # Use agent_contract from bus if provided, otherwise build from agent_manager
        agent_contract = kwargs.get("agent_contract", None)

        logger.info(
            f"| 🧠 PlanningAgent round {round_number}/{self.max_rounds} "
            f"(session={ctx.id})"
        )

        # ------------------------------------------------------------------
        # Update plan.md with results from the PREVIOUS round
        # ------------------------------------------------------------------
        plan_file = await self._get_plan_file(ctx.id, task)

        if round_results and plan_file.rounds:
            plan_file.rounds[-1].results = round_results

        # ------------------------------------------------------------------
        # Build LLM messages via prompt_manager
        # ------------------------------------------------------------------
        history_text = execution_history if execution_history else "(no rounds completed yet)"

        if agent_contract is None:
            agent_contract = await self._get_contract_context()

        # Shared context (user message) — same for all three steps
        context_message = await prompt_manager.get_agent_message(
            self.prompt_name,
            modules={
                "task": task,
                "files": files or [],
                "agent_contract": agent_contract,
                "round_number": str(round_number),
                "max_rounds": str(self.max_rounds),
                "execution_history": history_text,
            },
        )

        # ------------------------------------------------------------------
        # Step 1 — Evaluate: is the task done?
        # ------------------------------------------------------------------
        try:
            eval_sys = await prompt_manager.get_system_message("planning_agent_evaluate")
            eval_output = await model_manager(
                model=self.model_name,
                messages=[eval_sys, context_message],
                response_format=EvaluationResult,
            )
            evaluation: EvaluationResult = eval_output.extra.parsed_model
        except Exception as exc:
            logger.error(f"| PlanningAgent evaluate error: {exc}", exc_info=True)
            evaluation = None

        if evaluation is None:
            evaluation = EvaluationResult(
                reasoning="Evaluation failed: LLM returned no structured decision.",
                is_done=False,
            )

        # Guard: if no execution history yet, task cannot be complete
        if evaluation.is_done and not execution_history:
            logger.warning("| PlanningAgent: evaluation said done but no history — forcing is_done=False")
            evaluation.is_done = False

        logger.info(f"| 🔍 Evaluate: is_done={evaluation.is_done}")

        # ------------------------------------------------------------------
        # Step 2a — Complete: extract concise final answer
        # ------------------------------------------------------------------
        if evaluation.is_done:
            try:
                complete_sys = await prompt_manager.get_system_message("planning_agent_complete")
                complete_output = await model_manager(
                    model=self.model_name,
                    messages=[complete_sys, context_message],
                    response_format=CompletionDecision,
                )
                completion: CompletionDecision = complete_output.extra.parsed_model
            except Exception as exc:
                logger.error(f"| PlanningAgent complete error: {exc}", exc_info=True)
                completion = None

            if completion is None:
                completion = CompletionDecision(
                    reasoning="Completion failed: LLM returned no structured decision.",
                    final_result="Unable to determine",
                )

            logger.info(f"| ✅ Final result: {completion.final_result}")

            plan_file.finalize(result=completion.final_result, success=True)
            await plan_file.save()
            logger.info("| PlanningAgent: task complete")

            return AgentResponse(
                success=True,
                message=completion.final_result,
                extra=AgentExtra(
                    data={
                        "decision": {
                            "is_done": True,
                            "final_result": completion.final_result,
                            "plan_update": "",
                            "dispatches": [],
                        },
                        "plan_path": plan_file.path,
                    },
                ),
            )

        # ------------------------------------------------------------------
        # Step 2b — Continue: plan next dispatches
        # ------------------------------------------------------------------
        try:
            continue_sys = await prompt_manager.get_system_message("planning_agent_continue")
            continue_output = await model_manager(
                model=self.model_name,
                messages=[continue_sys, context_message],
                response_format=ContinuationDecision,
            )
            continuation: ContinuationDecision = continue_output.extra.parsed_model
        except Exception as exc:
            logger.error(f"| PlanningAgent continue error: {exc}", exc_info=True)
            continuation = None

        if continuation is None:
            continuation = ContinuationDecision(
                reasoning="Planning failed: LLM returned no structured decision.",
                plan_update="Planning failed due to LLM error.",
                dispatches=[],
            )

        logger.info(f"| 📋 Plan: {continuation.plan_update}")

        # ------------------------------------------------------------------
        # Update plan.md with THIS round's continuation decision
        # ------------------------------------------------------------------
        if continuation.dispatches:
            delivery = "BROADCAST" if len(continuation.dispatches) > 1 else "UNICAST"
            agent_names = [d.agent_name for d in continuation.dispatches]
            subtasks = {d.agent_name: d.task for d in continuation.dispatches}

            plan_round = PlanRound(
                number=round_number,
                goal=continuation.plan_update,
                agents=agent_names,
                delivery_mode=delivery,
                subtasks=subtasks,
                results={},
            )
            plan_file.add_round(plan_round)
            logger.info(f"| PlanningAgent: dispatching {agent_names}")

        await plan_file.save()

        # If dispatches is empty (LLM failure fallback), call the completion step
        # to extract a concise final answer rather than returning raw execution history.
        if not continuation.dispatches:
            logger.warning("| PlanningAgent: empty dispatches — running completion step for final answer")
            try:
                complete_sys = await prompt_manager.get_system_message("planning_agent_complete")
                complete_output = await model_manager(
                    model=self.model_name,
                    messages=[complete_sys, context_message],
                    response_format=CompletionDecision,
                )
                completion = complete_output.extra.parsed_model
            except Exception as exc:
                logger.error(f"| PlanningAgent fallback complete error: {exc}", exc_info=True)
                completion = None

            if completion is None:
                completion = CompletionDecision(
                    reasoning="Completion failed after empty dispatches.",
                    final_result="Unable to determine",
                )

            logger.info(f"| ✅ Fallback final result: {completion.final_result}")
            plan_file.finalize(result=completion.final_result, success=True)
            await plan_file.save()
            return AgentResponse(
                success=True,
                message=completion.final_result,
                extra=AgentExtra(
                    data={
                        "decision": {
                            "is_done": True,
                            "final_result": completion.final_result,
                            "plan_update": continuation.plan_update,
                            "dispatches": [],
                        },
                        "plan_path": plan_file.path,
                    },
                ),
            )

        return AgentResponse(
            success=True,
            message=continuation.plan_update,
            extra=AgentExtra(
                data={
                    "decision": {
                        "is_done": False,
                        "final_result": None,
                        "plan_update": continuation.plan_update,
                        "dispatches": [d.model_dump() for d in continuation.dispatches],
                    },
                    "plan_path": plan_file.path,
                },
            ),
        )
