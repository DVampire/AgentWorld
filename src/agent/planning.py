"""Planning agent — LLM reasoning + plan.md management + dispatch derivation.

Responsibility boundary
-----------------------
The PlanningAgent has exactly **two** responsibilities:

1. **LLM reasoning**: given a task context (task, available agents,
   execution history), produce a structured decision for the current phase.
2. **plan.md management**: maintain a ``plan.md`` file in ``workdir/<session_id>/``
   that records every round's decisions, results, and verifications.

It does **NOT**:
- Import or call the AgentBus.
- Run a multi-round loop.

The bus calls this agent once per round via agent manager and reads the
returned ``AgentResponse.extra.data["decision"]`` to decide what to do next.
Dispatches for the next round are derived directly from ``planned_steps`` in
code — no separate LLM call is needed.

Planning lifecycle — 2 phases per iteration
--------------------------------------------

  Phase 1 — Plan  (round 1, no history; or Replan when not done)
  ┌──────────────────────────────────────────────────────────────────┐
  │  LLM produces a comprehensive multi-round plan:                  │
  │  • planned_steps — ALL anticipated steps across ALL rounds       │
  │  Writes Plan N + full todo/flowchart to plan.md.                 │
  │  Code derives next round's dispatches from planned_steps         │
  │  (steps with the lowest round_number) and returns AgentResponse. │
  └──────────────────────────────────────────────────────────────────┘
      ↓  bus dispatches Round N agents, collects results
      ↓  bus writes results to plan.md, calls agent again

  Phase 2 — Verify  (after each round's results arrive)
  ┌──────────────────────────────────────────────────────────────────┐
  │  LLM reads plan.md (with results), evaluates task completion.    │
  │  Writes Verification N entry to plan.md.                         │
  │  → is_done=True  : return final result                           │
  │  → is_done=False : proceed to Replan (back to Phase 1)           │
  └──────────────────────────────────────────────────────────────────┘

plan.md execution log structure
---------------------------------
  Plan 1  →  Round 1  →  Verification 1
  Plan 2  →  Round 2  →  Verification 2
  …
  Final Result

plan.md file sections
---------------------
  ┌─────────────────────────────────────────────┐
  │  PlanFile                                   │
  │  ├── todo_list     : TodoList               │
  │  ├── flow_chart    : FlowChart              │
  │  ├── exec_history  : ExecutionHistory       │
  │  └── final_result  : FinalResult            │
  └─────────────────────────────────────────────┘

Todo step statuses
------------------
  [ ] pending  — not yet executed
  [x] done     — executed (result recorded in Execution Log)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.message.types import HumanMessage, SystemMessage
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext
from src.tool.types import Tool, ToolExtra, ToolResponse
from src.utils import PlanFile, TodoStep, FlowChartStep, make_plan_path


# ---------------------------------------------------------------------------
# LLM structured-output schemas
# ---------------------------------------------------------------------------


class PlannedStep(BaseModel):
    """One anticipated step in the initial comprehensive plan."""

    agent_name: str = Field(description="Exact agent name.")
    task: str = Field(description="Self-contained task description for this agent.")
    description: str = Field(
        default="",
        description="One-sentence summary of this step for the Todo List (≤15 words).",
    )
    files: List[str] = Field(default_factory=list, description="Optional file paths.")
    round_number: int = Field(
        description=(
            "Which execution round this step belongs to (1-based). "
            "Steps with the same round_number run concurrently."
        )
    )
    priority: str = Field(
        default="medium",
        description="Step priority: 'high' (🔴 blocking / critical path), 'medium' (🟡 normal), 'low' (🟢 optional / nice-to-have).",
    )


class PlanDecision(BaseModel):
    """Plan step output — used for both initial planning and plan updates.

    On init (no history): planned_steps covers ALL rounds upfront.
    On update (has history): planned_steps covers remaining rounds only.
    """

    reasoning: str = Field(
        description="How the task was decomposed (init) or what changed and why (update)."
    )
    planned_steps: List[PlannedStep] = Field(
        description=(
            "All anticipated steps. Steps sharing a round_number execute concurrently. "
            "Must be non-empty."
        )
    )


class EvaluationResult(BaseModel):
    """Verify step — decide whether the task is fully complete, and extract the
    final answer if it is."""

    reasoning: str = Field(
        description=(
            "Step-by-step evaluation: what has been accomplished, "
            "what is still missing or wrong, and the final verdict."
        )
    )
    is_done: bool = Field(
        description=(
            "True only if the entire original task is fully and correctly complete "
            "based on the execution history. False if any step failed or is missing."
        )
    )
    final_result: str = Field(
        description=(
            "Always required, never null. "
            "If is_done=False: set to 'task incomplete'. "
            "If is_done=True: the concise final answer — number, short phrase, or comma-separated list. "
            "No units/articles/punctuation unless required. LaTeX in $...$."
        )
    )
    reconciliation_task: Optional[str] = Field(
        default=None,
        description=(
            "Required when is_done=False due to a contradiction. "
            "Self-contained task for deep_analyzer_v2_agent to resolve the conflict independently. "
            "Null in all other cases."
        )
    )




# ---------------------------------------------------------------------------
# PlanRound — raw data record shared by the four section classes
# ---------------------------------------------------------------------------

class PlanRound(BaseModel):
    """Execution record for one planning round."""

    number: int
    goal: str
    agents: List[str]
    delivery_mode: str              # "UNICAST" | "BROADCAST"
    subtasks: List[str]             # parallel to agents — one task per dispatch
    results: List[Dict[str, Any]]   # parallel to agents — one result dict per dispatch
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class PlanEntry(BaseModel):
    """Planning agent decision record (reasoning before dispatching a round)."""

    number: int
    reasoning: str
    planned_steps: List[PlannedStep] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class VerificationEntry(BaseModel):
    """Verification record written after each round's results are collected."""

    number: int       # matches the round number being verified
    reasoning: str
    is_done: bool
    reconciliation_task: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# ===========================================================================
# PlanningHistoryEntry — per-round execution record for PlanningPlanFile
# ===========================================================================

class PlanningHistoryEntry:
    """One interleaved entry in the planning execution log.

    Wraps PlanEntry | PlanRound | VerificationEntry for render dispatch.
    """

    def __init__(
        self,
        plan_entries: List[PlanEntry],
        rounds: List[PlanRound],
        verifications: List[VerificationEntry],
    ) -> None:
        self._plan_entries = plan_entries
        self._rounds = rounds
        self._verifications = verifications

    @staticmethod
    def _render_plan_entry(e: PlanEntry) -> List[str]:
        lines: List[str] = [f"### Plan {e.number} — {e.timestamp}", ""]
        for line in e.reasoning.splitlines():
            lines.append(f"> {line}" if line.strip() else ">")
        lines.append("")
        if e.planned_steps:
            lines.append("**Planned steps:**")
            lines.append("")
            lines.append("| Round | Agent | Task |")
            lines.append("|-------|-------|------|")
            for s in e.planned_steps:
                task_short = s.task.replace("|", "\\|")
                lines.append(f"| {s.round_number} | `{s.agent_name}` | {task_short} |")
            lines.append("")
        lines += ["---", ""]
        return lines

    @staticmethod
    def _render_verification(v: VerificationEntry) -> List[str]:
        status = "✅ Task complete." if v.is_done else "⏳ Not yet complete."
        lines: List[str] = [f"### Verification {v.number} — {v.timestamp}", ""]
        lines.append(f"> **{status}**")
        lines.append(">")
        for line in v.reasoning.splitlines():
            lines.append(f"> {line}" if line.strip() else ">")
        if v.reconciliation_task:
            lines.append(">")
            lines.append("> **Reconciliation needed:**")
            lines.append(f"> {v.reconciliation_task}")
        lines.append("")
        lines += ["---", ""]
        return lines

    @staticmethod
    def _render_round(r: PlanRound) -> List[str]:
        lines: List[str] = [f"### Round {r.number} — {r.timestamp}", ""]
        lines.append(f"> {r.goal}")
        lines.append("")
        if r.agents:
            mode_label = "Concurrent" if r.delivery_mode == "BROADCAST" else "Single agent"
            lines.append(
                f"**Dispatched ({r.delivery_mode} / {mode_label}):** "
                f"{', '.join(f'`{a}`' for a in r.agents)}"
            )
            lines.append("")
            for i, a in enumerate(r.agents):
                st = r.subtasks[i] if i < len(r.subtasks) else ""
                res = r.results[i] if i < len(r.results) else {}
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
        lines += ["---", ""]
        return lines

    def render(self) -> List[str]:
        """Interleave plan → round → verification entries in number order."""
        lines: List[str] = []
        if not self._rounds and not self._plan_entries and not self._verifications:
            lines += ["*(planning in progress...)*", ""]
            return lines

        plan_by_num  = {e.number: e for e in self._plan_entries}
        round_by_num = {r.number: r for r in self._rounds}
        verif_by_num = {v.number: v for v in self._verifications}
        all_numbers  = sorted(set(plan_by_num) | set(round_by_num) | set(verif_by_num))

        for num in all_numbers:
            if num in plan_by_num:
                lines += self._render_plan_entry(plan_by_num[num])
            if num in round_by_num:
                lines += self._render_round(round_by_num[num])
            if num in verif_by_num:
                lines += self._render_verification(verif_by_num[num])
        return lines


# ===========================================================================
# PlanningPlanFile — PlanFile subclass specialised for PlanningAgent
# ===========================================================================

class PlanningPlanFile(PlanFile):
    """PlanFile subclass for PlanningAgent.

    Uses generic TodoList / FlowChart / ExecutionHistory from plan_file.py
    but adds planning-specific mutation helpers and a custom ExecutionHistory
    renderer (PlanningHistoryEntry) that interleaves Plan/Round/Verification.
    """

    def __init__(self, path: str, task: str, session_id: str = "") -> None:
        super().__init__(path=path, task=task, session_id=session_id)
        # Planning-specific entry lists (not stored in generic exec_history)
        self._plan_entries: List[PlanEntry] = []
        self._rounds: List[PlanRound] = []
        self._verifications: List[VerificationEntry] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rounds(self) -> List[PlanRound]:
        return self._rounds

    @property
    def status(self) -> str:
        if self.final_result.is_set:
            return "done" if self.final_result.success else "failed"
        return "running"

    # ------------------------------------------------------------------
    # Mutation — todo + flowchart
    # ------------------------------------------------------------------

    def initialize_plan(self, decision: PlanDecision) -> None:
        """Populate todo list and flowchart with ALL anticipated steps upfront."""
        by_round: Dict[int, List[PlannedStep]] = {}
        for step in decision.planned_steps:
            by_round.setdefault(step.round_number, []).append(step)

        todo_steps = []
        for rnum in sorted(by_round.keys()):
            for s in by_round[rnum]:
                sid = f"step-{len(todo_steps)+1}-{s.agent_name.replace('-','_').replace('.','_')}"
                todo_steps.append(TodoStep(
                    id=sid,
                    agent_name=s.agent_name,
                    description=s.description or s.task,
                    status="pending",
                    priority=s.priority,
                    round_number=s.round_number,
                ))
        self.todo_list.add_steps(todo_steps)

        by_round_agents: Dict[int, List[str]] = {}
        for s in decision.planned_steps:
            by_round_agents.setdefault(s.round_number, []).append(s.agent_name)
        for rnum in sorted(by_round_agents.keys()):
            self.flow_chart.add_step(FlowChartStep(
                number=rnum,
                agents=by_round_agents[rnum],
                is_planned=True,
            ))

    def update_plan(self, decision: PlanDecision) -> None:
        """Append only new rounds not already in todo/flowchart."""
        existing_todo_rounds = {s.round_number for s in self.todo_list._steps}
        existing_flow_rounds = {s.number for s in self.flow_chart._steps}

        by_round: Dict[int, List[PlannedStep]] = {}
        for step in decision.planned_steps:
            by_round.setdefault(step.round_number, []).append(step)

        new_todo: List[TodoStep] = []
        for rnum in sorted(by_round.keys()):
            steps = by_round[rnum]
            if rnum not in existing_todo_rounds:
                for s in steps:
                    sid = f"step-{len(self.todo_list._steps)+len(new_todo)+1}-{s.agent_name.replace('-','_').replace('.','_')}"
                    new_todo.append(TodoStep(
                        id=sid,
                        agent_name=s.agent_name,
                        description=s.description or s.task,
                        status="pending",
                        priority=s.priority,
                        round_number=rnum,
                    ))
            if rnum not in existing_flow_rounds:
                agents = [s.agent_name for s in steps]
                self.flow_chart.add_step(FlowChartStep(
                    number=rnum,
                    agents=agents,
                    is_planned=True,
                ))
        if new_todo:
            self.todo_list.add_steps(new_todo)

    def add_round(self, round_: PlanRound) -> None:
        """Record a dispatched round — update todo, flowchart, and local round list."""
        self._rounds.append(round_)

        # Sync todo: update agent names for existing planned steps, append extras
        existing = [s for s in self.todo_list._steps if s.round_number == round_.number]
        for i, step in enumerate(existing):
            if i < len(round_.agents):
                step.agent_name = round_.agents[i]
        for i in range(len(existing), len(round_.agents)):
            agent = round_.agents[i]
            task_desc = round_.subtasks[i] if i < len(round_.subtasks) else ""
            sid = f"step-{len(self.todo_list._steps)+1}-{agent.replace('-','_').replace('.','_')}"
            self.todo_list.add_steps([TodoStep(
                id=sid,
                agent_name=agent,
                description=task_desc,
                status="pending",
                priority="medium",
                round_number=round_.number,
            )])

        # Sync flowchart: mark round as dispatched (is_planned=False)
        self.flow_chart.add_step(FlowChartStep(
            number=round_.number,
            agents=round_.agents,
            is_planned=False,
        ))

    def apply_round_results(
        self,
        results,
        summaries: Optional[List[str]] = None,
    ) -> None:
        if not self._rounds:
            return
        last = self._rounds[-1]
        # bus passes results as dict {agent_name: {result, error, ...}};
        # _render_round indexes by position, so convert to an ordered list.
        if isinstance(results, dict):
            last.results = [results.get(a, {}) for a in last.agents]
        else:
            last.results = results
        self.todo_list.complete_round(round_number=last.number, summaries=summaries)

    def add_plan_entry(self, entry: PlanEntry) -> None:
        self._plan_entries.append(entry)

    def add_verification_entry(self, entry: VerificationEntry) -> None:
        for i, v in enumerate(self._verifications):
            if v.number == entry.number:
                self._verifications[i] = entry
                return
        self._verifications.append(entry)

    def last_round(self) -> Optional[PlanRound]:
        return self._rounds[-1] if self._rounds else None

    def last_plan_reasoning(self) -> str:
        return self._plan_entries[-1].reasoning if self._plan_entries else ""

    # ------------------------------------------------------------------
    # Override render — use PlanningHistoryEntry for exec log section
    # ------------------------------------------------------------------

    def render(self) -> str:
        entry = PlanningHistoryEntry(
            plan_entries=self._plan_entries,
            rounds=self._rounds,
            verifications=self._verifications,
        )
        parts: List[str] = [
            self.todo_list.render(),
            "",
            self.flow_chart.render(),
            "",
            "## Execution Log\n",
        ]
        parts.extend(entry.render())
        final = self.final_result.render()
        if final:
            parts.append(final)
        return "\n".join(parts)


# ===========================================================================
# PlanTool — wraps _run_plan LLM call
# ===========================================================================

class PlanTool(Tool):
    """Calls the planning LLM to produce or update a PlanDecision."""

    name: str = "plan_tool"
    description: str = "Calls the planning LLM to produce or update a structured execution plan."
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)
    model_name: str = Field(default="")

    def __init__(self, model_name: str, require_grad: bool = False, **kwargs) -> None:
        super().__init__(model_name=model_name, require_grad=require_grad, **kwargs)

    async def __call__(self, **kwargs) -> ToolResponse:
        context_vars: Dict[str, Any] = kwargs.get("context_vars", {})
        try:
            sys_msg = await prompt_manager.get_system_message("planning_agent_plan")
            context_message = await prompt_manager.get_agent_message(
                "planning_agent_plan", modules=context_vars
            )
            output = await model_manager(
                model=self.model_name,
                messages=[sys_msg, context_message],
                response_format=PlanDecision,
                caller="planner/plan",
            )
            decision: PlanDecision = output.extra.parsed_model
            return ToolResponse(success=True, message=decision.reasoning,
                                extra=ToolExtra(parsed_model=decision))
        except Exception as exc:
            logger.error(f"| PlanTool error: {exc}", exc_info=True)
            return ToolResponse(success=False, message=str(exc), extra=ToolExtra())


# ===========================================================================
# VerifyTool — wraps planning verify LLM call
# ===========================================================================

class VerifyTool(Tool):
    """Calls the verification LLM to decide whether the task is complete."""

    name: str = "planning_verify_tool"
    description: str = "Evaluates task completion after a round's results are collected."
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)
    model_name: str = Field(default="")

    def __init__(self, model_name: str, require_grad: bool = False, **kwargs) -> None:
        super().__init__(model_name=model_name, require_grad=require_grad, **kwargs)

    async def __call__(self, **kwargs) -> ToolResponse:
        context_vars: Dict[str, Any] = kwargs.get("context_vars", {})
        try:
            verify_sys = await prompt_manager.get_system_message("planning_agent_verify")
            context_message = await prompt_manager.get_agent_message(
                "planning_agent_verify", modules=context_vars
            )
            verify_output = await model_manager(
                model=self.model_name,
                messages=[verify_sys, context_message],
                caller="planner/verify",
                response_format=EvaluationResult,
            )
            evaluation: EvaluationResult = verify_output.extra.parsed_model
            return ToolResponse(success=True, message=evaluation.reasoning,
                                extra=ToolExtra(parsed_model=evaluation))
        except Exception as exc:
            logger.error(f"| VerifyTool error: {exc}", exc_info=True)
            return ToolResponse(success=False, message=str(exc), extra=ToolExtra())


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

@AGENT.register_module(force=True)
class PlanningAgent(Agent):
    """Pure LLM planning agent.

    One LLM interaction per invocation (initial plan or evaluate+continue/complete).
    The AgentBus owns the multi-round execution loop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="planning_agent")
    description: str = Field(
        default=(
            "Decomposes tasks into a comprehensive upfront plan, then drives "
            "execution round-by-round, updating the plan as results arrive. "
            "Returns a PlanDecision; the AgentBus drives the loop."
        ),
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

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
            prompt_name=prompt_name,
            memory_name=memory_name,
            require_grad=require_grad,
            **kwargs,
        )
        self.max_rounds = max_rounds
        self._plan_files: Dict[str, PlanningPlanFile] = {}
        self.plan_tool = PlanTool(model_name=self.model_name)
        self.verify_tool = VerifyTool(model_name=self.model_name)

    async def initialize(self) -> None:
        await super().initialize()

    # ------------------------------------------------------------------
    # Agent contract
    # ------------------------------------------------------------------

    async def _get_contract_context(self, agent_names: Optional[List[str]] = None) -> str:
        from src.agent.server import agent_manager
        agent_names = agent_names or await agent_manager.list()
        agent_names = [n for n in agent_names if n != "planning_agent"]
        await agent_manager.set_contract(agent_names)
        return await agent_manager.get_contract()

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _extract_round_results(self, results) -> List[str]:
        """Extract sub-agent messages directly — already structured as Reasoning + Answer."""
        # bus.py passes results as dict {agent_name: {result, error, ...}}
        items = results.values() if isinstance(results, dict) else results
        return [str(r.get("result") or r.get("error") or "") for r in items]

    # ------------------------------------------------------------------
    # PlanFile lifecycle
    # ------------------------------------------------------------------

    async def _get_plan_file(self, session_id: str, task: str) -> PlanningPlanFile:
        existing = self._plan_files.get(session_id)
        if existing is not None and existing.task == task:
            return existing
        # New session or different task — always create a fresh PlanFile so
        # stale state from a previous (possibly incomplete) session never
        # contaminates a new task.
        plan_path = make_plan_path(workdir=self.workdir, session_id=session_id, suffix="plan")
        plan_file = PlanningPlanFile(path=plan_path, task=task, session_id=session_id)
        self._plan_files[session_id] = plan_file
        return plan_file

    def _release_plan_file(self, session_id: str) -> None:
        """Release the in-memory PlanFile for a completed session."""
        self._plan_files.pop(session_id, None)

    def release_session(self, session_id: str) -> None:
        """Release the PlanFile for a session that did not complete normally.

        The bus must call this when a session is abandoned (e.g. max_rounds
        exceeded, user cancellation) to prevent unbounded memory growth.
        """
        if session_id in self._plan_files:
            logger.info(f"| PlanningAgent: releasing abandoned session {session_id}")
            self._plan_files.pop(session_id)

    def _get_plan_content(self, session_id: str) -> str:
        plan_file = self._plan_files.get(session_id)
        return plan_file.render() if plan_file else ""

    # ------------------------------------------------------------------
    # Internal helper — finalize plan.md and build the done AgentResponse
    # ------------------------------------------------------------------

    async def _build_done_response(self, final_result: str, plan_file: PlanningPlanFile, success: bool = True, reasoning: Optional[str] = None) -> AgentResponse:
        """Finalize plan.md, persist the final state to disk, and return the
        terminal AgentResponse.

        Must be async so that ``plan_file.save()`` is called AFTER
        ``finalize()``; this guarantees the ``## Final Result`` section and
        the flowchart's Done/Failed terminal node are actually written.
        """
        plan_file.finalize(answer=final_result, success=success, reasoning=reasoning)
        await plan_file.save()
        self._release_plan_file(plan_file.session_id)
        return AgentResponse(
            success=success,
            message=final_result,
            extra=AgentExtra(
                data={
                    "decision": {
                        "is_done": True,
                        "final_result": final_result,
                        "dispatches": [],
                    },
                    "plan_path": plan_file.path,
                },
            ),
        )

    # ------------------------------------------------------------------
    # Phase 1 — Plan (init or replan)
    # ------------------------------------------------------------------

    async def _run_plan(
        self,
        context_vars: Dict[str, Any],
        plan_file: PlanningPlanFile,
        round_number: int,
        is_init: bool,
    ) -> Optional[AgentResponse]:
        tool_resp = await self.plan_tool(context_vars=context_vars)
        if not tool_resp.success:
            logger.error(f"| PlanningAgent plan error: {tool_resp.message}")
            return None

        decision: PlanDecision = tool_resp.extra.parsed_model
        if not decision or not decision.planned_steps:
            logger.warning("| PlanningAgent: plan returned no steps")
            return None

        if is_init:
            min_round = min(s.round_number for s in decision.planned_steps)
            if min_round != 1:
                logger.warning(
                    f"| PlanningAgent: plan has no round-1 steps (min={min_round}), normalizing"
                )
                offset = 1 - min_round
                for s in decision.planned_steps:
                    s.round_number += offset
            logger.info(
                f"| 📋 Plan: {decision.reasoning} "
                f"({len(decision.planned_steps)} steps across "
                f"{len({s.round_number for s in decision.planned_steps})} rounds)"
            )
            plan_file.initialize_plan(decision)
        else:
            logger.info(
                f"| 📋 Plan update: {decision.reasoning} "
                f"({len(decision.planned_steps)} remaining steps)"
            )
            plan_file.update_plan(decision)

        plan_file.add_plan_entry(PlanEntry(
            number=round_number,
            reasoning=decision.reasoning,
            planned_steps=decision.planned_steps,
        ))

        next_round = min(s.round_number for s in decision.planned_steps)
        next_steps = [s for s in decision.planned_steps if s.round_number == next_round]

        delivery = "BROADCAST" if len(next_steps) > 1 else "UNICAST"
        round_ = PlanRound(
            number=next_round,
            goal=decision.reasoning,
            agents=[s.agent_name for s in next_steps],
            delivery_mode=delivery,
            subtasks=[s.task for s in next_steps],
            results=[{} for _ in next_steps],
        )
        plan_file.add_round(round_)
        logger.info(f"| PlanningAgent: dispatching {round_.agents}")
        await plan_file.save()

        return AgentResponse(
            success=True,
            message=decision.reasoning,
            extra=AgentExtra(
                data={
                    "decision": {
                        "is_done": False,
                        "final_result": None,
                        "plan_update": decision.reasoning,
                        "dispatches": [{"agent_name": s.agent_name, "task": s.task, "files": s.files} for s in next_steps],
                    },
                    "plan_path": plan_file.path,
                },
            ),
        )

    # ------------------------------------------------------------------
    # Phase 2 — Verify (after results arrive)
    # ------------------------------------------------------------------

    async def _extract_final_result(self, plan_file: PlanningPlanFile, context_vars: Dict[str, Any]) -> Optional[str]:
        """Fallback: ask the LLM to extract the final answer from the full plan.md content."""
        task = context_vars.get("task", "")
        try:
            extract_output = await model_manager(
                model=self.model_name,
                messages=[
                    SystemMessage(content=(
                        "You are a precise answer extractor. "
                        "Given the task and the full execution plan (with all step results), "
                        "extract ONLY the final answer to the task. "
                        "Reply with the bare answer only — no explanation, no punctuation, no extra words."
                    )),
                    HumanMessage(content=(
                        f"Task: {task}\n\n"
                        f"Plan execution log:\n{plan_file.render()}\n\n"
                        "What is the final answer?"
                    )),
                ],
                caller="planner/extract_final_result",
            )
            extracted = (extract_output.message or "").strip()
            if extracted:
                logger.info(f"| 🔍 LLM-extracted final_result: {extracted!r}")
                return extracted
        except Exception as exc:
            logger.error(f"| PlanningAgent extract_final_result error: {exc}")
        return None

    async def _run_verify(
        self,
        context_vars: Dict[str, Any],
        plan_file: PlanningPlanFile,
        verify_number: int,
    ) -> Optional[AgentResponse]:
        tool_resp = await self.verify_tool(context_vars=context_vars)
        if tool_resp.success and tool_resp.extra and tool_resp.extra.parsed_model:
            evaluation: EvaluationResult = tool_resp.extra.parsed_model
        else:
            logger.error(f"| PlanningAgent verify error: {tool_resp.message}")
            evaluation = EvaluationResult(
                reasoning="Verification failed: LLM returned no structured decision.",
                is_done=False,
                final_result="task incomplete",
            )

        if evaluation.is_done and not plan_file.rounds:
            logger.warning("| PlanningAgent: verify said done but no history — forcing is_done=False")
            evaluation.is_done = False

        logger.info(f"| 🔍 Verification {verify_number}: is_done={evaluation.is_done}")

        plan_file.add_verification_entry(VerificationEntry(
            number=verify_number,
            reasoning=evaluation.reasoning,
            is_done=evaluation.is_done,
            reconciliation_task=evaluation.reconciliation_task,
        ))
        await plan_file.save()

        if evaluation.is_done:
            final_result = evaluation.final_result
            if not final_result:
                logger.warning("| ⚠️ final_result was null despite is_done=True — extracting via LLM from plan.md")
                final_result = await self._extract_final_result(plan_file, context_vars)
            final_result = final_result or "Unable to determine"
            logger.info(f"| ✅ Final result: {final_result}")
            return await self._build_done_response(final_result, plan_file, reasoning=evaluation.reasoning)

        return None

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

        Round 1 (no history): runs the initial planning step to create a
        comprehensive upfront plan and dispatch the first agents.

        Round N (has history): runs evaluate → complete or continue,
        updating the plan based on sub-agent results.
        """
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()

        round_number  = kwargs.get("round_number", 1)
        round_results = kwargs.get("round_results", [])
        agent_contract = kwargs.get("agent_contract", None)

        logger.info(
            f"| 🧠 PlanningAgent round {round_number}/{self.max_rounds} "
            f"(session={ctx.id})"
        )

        plan_file = await self._get_plan_file(ctx.id, task)

        # ------------------------------------------------------------------
        # Apply results from the previous round
        # ------------------------------------------------------------------
        if round_results:
            if plan_file.rounds:
                summaries = self._extract_round_results(round_results)
                plan_file.apply_round_results(results=round_results, summaries=summaries)
                await plan_file.save()
            else:
                logger.warning("| PlanningAgent: round_results provided but no rounds recorded — ignoring")

        # ------------------------------------------------------------------
        # Build context variables (each phase builds its own message)
        # ------------------------------------------------------------------
        if agent_contract is None:
            agent_contract = await self._get_contract_context()

        context_vars = {
            "task": task,
            "files": files or [],
            "agent_contract": agent_contract,
            "round_number": str(round_number),
            "max_rounds": str(self.max_rounds),
            "plan": self._get_plan_content(ctx.id),
        }

        # ------------------------------------------------------------------
        # Phase 1 — Plan init (round 1, no prior history)
        # ------------------------------------------------------------------
        if round_number == 1 and not plan_file.rounds:
            response = await self._run_plan(context_vars, plan_file, round_number, is_init=True)
            if response is None:
                return await self._build_done_response("Unable to determine", plan_file, success=False)
            return response

        # ------------------------------------------------------------------
        # Phase 2 — Verify (after each round's results arrive)
        # Returns a final AgentResponse if done, None if not done.
        # verify_number matches the round whose results just arrived (1-based).
        # ------------------------------------------------------------------
        verify_number = max(1, round_number - 1)
        result = await self._run_verify(context_vars, plan_file, verify_number=verify_number)
        if result is not None:
            return result

        # ------------------------------------------------------------------
        # Phase 1 (Replan) — update plan and derive next round's dispatches
        # ------------------------------------------------------------------
        response = await self._run_plan(context_vars, plan_file, round_number, is_init=False)
        if response is None:
            return await self._build_done_response("Unable to determine", plan_file, success=False)
        return response
