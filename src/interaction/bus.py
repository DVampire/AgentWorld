"""AgentBus — async message bus that drives the planner loop.

Architecture
------------
::

    submit(Task)
       │
       ▼
    session queue  (one per session_id)
       │
       ▼
    session worker  (one asyncio.Task per session_id)
       │
       ├─── Round 1: call planner → get PlanDecision
       │       │
       │       ├── dispatches: [agent_a, agent_b]   (concurrent via gather)
       │       └── collect results
       │
       ├─── Round 2: call planner again (with results) → get PlanDecision
       │       │
       │       └── dispatches: [agent_c]             (unicast)
       │
       └─── Round N: planner returns is_done=True → resolve caller Future

Concurrency model
~~~~~~~~~~~~~~~~~
Cross-session
    Each ``session_id`` owns a dedicated ``asyncio.Queue`` + worker.
    Different sessions run in complete isolation.

Intra-round
    All agents listed in one ``PlanDecision.dispatches`` are dispatched
    concurrently via ``asyncio.gather``.

Correlation
    ``submit()`` registers an ``asyncio.Future`` keyed by ``correlation_id``.
    The worker resolves it when the planning loop finishes (or errors).

Usage
~~~~~
::

    from src.interaction import bus
    from src.task import Task
    from src.session import SessionContext

    await bus.initialize()                # sync agents from agent manager
    ctx = SessionContext()
    task = Task(content="...", session_id=ctx.id)
    response = await bus.submit(task, ctx=ctx)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from src.config import config
from src.agent.server import agent_manager
from src.interaction.types import BusEvent, BusMessage, BusMessageType, DeliveryMode
from src.logger import logger
from src.session import SessionContext
from src.task import Task
from src.tracer import Tracer, Record

_DEFAULT_MAX_ROUNDS = 10


class AgentBus:
    """Session-isolated message bus that orchestrates the planner loop."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, planner_name: str = "planning_agent") -> None:
        self.planner_name = planner_name

        self._session_queues: Dict[str, asyncio.Queue] = {}
        self._session_workers: Dict[str, asyncio.Task] = {}
        self._session_contexts: Dict[str, SessionContext] = {}

        self._pending_responses: Dict[str, asyncio.Future] = {}

        self._known_agents: Dict[str, str] = {}  # name → description
        self._event_log: List[BusEvent] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # Agent registry  (synced from agent manager)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Sync agent names from agent manager into the bus registry."""
        self.workdir = config.workdir
        os.makedirs(os.path.join(self.workdir, "tracer"), exist_ok=True)
        try:
            agent_names = await agent_manager.list()
            for name in agent_names:
                info = await agent_manager.get_info(name)
                if info and name not in self._known_agents:
                    self._known_agents[name] = info.description or ""
            logger.info(
                f"| Bus: synced {len(self._known_agents)} agents from agent manager: "
                f"{list(self._known_agents.keys())}"
            )
        except Exception as exc:
            logger.warning(f"| Bus: failed to sync agents from agent manager: {exc}")

    async def register_agent(self, name: str, description: str = "") -> None:
        self._known_agents[name] = description

    async def unregister_agent(self, name: str) -> None:
        self._known_agents.pop(name, None)

    async def list(self) -> List[str]:
        return list(self._known_agents.keys())

    async def is_agent(self, name: str) -> bool:
        return name in self._known_agents

    async def get_agent_description(self, name: str) -> Optional[str]:
        return self._known_agents.get(name)

    # ------------------------------------------------------------------
    # Public API — submit
    # ------------------------------------------------------------------

    async def submit(
        self,
        task: Task,
        ctx: Optional[SessionContext] = None,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
    ) -> BusMessage:
        """Submit a task; the bus runs the full planner loop and returns the final result.

        Args:
            task:       Top-level work unit.
            ctx:        Session context.  Auto-created if ``None``.
            max_rounds: Maximum number of planner iterations.

        Returns:
            Terminal ``BusMessage`` (RESPONSE or ERROR).
        """
        if ctx is None:
            ctx = SessionContext()
        if task.session_id is None:
            task.session_id = ctx.id

        await self._ensure_session(task.session_id, ctx)

        msg = BusMessage.task_message(
            session_id=task.session_id,
            task_id=task.id,
            content=task.content,
            files=task.files,
            recipients=[self.planner_name],
            delivery_mode=DeliveryMode.UNICAST,
        )
        msg.payload["max_rounds"] = max_rounds

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg.correlation_id] = future

        task.mark_running()
        await self._session_queues[task.session_id].put(msg)
        self._log_event(msg, "message_enqueued")
        logger.info(f"| Bus: task '{task.id}' enqueued (session='{task.session_id}')")

        try:
            response: BusMessage = await future
            if response.type == BusMessageType.ERROR:
                task.mark_failed()
            else:
                task.mark_done()
            return response
        except asyncio.CancelledError:
            task.mark_cancelled()
            raise
        except Exception:
            task.mark_failed()
            raise

    # ------------------------------------------------------------------
    # Public API — direct dispatch (used ONLY for non-planner agents)
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        agent_name: str,
        message: BusMessage,
        ctx: Optional[SessionContext] = None,
    ) -> BusMessage:
        """Dispatch a single message to a single agent via agent manager.

        This is a **direct call**, not queued.  The bus uses it internally
        to call sub-agents inside the planner loop, and it can also be used
        externally for one-off agent calls outside the planning flow.
        """
        ctx = ctx or self._session_contexts.get(message.session_id)
        return await self._call_agent(agent_name, message, ctx=ctx)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel all workers and unblock all pending Futures."""
        logger.info("| Bus: shutting down")
        for _, worker in list(self._session_workers.items()):
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        for _, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(RuntimeError("AgentBus shut down"))
        self._session_workers.clear()
        self._session_queues.clear()
        self._session_contexts.clear()
        self._pending_responses.clear()
        logger.info("| Bus: shutdown complete")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    async def get_event_log(self, session_id: Optional[str] = None) -> List[BusEvent]:
        if session_id:
            return [e for e in self._event_log if e.session_id == session_id]
        return list(self._event_log)

    async def list_sessions(self) -> List[str]:
        return list(self._session_queues.keys())

    # ------------------------------------------------------------------
    # Internal — session management
    # ------------------------------------------------------------------

    async def _ensure_session(
        self, session_id: str, ctx: SessionContext
    ) -> None:
        async with self._get_lock():
            if session_id not in self._session_queues:
                self._session_queues[session_id] = asyncio.Queue()
                self._session_contexts[session_id] = ctx
                worker = asyncio.create_task(
                    self._session_worker(session_id),
                    name=f"bus-worker-{session_id}",
                )
                self._session_workers[session_id] = worker
                logger.info(f"| Bus: session worker created for '{session_id}'")

    # ------------------------------------------------------------------
    # Internal — session worker (drains the queue)
    # ------------------------------------------------------------------

    async def _session_worker(self, session_id: str) -> None:
        """Long-lived coroutine: processes tasks sequentially within a session."""
        queue = self._session_queues[session_id]
        logger.info(f"| Bus: worker started (session='{session_id}')")

        while True:
            try:
                message: BusMessage = await queue.get()
                try:
                    await self._run_planner_loop(message)
                except Exception as exc:
                    logger.error(
                        f"| Bus: error in planner loop for '{message.id}': {exc}",
                        exc_info=True,
                    )
                    self._resolve_future(
                        message.correlation_id,
                        BusMessage.error_message(
                            session_id=message.session_id,
                            task_id=message.task_id,
                            correlation_id=message.correlation_id,
                            error=str(exc),
                            parent_id=message.id,
                        ),
                    )
                finally:
                    queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"| Bus: worker cancelled (session='{session_id}')")
                break

    # ------------------------------------------------------------------
    # Internal — THE planner loop  (the core of the bus)
    # ------------------------------------------------------------------

    async def _run_planner_loop(self, message: BusMessage) -> None:
        """Drive the full planner→dispatch→collect→repeat cycle for one task.

        This method is the heart of the bus.  It:

        1. Calls the planner agent via agent manager to get a ``PlanDecision``.
        2. If ``is_done`` → resolve the caller's Future and return.
        3. Otherwise, dispatch all listed sub-agents concurrently.
        4. Feed results back to the planner in the next round.
        5. Repeat until ``is_done`` or ``max_rounds`` is exhausted.
        """
        # TTL check
        if message.is_expired():
            self._log_event(message, "ttl_expired")
            self._resolve_future(
                message.correlation_id,
                BusMessage.error_message(
                    session_id=message.session_id,
                    task_id=message.task_id,
                    correlation_id=message.correlation_id,
                    error=f"Message expired (TTL={message.ttl}s)",
                    parent_id=message.id,
                ),
            )
            return

        session_id = message.session_id
        task_id = message.task_id
        task_content = message.payload.get("content", "")
        task_files = message.payload.get("files", [])
        max_rounds = message.payload.get("max_rounds", _DEFAULT_MAX_ROUNDS)
        ctx = self._session_contexts.get(session_id)

        # Build agent contract from bus-known agents only (exclude planner)
        await agent_manager.set_contract(self._known_agents)
        agent_contract = await agent_manager.get_contract()

        round_results: Dict[str, Any] = {}  # results from previous round
        execution_history = ""               # cumulative text log

        # Per-session tracer file — each ctx gets its own file to support concurrency
        tracer_dir = os.path.join(self.workdir, "tracer")
        os.makedirs(tracer_dir, exist_ok=True)
        tracer_save_path = os.path.join(tracer_dir, f"{ctx.id}.json")

        tracer = Tracer()
        if os.path.exists(tracer_save_path):
            await tracer.load_from_json(tracer_save_path)

        for round_num in range(1, max_rounds + 1):
            logger.info(
                f"| Bus: planner round {round_num}/{max_rounds} "
                f"(session='{session_id}')"
            )
            self._log_event(message, "planner_round_start", detail=str(round_num))

            record = Record()
            record.observation = {"task": task_content, "round": round_num}

            # ---- 1. Call the planner ----
            decision_dict = await self._call_planner(
                task_content=task_content,
                task_files=task_files,
                ctx=ctx,
                task_id=task_id,
                round_number=round_num,
                max_rounds=max_rounds,
                agent_contract=agent_contract,
                execution_history=execution_history,
                round_results=round_results,
            )

            if decision_dict is None:
                logger.error("| Bus: planner returned None — aborting")
                record.observation["status"] = "error"
                record.action = {"agent_name": self.planner_name, "error": "Planner returned no decision"}
                await tracer.add_record(observation=record.observation, action=record.action, task_id=task_id, ctx=ctx)
                await tracer.save_to_json(tracer_save_path)
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.error_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        error="Planner returned no decision",
                        parent_id=message.id,
                    ),
                )
                return

            is_done = decision_dict.get("is_done", False)
            final_result = decision_dict.get("final_result")
            dispatches = decision_dict.get("dispatches", [])
            plan_update = decision_dict.get("plan_update", "")

            record.observation["plan_update"] = plan_update

            # ---- 2. Done? ----
            if is_done:
                logger.info(f"| Bus: planner signalled done (round {round_num})")
                if not final_result:
                    logger.warning("| Bus: planner set is_done=True but final_result is empty")
                record.observation["status"] = "done"
                record.action = {"agent_name": self.planner_name, "final_result": final_result or plan_update}
                await tracer.add_record(observation=record.observation, action=record.action, task_id=task_id, ctx=ctx)
                await tracer.save_to_json(tracer_save_path)
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.response_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        sender=self.planner_name,
                        success=True,
                        result=final_result or plan_update,
                    ),
                )
                return

            # ---- 3. No dispatches and not done → stall ----
            if not dispatches:
                logger.warning(
                    f"| Bus: planner returned no dispatches in round {round_num} — aborting"
                )
                record.observation["status"] = "error"
                record.action = {"agent_name": self.planner_name, "error": "Planner stalled (no dispatches and not done)"}
                await tracer.add_record(observation=record.observation, action=record.action, task_id=task_id, ctx=ctx)
                await tracer.save_to_json(tracer_save_path)
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.error_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        error="Planner stalled (no dispatches and not done)",
                        parent_id=message.id,
                    ),
                )
                return

            # ---- 4. Validate agent names ----
            known = set(self._known_agents.keys())
            valid_dispatches = []
            for d in dispatches:
                name = d["agent_name"]
                if name in known:
                    valid_dispatches.append(d)
                else:
                    logger.error(
                        f"| Bus: unknown agent '{name}' — skipping "
                        f"(known: {list(known)})"
                    )
            dispatches = valid_dispatches

            if not dispatches:
                logger.warning(
                    "| Bus: all dispatched agent names were invalid — aborting round"
                )
                execution_history += (
                    f"=== Round {round_num} ===\n"
                    f"ERROR: All agent names were invalid.\n\n"
                )
                continue

            agent_names = [d["agent_name"] for d in dispatches]
            delivery = "BROADCAST" if len(agent_names) > 1 else "UNICAST"
            logger.info(f"| Bus: {delivery} → {agent_names}")

            sub_messages = [
                BusMessage(
                    type=BusMessageType.PLAN,
                    session_id=session_id,
                    task_id=task_id,
                    sender="bus",
                    recipients=[d["agent_name"]],
                    delivery_mode=DeliveryMode.UNICAST,
                    payload={
                        "content": d["task"],
                        "files": d.get("files", []),
                    },
                )
                for d in dispatches
            ]

            raw_responses = await asyncio.gather(
                *[
                    self._call_agent(m.recipients[0], m, ctx=ctx)
                    for m in sub_messages
                ],
                return_exceptions=True,
            )

            # ---- 5. Collect results ----
            round_results = {}
            history_parts: List[str] = [
                f"=== Round {round_num} ===",
                f"Plan: {plan_update}",
                f"Dispatched ({delivery}): {', '.join(agent_names)}",
            ]

            dispatch_records: List[Dict[str, Any]] = []

            for d, raw in zip(dispatches, raw_responses):
                name = d["agent_name"]
                subtask = d["task"]
                history_parts.append(f"  {name} subtask: {subtask}")

                if isinstance(raw, BaseException):
                    logger.error(f"| Bus: agent '{name}' error: {raw}")
                    round_results[name] = {"success": False, "error": str(raw)}
                    history_parts.append(f"  FAIL {name}: {str(raw)}")
                    dispatch_records.append({"agent": name, "task": subtask, "success": False, "error": str(raw)})
                else:
                    ok = raw.payload.get("success", False)
                    result_text = str(
                        raw.payload.get("result") or raw.payload.get("error") or ""
                    )
                    round_results[name] = {
                        "success": ok,
                        "result": result_text,
                        "error": raw.payload.get("error"),
                    }
                    tag = "OK" if ok else "FAIL"
                    history_parts.append(f"  {tag} {name}: {result_text}")
                    logger.info(f"| Bus: agent '{name}' → {tag}")
                    dispatch_records.append({"agent": name, "task": subtask, "success": ok, "result": result_text, "error": raw.payload.get("error")})

            history_parts.append("")
            execution_history += "\n".join(history_parts) + "\n"

            record.observation["status"] = "running"
            record.action = {
                "agents": [{"agent_name": r["agent"], "task": r["task"], "success": r["success"], "result": r.get("result"), "error": r.get("error")} for r in dispatch_records],
            }
            await tracer.add_record(observation=record.observation, action=record.action, task_id=task_id, ctx=ctx)
            await tracer.save_to_json(tracer_save_path)

            self._log_event(message, "round_complete", detail=str(round_num))

        # ---- Max rounds exhausted ----
        logger.warning(f"| Bus: max rounds ({max_rounds}) reached")
        record = Record()
        record.observation = {"task": task_content, "round": max_rounds, "status": "max_rounds_exceeded"}
        record.action = {"agent_name": self.planner_name, "error": f"Planner did not finish within {max_rounds} rounds"}
        await tracer.add_record(observation=record.observation, action=record.action, task_id=task_id, ctx=ctx)
        await tracer.save_to_json(tracer_save_path)
        self._resolve_future(
            message.correlation_id,
            BusMessage.error_message(
                session_id=session_id,
                task_id=task_id,
                correlation_id=message.correlation_id,
                error=f"Planner did not finish within {max_rounds} rounds",
                parent_id=message.id,
            ),
        )

        # Tell the planner to finalize plan.md as failed
        await self._call_planner(
            task_content=task_content,
            task_files=task_files,
            ctx=ctx,
            task_id=task_id,
            round_number=max_rounds + 1,
            max_rounds=max_rounds,
            agent_contract=agent_contract,
            execution_history=execution_history,
            round_results=round_results,
        )

    # ------------------------------------------------------------------
    # Internal — call planner via agent manager and extract PlanDecision dict
    # ------------------------------------------------------------------

    async def _call_planner(
        self,
        task_content: str,
        task_files: List[str],
        ctx: Optional[SessionContext],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Call the planner via agent manager and return the raw PlanDecision dict."""
        try:
            agent_response = await agent_manager(
                name=self.planner_name,
                input={
                    "task": task_content,
                    "files": task_files if task_files else None,
                },
                ctx=ctx,
                **kwargs,
            )
            if hasattr(agent_response, "extra") and agent_response.extra:
                data = agent_response.extra.data or {}
                return data.get("decision")
        except Exception as exc:
            logger.error(f"| Bus: planner call failed: {exc}", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Internal — call any agent via agent manager → BusMessage
    # ------------------------------------------------------------------

    async def _call_agent(
        self,
        agent_name: str,
        message: BusMessage,
        ctx: Optional[SessionContext] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BusMessage:
        """Call one agent via agent manager, wrapping the result in a BusMessage.

        Never raises — errors are returned as ERROR-type BusMessages.
        """
        self._log_event(message, "agent_dispatched", agent_name=agent_name)
        logger.info(f"| Bus: dispatching → '{agent_name}'")

        try:
            call_kwargs: Dict[str, Any] = {}
            if extra_kwargs:
                call_kwargs.update(extra_kwargs)

            agent_result = await agent_manager(
                name=agent_name,
                input={
                    "task": message.payload.get("content", ""),
                    "files": message.payload.get("files", []),
                },
                ctx=ctx,
                **call_kwargs,
            )

            success = getattr(agent_result, "success", bool(agent_result))
            result_data = getattr(agent_result, "message", str(agent_result))
            error_str = None if success else result_data

            return BusMessage.response_message(
                session_id=message.session_id,
                task_id=message.task_id,
                correlation_id=message.correlation_id,
                sender=agent_name,
                success=success,
                result=result_data,
                error=error_str,
                parent_id=message.id,
            )

        except Exception as exc:
            logger.error(f"| Bus: agent '{agent_name}' raised: {exc}", exc_info=True)
            return BusMessage.error_message(
                session_id=message.session_id,
                task_id=message.task_id,
                correlation_id=message.correlation_id,
                error=str(exc),
                parent_id=message.id,
            )

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _resolve_future(self, correlation_id: str, response: BusMessage) -> None:
        future = self._pending_responses.pop(correlation_id, None)
        if future is not None and not future.done():
            future.set_result(response)

    def _log_event(
        self,
        message: BusMessage,
        event_type: str,
        agent_name: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        self._event_log.append(
            BusEvent(
                session_id=message.session_id,
                task_id=message.task_id,
                message_id=message.id,
                event_type=event_type,
                agent_name=agent_name,
                detail=detail,
            )
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

bus = AgentBus()
