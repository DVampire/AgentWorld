"""
Memory system backed by heartbeat's JSONL learning schema
(github.com/uameer/heartbeat).

Maps Autogenesis ChatEvents to heartbeat learning entries
(type, key, insight, confidence, source) without requiring an LLM call —
entries are derived deterministically from EventType and event data, matching
heartbeat's native ``write_learning()`` schema exactly.

Learning type mapping:
    TASK_START        → observation  (confidence 5)
    TOOL_STEP         → observation  (confidence 6)
    TASK_END          → pattern      (confidence 7)
    OPTIMIZATION_STEP → architecture (confidence 7)

The JSONL file is written to:
    <base_dir>/.heartbeat/memory/learnings.jsonl
"""

import json
import os
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.logger import logger
from src.memory.types import ChatEvent, EventType, Importance, Memory
from src.registry import MEMORY_SYSTEM
from src.session import SessionContext
from src.utils import dedent, file_lock, generate_unique_id


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class HeartbeatSummary(BaseModel):
    """Session-level summary aggregated from a batch of heartbeat JSONL entries."""

    id: str = Field(description="Unique identifier")
    importance: Importance = Field(description="Importance level")
    content: str = Field(description="Human-readable description of the batch that was flushed")
    entry_count: int = Field(default=0, description="Number of JSONL entries written in this batch")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch flush timestamp")

    def __str__(self) -> str:
        return dedent(f"""<heartbeat_summary>
            ID: {self.id}
            Importance: {self.importance.value}
            Content: {self.content}
            Entry Count: {self.entry_count}
            Timestamp: {self.timestamp}
            </heartbeat_summary>""")

    def __repr__(self) -> str:
        return self.__str__()


class HeartbeatInsight(BaseModel):
    """Individual learning entry in heartbeat's native schema, derived from a ChatEvent."""

    id: str = Field(description="Unique identifier")
    importance: Importance = Field(description="Importance level")
    content: str = Field(description="One-sentence insight (the 'insight' field in heartbeat JSONL)")
    learning_type: str = Field(
        description="Heartbeat learning type: pattern | pitfall | observation | architecture"
    )
    key: str = Field(description="2-5-word kebab-case identifier")
    confidence: int = Field(description="Confidence score 1-10")
    source_event_id: str = Field(description="ID of the originating ChatEvent")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")

    def __str__(self) -> str:
        return dedent(f"""<heartbeat_insight>
            ID: {self.id}
            Importance: {self.importance.value}
            Type: {self.learning_type}
            Key: {self.key}
            Content: {self.content}
            Confidence: {self.confidence}/10
            Source Event: {self.source_event_id}
            Tags: {self.tags}
            </heartbeat_insight>""")

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Event → heartbeat JSONL mapping
# ---------------------------------------------------------------------------

_LEARNING_TYPE: Dict[EventType, str] = {
    EventType.TASK_START: "observation",
    EventType.TOOL_STEP: "observation",
    EventType.TASK_END: "pattern",
    EventType.OPTIMIZATION_STEP: "architecture",
}

_CONFIDENCE: Dict[EventType, int] = {
    EventType.TASK_START: 5,
    EventType.TOOL_STEP: 6,
    EventType.TASK_END: 7,
    EventType.OPTIMIZATION_STEP: 7,
}

def _make_key(agent_name: Optional[str], event_type: EventType) -> str:
    """Build a kebab-case key from agent name and event type, capped at 50 chars."""
    slug = re.sub(r"[^a-z0-9]+", "-", (agent_name or "agent").lower()).strip("-")
    return f"{slug[:30]}-{event_type.value.replace('_', '-')}"[:50]


def _make_insight(event: ChatEvent) -> str:
    """Derive a one-sentence insight string from event data."""
    if not event.data:
        return f"Step {event.step_number}: {event.event_type.value} with no payload."
    data_str = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))[:120]
    return f"Step {event.step_number} ({event.event_type.value}): {data_str}"


def _event_to_jsonl_entry(event: ChatEvent) -> dict:
    """Convert a ChatEvent to a heartbeat JSONL learning entry dict."""
    return {
        "ts": event.timestamp.isoformat(),
        "type": _LEARNING_TYPE.get(event.event_type, "observation"),
        "key": _make_key(event.agent_name, event.event_type),
        "insight": _make_insight(event),
        "confidence": _CONFIDENCE.get(event.event_type, 5),
        "source": event.agent_name or "autogenesis",
    }


def _entry_to_insight(entry: dict, source_event_id: str) -> HeartbeatInsight:
    """Map a heartbeat JSONL entry dict to a HeartbeatInsight model."""
    confidence = entry["confidence"]
    if confidence >= 7:
        importance = Importance.HIGH
    elif confidence >= 5:
        importance = Importance.MEDIUM
    else:
        importance = Importance.LOW
    return HeartbeatInsight(
        id=generate_unique_id("hb_insight"),
        importance=importance,
        content=entry["insight"],
        learning_type=entry["type"],
        key=entry["key"],
        confidence=confidence,
        source_event_id=source_event_id,
        tags=[entry["type"], entry["source"]],
        timestamp=datetime.fromisoformat(entry["ts"]),
    )


# ---------------------------------------------------------------------------
# Per-session container
# ---------------------------------------------------------------------------

class HeartbeatCombinedMemory:
    """Per-session container that flushes events to heartbeat JSONL without LLM calls."""

    def __init__(
        self,
        max_summaries: int = 20,
        max_insights: int = 100,
        jsonl_path: Optional[Path] = None,
    ) -> None:
        self.max_summaries = max_summaries
        self.max_insights = max_insights
        self.jsonl_path = jsonl_path

        self.events: List[ChatEvent] = []
        self.summaries: List[HeartbeatSummary] = []
        self.insights: List[HeartbeatInsight] = []
        self._pending: List[ChatEvent] = []

    async def add_event(self, event: Union[ChatEvent, List[ChatEvent]]) -> None:
        """Append events and stage them for the next JSONL flush."""
        if isinstance(event, ChatEvent):
            events = [event]
        else:
            events = event
        for evt in events:
            self.events.append(evt)
            self._pending.append(evt)

    async def check_and_process_memory(self) -> None:
        """Flush any pending events to heartbeat JSONL entries."""
        if not self._pending:
            return
        await self._flush_pending()

    async def _flush_pending(self) -> None:
        """Convert pending events to JSONL, append to file, and update insight/summary lists."""
        new_insights: List[HeartbeatInsight] = []
        jsonl_lines: List[str] = []

        for evt in self._pending:
            entry = _event_to_jsonl_entry(evt)
            jsonl_lines.append(json.dumps(entry, ensure_ascii=False))
            new_insights.append(_entry_to_insight(entry, evt.id))

        if self.jsonl_path and jsonl_lines:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(jsonl_lines) + "\n")

        self.insights.extend(new_insights)
        await self._sort_and_limit_insights()

        if new_insights:
            importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
            batch_importance = min(new_insights, key=lambda i: importance_order[i.importance]).importance
            summary = HeartbeatSummary(
                id=generate_unique_id("hb_summary"),
                importance=batch_importance,
                content=(
                    f"Flushed {len(new_insights)} heartbeat entr"
                    f"{'y' if len(new_insights) == 1 else 'ies'} "
                    f"({len(set(i.learning_type for i in new_insights))} learning type(s)) "
                    f"from {len(self._pending)} event(s)."
                ),
                entry_count=len(new_insights),
            )
            self.summaries.append(summary)
            await self._sort_and_limit_summaries()

        self._pending.clear()

    async def _sort_and_limit_summaries(self) -> None:
        importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.summaries.sort(key=lambda x: importance_order[x.importance])
        if len(self.summaries) > self.max_summaries:
            self.summaries = self.summaries[: self.max_summaries]

    async def _sort_and_limit_insights(self) -> None:
        importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.insights.sort(key=lambda x: importance_order[x.importance])
        if len(self.insights) > self.max_insights:
            self.insights = self.insights[: self.max_insights]

    def clear(self) -> None:
        """Clear all session memory."""
        self.events.clear()
        self.summaries.clear()
        self.insights.clear()
        self._pending.clear()

    def size(self) -> int:
        """Return total event count."""
        return len(self.events)

    async def get_event(self, n: Optional[int] = None) -> List[ChatEvent]:
        if n is None:
            return self.events
        return self.events[-n:] if len(self.events) > n else self.events

    async def get_summary(self, n: Optional[int] = None) -> List[HeartbeatSummary]:
        if n is None:
            return self.summaries
        return self.summaries[-n:] if len(self.summaries) > n else self.summaries

    async def get_insight(self, n: Optional[int] = None) -> List[HeartbeatInsight]:
        if n is None:
            return self.insights
        return self.insights[-n:] if len(self.insights) > n else self.insights


# ---------------------------------------------------------------------------
# Memory system
# ---------------------------------------------------------------------------

@MEMORY_SYSTEM.register_module(force=True)
class HeartbeatMemorySystem(Memory):
    """Memory system backed by heartbeat's JSONL learning schema.

    Converts Autogenesis ChatEvents to heartbeat learning entries and appends
    them to a ``learnings.jsonl`` file under ``<base_dir>/.heartbeat/memory/``.
    No LLM is required: entries are derived deterministically from EventType
    and event data, matching heartbeat's native ``write_learning()`` schema.
    """

    require_grad: bool = Field(default=False, description="Whether the memory system requires gradients")

    def __init__(
        self,
        base_dir: Optional[str] = None,
        max_summaries: int = 20,
        max_insights: int = 100,
        require_grad: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(require_grad=require_grad, **kwargs)

        if base_dir is not None:
            self.base_dir = base_dir

        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| Heartbeat memory system base directory: {self.base_dir}")
        self.save_path = (
            os.path.join(self.base_dir, "memory_system.json") if self.base_dir else None
        )
        self._jsonl_path: Optional[Path] = (
            Path(self.base_dir) / ".heartbeat" / "memory" / "learnings.jsonl"
            if self.base_dir
            else None
        )

        self.max_summaries = max_summaries
        self.max_insights = max_insights

        self._session_memory_cache: Dict[str, HeartbeatCombinedMemory] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._cache_lock = asyncio.Lock()
        self._pending_process_tasks: Dict[str, asyncio.Task] = {}

    async def _get_or_create_session_memory(
        self, id: str
    ) -> tuple[HeartbeatCombinedMemory, asyncio.Lock]:
        """Get or create a HeartbeatCombinedMemory instance with proper locking."""
        async with self._cache_lock:
            if id not in self._session_locks:
                self._session_locks[id] = asyncio.Lock()
            if id not in self._session_memory_cache:
                self._session_memory_cache[id] = HeartbeatCombinedMemory(
                    max_summaries=self.max_summaries,
                    max_insights=self.max_insights,
                    jsonl_path=self._jsonl_path,
                )
                logger.info(f"| 📝 Created new heartbeat memory cache for id: {id}")
            return self._session_memory_cache[id], self._session_locks[id]

    async def _cleanup_session_memory(self, id: str) -> None:
        """Remove session memory from cache."""
        async with self._cache_lock:
            if id in self._session_memory_cache:
                del self._session_memory_cache[id]
            if id in self._session_locks:
                del self._session_locks[id]

    async def _process_memory_background(self, id: str) -> None:
        """Background task: flush heartbeat JSONL entries without blocking callers."""
        try:
            session_memory, session_lock = await self._get_or_create_session_memory(id)
            async with session_lock:
                await session_memory.check_and_process_memory()
            if self.save_path:
                await self.save_to_json(self.save_path)
        except Exception as e:
            logger.warning(f"| ⚠️ Background heartbeat memory processing failed: {e}")
        finally:
            current_task = asyncio.current_task()
            if self._pending_process_tasks.get(id) is current_task:
                self._pending_process_tasks.pop(id, None)

    async def start_session(self, ctx: SessionContext = None, **kwargs) -> str:
        """Start a new heartbeat memory session."""
        if self.save_path and os.path.exists(self.save_path):
            await self.load_from_json(self.save_path)

        if ctx is None:
            ctx = SessionContext()
        id = ctx.id
        await self._get_or_create_session_memory(id)
        logger.info(f"| Started heartbeat memory session: {id}")
        return id

    async def end_session(self, ctx: SessionContext = None, **kwargs) -> None:
        """End session: wait for in-flight JSONL flush, save state, clean up."""
        if ctx is None:
            return
        id = ctx.id

        if id in self._pending_process_tasks:
            try:
                await asyncio.wait_for(self._pending_process_tasks[id], timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"| ⚠️ Timeout waiting for heartbeat memory flush on session {id}"
                )
            except Exception as e:
                logger.warning(f"| ⚠️ Error waiting for heartbeat memory flush: {e}")

        if self.save_path:
            await self.save_to_json(self.save_path)

        await self._cleanup_session_memory(id)
        logger.info(f"| Ended heartbeat memory session: {id}")

    async def add_event(
        self,
        step_number: int,
        event_type: Any,
        data: Any,
        agent_name: str,
        task_id: Optional[str] = None,
        ctx: SessionContext = None,
        **kwargs,
    ) -> None:
        """Add an event to the heartbeat memory system."""
        if ctx is None:
            logger.warning("| No context available for add_event")
            return
        id = ctx.id

        if not isinstance(event_type, EventType):
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    logger.warning(
                        f"| ⚠️ Invalid event_type '{event_type}', defaulting to TOOL_STEP"
                    )
                    event_type = EventType.TOOL_STEP
            else:
                logger.warning(
                    f"| ⚠️ Invalid event_type type '{type(event_type)}', defaulting to TOOL_STEP"
                )
                event_type = EventType.TOOL_STEP

        event = ChatEvent(
            id=generate_unique_id("hb_event"),
            step_number=step_number,
            event_type=event_type,
            data=data,
            agent_name=agent_name,
            task_id=task_id,
            session_id=id,
        )

        session_memory, session_lock = await self._get_or_create_session_memory(id)
        async with session_lock:
            await session_memory.add_event(event)

        task = asyncio.create_task(self._process_memory_background(id))
        self._pending_process_tasks[id] = task

        if self.save_path:
            await self.save_to_json(self.save_path)

    async def clear_session(self, ctx: SessionContext = None, **kwargs) -> None:
        """Clear a specific session."""
        if ctx is None:
            return
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                self._session_memory_cache[id].clear()
        await self._cleanup_session_memory(id)

    async def get_event(
        self, ctx: SessionContext = None, n: Optional[int] = None, **kwargs
    ) -> List[ChatEvent]:
        """Get events from session."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_event(n=n)
        return []

    async def get_summary(
        self, ctx: SessionContext = None, n: Optional[int] = None, **kwargs
    ) -> List[HeartbeatSummary]:
        """Get batch-flush summaries from session."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_summary(n=n)
        return []

    async def get_insight(
        self, ctx: SessionContext = None, n: Optional[int] = None, **kwargs
    ) -> List[HeartbeatInsight]:
        """Get heartbeat learning insights from session."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_insight(n=n)
        return []

    async def save_to_json(self, file_path: str) -> str:
        """Persist memory state to JSON (mirrors heartbeat JSONL for portability)."""
        async with file_lock(file_path):
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            metadata = {
                "memory_system_type": "heartbeat_memory_system",
                "session_ids": list(self._session_memory_cache.keys()),
            }

            sessions: Dict[str, Any] = {}
            async with self._cache_lock:
                for id, session_memory in self._session_memory_cache.items():
                    sessions[id] = {
                        "session_memory": {
                            "events": [e.model_dump(mode="json") for e in session_memory.events],
                            "summaries": [
                                s.model_dump(mode="json") for s in session_memory.summaries
                            ],
                            "insights": [
                                i.model_dump(mode="json") for i in session_memory.insights
                            ],
                        }
                    }

            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"metadata": metadata, "sessions": sessions},
                    fh,
                    indent=4,
                    ensure_ascii=False,
                )

            logger.debug(f"| 💾 Heartbeat memory saved to {file_path}")
            return str(file_path)

    async def load_from_json(self, file_path: str) -> bool:
        """Restore memory state from JSON."""
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Heartbeat memory file not found: {file_path}")
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    load_data = json.load(fh)

                if "metadata" not in load_data or "sessions" not in load_data:
                    raise ValueError(
                        f"Invalid heartbeat memory format, got keys: {list(load_data.keys())}"
                    )

                sessions_data = load_data.get("sessions", {})
                async with self._cache_lock:
                    for id, session_data in sessions_data.items():
                        if id not in self._session_memory_cache:
                            self._session_memory_cache[id] = HeartbeatCombinedMemory(
                                max_summaries=self.max_summaries,
                                max_insights=self.max_insights,
                                jsonl_path=self._jsonl_path,
                            )
                            self._session_locks[id] = asyncio.Lock()

                        sm = self._session_memory_cache[id]
                        sm_data = session_data.get("session_memory", {})

                        if "events" in sm_data:
                            events = []
                            for ed in sm_data["events"]:
                                if ed.get("timestamp"):
                                    ed["timestamp"] = datetime.fromisoformat(ed["timestamp"])
                                if ed.get("event_type"):
                                    ed["event_type"] = EventType(ed["event_type"])
                                events.append(ChatEvent(**ed))
                            sm.events = events

                        if "summaries" in sm_data:
                            summaries = []
                            for sd in sm_data["summaries"]:
                                if sd.get("timestamp"):
                                    sd["timestamp"] = datetime.fromisoformat(sd["timestamp"])
                                if sd.get("importance"):
                                    sd["importance"] = Importance(sd["importance"])
                                summaries.append(HeartbeatSummary(**sd))
                            sm.summaries = summaries

                        if "insights" in sm_data:
                            insights = []
                            for idata in sm_data["insights"]:
                                if idata.get("timestamp"):
                                    idata["timestamp"] = datetime.fromisoformat(idata["timestamp"])
                                if idata.get("importance"):
                                    idata["importance"] = Importance(idata["importance"])
                                insights.append(HeartbeatInsight(**idata))
                            sm.insights = insights

                logger.info(f"| 📂 Heartbeat memory loaded from {file_path}")
                return True

            except Exception as e:
                logger.error(
                    f"| ❌ Failed to load heartbeat memory from {file_path}: {e}",
                    exc_info=True,
                )
                return False
