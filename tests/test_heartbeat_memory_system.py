"""
Tests for HeartbeatMemorySystem and its supporting types/helpers.

Run with: pytest tests/test_heartbeat_memory_system.py -v

These are pure unit / lightweight integration tests that do not require a
running model server — HeartbeatMemorySystem derives all entries
deterministically from event data.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

from src.memory.heartbeat_memory_system import (
    HeartbeatCombinedMemory,
    HeartbeatInsight,
    HeartbeatMemorySystem,
    HeartbeatSummary,
    _entry_to_insight,
    _event_to_jsonl_entry,
    _make_insight,
    _make_key,
)
from src.memory.types import ChatEvent, EventType, Importance
from src.session import SessionContext
from src.utils import generate_unique_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    step: int = 0,
    event_type: EventType = EventType.TOOL_STEP,
    agent_name: str = "test_agent",
    data: dict = None,
    session_id: str = "sess_test",
) -> ChatEvent:
    return ChatEvent(
        id=generate_unique_id("ev"),
        step_number=step,
        event_type=event_type,
        data=data or {"key": "value"},
        agent_name=agent_name,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Unit tests: mapping helpers
# ---------------------------------------------------------------------------

def test_make_key_contains_event_type():
    key = _make_key("my_agent", EventType.TOOL_STEP)
    assert "tool-step" in key
    assert len(key) <= 50


def test_make_key_no_agent_name():
    key = _make_key(None, EventType.TASK_END)
    assert "task-end" in key
    assert "agent" in key


def test_make_key_special_chars_stripped():
    key = _make_key("My Agent (v2)!", EventType.TASK_START)
    assert re.match(r"^[a-z0-9\-]+$", key), f"Expected kebab-case only, got: {key!r}"


def test_make_insight_with_data():
    event = make_event(step=3, data={"action": "buy", "qty": 10})
    text = _make_insight(event)
    assert "3" in text
    assert "buy" in text


def test_make_insight_empty_data():
    event = make_event(data={})
    text = _make_insight(event)
    assert "no payload" in text.lower() or "tool_step" in text.lower()


def test_event_to_jsonl_entry_schema():
    event = make_event(event_type=EventType.TASK_END)
    entry = _event_to_jsonl_entry(event)
    assert set(entry.keys()) == {"ts", "type", "key", "insight", "confidence", "source"}
    assert entry["type"] == "pattern"
    assert entry["confidence"] == 7
    assert entry["source"] == "test_agent"


def test_event_to_jsonl_entry_optimization():
    event = make_event(event_type=EventType.OPTIMIZATION_STEP)
    entry = _event_to_jsonl_entry(event)
    assert entry["type"] == "architecture"
    assert entry["confidence"] == 7


def test_event_to_jsonl_entry_task_start():
    event = make_event(event_type=EventType.TASK_START)
    entry = _event_to_jsonl_entry(event)
    assert entry["type"] == "observation"
    assert entry["confidence"] == 5


def test_entry_to_insight_high_confidence():
    entry = {
        "ts": datetime.now().isoformat(),
        "type": "pattern",
        "key": "agent-task-end",
        "insight": "Task completed successfully.",
        "confidence": 8,
        "source": "test_agent",
    }
    insight = _entry_to_insight(entry, "ev_001")
    assert insight.importance == Importance.HIGH
    assert insight.learning_type == "pattern"
    assert insight.confidence == 8
    assert insight.source_event_id == "ev_001"
    assert "pattern" in insight.tags


def test_entry_to_insight_low_confidence():
    entry = {
        "ts": datetime.now().isoformat(),
        "type": "observation",
        "key": "agent-task-start",
        "insight": "Task started.",
        "confidence": 3,
        "source": "agent",
    }
    insight = _entry_to_insight(entry, "ev_002")
    assert insight.importance == Importance.LOW


# ---------------------------------------------------------------------------
# Unit tests: HeartbeatCombinedMemory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_combined_memory_add_and_flush(tmp_path):
    jsonl_file = tmp_path / "learnings.jsonl"
    mem = HeartbeatCombinedMemory(jsonl_path=jsonl_file)

    event = make_event()
    await mem.add_event(event)
    assert mem.size() == 1
    assert len(mem._pending) == 1

    await mem.check_and_process_memory()

    assert len(mem._pending) == 0
    assert len(mem.insights) == 1
    assert len(mem.summaries) == 1
    assert jsonl_file.exists()

    lines = [ln for ln in jsonl_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert {"ts", "type", "key", "insight", "confidence", "source"}.issubset(parsed.keys())


@pytest.mark.asyncio
async def test_combined_memory_multiple_event_types(tmp_path):
    mem = HeartbeatCombinedMemory(jsonl_path=tmp_path / "learnings.jsonl")
    for i in range(4):
        await mem.add_event(make_event(step=i, event_type=EventType.TOOL_STEP))
    await mem.add_event(make_event(step=4, event_type=EventType.TASK_END))
    await mem.check_and_process_memory()

    assert len(mem.insights) == 5
    learning_types = {i.learning_type for i in mem.insights}
    assert "observation" in learning_types
    assert "pattern" in learning_types


@pytest.mark.asyncio
async def test_combined_memory_no_duplicate_flush():
    mem = HeartbeatCombinedMemory()
    await mem.add_event(make_event())
    await mem.check_and_process_memory()
    count_after_first = len(mem.insights)

    # Second flush with no new events should be a no-op
    await mem.check_and_process_memory()
    assert len(mem.insights) == count_after_first


@pytest.mark.asyncio
async def test_combined_memory_clear():
    mem = HeartbeatCombinedMemory()
    await mem.add_event(make_event())
    await mem.check_and_process_memory()
    mem.clear()
    assert mem.size() == 0
    assert len(mem.insights) == 0
    assert len(mem.summaries) == 0
    assert len(mem._pending) == 0


@pytest.mark.asyncio
async def test_combined_memory_get_with_limit(tmp_path):
    mem = HeartbeatCombinedMemory(jsonl_path=tmp_path / "learnings.jsonl")
    for i in range(10):
        await mem.add_event(make_event(step=i))
    await mem.check_and_process_memory()

    events = await mem.get_event(n=3)
    assert len(events) == 3

    insights = await mem.get_insight(n=2)
    assert len(insights) == 2

    summaries = await mem.get_summary(n=1)
    assert len(summaries) == 1


@pytest.mark.asyncio
async def test_combined_memory_max_insights_cap():
    mem = HeartbeatCombinedMemory(max_insights=3)
    for i in range(10):
        await mem.add_event(make_event(step=i))
    await mem.check_and_process_memory()
    assert len(mem.insights) <= 3


@pytest.mark.asyncio
async def test_combined_memory_jsonl_not_written_without_path():
    mem = HeartbeatCombinedMemory(jsonl_path=None)
    await mem.add_event(make_event())
    await mem.check_and_process_memory()
    # Should still update in-memory structures
    assert len(mem.insights) == 1


# ---------------------------------------------------------------------------
# Integration tests: HeartbeatMemorySystem lifecycle
# ---------------------------------------------------------------------------

async def _wait_for_flush(system: HeartbeatMemorySystem, session_id: str) -> None:
    """Await the pending background flush task for a session, if any."""
    task = system._pending_process_tasks.get(session_id)
    if task:
        await task


@pytest.mark.asyncio
async def test_system_start_and_end_session(tmp_path):
    system = HeartbeatMemorySystem(base_dir=str(tmp_path))
    ctx = SessionContext()

    session_id = await system.start_session(ctx=ctx)
    assert session_id == ctx.id
    assert ctx.id in system._session_memory_cache

    await system.end_session(ctx=ctx)
    assert ctx.id not in system._session_memory_cache


@pytest.mark.asyncio
async def test_system_add_events_and_retrieve(tmp_path):
    system = HeartbeatMemorySystem(base_dir=str(tmp_path))
    ctx = SessionContext()
    await system.start_session(ctx=ctx)

    for i in range(3):
        await system.add_event(
            step_number=i,
            event_type=EventType.TOOL_STEP,
            data={"step": i},
            agent_name="agent_x",
            ctx=ctx,
        )
    await system.add_event(
        step_number=3,
        event_type=EventType.TASK_END,
        data={"result": "ok"},
        agent_name="agent_x",
        ctx=ctx,
    )

    await _wait_for_flush(system, ctx.id)

    events = await system.get_event(ctx=ctx)
    assert len(events) == 4

    insights = await system.get_insight(ctx=ctx)
    assert len(insights) >= 1
    assert any(i.learning_type == "pattern" for i in insights)

    await system.end_session(ctx=ctx)


@pytest.mark.asyncio
async def test_system_jsonl_written_to_disk(tmp_path):
    system = HeartbeatMemorySystem(base_dir=str(tmp_path))
    ctx = SessionContext()
    await system.start_session(ctx=ctx)

    await system.add_event(
        step_number=0,
        event_type=EventType.TASK_END,
        data={"result": "done"},
        agent_name="my_agent",
        ctx=ctx,
    )

    await _wait_for_flush(system, ctx.id)
    await system.end_session(ctx=ctx)

    jsonl_path = tmp_path / ".heartbeat" / "memory" / "learnings.jsonl"
    assert jsonl_path.exists(), "learnings.jsonl should have been created"

    lines = [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 1
    entry = json.loads(lines[0])
    assert entry["type"] == "pattern"
    assert entry["source"] == "my_agent"
    assert entry["confidence"] == 7


@pytest.mark.asyncio
async def test_system_event_type_string_coercion(tmp_path):
    system = HeartbeatMemorySystem(base_dir=str(tmp_path))
    ctx = SessionContext()
    await system.start_session(ctx=ctx)

    await system.add_event(
        step_number=0,
        event_type="tool_step",
        data={},
        agent_name="agent",
        ctx=ctx,
    )
    await _wait_for_flush(system, ctx.id)
    events = await system.get_event(ctx=ctx)
    assert len(events) == 1
    assert events[0].event_type == EventType.TOOL_STEP

    await system.end_session(ctx=ctx)


@pytest.mark.asyncio
async def test_system_save_and_load_json(tmp_path):
    system = HeartbeatMemorySystem(base_dir=str(tmp_path))
    ctx = SessionContext()
    await system.start_session(ctx=ctx)

    await system.add_event(
        step_number=0,
        event_type=EventType.TOOL_STEP,
        data={"x": 1},
        agent_name="agent_a",
        ctx=ctx,
    )
    await _wait_for_flush(system, ctx.id)

    save_path = str(tmp_path / "memory_system.json")
    await system.save_to_json(save_path)

    system2 = HeartbeatMemorySystem(base_dir=str(tmp_path))
    loaded = await system2.load_from_json(save_path)
    assert loaded is True
    assert ctx.id in system2._session_memory_cache
    restored_events = await system2.get_event(ctx=ctx)
    assert len(restored_events) == 1

    await system.end_session(ctx=ctx)


@pytest.mark.asyncio
async def test_system_clear_session(tmp_path):
    system = HeartbeatMemorySystem(base_dir=str(tmp_path))
    ctx = SessionContext()
    await system.start_session(ctx=ctx)

    await system.add_event(
        step_number=0,
        event_type=EventType.TOOL_STEP,
        data={},
        agent_name="a",
        ctx=ctx,
    )
    await _wait_for_flush(system, ctx.id)
    await system.clear_session(ctx=ctx)
    assert ctx.id not in system._session_memory_cache


@pytest.mark.asyncio
async def test_system_no_ctx_is_safe(tmp_path):
    system = HeartbeatMemorySystem(base_dir=str(tmp_path))
    # None ctx should not raise
    await system.add_event(step_number=0, event_type=EventType.TOOL_STEP, data={}, agent_name="a", ctx=None)
    await system.end_session(ctx=None)
    await system.clear_session(ctx=None)
    assert await system.get_event(ctx=None) == []
    assert await system.get_summary(ctx=None) == []
    assert await system.get_insight(ctx=None) == []


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
