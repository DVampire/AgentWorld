"""Memory types for the memory module."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class EventType(Enum):
    """Types of events in chat history."""
    HUMAN_MESSAGE = "human_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class ChatEvent:
    """A single event in the chat history."""
    id: str
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None


@dataclass
class Session:
    """A session containing a flat list of events (optionally tied to tasks)."""
    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    events: List[ChatEvent] = field(default_factory=list)
    
    def add_event(self, event: ChatEvent):
        self.events.append(event)
    
    def get_events(
        self,
        task_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[ChatEvent]:
        results: List[ChatEvent] = self.events
        if task_id is not None:
            results = [e for e in results if e.task_id == task_id]
        if event_types:
            results = [e for e in results if e.type in event_types]
        return sorted(results, key=lambda e: e.timestamp)
    
    def get_stats(self) -> Dict[str, Any]:
        total_events = len(self.events)
        unique_task_ids = {e.task_id for e in self.events if e.task_id}
        event_type_counts: Dict[str, int] = {}
        for e in self.events:
            key = e.type.value
            event_type_counts[key] = event_type_counts.get(key, 0) + 1
        return {
            "session_id": self.id,
            "session_name": self.name,
            "total_events": total_events,
            "total_tasks": len(unique_task_ids),
            "events_by_type": event_type_counts,
            "created_at": self.created_at.isoformat()
        }
