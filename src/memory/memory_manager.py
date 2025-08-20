"""Memory manager for session-based memory operations."""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from .memory_store import MemoryStore, InMemoryStore, FileMemoryStore
from .memory_types import ChatEvent, EventType, Session
from src.logger import logger


class MemoryManager:
    """Session-based memory manager for agent execution history."""
    
    def __init__(self, store: Optional[MemoryStore] = None):
        """Initialize memory manager with a store."""
        self.store = store or InMemoryStore()
        self.current_session: Optional[Session] = None
        self.current_task_id: Optional[str] = None
    
    def create_session(self, session_id: str, session_name: str) -> Session:
        """Create a new session (should be called when agent starts)."""
        self.current_session = self.store.create_session(session_id, session_name)
        logger.info(f"🆕 Created session: {session_id} - {session_name}")
        return self.current_session
    
    def create_task(self, task_description: str, task_id: str) -> str:
        """Create a new task in the current session (should be called by LLM)."""
        if not self.current_session:
            raise ValueError("No current session. Call create_session() first.")
        
        # Set current task id
        self.current_task_id = task_id
        # Add the task description as a human message (flat event, with task_id/session_id)
        task_event = ChatEvent(
            id=f"human_message_{str(uuid.uuid4())}",
            type=EventType.HUMAN_MESSAGE,
            content=task_description,
            session_id=self.current_session.id if self.current_session else None,
            task_id=self.current_task_id
        )
        self.store.add_event(task_event)
        
        logger.info(f"🆕 Created task: {task_id} - {task_description}")
        return task_id
    
    def add_event(
        self,
        event_type: EventType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None
    ) -> str:
        """Add an event to the current task."""
        if not self.current_session:
            raise ValueError("No current session. Call create_session() first.")
        
        if not self.current_task_id:
            raise ValueError("No current task. Call create_task() first.")
        
        if metadata is None:
            metadata = {}
        
        # Auto-generate event ID: {event_type}_{uuid}
        event_id = f"{event_type.value}_{str(uuid.uuid4())}"
        
        event = ChatEvent(
            id=event_id,
            type=event_type,
            content=content,
            metadata=metadata,
            agent_name=agent_name,
            session_id=self.current_session.id if self.current_session else None,
            task_id=self.current_task_id
        )
        
        # Add event to the current session (flat)
        self.store.add_event(event)
        
        return event_id
    
    def get_events(
        self,
        task_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[ChatEvent]:
        """Get events from the current session."""
        return self.store.get_events(task_id=task_id, event_types=event_types)
    
    def format_full_history(
        self,
        max_events: int = 20
    ) -> str:
        """Format full history for prompts - all events in chronological order."""
        events = self.get_events()
        
        if not events:
            return ""
        
        # Limit events if specified
        if max_events:
            events = events[-max_events:]  # Get the most recent events
        
        formatted_events = []
        for event in events:
            if event.type == EventType.HUMAN_MESSAGE:
                formatted_events.append(f"Human: {event.content}")
            elif event.type == EventType.ASSISTANT_MESSAGE:
                formatted_events.append(f"Assistant: {event.content}")
            elif event.type == EventType.TOOL_CALL:
                formatted_events.append(f"Calling tool: {event.content}")
            elif event.type == EventType.TOOL_RESULT:
                formatted_events.append(event.content)
            elif event.type == EventType.ERROR:
                formatted_events.append(event.content)
            elif event.type == EventType.SYSTEM_MESSAGE:
                formatted_events.append(f"System: {event.content}")
        
        return "\n".join(formatted_events)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        if not self.current_session:
            return {
                "session_id": None,
                "session_name": None,
                "total_events": 0,
                "total_tasks": 0
            }
        
        return self.current_session.get_stats()
    
    def clear_session(self) -> bool:
        """Clear the current session (should be called when agent ends)."""
        if self.current_session:
            logger.info(f"🗑️ Clearing session: {self.current_session.id}")
        
        self.current_session = None
        self.current_task_id = None
        return self.store.clear_session()
    
    def get_current_session(self) -> Optional[Session]:
        """Get the current session."""
        return self.current_session
    
    def get_current_task_id(self) -> Optional[str]:
        """Get the current task ID."""
        return self.current_task_id


