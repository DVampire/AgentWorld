"""Memory store for session-based storage."""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from .memory_types import ChatEvent, EventType, Session


class MemoryStore:
    """Base memory store interface."""
    
    def get_current_session(self) -> Optional[Session]:
        """Get the current session."""
        raise NotImplementedError
    
    def create_session(self, session_id: str, session_name: str) -> Session:
        """Create a new session."""
        raise NotImplementedError
    
    def add_event(self, event: ChatEvent) -> str:
        """Add an event to the current session (flat event list)."""
        raise NotImplementedError
    
    def get_events(
        self,
        task_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[ChatEvent]:
        """Get events from the current session."""
        raise NotImplementedError
    
    def clear_session(self) -> bool:
        """Clear the current session."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """In-memory storage for sessions."""
    
    def __init__(self):
        self.current_session: Optional[Session] = None
    
    def get_current_session(self) -> Optional[Session]:
        """Get the current session."""
        return self.current_session
    
    def create_session(self, session_id: str, session_name: str) -> Session:
        """Create a new session."""
        self.current_session = Session(id=session_id, name=session_name)
        return self.current_session
    
    def add_event(self, event: ChatEvent) -> str:
        """Add an event to the current session (flat event list)."""
        if not self.current_session:
            raise ValueError("No current session. Call create_session() first.")
        # Ensure event has session_id
        if not event.session_id:
            event.session_id = self.current_session.id
        # Append to session events
        self.current_session.add_event(event)
        return event.id
    
    def get_events(
        self,
        task_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[ChatEvent]:
        """Get events from the current session."""
        if not self.current_session:
            return []
        
        # Flat session events; optional filter by task_id
        return self.current_session.get_events(task_id=task_id, event_types=event_types)
    
    def clear_session(self) -> bool:
        """Clear the current session."""
        self.current_session = None
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        if not self.current_session:
            return {
                "current_session": None,
                "total_events": 0,
                "total_tasks": 0
            }
        
        session_stats = self.current_session.get_stats()
        return {
            "current_session": session_stats,
            "total_events": session_stats["total_events"],
            "total_tasks": session_stats["total_tasks"]
        }


class FileMemoryStore(MemoryStore):
    """File-based storage for sessions."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.current_session: Optional[Session] = None
        self._load_session()
    
    def _load_session(self):
        """Load session from file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('current_session'):
                        session_data = data['current_session']
                        self.current_session = Session(
                            id=session_data['id'],
                            name=session_data['name'],
                            created_at=datetime.fromisoformat(session_data['created_at'])
                        )
                        
                        # Load flat events
                        for event_data in session_data.get('events', []):
                            event = ChatEvent(
                                id=event_data['id'],
                                type=EventType(event_data['type']),
                                timestamp=datetime.fromisoformat(event_data['timestamp']),
                                content=event_data['content'],
                                metadata=event_data.get('metadata', {}),
                                agent_name=event_data.get('agent_name'),
                                session_id=event_data.get('session_id'),
                                task_id=event_data.get('task_id')
                            )
                            self.current_session.add_event(event)
            except Exception as e:
                print(f"Warning: Failed to load session: {e}")
    
    def _save_session(self):
        """Save session to file."""
        try:
            data = {
                'current_session': None,
                'last_updated': datetime.now().isoformat()
            }
            
            if self.current_session:
                session_data = {
                    'id': self.current_session.id,
                    'name': self.current_session.name,
                    'created_at': self.current_session.created_at.isoformat(),
                    'events': []
                }
                # Save flat events
                for event in self.current_session.events:
                    event_data = {
                        'id': event.id,
                        'type': event.type.value,
                        'timestamp': event.timestamp.isoformat(),
                        'content': event.content,
                        'metadata': event.metadata,
                        'agent_name': event.agent_name,
                        'session_id': event.session_id,
                        'task_id': event.task_id
                    }
                    session_data['events'].append(event_data)
                data['current_session'] = session_data
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Warning: Failed to save session: {e}")
    
    def get_current_session(self) -> Optional[Session]:
        """Get the current session."""
        return self.current_session
    
    def create_session(self, session_id: str, session_name: str) -> Session:
        """Create a new session."""
        self.current_session = Session(id=session_id, name=session_name)
        self._save_session()
        return self.current_session
    
    def add_event(self, event: ChatEvent) -> str:
        """Add an event to the current session (flat event list)."""
        if not self.current_session:
            raise ValueError("No current session. Call create_session() first.")
        # Ensure event has session_id
        if not event.session_id:
            event.session_id = self.current_session.id
        self.current_session.add_event(event)
        self._save_session()
        return event.id
    
    def get_events(
        self,
        task_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[ChatEvent]:
        """Get events from the current session."""
        if not self.current_session:
            return []
        
        # Flat session events; optional filter by task_id
        return self.current_session.get_events(task_id=task_id, event_types=event_types)
    
    def clear_session(self) -> bool:
        """Clear the current session."""
        self.current_session = None
        self._save_session()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        if not self.current_session:
            return {
                "current_session": None,
                "total_events": 0,
                "total_tasks": 0,
                "file_path": self.file_path
            }
        
        session_stats = self.current_session.get_stats()
        return {
            "current_session": session_stats,
            "total_events": session_stats["total_events"],
            "total_tasks": session_stats["total_tasks"],
            "file_path": self.file_path
        }
