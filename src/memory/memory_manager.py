"""Memory manager for session-based memory operations."""
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.logger import logger
from src.memory.memory_store import MemoryStore

class MemoryManager:
    """Session-based memory manager for agent execution history."""
    def __init__(self):
        self.memory_store = MemoryStore()
    
    def start_session(self, session_id: str, description: str):
        self.memory_store.start_session(session_id, description)
    
    def end_session(self):
        self.memory_store.end_session()
    
    def add_event(self, 
                  step_number: int,
                  event_type: str,
                  data: Any,
                  agent_name: str,
                  task_id: Optional[str] = None,
                  extra: Optional[Dict[str, Any]] = None,
                  **kwargs,
                  ):
        
        self.memory_store.add_event(step_number,
                                    event_type, 
                                    data, 
                                    agent_name,
                                    task_id,
                                    extra,
                                    **kwargs)
    
    def get_events(self, num: int = 5):
        return self.memory_store.get_events(num)