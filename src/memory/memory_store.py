from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class EventType(Enum):
    TASK_START = "task_start"
    ACTION_STEP = "action_step"
    TASK_END = "task_end"

class ChatEvent(BaseModel):
    id: str = Field(..., description="The unique identifier for the event.")
    step_number: int = Field(..., description="The step number of the event.")
    event_type: EventType = Field(..., description="The type of the event.")
    timestamp: datetime = Field(default_factory=datetime.now, description="The timestamp of the event.")
    data: Dict[str, Any] = Field(default_factory=Dict[str, Any], description="The data of the event.")
    agent_name: Optional[str] = Field(None, description="The name of the agent that generated the event.")
    session_id: Optional[str] = Field(None, description="The session ID of the event.")
    task_id: Optional[str] = Field(None, description="The task ID of the event.")
    extra: Optional[Dict[str, Any]] = Field(None, description="The extra information of the event.")

class SessionInfo(BaseModel):
    session_id: str = Field(..., description="The unique identifier for the session.")
    description: str = Field(..., description="The description of the session.")

class Session(BaseModel):
    id: str = Field(..., description="The unique identifier for the session.")
    description: str = Field(..., description="The description of the session.")
    created_at: datetime = Field(..., description="The timestamp of the session.")
    events: List[ChatEvent] = Field(..., description="The events in the session.")
    
    def add_event(self, event: ChatEvent):
        self.events.append(event)
    
    def get_events(self, num: int = 5):
        return self.events[-min(num, len(self.events)):]
    

class MemoryStore:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: Optional[str] = None
        self.current_session: Optional[Session] = None
        
    def start_session(self, session_id: str, description: str):
        self.current_session_id = session_id
        self.current_session = Session(id=session_id, 
                                       description=description,
                                       created_at=datetime.now().isoformat(), 
                                       events=[]
                                       )
    def end_session(self):
        self.current_session_id = None
        self.current_session = None
        
    def add_event(self, 
                  step_number: int,
                  event_type: str,
                  data: Any,
                  agent_name: str,
                  task_id: Optional[str] = None,
                  extra: Optional[Dict[str, Any]] = None,
                  **kwargs,
                  ):
        
        event_id = str(uuid.uuid4())
        event_type = EventType(event_type)
        timestamp = datetime.now().isoformat()

        task_id = str(task_id)
        
        event = ChatEvent(id=event_id,
                          step_number=step_number,
                          event_type=event_type,
                          timestamp=timestamp,
                          data=data,
                          agent_name=agent_name,
                          task_id=task_id,
                          extra=extra)
        self.current_session.add_event(event)
    
    def get_events(self, num: int = 5):
        return self.current_session.get_events(num)
    