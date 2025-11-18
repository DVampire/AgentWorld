from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import json

from src.utils import dedent

class EventType(Enum):
    TASK_START = "task_start"
    ACTION_STEP = "action_step"
    TASK_END = "task_end"

class ChatEvent(BaseModel):
    id: str = Field(..., description="The unique identifier for the event.")
    step_number: int = Field(..., description="The step number of the event.")
    event_type: EventType = Field(..., description="The type of the event.")
    timestamp: datetime = Field(default_factory=datetime.now, description="The timestamp of the event.")
    data: Dict[str, Any] = Field(default_factory=dict, description="The data of the event.")
    agent_name: Optional[str] = Field(None, description="The name of the agent that generated the event.")
    session_id: Optional[str] = Field(None, description="The session ID of the event.")
    task_id: Optional[str] = Field(None, description="The task ID of the event.")
    
    def __str__(self):
        string = dedent(f"""<chat_event>
            ID: {self.id}
            Step Number: {self.step_number}
            Event Type: {self.event_type}
            Timestamp: {self.timestamp}
            Agent Name: {self.agent_name}
            Session ID: {self.session_id}
            Task ID: {self.task_id}
            Data: {json.dumps(self.data)}
            </chat_event>""")
        return string
    
    def __repr__(self):
        return self.__str__()

class SessionInfo(BaseModel):
    session_id: str = Field(..., description="The unique identifier for the session.")
    start_time: Optional[datetime] = Field(None, description="The start time of the session.")
    end_time: Optional[datetime] = Field(None, description="The end time of the session.")
    agent_name: Optional[str] = Field(None, description="The name of the agent.")
    task_id: Optional[str] = Field(None, description="The task ID.")
    description: Optional[str] = Field(None, description="The description of the session.")

class Importance(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Insight(BaseModel):
    id: str = Field(..., description="The unique identifier for the insight.")
    content: str = Field(..., description="The insight content")
    importance: Importance = Field(..., description="Importance level")
    source_event_id: Optional[str] = Field(None, description="ID of the event that generated this insight")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    def __str__(self):
        string = dedent(f"""<insight>
            ID: {self.id}
            Content: {self.content}
            Importance: {self.importance}
            Source Event ID: {self.source_event_id}
            Tags: {self.tags}
            </insight>""")
        return string
    
    def __repr__(self):
        return self.__str__()

class Summary(BaseModel):
    id: str = Field(..., description="The unique identifier for the summary.")
    importance: Importance = Field(..., description="Importance level")
    content: str = Field(..., description="The summary content")
    
    def __str__(self):
        string = dedent(f"""<summary>
            ID: {self.id}
            Importance: {self.importance}
            Content: {self.content}
            </summary>""")
        return string
    
    def __repr__(self):
        return self.__str__()