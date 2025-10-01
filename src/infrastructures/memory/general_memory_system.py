"""
Memory system that combines different types of memory for comprehensive agent memory management.
Architecture:
- MemorySystem: Overall external interface
- SessionMemory: Manages multiple sessions, each with:
  - CombinedMemory: Combines summary and insight extraction
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import json
from datetime import datetime

from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field

from src.logger import logger
from src.infrastructures.memory.types import ChatEvent, Summary, Insight, EventType, Importance, SessionInfo
from src.infrastructures.models import model_manager
from src.utils import dedent
from src.infrastructures.registry import MEMORY_SYSTEM

class CombinedMemoryOutput(BaseModel):
    """Structured output for combined summary and insight generation"""
    summaries: List[Summary] = Field(description="List of summary points")
    insights: List[Insight] = Field(description="List of insights extracted from the conversation")

class ProcessDecision(BaseModel):
    should_process: bool = Field(description="Whether to process the memory")
    reason: str = Field(description="Reason for the decision")

class CombinedMemory:
    """Combined memory that handles both summaries and insights using structured output"""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1", 
                 max_summaries: int = 20,
                 max_insights: int = 100, 
                 ):
        
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
        
        self.llm = None
        self.events: List[ChatEvent] = []
        # Store the candidate chat history that not been processed yet
        self.candidate_chat_history = ChatMessageHistory()
        self.summaries: List[Summary] = []
        self.insights: List[Insight] = []
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM for combined memory processing"""
        try:
            self.llm = model_manager.get(self.model_name)
        except Exception as e:
            logger.warning(f"Could not initialize LLM for combined memory: {e}")
            self.llm = None
    
    async def add_event(self, event: Union[ChatEvent, List[ChatEvent]]):
        """Process conversation events and extract both summaries and insights"""
        if not self.llm:
            return
        
        # Add events to chat history
        if isinstance(event, ChatEvent):
            events = [event]
        else:
            events = event
            
        for event in events:
            self.events.append(event)
            if event.event_type == EventType.ACTION_STEP or event.event_type == EventType.TASK_END:
                content = str(event)
                if event.agent_name:
                    # AI output
                    self.candidate_chat_history.add_ai_message(content)
                else:
                    # User input
                    self.candidate_chat_history.add_user_message(content)
        
        # Let LLM decide if we need to process and generate summaries/insights
        should_process = await self._check_should_process_memory()
        if should_process:
            await self._process_memory()
            
    async def _get_new_lines_text(self) -> str:
        """Get new lines from chat history"""
        new_lines = []
        for msg in self.candidate_chat_history.messages:
            if isinstance(msg, HumanMessage):
                new_lines.append(
                dedent(f"""
                <human>
                {msg.content}
                </human>
                """)
                )
            elif isinstance(msg, AIMessage):
                new_lines.append(
                dedent(f"""
                <ai>
                {msg.content}
                </ai>
                """)
                )
        new_lines_text = chr(10).join(new_lines)
        
        return new_lines_text
    
    async def _get_current_memory_text(self) -> str:
        """Get current memory text"""
        current_memory = dedent(f"""<summaries>
            {chr(10).join([str(summary) for summary in self.summaries])}
            </summaries>
            <insights>
            {chr(10).join([str(insight) for insight in self.insights])}
            </insights>""")
        return current_memory

    async def _check_should_process_memory(self) -> bool:
        """Check if we should process memory based on conversation content"""
        if not self.llm or len(self.candidate_chat_history.messages) <= 3: # If there are fewer than 3 events, do not process the memory.
            return False
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        # Create decision prompt
        decision_prompt = dedent(f"""You are analyzing a conversation to decide whether to process it and generate summaries and insights.

        Current conversation has {self.size()} events.

        Decision criteria:
        1. If there are fewer than 3 events, do not process the memory.
        2. If the conversation is repetitive or doesn't add new information, do not process the memory.
        3. If there are significant new insights, decisions, or learnings, process the memory.
        4. If the conversation is getting long (more than 5 events), process the memory.

        Current memory:
        {current_memory}

        New conversation events:
        {new_lines}

        Decide if you should process the memory.""")
                
        try:
            # Use structured LLM for reliable JSON output
            structured_llm = self.llm.with_structured_output(ProcessDecision)
            decision_response = await structured_llm.ainvoke(decision_prompt)
            logger.info(f"| Memory processing decision: {decision_response.should_process} - {decision_response.reason}")
            return decision_response.should_process
                
        except Exception as e:
            logger.warning(f"Failed to check if should process memory: {e}")
            return False
    
    async def _process_memory(self):
        """Process memory and generate summaries and insights"""
        if not self.llm or not self.candidate_chat_history.messages:
            return
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        # Create processing prompt using the combined template
        prompt = dedent(f"""Analyze the conversation events and extract both summaries and insights.

        <intro>
        For summaries, focus on:
        1. Key decisions and actions taken
        2. Important information exchanged
        3. Task progress and outcomes

        For insights, look for:
        1. Successful strategies and patterns
        2. Mistakes or failures to avoid
        3. Key learnings and realizations
        4. Actionable insights that could help improve future performance

        Avoid repeating information already in the summaries or insights.
        If there is nothing new, do not add a new entry.
        </intro>

        <output_format>
        You must respond with a valid JSON object containing both "summaries" and "insights" arrays.
        - "summaries": array of objects, each with:
            - "id": string (the unique identifier for the summary)
            - "importance": string ("high", "medium", or "low")
            - "content": string (the summary content)
        - "insights": array of objects, each with:
            - "id": string (the unique identifier for the insight)
            - "importance": string ("high", "medium", or "low")
            - "content": string (the insight text)
            - "source_event_id": string (the ID of the event that generated this insight)
            - "tags": array of strings (categorization tags)
        </output_format>

        Current memory:
        {current_memory}

        New conversation events:
        {new_lines}

        Based on the current memory and new conversation events, generate new summaries and insights.
        """)
        
        try:
            structured_llm = self.llm.with_structured_output(CombinedMemoryOutput)
            response = await structured_llm.ainvoke(prompt)
            
            new_summaries = response.summaries
            new_insights = response.insights
            
            # Update summaries and insights
            self.summaries.extend(new_summaries)
            self.insights.extend(new_insights)
            
            # Sort and limit summaries and insights
            await self._sort_and_limit_summaries()
            await self._sort_and_limit_insights()
            
            # Clear candidate chat history
            self.candidate_chat_history.clear()
            
        except Exception as e:
            logger.warning(f"Failed to process memory: {e}")
    
    async def _sort_and_limit_insights(self):
        """Sort insights by importance and limit count"""
        # Sort by importance: high > medium > low
        importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.insights.sort(key=lambda x: importance_order[x.importance])
        
        # Limit count
        if len(self.insights) > self.max_insights:
            self.insights = self.insights[:self.max_insights]

    async def _sort_and_limit_summaries(self):
        """Sort summaries by importance and limit count"""
        importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.summaries.sort(key=lambda x: importance_order[x.importance])
        
        # Limit count
        if len(self.summaries) > self.max_summaries:
            self.summaries = self.summaries[:self.max_summaries]
    
    def clear(self):
        """Clear all memory"""
        self.events.clear()
        self.candidate_chat_history.clear()
        self.summaries.clear()
        self.insights.clear()
    
    def size(self) -> int:
        """Return current event count"""
        return len(self.events)
    
    async def get_event(self, n: Optional[int] = None) -> List[ChatEvent]:
        if n is None:
            return self.events
        
        return self.events[-n:] if len(self.events) > n else self.events
    
    async def get_summary(self, n: Optional[int] = None) -> List[Summary]:
        if n is None:
            return self.summaries
        return self.summaries[-n:] if len(self.summaries) > n else self.summaries
    
    async def get_insight(self, n: Optional[int] = None) -> List[Insight]:
        if n is None:
            return self.insights
        return self.insights[-n:] if len(self.insights) > n else self.insights

@MEMORY_SYSTEM.register_module(name="general_memory_system", force=True)
class GeneralMemorySystem:
    """Memory system that combines different types of memory for comprehensive agent memory management."""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1",
                 max_summaries: int = 20,
                 max_insights: int = 100
                 ):
        
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
    
        self.session_memory: Dict[str, CombinedMemory] = {}
        self.session_info: Dict[str, SessionInfo] = {}
        self.current_session_id: Optional[str] = None
        
    async def _check_session_id(self, session_id: Optional[str] = None):
        if session_id is None:
            session_id = self.current_session_id
        return session_id

    async def start_session(self, 
                            session_id: str, 
                            agent_name: Optional[str] = None, 
                            task_id: Optional[str] = None, 
                            description: Optional[str] = None
                            ) -> str:
        """Start new session with MemorySystem"""
        session_info = SessionInfo(
            session_id=session_id,
            agent_name=agent_name,
            task_id=task_id,
            description=description
        )
        self.session_info[session_id] = session_info
        self.current_session_id = session_id
        
        # Initialize CombinedMemory for this session
        self.session_memory[session_id] = CombinedMemory(
            model_name=self.model_name, 
            max_summaries=self.max_summaries,
            max_insights=self.max_insights
        )
        
        return session_id
    
    async def end_session(self, session_id: Optional[str] = None):
        """End session"""
        session_id = await self._check_session_id(session_id)
            
        if session_id and session_id in self.session_info:
            self.session_info[session_id].end_time = datetime.now()
            
            if session_id == self.current_session_id:
                self.current_session_id = None
    
    async def add_event(self,
                        step_number: int,
                        event_type,
                        data: Any,
                        agent_name: str,
                        task_id: Optional[str] = None,
                        ):
        event_id = "event_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        session_id = self.current_session_id
        
        event = ChatEvent(
            id=event_id,
            step_number=step_number,
            event_type=event_type,
            data=data,
            agent_name=agent_name,
            task_id=task_id,
            session_id=session_id
        )
    
        if session_id and session_id in self.session_memory:
            await self.session_memory[session_id].add_event(event)
    
    async def get_session_info(self, session_id: Optional[str] = None) -> Optional[SessionInfo]:
        session_id = await self._check_session_id(session_id)
            
        """Get session info"""
        return self.session_info.get(session_id)
    
    async def clear_session(self, session_id: Optional[str] = None):
        session_id = await self._check_session_id(session_id)
            
        """Clear specific session"""
        if session_id in self.session_info:
            del self.session_info[session_id]
            
            # Clear CombinedMemory
            if session_id in self.session_memory:
                await self.session_memory[session_id].clear()
                del self.session_memory[session_id]
            
            if session_id == self.current_session_id:
                self.current_session_id = None
                
    async def clear(self):
        """Clear all sessions"""
        for session_id in list(self.session_info.keys()):
            await self.clear_session(session_id)
            
    async def get_event(self, n: Optional[int] = None) -> List[ChatEvent]:
        session_id = self.current_session_id
        return await self.session_memory[session_id].get_event(n=n)
    
    async def get_summary(self, n: Optional[int] = None) -> List[Summary]:
        session_id = self.current_session_id
        return await self.session_memory[session_id].get_summary(n=n)
    
    async def get_insight(self, n: Optional[int] = None) -> List[Insight]:
        session_id = self.current_session_id
        return await self.session_memory[session_id].get_insight(n=n)