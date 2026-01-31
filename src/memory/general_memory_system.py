"""
Memory system that combines different types of memory for comprehensive agent memory management.
Architecture:
- MemorySystem: Overall external interface
- SessionMemory: Manages multiple sessions, each with:
  - CombinedMemory: Combines summary and insight extraction
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import json
import os
import asyncio

from src.logger import logger
from src.model import model_manager
from src.utils import dedent, generate_unique_id
from src.message.types import HumanMessage, AssistantMessage, Message, SystemMessage
from src.memory.types import ChatEvent, Summary, Insight, EventType, Importance, Memory, MemoryContext
from src.utils import file_lock
from src.registry import MEMORY_SYSTEM

class CombinedMemoryOutput(BaseModel):
    """Structured output for combined summary and insight generation"""
    summaries: List[Summary] = Field(description="List of summary points")
    insights: List[Insight] = Field(description="List of insights extracted from the conversation")

class ProcessDecision(BaseModel):
    should_process: bool = Field(description="Whether to process the memory")
    reasoning: str = Field(description="Reasoning for the decision")

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
        
        self.events: List[ChatEvent] = []
        # Store the candidate chat history that not been processed yet
        self.candidate_chat_history: List[Message] = []
        self.summaries: List[Summary] = []
        self.insights: List[Insight] = []
    
    async def add_event(self, event: Union[ChatEvent, List[ChatEvent]]):
        """Process conversation events and extract both summaries and insights"""
        # Add events to chat history
        if isinstance(event, ChatEvent):
            events = [event]
        else:
            events = event
            
        for event in events:
            self.events.append(event)
            if event.event_type == EventType.TOOL_STEP or event.event_type == EventType.TASK_END:
                content = str(event)
                if event.agent_name:
                    # AI output
                    self.candidate_chat_history.append(AssistantMessage(content=content))
                else:
                    # User input
                    self.candidate_chat_history.append(HumanMessage(content=content))
        
        # Let LLM decide if we need to process and generate summaries/insights
        should_process = await self._check_should_process_memory()
        if should_process:
            await self._process_memory()
            
    async def _get_new_lines_text(self) -> str:
        """Get new lines from chat history"""
        new_lines = []
        for msg in self.candidate_chat_history:
            if isinstance(msg, HumanMessage):
                new_lines.append(
                dedent(f"""
                <human>
                {msg.content}
                </human>
                """)
                )
            elif isinstance(msg, AssistantMessage):
                new_lines.append(
                dedent(f"""
                <assistant>
                {msg.content}
                </assistant>
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
        if len(self.candidate_chat_history) <= 3:  # If there are fewer than 3 events, do not process the memory.
            return False
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        # Create decision prompt
        decision_prompt = dedent(f"""You are analyzing a conversation to decide whether to process it and generate summaries and insights.
        Current conversation has {self.size()} events.

        Decision criteria:
        1. If there are fewer than 5 events, do not process the memory.
        2. If the conversation is repetitive or doesn't add new information, do not process the memory.
        3. If there are significant new insights, decisions, or learnings, process the memory.
        4. If the conversation is getting long (more than 10 events), process the memory.

        Current memory:
        {current_memory}

        New conversation events:
        {new_lines}

        Decide if you should process the memory.""")
                
        try:
            # Build messages
            messages = [
                SystemMessage(content="You are a memory processing decision system. Always respond with valid JSON."),
                HumanMessage(content=decision_prompt)
            ]
            
            # Call model manager with BaseModel response_format
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ProcessDecision
            )
            if not response.extra or not response.extra.parsed_model:
                logger.warning("Response does not contain parsed_model")
                return False
            processed_decision_response = response.extra.parsed_model
            should_process = processed_decision_response.should_process
            reasoning = processed_decision_response.reasoning
            
            logger.info(f"| Memory processing decision: {should_process} - {reasoning}")
            return should_process
                
        except Exception as e:
            logger.warning(f"Failed to check if should process memory: {e}")
            return False
    
    async def _process_memory(self):
        """Process memory and generate summaries and insights"""
        if not self.candidate_chat_history:
            return
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        # Create processing prompt using the combined template
        prompt = dedent(f"""Analyze the conversation events and extract both summaries and insights.
        <intro>
        For summaries, focus on:
        1. Key decisions and tools taken
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
            # Build messages
            messages = [
                SystemMessage(content="You are a memory processing system. Always respond with valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            # Call model manager with BaseModel response_format
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=CombinedMemoryOutput
            )
            
            # Check if response was successful and contains parsed model
            if not response.success:
                raise ValueError(f"Model call failed: {response.message}")
            
            if not response.extra or not response.extra.parsed_model:
                raise ValueError(f"Response does not contain parsed_model. Response: {response.message}")
            
            combined_memory_output_response = response.extra.parsed_model
            
            new_summaries = combined_memory_output_response.summaries
            new_insights = combined_memory_output_response.insights
            
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

@MEMORY_SYSTEM.register_module(force=True)
class GeneralMemorySystem(Memory):
    """Memory system that combines different types of memory for comprehensive agent memory management."""
    
    require_grad: bool = Field(default=False, description="Whether the memory system requires gradients")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 model_name: str = "gpt-4.1",
                 max_summaries: int = 10,
                 max_insights: int = 10,
                 require_grad: bool = False,
                 **kwargs
                 ):
        super().__init__(require_grad=require_grad, **kwargs)
        
        if base_dir is not None:
            self.base_dir = base_dir
        
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| General memory system base directory: {self.base_dir}")
        self.save_path = os.path.join(self.base_dir, "memory_system.json")
            
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
    
        # Per-session cache and locks for concurrent safety
        # Key: session_id (str), Value: CombinedMemory instance
        self._session_memory_cache: Dict[str, CombinedMemory] = {}
        # Key: session_id (str), Value: asyncio.Lock for that session
        self._session_locks: Dict[str, asyncio.Lock] = {}
        # Lock for managing the cache dictionaries themselves
        self._cache_lock = asyncio.Lock()

    async def _get_or_create_session_memory(self, id: str) -> tuple[CombinedMemory, asyncio.Lock]:
        """Get or create a CombinedMemory instance for the given id with proper locking.
        
        Args:
            id: The unique identifier for the session
            
        Returns:
            tuple[CombinedMemory, asyncio.Lock]: The session memory instance and its lock
        """
        async with self._cache_lock:
            # Get or create lock for this session
            if id not in self._session_locks:
                self._session_locks[id] = asyncio.Lock()
            
            # Get or create session memory for this session
            if id not in self._session_memory_cache:
                self._session_memory_cache[id] = CombinedMemory(
                    model_name=self.model_name,
                    max_summaries=self.max_summaries,
                    max_insights=self.max_insights
                )
                logger.info(f"| 📝 Created new session memory cache for id: {id}")
            else:
                logger.debug(f"| 📂 Using existing session memory cache for id: {id}")
            
            return self._session_memory_cache[id], self._session_locks[id]

    async def _cleanup_session_memory(self, id: str):
        """Remove session memory from cache."""
        async with self._cache_lock:
            if id in self._session_memory_cache:
                del self._session_memory_cache[id]
                logger.info(f"| 🧹 Removed session memory from cache: {id}")
            if id in self._session_locks:
                del self._session_locks[id]

    async def start_session(self,
                            agent_name: Optional[str] = None, 
                            task_id: Optional[str] = None, 
                            description: Optional[str] = None,
                            ctx: MemoryContext = None, 
                            **kwargs
                            ) -> str:
        """Start new session with MemorySystem. Automatically loads from JSON if file exists."""
        # Auto-load from JSON if file exists and save_path is set
        if self.save_path and os.path.exists(self.save_path):
            logger.info(f"| 📂 Loading memory from JSON: {self.save_path}")
            await self.load_from_json(self.save_path)
            logger.info(f"| ✅ Memory loaded from JSON")
            
        if ctx is None:
            ctx = MemoryContext()
        id = ctx.id
        
        # Initialize CombinedMemory for this session (with proper locking)
        await self._get_or_create_session_memory(id)
        
        return id
    
    async def end_session(self, ctx: MemoryContext = None, **kwargs):
        """End session. Automatically saves to JSON if save_path is set."""
        id = ctx.id
        
        # Save to JSON if save_path is set
        if self.save_path:
            await self.save_to_json(self.save_path)
        
        # Cleanup session memory
        await self._cleanup_session_memory(id)
    
    async def add_event(self,
                        step_number: int,
                        event_type,
                        data: Any,
                        agent_name: str,
                        task_id: Optional[str] = None,
                        ctx: MemoryContext = None,
                        **kwargs):
        """Add event to memory system.
        
        Args:
            step_number: Step number
            event_type: Event type (EventType enum)
            data: Event data
            agent_name: Agent name
            task_id: Optional task ID
            ctx: Memory context
        """
        id = ctx.id
        
        # Ensure event_type is EventType enum
        if not isinstance(event_type, EventType):
            # Try to convert string to EventType
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    logger.warning(f"| ⚠️ Invalid event_type '{event_type}', defaulting to TOOL_STEP")
                    event_type = EventType.TOOL_STEP
            else:
                logger.warning(f"| ⚠️ Invalid event_type type '{type(event_type)}', defaulting to TOOL_STEP")
                event_type = EventType.TOOL_STEP
        
        event_id = generate_unique_id(prefix="event")
        
        event = ChatEvent(
            id=event_id,
            step_number=step_number,
            event_type=event_type,
            data=data,
            agent_name=agent_name,
            task_id=task_id,
            session_id = id
        )
    
        # Get session memory with proper locking
        session_memory, session_lock = await self._get_or_create_session_memory(id)
        async with session_lock:
            await session_memory.add_event(event)
        
        # Auto-save to JSON if save_path is set
        if self.save_path:
            await self.save_to_json(self.save_path)
    
    async def get(self, ctx: MemoryContext = None) -> CombinedMemory:
        """Get session info"""
        id = ctx.id
        if id in self._session_memory_cache:
            return self._session_memory_cache[id]
        return None
    
    async def clear_session(self, ctx: MemoryContext = None):
        """Clear specific session"""
        id = ctx.id
        if id in self._session_memory_cache:
            await self._session_memory_cache[id].clear()
            await self._cleanup_session_memory(id)
            
    async def get_event(self, n: Optional[int] = None, ctx: MemoryContext = None, **kwargs) -> List[ChatEvent]:
        """Get events from memory system.
        
        Args:
            n: Number of events to retrieve. If None, returns all events.
            ctx: Memory context
            
        Returns:
            List of events
        """
        id = ctx.id
        if id in self._session_memory_cache:
            return await self._session_memory_cache[id].get_event(n=n)
        return []
    
    async def get_summary(self, n: Optional[int] = None, ctx: MemoryContext = None, **kwargs) -> List[Summary]:
        """Get summaries from memory system.
        
        Args:
            n: Number of summaries to retrieve. If None, returns all summaries.
            ctx: Memory context
            
        Returns:
            List of summaries
        """
        id = ctx.id
        if id in self._session_memory_cache:
            return await self._session_memory_cache[id].get_summary(n=n)
        return []
    
    async def get_insight(self, n: Optional[int] = None, ctx: MemoryContext = None, **kwargs) -> List[Insight]:
        """Get insights from memory system.
        
        Args:
            n: Number of insights to retrieve. If None, returns all insights.
            ctx: Memory context
        Returns:
            List of insights
        """
        id = ctx.id
        if id in self._session_memory_cache:
            return await self._session_memory_cache[id].get_insight(n=n)
        return []
    
    async def save_to_json(self, file_path: str) -> str:
        """Save memory system state to JSON file.
        
        Structure:
        {
            "metadata": {
                "memory_system_type": str,
                "session_ids": [str, ...]
            },
            "sessions": {
                "session_id": {
                    "session_memory": {
                        "events": [...],
                        "summaries": [...],
                        "insights": [...]
                    }
                },
                ...
            }
        }
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to the saved file
        """
        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare metadata
            metadata = {
                "memory_system_type": "general_memory_system",
                "session_ids": list(self._session_memory_cache.keys())
            }
            
            # Prepare sessions data
            sessions = {}
            async with self._cache_lock:
                for id in self._session_memory_cache.keys():
                    session_memory = self._session_memory_cache[id]
                    session_data = {
                        "session_memory": {
                            "events": [event.model_dump(mode="json") for event in session_memory.events],
                            "summaries": [summary.model_dump(mode="json") for summary in session_memory.summaries],
                            "insights": [insight.model_dump(mode="json") for insight in session_memory.insights],
                        }
                    }
                    sessions[id] = session_data
            
            # Prepare save data
            save_data = {
                "metadata": metadata,
                "sessions": sessions
            }
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.debug(f"| 💾 Memory saved to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: str) -> bool:
        """Load memory system state from JSON file.
        
        Expected format:
        {
            "metadata": {
                "memory_system_type": str,
                "session_ids": [str, ...]
            },
            "sessions": {
                "id": {
                    "session_memory": {
                        "events": [...],
                        "summaries": [...],
                        "insights": [...]
                    }
                },
                ...
            }
        }
        
        Args:
            file_path: File path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        logger.debug(f"| 🔒 Acquiring file lock for: {file_path}")
        async with file_lock(file_path):
            logger.debug(f"| 🔓 File lock acquired for: {file_path}")
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️  Memory file not found: {file_path}")
                return False
            
            try:
                logger.debug(f"| 📖 Reading JSON file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                logger.debug(f"| ✅ JSON file read successfully")
                
                # Validate format
                if "metadata" not in load_data or "sessions" not in load_data:
                    raise ValueError(
                        f"Invalid memory format. Expected {{'metadata': {{...}}, 'sessions': {{...}}}}, "
                        f"got keys: {list(load_data.keys())}"
                    )
                
                # Restore sessions
                sessions_data = load_data.get("sessions", {})
                logger.debug(f"| 📊 Restoring {len(sessions_data)} sessions from JSON")
                
                async with self._cache_lock:
                    for id, session_data in sessions_data.items():
                        logger.debug(f"| 🔄 Restoring session: {id}")
                        
                        # Ensure session memory exists (skip auto-load to avoid recursion)
                        if id not in self._session_memory_cache:
                            # Create CombinedMemory directly without calling start_session to avoid recursion
                            self._session_memory_cache[id] = CombinedMemory(
                                model_name=self.model_name, 
                                max_summaries=self.max_summaries,
                                max_insights=self.max_insights
                            )
                            self._session_locks[id] = asyncio.Lock()
                        
                        session_memory = self._session_memory_cache[id]
                        session_memory_data = session_data.get("session_memory", {})
                        
                        # Restore events
                        if "events" in session_memory_data:
                            events = []
                            for event_data in session_memory_data["events"]:
                                if event_data.get("timestamp"):
                                    event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                                if event_data.get("event_type"):
                                    event_data["event_type"] = EventType(event_data["event_type"])
                                events.append(ChatEvent(**event_data))
                            session_memory.events = events
                        
                        # Restore summaries
                        if "summaries" in session_memory_data:
                            summaries = []
                            for summary_data in session_memory_data["summaries"]:
                                if summary_data.get("importance"):
                                    summary_data["importance"] = Importance(summary_data["importance"])
                                summaries.append(Summary(**summary_data))
                            session_memory.summaries = summaries
                        
                        # Restore insights
                        if "insights" in session_memory_data:
                            insights = []
                            for insight_data in session_memory_data["insights"]:
                                if insight_data.get("importance"):
                                    insight_data["importance"] = Importance(insight_data["importance"])
                                insights.append(Insight(**insight_data))
                            session_memory.insights = insights
                
                logger.info(f"| 📂 Memory loaded from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load memory from {file_path}: {e}", exc_info=True)
                return False