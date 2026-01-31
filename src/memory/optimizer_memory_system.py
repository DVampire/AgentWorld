"""
Memory system for recording optimizer optimization history and experiences.
Used for agent self-evolution by learning from past optimization experiences.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import json
import os
import asyncio

from src.logger import logger
from src.model import model_manager
from src.utils import dedent, file_lock
from src.message.types import HumanMessage, AssistantMessage, Message, SystemMessage
from src.memory.types import ChatEvent, Summary, Insight, EventType, Importance, Memory, MemoryContext
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
            if event.event_type == EventType.OPTIMIZATION_STEP or event.event_type == EventType.TOOL_STEP or event.event_type == EventType.TASK_END:
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
        
        # Create processing prompt using the combined template, optimized for optimization learning
        prompt = dedent(f"""Analyze the optimization events and extract both summaries and insights.
        <intro>
        This is an optimizer memory system that records optimization experiences for agent self-evolution.
        Focus on extracting actionable knowledge that can help future optimizations.
        
        For summaries, focus on:
        1. Key optimization decisions and variable changes
        2. Which variables were optimized and why
        3. Task progress and optimization outcomes
        4. Reflection analysis highlights
        
        For insights, look for:
        1. **Effective optimization strategies**: Which variable types benefit most from optimization? What patterns lead to successful improvements?
        2. **Variable-specific learnings**: What works well for prompt variables vs tool code vs solution variables?
        3. **Reflection patterns**: What kinds of reflection analysis lead to better improvements?
        4. **Common pitfalls**: What mistakes should be avoided in future optimizations?
        5. **Task-specific patterns**: What optimization strategies work best for different task types?
        6. **Improvement patterns**: What changes to variables typically lead to better results?
        7. **Optimization effectiveness**: Which optimization steps showed the most improvement?

        Avoid repeating information already in the summaries or insights.
        If there is nothing new, do not add a new entry.
        Prioritize insights that can be directly applied to future optimization tasks.
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
            - "tags": array of strings (categorization tags like "variable_optimization", "reflection_pattern", "effective_strategy", "common_pitfall", etc.)
        </output_format>

        Current memory:
        {current_memory}

        New conversation events:
        {new_lines}

        Based on the current memory and new conversation events, generate new summaries and insights that will help future optimizations.
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
class OptimizerMemorySystem(Memory):
    """Memory system for recording optimizer optimization history and experiences."""
    
    require_grad: bool = Field(default=False, description="Whether the memory system requires gradients")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 model_name: str = "gpt-4.1",
                 max_summaries: int = 10,
                 max_insights: int = 10,
                 require_grad: bool = False,
                 **kwargs):
        super().__init__(require_grad=require_grad, **kwargs)
        
        if base_dir is not None:
            self.base_dir = base_dir
        
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| Optimizer memory system base directory: {self.base_dir}")
        self.save_path = os.path.join(self.base_dir, "optimizer_memory.json")
            
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
    
        # Per-session cache and locks for concurrent safety
        self._session_memory_cache: Dict[str, CombinedMemory] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._cache_lock = asyncio.Lock()

    async def _get_or_create_session_memory(self, id: str) -> tuple[CombinedMemory, asyncio.Lock]:
        """Get or create a CombinedMemory instance for the given id with proper locking."""
        async with self._cache_lock:
            if id not in self._session_locks:
                self._session_locks[id] = asyncio.Lock()
            if id not in self._session_memory_cache:
                self._session_memory_cache[id] = CombinedMemory(
                    model_name=self.model_name,
                    max_summaries=self.max_summaries,
                    max_insights=self.max_insights
                )
                logger.info(f"| 📝 Created new optimizer memory cache for id: {id}")
            return self._session_memory_cache[id], self._session_locks[id]

    async def _cleanup_session_memory(self, id: str):
        """Remove session memory from cache."""
        async with self._cache_lock:
            if id in self._session_memory_cache:
                del self._session_memory_cache[id]
                logger.info(f"| 🧹 Removed optimizer memory from cache: {id}")
            if id in self._session_locks:
                del self._session_locks[id]

    async def start_session(self, ctx: MemoryContext = None, **kwargs) -> str:
        """Start new session with MemorySystem. Automatically loads from JSON if file exists."""
        if self.save_path and os.path.exists(self.save_path):
            logger.info(f"| 📂 Loading optimizer memory from JSON: {self.save_path}")
            await self.load_from_json(self.save_path)
            logger.info(f"| ✅ Optimizer memory loaded from JSON")
        
        if ctx is None:
            ctx = MemoryContext()
        id = ctx.id
        await self._get_or_create_session_memory(id)
        return id
    
    async def end_session(self, ctx: MemoryContext = None, **kwargs):
        """End session. Automatically saves to JSON if save_path is set."""
        if ctx is None:
            return
        id = ctx.id
        if self.save_path:
            await self.save_to_json(self.save_path)
        
        # Cleanup session memory
        await self._cleanup_session_memory(id)
    
    async def add_event(self,
                        step_number: int,
                        event_type,
                        data: Any,
                        agent_name: str,
                        ctx: MemoryContext = None,
                        task_id: Optional[str] = None,
                        **kwargs):
        """Add event to optimizer memory system.
        
        Args:
            step_number: Step number (optimization step)
            event_type: Event type (use EventType.OPTIMIZATION_STEP with "variable_changes" in data for full optimization records)
            data: Event data (dict containing optimization information)
            agent_name: Agent name
            ctx: Memory context (required)
            task_id: Optional task ID
            **kwargs: Additional arguments (can include optimization-specific fields)
        """
        if ctx is None:
            logger.warning("| No context available for add_event")
            return
        id = ctx.id
        
        # Ensure event_type is EventType enum
        if not isinstance(event_type, EventType):
            # Try to convert string to EventType
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    logger.warning(f"| ⚠️ Invalid event_type '{event_type}', defaulting to OPTIMIZATION_STEP")
                    event_type = EventType.OPTIMIZATION_STEP
            else:
                logger.warning(f"| ⚠️ Invalid event_type type '{type(event_type)}', defaulting to OPTIMIZATION_STEP")
                event_type = EventType.OPTIMIZATION_STEP
        
        # Build event data string from optimization information
        event_data_str = ""
        if isinstance(data, dict):
            # Check if this is a full optimization step record
            # Optimization steps use EventType.OPTIMIZATION_STEP with "variable_changes" in data
            if event_type == EventType.OPTIMIZATION_STEP and "variable_changes" in data:
                # Format comprehensive optimization step data
                event_parts = []
                event_parts.append(f"=== Optimization Step {step_number} ===")
                if data.get("task"):
                    event_parts.append(f"Task: {data['task']}")
                if data.get("reflection_analysis"):
                    event_parts.append(f"\n--- Reflection Analysis ---\n{data['reflection_analysis']}")
                
                # Record variable changes
                if data.get("variable_changes"):
                    event_parts.append("\n--- Variable Changes ---")
                    for var_name, var_data in data["variable_changes"].items():
                        var_type = var_data.get("type", "")
                        before_value = var_data.get("before", "")
                        after_value = var_data.get("after", "")
                        event_parts.append(f"\nVariable: {var_name} (Type: {var_type})")
                        event_parts.append(f"Before:\n{before_value}")
                        event_parts.append(f"After:\n{after_value}")
                
                if data.get("execution_result"):
                    event_parts.append(f"\n--- Execution Result ---\n{data['execution_result']}")
                
                event_data_str = "\n".join(event_parts)
            else:
                # Handle legacy format or simple optimization events
                variable_name = data.get("variable_name", kwargs.get("variable_name", ""))
                before_value = data.get("before_value", kwargs.get("before_value", ""))
                after_value = data.get("after_value", kwargs.get("after_value", ""))
                execution_result = data.get("execution_result", kwargs.get("execution_result"))
                reflection_analysis = data.get("reflection_analysis", kwargs.get("reflection_analysis"))
                reward = data.get("reward", kwargs.get("reward"))
                loss = data.get("loss", kwargs.get("loss"))
                advantage = data.get("advantage", kwargs.get("advantage"))
                
                # Build event data string with more context for optimization learning
                event_parts = []
                if variable_name:
                    event_parts.append(f"Variable Name: {variable_name}")
                if before_value:
                    event_parts.append(f"Before Value:\n{before_value}")
                if after_value:
                    event_parts.append(f"After Value:\n{after_value}")
                if execution_result:
                    event_parts.append(f"Execution Result:\n{execution_result}")
                if reflection_analysis:
                    event_parts.append(f"Reflection Analysis:\n{reflection_analysis}")
                if reward is not None:
                    event_parts.append(f"Reward: {reward}")
                if loss is not None:
                    event_parts.append(f"Loss: {loss}")
                if advantage is not None:
                    event_parts.append(f"Advantage: {advantage}")
                
                event_data_str = "\n\n".join(event_parts)
        else:
            event_data_str = str(data)
        
        event_id = "event_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        event = ChatEvent(
            id=event_id,
            step_number=step_number,
            event_type=event_type,  # Already validated and converted above
            data={"content": event_data_str, "raw_data": data},
            agent_name=agent_name,
            task_id=task_id,
            session_id=id
        )
    
        session_memory, session_lock = await self._get_or_create_session_memory(id)
        async with session_lock:
            await session_memory.add_event(event)
        
        if self.save_path:
            await self.save_to_json(self.save_path)
    
    async def clear_session(self, ctx: MemoryContext = None, **kwargs):
        """Clear specific session"""
        if ctx is None:
            return
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                await self._session_memory_cache[id].clear()
        await self._cleanup_session_memory(id)
                
    async def clear(self):
        """Clear all sessions"""
        async with self._cache_lock:
            for id in list(self._session_memory_cache.keys()):
                await self._session_memory_cache[id].clear()
            self._session_memory_cache.clear()
            self._session_locks.clear()
            
    async def get_event(self, ctx: MemoryContext = None, n: Optional[int] = None, **kwargs) -> List[ChatEvent]:
        """Get events from memory system."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_event(n=n)
        return []
    
    async def get_summary(self, ctx: MemoryContext = None, n: Optional[int] = None, **kwargs) -> List[Summary]:
        """Get summaries from memory system."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_summary(n=n)
        return []
    
    async def get_insight(self, ctx: MemoryContext = None, n: Optional[int] = None, **kwargs) -> List[Insight]:
        """Get insights from memory system."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_insight(n=n)
        return []
    
    async def get_optimization_experience(self,
                                         variable_type: Optional[str] = None,
                                         tag: Optional[str] = None,
                                         n: Optional[int] = 10,
                                         **kwargs) -> Dict[str, Any]:
        """Get optimization experience from all sessions for learning.
        
        Args:
            variable_type: Optional variable type filter (e.g., "system_prompt", "tool_code")
            tag: Optional tag filter (e.g., "effective_strategy", "common_pitfall")
            n: Maximum number of insights/summaries to return
        """
        all_insights = []
        all_summaries = []
        
        async with self._cache_lock:
            for id, session_memory in self._session_memory_cache.items():
                session_insights = await session_memory.get_insight()
                session_summaries = await session_memory.get_summary()
                
                if tag:
                    session_insights = [
                        insight for insight in session_insights
                        if hasattr(insight, 'tags') and tag in insight.tags
                    ]
                
                if variable_type:
                    session_insights = [
                        insight for insight in session_insights
                        if variable_type.lower() in insight.content.lower()
                    ]
                    session_summaries = [
                        summary for summary in session_summaries
                        if variable_type.lower() in summary.content.lower()
                    ]
                
                all_insights.extend(session_insights)
                all_summaries.extend(session_summaries)
        
        importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        all_insights.sort(key=lambda x: importance_order.get(x.importance, 2))
        all_summaries.sort(key=lambda x: importance_order.get(x.importance, 2))
        
        if n:
            all_insights = all_insights[:n]
            all_summaries = all_summaries[:n]
        
        return {
            "insights": all_insights,
            "summaries": all_summaries,
            "total_sessions": len(self._session_memory_cache)
        }
    
    async def save_to_json(self, file_path: str) -> str:
        """Save memory system state to JSON file."""
        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            metadata = {
                "memory_system_type": "optimizer_memory_system",
                "session_ids": list(self._session_memory_cache.keys())
            }
            
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
            
            save_data = {"metadata": metadata, "sessions": sessions}
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.debug(f"| 💾 Optimizer memory saved to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: str) -> bool:
        """Load memory system state from JSON file."""
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️  Optimizer memory file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                if "metadata" not in load_data or "sessions" not in load_data:
                    raise ValueError(f"Invalid optimizer memory format")
                
                sessions_data = load_data.get("sessions", {})
                
                async with self._cache_lock:
                    for id, session_data in sessions_data.items():
                        if id not in self._session_memory_cache:
                            self._session_memory_cache[id] = CombinedMemory(
                                model_name=self.model_name, 
                                max_summaries=self.max_summaries,
                                max_insights=self.max_insights
                            )
                            self._session_locks[id] = asyncio.Lock()
                        
                        session_memory = self._session_memory_cache[id]
                        session_memory_data = session_data.get("session_memory", {})
                        
                        if "events" in session_memory_data:
                            events = []
                            for event_data in session_memory_data["events"]:
                                if event_data.get("timestamp"):
                                    event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                                if event_data.get("event_type"):
                                    event_data["event_type"] = EventType(event_data["event_type"])
                                events.append(ChatEvent(**event_data))
                            session_memory.events = events
                        
                        if "summaries" in session_memory_data:
                            summaries = []
                            for summary_data in session_memory_data["summaries"]:
                                if summary_data.get("importance"):
                                    summary_data["importance"] = Importance(summary_data["importance"])
                                summaries.append(Summary(**summary_data))
                            session_memory.summaries = summaries
                        
                        if "insights" in session_memory_data:
                            insights = []
                            for insight_data in session_memory_data["insights"]:
                                if insight_data.get("importance"):
                                    insight_data["importance"] = Importance(insight_data["importance"])
                                insights.append(Insight(**insight_data))
                            session_memory.insights = insights
                
                logger.info(f"| 📂 Optimizer memory loaded from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load optimizer memory from {file_path}: {e}", exc_info=True)
                return False
