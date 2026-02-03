"""
Offline trading memory system for tracking perpetual futures decisions, performance patterns, and extracting actionable insights.

Architecture:
- OfflineTradingMemorySystem: Specialized memory for offline trading agents
- Focuses on: decision rationale, win/loss patterns, strategy effectiveness, market conditions
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import json
import os
import asyncio

from src.memory.types import ChatEvent, EventType, Importance, Memory
from src.session import SessionContext
from src.model import model_manager
from src.message.types import HumanMessage, AssistantMessage, Message, SystemMessage
from src.logger import logger
from src.utils import dedent, file_lock
from src.registry import MEMORY_SYSTEM


class OfflineTradingSummary(BaseModel):
    """Summary of offline perpetual futures decisions, highlighting the reasoning, execution, and outcomes"""
    id: str = Field(description="Unique identifier")
    importance: Importance = Field(description="Importance level")
    content: str = Field(description="Narrative of the decisions taken, rationale, and observed results")
    trade_count: int = Field(default=0, description="Number of offline trades captured in this period")
    profit_loss: float = Field(default=0.0, description="Cumulative profit/loss percentage driven by those decisions")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        return f"[{self.importance.value}] {self.content} (Trades: {self.trade_count}, P/L: {self.profit_loss:.2f}%)"


class OfflineTradingInsight(BaseModel):
    """Insight capturing how offline trading decisions impacted P/L, highlighting lessons from wins and losses"""
    id: str = Field(description="Unique identifier")
    importance: Importance = Field(description="Importance level")
    content: str = Field(description="Lesson learned about decision quality, execution, or market response")
    insight_type: str = Field(description="Type: winning_pattern, losing_pattern, risk_lesson, market_condition")
    related_trades: List[str] = Field(default_factory=list, description="Related trade IDs or periods")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        return f"[{self.importance.value}|{self.insight_type}] {self.content} (Tags: {', '.join(self.tags)})"


class OfflineTradingMemoryOutput(BaseModel):
    """Structured output for offline trading memory generation"""
    summaries: List[OfflineTradingSummary] = Field(description="List of offline trading decision summaries")
    insights: List[OfflineTradingInsight] = Field(description="List of insights derived from decision outcomes")


class ProcessDecision(BaseModel):
    should_process: bool = Field(description="Whether to process the trading memory")
    reasoning: str = Field(description="Reasoning for the decision")


class OfflineTradingCombinedMemory:
    """Offline trading combined memory that tracks perpetual futures decisions and resulting insights"""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1", 
                 max_summaries: int = 15,
                 max_insights: int = 50):
        
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
        
        self.events: List[ChatEvent] = []
        self.candidate_chat_history: List[Message] = []
        self.summaries: List[OfflineTradingSummary] = []
        self.insights: List[OfflineTradingInsight] = []
    
    async def add_event(self, event: Union[ChatEvent, List[ChatEvent]]):
        """Process trading events and extract summaries and insights"""
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
                    self.candidate_chat_history.append(AssistantMessage(content=content))
                else:
                    self.candidate_chat_history.append(HumanMessage(content=content))
        
        # Check if we should process trading memory
        should_process = await self._check_should_process_memory()
        if should_process:
            await self._process_trading_memory()
            
    async def _get_new_lines_text(self) -> str:
        """Get new trading events from chat history"""
        new_lines = []
        for msg in self.candidate_chat_history:
            if isinstance(msg, HumanMessage):
                new_lines.append(f"<market_state>\n{msg.content}\n</market_state>")
            elif isinstance(msg, AssistantMessage):
                new_lines.append(f"<trading_action>\n{msg.content}\n</trading_action>")
        return "\n".join(new_lines)
    
    async def _get_current_memory_text(self) -> str:
        """Get current trading memory text"""
        summaries_text = "\n".join([str(s) for s in self.summaries]) if self.summaries else "No summaries yet"
        insights_text = "\n".join([str(i) for i in self.insights]) if self.insights else "No insights yet"
        
        current_memory = dedent(f"""<trading_summaries>
            {summaries_text}
            </trading_summaries>
            <trading_insights>
            {insights_text}
            </trading_insights>""")
        return current_memory

    async def _check_should_process_memory(self) -> bool:
        """Check if we should process trading memory"""
        if len(self.candidate_chat_history) <= 2:
            return False
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        decision_prompt = dedent(f"""You are analyzing offline trading (perpetual futures) events to decide whether to process them into decision summaries and insights.
        Current trading session has {self.size()} events.

        Decision criteria for OFFLINE TRADING memory:
        1. If there are fewer than 2 trading events, do not process
        2. If the trading actions are repetitive without new outcomes, do not process
        3. If there are completed decision cycles (e.g., LONG→CLOSE_LONG or SHORT→CLOSE_SHORT) with results, PROCESS
        4. If there are significant profit/loss events (>2% change), PROCESS
        5. If there are clear trading patterns or lessons emerging, PROCESS
        6. If conversation exceeds 4 trading events, PROCESS

        Current trading memory:
        {current_memory}

        New trading events:
        {new_lines}

        Decide if you should process the trading memory.""")
                
        try:
            # Build messages
            messages = [
                SystemMessage(content="You are a trading memory processing decision system. Always respond with valid JSON."),
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
            
            logger.info(f"| Offline trading memory processing decision: {should_process} - {reasoning}")
            return should_process
                
        except Exception as e:
            logger.warning(f"Failed to check if should process offline trading memory: {e}")
            return False
    
    async def _process_trading_memory(self):
        """Process trading memory and generate trading-specific summaries and insights"""
        if not self.candidate_chat_history:
            return
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        prompt = dedent(f"""Analyze the offline trading (perpetual futures) events and extract decision summaries and outcome-driven insights.
        <intro>
        For OFFLINE TRADING SUMMARIES, focus on:
        1. Decisions executed (LONG/SHORT/HOLD/CLOSE) and the reasoning behind them
        2. Market context (trend, volatility, liquidity) during those decisions
        3. Execution results: profit/loss impact, holding durations, notable slippage
        4. Overall effectiveness of the decision-making during this window
        
        For OFFLINE TRADING INSIGHTS, extract:
        1. WINNING PATTERNS: Decision approaches that consistently produced gains.
        2. LOSING PATTERNS: Decision mistakes that led to losses or missed opportunities.
        3. RISK LESSONS: How stop placement, position sizing, or capital usage impacted outcomes.
        4. MARKET CONDITIONS: Observations about the market regime affecting these decisions.
        
        Insight types: "winning_pattern", "losing_pattern", "risk_lesson", "market_condition"
        
        Avoid repeating information already in memory.
        Focus on ACTIONABLE insights that can improve future trading decisions.
        </intro>

        <output_format>
        Respond with JSON containing:
        - "summaries": array of offline trading summaries with:
            - "id": unique identifier
            - "importance": "high", "medium", or "low"
            - "content": summary of trading actions and outcomes
            - "trade_count": number of trades
            - "profit_loss": cumulative profit/loss percentage
        - "insights": array of offline trading insights with:
            - "id": unique identifier
            - "importance": "high", "medium", or "low"
            - "content": the trading insight
            - "insight_type": one of [winning_pattern, losing_pattern, risk_lesson, market_condition]
            - "related_trades": list of related trade identifiers
            - "tags": categorization tags (e.g., ["volatility", "momentum", "news-driven"])
        </output_format>

        Current trading memory:
        {current_memory}

        New trading events:
        {new_lines}

        Generate new trading summaries and insights based on the events.""")
        
        try:
            # Build messages
            messages = [
                SystemMessage(content="You are a trading memory processing system. Always respond with valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            # Call model manager with BaseModel response_format
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=OfflineTradingMemoryOutput
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
            
            logger.info(f"| Generated {len(new_summaries)} trading summaries and {len(new_insights)} trading insights")
            
            # Sort and limit
            await self._sort_and_limit_summaries()
            await self._sort_and_limit_insights()
            
            # Clear candidate chat history
            self.candidate_chat_history.clear()
            
        except Exception as e:
            logger.warning(f"Failed to process trading memory: {e}")
    
    async def _sort_and_limit_insights(self):
        """Sort trading insights by importance and limit count"""
        importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.insights.sort(key=lambda x: importance_order[x.importance])
        
        if len(self.insights) > self.max_insights:
            self.insights = self.insights[:self.max_insights]

    async def _sort_and_limit_summaries(self):
        """Sort trading summaries by importance and limit count"""
        importance_order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.summaries.sort(key=lambda x: importance_order[x.importance])
        
        if len(self.summaries) > self.max_summaries:
            self.summaries = self.summaries[:self.max_summaries]
    
    def clear(self):
        """Clear all trading memory"""
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
    
    async def get_summary(self, n: Optional[int] = None) -> List[OfflineTradingSummary]:
        if n is None:
            return self.summaries
        return self.summaries[-n:] if len(self.summaries) > n else self.summaries
    
    async def get_insight(self, n: Optional[int] = None) -> List[OfflineTradingInsight]:
        if n is None:
            return self.insights
        return self.insights[-n:] if len(self.insights) > n else self.insights


@MEMORY_SYSTEM.register_module(force=True)
class OfflineTradingMemorySystem(Memory):
    """Offline trading memory system focused on perpetual futures decision tracking and learning"""
    
    require_grad: bool = Field(default=False, description="Whether the memory system requires gradients")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 model_name: str = "gpt-4.1",
                 max_summaries: int = 15,
                 max_insights: int = 50,
                 require_grad: bool = False,
                 **kwargs):
        super().__init__(require_grad=require_grad, **kwargs)
        
        if base_dir is not None:
            self.base_dir = base_dir
        
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| Offline trading memory system base directory: {self.base_dir}")
        self.save_path = os.path.join(self.base_dir, "memory_system.json") if self.base_dir else None
            
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
    
        # Per-session cache and locks for concurrent safety
        self._session_memory_cache: Dict[str, OfflineTradingCombinedMemory] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._cache_lock = asyncio.Lock()

    async def _get_or_create_session_memory(self, id: str) -> tuple[OfflineTradingCombinedMemory, asyncio.Lock]:
        """Get or create a session memory instance with proper locking."""
        async with self._cache_lock:
            if id not in self._session_locks:
                self._session_locks[id] = asyncio.Lock()
            if id not in self._session_memory_cache:
                self._session_memory_cache[id] = OfflineTradingCombinedMemory(
                    model_name=self.model_name,
                    max_summaries=self.max_summaries,
                    max_insights=self.max_insights
                )
                logger.info(f"| 📝 Created new offline trading memory cache for id: {id}")
            return self._session_memory_cache[id], self._session_locks[id]

    async def _cleanup_session_memory(self, id: str):
        """Remove session memory from cache."""
        async with self._cache_lock:
            if id in self._session_memory_cache:
                del self._session_memory_cache[id]
            if id in self._session_locks:
                del self._session_locks[id]

    async def start_session(self, ctx: SessionContext = None, **kwargs) -> str:
        """Start new trading session."""
        if self.save_path and os.path.exists(self.save_path):
            await self.load_from_json(self.save_path)
        
        if ctx is None:
            ctx = SessionContext()
        id = ctx.id
        await self._get_or_create_session_memory(id)
        logger.info(f"| Started trading memory session: {id}")
        return id
    
    async def end_session(self, ctx: SessionContext = None, **kwargs):
        """End trading session."""
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
                        task_id: Optional[str] = None,
                        ctx: SessionContext = None,
                        **kwargs):
        """Add trading event to memory"""
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
                    logger.warning(f"| ⚠️ Invalid event_type '{event_type}', defaulting to TOOL_STEP")
                    event_type = EventType.TOOL_STEP
            else:
                logger.warning(f"| ⚠️ Invalid event_type type '{type(event_type)}', defaulting to TOOL_STEP")
                event_type = EventType.TOOL_STEP
        
        event_id = "trade_event_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        event = ChatEvent(
            id=event_id,
            step_number=step_number,
            event_type=event_type,
            data=data,
            agent_name=agent_name,
            task_id=task_id,
            session_id=id
        )
    
        session_memory, session_lock = await self._get_or_create_session_memory(id)
        async with session_lock:
            await session_memory.add_event(event)
        
        if self.save_path:
            await self.save_to_json(self.save_path)
    
    async def clear_session(self, ctx: SessionContext = None, **kwargs):
        """Clear specific trading session"""
        if ctx is None:
            return
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                self._session_memory_cache[id].clear()
        await self._cleanup_session_memory(id)
                
    async def clear(self):
        """Clear all trading sessions"""
        async with self._cache_lock:
            for id in list(self._session_memory_cache.keys()):
                self._session_memory_cache[id].clear()
            self._session_memory_cache.clear()
            self._session_locks.clear()
            
    async def get_event(self, ctx: SessionContext = None, n: Optional[int] = None, **kwargs) -> List[ChatEvent]:
        """Get events from memory system."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_event(n=n)
        return []
    
    async def get_summary(self, ctx: SessionContext = None, n: Optional[int] = None, **kwargs) -> List[OfflineTradingSummary]:
        """Get summaries from memory system."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_summary(n=n)
        return []
    
    async def get_insight(self, ctx: SessionContext = None, n: Optional[int] = None, **kwargs) -> List[OfflineTradingInsight]:
        """Get insights from memory system."""
        if ctx is None:
            return []
        id = ctx.id
        async with self._cache_lock:
            if id in self._session_memory_cache:
                return await self._session_memory_cache[id].get_insight(n=n)
        return []
    
    async def save_to_json(self, file_path: str) -> str:
        """Save memory system state to JSON file."""
        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            metadata = {
                "memory_system_type": "offline_trading_memory_system",
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
            
            logger.debug(f"| 💾 Memory saved to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: str) -> bool:
        """Load memory system state from JSON file."""
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️  Memory file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                if "metadata" not in load_data or "sessions" not in load_data:
                    raise ValueError(f"Invalid memory format")
                
                sessions_data = load_data.get("sessions", {})
                
                async with self._cache_lock:
                    for id, session_data in sessions_data.items():
                        if id not in self._session_memory_cache:
                            self._session_memory_cache[id] = OfflineTradingCombinedMemory(
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
                                if summary_data.get("timestamp"):
                                    summary_data["timestamp"] = datetime.fromisoformat(summary_data["timestamp"])
                                if summary_data.get("importance"):
                                    summary_data["importance"] = Importance(summary_data["importance"])
                                summaries.append(OfflineTradingSummary(**summary_data))
                            session_memory.summaries = summaries
                        
                        if "insights" in session_memory_data:
                            insights = []
                            for insight_data in session_memory_data["insights"]:
                                if insight_data.get("timestamp"):
                                    insight_data["timestamp"] = datetime.fromisoformat(insight_data["timestamp"])
                                if insight_data.get("importance"):
                                    insight_data["importance"] = Importance(insight_data["importance"])
                                insights.append(OfflineTradingInsight(**insight_data))
                            session_memory.insights = insights
                
                logger.info(f"| 📂 Memory loaded from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load memory from {file_path}: {e}", exc_info=True)
                return False

