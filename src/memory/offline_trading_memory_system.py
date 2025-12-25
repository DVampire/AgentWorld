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

from src.memory.types import ChatEvent, EventType, Importance, SessionInfo, Memory
from src.model import model_manager
from src.message.types import HumanMessage, AssistantMessage, Message, SystemMessage
from src.logger import logger
from src.utils import dedent, file_lock


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
    reason: str = Field(description="Reason for the decision")


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
            reason = processed_decision_response.reason
            
            logger.info(f"| Offline trading memory processing decision: {should_process} - {reason}")
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


class OfflineTradingMemorySystem(Memory):
    """Offline trading memory system focused on perpetual futures decision tracking and learning"""
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 model_name: str = "gpt-4.1",
                 max_summaries: int = 15,
                 max_insights: int = 50,
                 **kwargs):
        super().__init__(**kwargs)
        
        if base_dir is not None:
            self.base_dir = base_dir
        
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| Offline trading memory system base directory: {self.base_dir}")
        self.save_path = os.path.join(self.base_dir, "memory_system.json") if self.base_dir else None
            
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
    
        self.session_memory: Dict[str, OfflineTradingCombinedMemory] = {}
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
                            description: Optional[str] = None) -> str:
        """Start new trading session. Automatically loads from JSON if file exists."""
        # Auto-load from JSON if file exists and save_path is set
        if self.save_path and os.path.exists(self.save_path):
            await self.load_from_json(self.save_path)
        
        session_info = SessionInfo(
            session_id=session_id,
            agent_name=agent_name,
            task_id=task_id,
            description=description
        )
        self.session_info[session_id] = session_info
        self.current_session_id = session_id
        
        # Initialize OfflineTradingCombinedMemory for this session if it doesn't exist
        if session_id not in self.session_memory:
            self.session_memory[session_id] = OfflineTradingCombinedMemory(
                model_name=self.model_name, 
                max_summaries=self.max_summaries,
                max_insights=self.max_insights
            )
        
        logger.info(f"| Started trading memory session: {session_id}")
        return session_id
    
    async def end_session(self, session_id: Optional[str] = None):
        """End trading session. Automatically saves to JSON if save_path is set."""
        session_id = await self._check_session_id(session_id)
            
        if session_id and session_id in self.session_info:
            self.session_info[session_id].end_time = datetime.now()
            
            if session_id == self.current_session_id:
                self.current_session_id = None
            
            # Auto-save to JSON if save_path is set
            if self.save_path:
                await self.save_to_json(self.save_path)
    
    async def add_event(self,
                        step_number: int,
                        event_type,
                        data: Any,
                        agent_name: str,
                        task_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        **kwargs):
        """Add trading event to memory
        
        Args:
            step_number: Step number
            event_type: Event type
            data: Event data
            agent_name: Agent name
            task_id: Optional task ID
            session_id: Optional session ID. If None, uses current session.
            **kwargs: Additional arguments
        """
        session_id = await self._check_session_id(session_id)
        if session_id is None:
            logger.warning("| No session ID available for add_event")
            return
        
        event_id = "trade_event_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        event = ChatEvent(
            id=event_id,
            step_number=step_number,
            event_type=event_type,
            data=data,
            agent_name=agent_name,
            task_id=task_id,
            session_id=session_id
        )
    
        if session_id in self.session_memory:
            await self.session_memory[session_id].add_event(event)
            
            # Auto-save to JSON if save_path is set
            if self.save_path:
                await self.save_to_json(self.save_path)
    
    async def get_session_info(self, session_id: Optional[str] = None) -> Optional[SessionInfo]:
        session_id = await self._check_session_id(session_id)
        return self.session_info.get(session_id)
    
    async def clear_session(self, session_id: Optional[str] = None):
        """Clear specific trading session"""
        session_id = await self._check_session_id(session_id)
            
        if session_id in self.session_info:
            del self.session_info[session_id]
            
            if session_id in self.session_memory:
                self.session_memory[session_id].clear()
                del self.session_memory[session_id]
            
            if session_id == self.current_session_id:
                self.current_session_id = None
                
    async def clear(self):
        """Clear all trading sessions"""
        for session_id in list(self.session_info.keys()):
            await self.clear_session(session_id)
            
    async def get_event(self, n: Optional[int] = None, session_id: Optional[str] = None) -> List[ChatEvent]:
        """Get events from memory system.
        
        Args:
            n: Number of events to retrieve. If None, returns all events.
            session_id: Optional session ID. If None, uses current session.
            
        Returns:
            List of events
        """
        session_id = await self._check_session_id(session_id)
        if session_id and session_id in self.session_memory:
            return await self.session_memory[session_id].get_event(n=n)
        return []
    
    async def get_summary(self, n: Optional[int] = None, session_id: Optional[str] = None) -> List[OfflineTradingSummary]:
        """Get summaries from memory system.
        
        Args:
            n: Number of summaries to retrieve. If None, returns all summaries.
            session_id: Optional session ID. If None, uses current session.
            
        Returns:
            List of summaries
        """
        session_id = await self._check_session_id(session_id)
        if session_id and session_id in self.session_memory:
            return await self.session_memory[session_id].get_summary(n=n)
        return []
    
    async def get_insight(self, n: Optional[int] = None, session_id: Optional[str] = None) -> List[OfflineTradingInsight]:
        """Get insights from memory system.
        
        Args:
            n: Number of insights to retrieve. If None, returns all insights.
            session_id: Optional session ID. If None, uses current session.
            
        Returns:
            List of insights
        """
        session_id = await self._check_session_id(session_id)
        if session_id and session_id in self.session_memory:
            return await self.session_memory[session_id].get_insight(n=n)
        return []
    
    async def save_to_json(self, file_path: str) -> str:
        """Save memory system state to JSON file.
        
        Structure:
        {
            "metadata": {
                "memory_system_type": str,
                "current_session_id": str,
                "session_ids": [str, ...]
            },
            "sessions": {
                "session_id": {
                    "session_info": {...},
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
                "memory_system_type": "offline_trading_memory_system",
                "current_session_id": self.current_session_id,
                "session_ids": list(self.session_info.keys())
            }
            
            # Prepare sessions data
            sessions = {}
            for session_id in self.session_info.keys():
                session_data = {
                    "session_info": None,
                    "session_memory": {
                        "events": [],
                        "summaries": [],
                        "insights": []
                    }
                }
                
                # Save session info
                if session_id in self.session_info:
                    session_data["session_info"] = self.session_info[session_id].model_dump(mode="json")
                
                # Save session memory
                if session_id in self.session_memory:
                    session_memory = self.session_memory[session_id]
                    session_data["session_memory"] = {
                        "events": [event.model_dump(mode="json") for event in session_memory.events],
                        "summaries": [summary.model_dump(mode="json") for summary in session_memory.summaries],
                        "insights": [insight.model_dump(mode="json") for insight in session_memory.insights],
                    }
                
                sessions[session_id] = session_data
            
            # Prepare save data
            save_data = {
                "metadata": metadata,
                "sessions": sessions
            }
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Memory saved to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: str) -> bool:
        """Load memory system state from JSON file.
        
        Expected format:
        {
            "metadata": {
                "memory_system_type": str,
                "current_session_id": str,
                "session_ids": [str, ...]
            },
            "sessions": {
                "session_id": {
                    "session_info": {...},
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
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️  Memory file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                # Validate format
                if "metadata" not in load_data or "sessions" not in load_data:
                    raise ValueError(
                        f"Invalid memory format. Expected {{'metadata': {{...}}, 'sessions': {{...}}}}, "
                        f"got keys: {list(load_data.keys())}"
                    )
                
                # Restore metadata
                metadata = load_data.get("metadata", {})
                self.current_session_id = metadata.get("current_session_id")
                
                # Restore sessions
                sessions_data = load_data.get("sessions", {})
                
                for session_id, session_data in sessions_data.items():
                    # Restore session info
                    session_info_data = session_data.get("session_info")
                    if session_info_data:
                        # Parse datetime strings
                        if session_info_data.get("start_time"):
                            session_info_data["start_time"] = datetime.fromisoformat(session_info_data["start_time"])
                        if session_info_data.get("end_time"):
                            session_info_data["end_time"] = datetime.fromisoformat(session_info_data["end_time"])
                        
                        self.session_info[session_id] = SessionInfo(**session_info_data)
                    
                    # Ensure session memory exists
                    if session_id not in self.session_memory:
                        await self.start_session(
                            session_id=session_id,
                            agent_name=self.session_info.get(session_id).agent_name if session_id in self.session_info else None,
                            task_id=self.session_info.get(session_id).task_id if session_id in self.session_info else None,
                            description=self.session_info.get(session_id).description if session_id in self.session_info else None,
                        )
                    
                    session_memory = self.session_memory[session_id]
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
                            if summary_data.get("timestamp"):
                                summary_data["timestamp"] = datetime.fromisoformat(summary_data["timestamp"])
                            if summary_data.get("importance"):
                                summary_data["importance"] = Importance(summary_data["importance"])
                            summaries.append(OfflineTradingSummary(**summary_data))
                        session_memory.summaries = summaries
                    
                    # Restore insights
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

