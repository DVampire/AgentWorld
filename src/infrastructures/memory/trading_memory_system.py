"""
Trading-specific memory system for tracking trading history, performance patterns, and extracting trading insights.

Architecture:
- TradingMemorySystem: Specialized memory for trading agents
- Focuses on: trade history, win/loss patterns, strategy effectiveness, market conditions
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import json

from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field

from src.logger import logger
from src.infrastructures.memory.types import ChatEvent, EventType, Importance, SessionInfo
from src.infrastructures.models import model_manager
from src.utils import dedent
from src.infrastructures.registry import MEMORY_SYSTEM


class TradingSummary(BaseModel):
    """Trading-specific summary focusing on trade records and market conditions"""
    id: str = Field(description="Unique identifier")
    importance: Importance = Field(description="Importance level")
    content: str = Field(description="Summary of trading actions and outcomes")
    trade_count: int = Field(default=0, description="Number of trades in this summary period")
    profit_loss: float = Field(default=0.0, description="Cumulative profit/loss in this period")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        return f"[{self.importance.value}] {self.content} (Trades: {self.trade_count}, P/L: {self.profit_loss:.2f}%)"


class TradingInsight(BaseModel):
    """Trading-specific insight focusing on strategy patterns and lessons learned"""
    id: str = Field(description="Unique identifier")
    importance: Importance = Field(description="Importance level")
    content: str = Field(description="Trading insight or lesson learned")
    insight_type: str = Field(description="Type: winning_pattern, losing_pattern, risk_lesson, market_condition")
    related_trades: List[str] = Field(default_factory=list, description="Related trade IDs or periods")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        return f"[{self.importance.value}|{self.insight_type}] {self.content} (Tags: {', '.join(self.tags)})"


class TradingMemoryOutput(BaseModel):
    """Structured output for trading memory generation"""
    summaries: List[TradingSummary] = Field(description="List of trading summaries")
    insights: List[TradingInsight] = Field(description="List of trading insights")


class ProcessDecision(BaseModel):
    should_process: bool = Field(description="Whether to process the trading memory")
    reason: str = Field(description="Reason for the decision")


class TradingCombinedMemory:
    """Trading-specific combined memory that handles trade summaries and trading insights"""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1", 
                 max_summaries: int = 15,
                 max_insights: int = 50):
        
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
        
        self.llm = None
        self.events: List[ChatEvent] = []
        self.candidate_chat_history = ChatMessageHistory()
        self.summaries: List[TradingSummary] = []
        self.insights: List[TradingInsight] = []
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM for trading memory processing"""
        try:
            self.llm = model_manager.get(self.model_name)
        except Exception as e:
            logger.warning(f"Could not initialize LLM for trading memory: {e}")
            self.llm = None
    
    async def add_event(self, event: Union[ChatEvent, List[ChatEvent]]):
        """Process trading events and extract summaries and insights"""
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
                    self.candidate_chat_history.add_ai_message(content)
                else:
                    self.candidate_chat_history.add_user_message(content)
        
        # Check if we should process trading memory
        should_process = await self._check_should_process_memory()
        if should_process:
            await self._process_trading_memory()
            
    async def _get_new_lines_text(self) -> str:
        """Get new trading events from chat history"""
        new_lines = []
        for msg in self.candidate_chat_history.messages:
            if isinstance(msg, HumanMessage):
                new_lines.append(f"<market_state>\n{msg.content}\n</market_state>")
            elif isinstance(msg, AIMessage):
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
        if not self.llm or len(self.candidate_chat_history.messages) <= 2:
            return False
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        decision_prompt = dedent(f"""You are analyzing trading events to decide whether to process them into trading summaries and insights.

        Current trading session has {self.size()} events.

        Decision criteria for TRADING memory:
        1. If there are fewer than 2 trading events, do not process
        2. If the trading actions are repetitive without new outcomes, do not process
        3. If there are completed trades (BUY→SELL or SELL→BUY cycles) with results, PROCESS
        4. If there are significant profit/loss events (>2% change), PROCESS
        5. If there are clear trading patterns or lessons emerging, PROCESS
        6. If conversation exceeds 4 trading events, PROCESS

        Current trading memory:
        {current_memory}

        New trading events:
        {new_lines}

        Decide if you should process the trading memory.""")
                
        try:
            structured_llm = self.llm.with_structured_output(ProcessDecision)
            decision_response = await structured_llm.ainvoke(decision_prompt)
            logger.info(f"| Trading memory processing decision: {decision_response.should_process} - {decision_response.reason}")
            return decision_response.should_process
                
        except Exception as e:
            logger.warning(f"Failed to check if should process trading memory: {e}")
            return False
    
    async def _process_trading_memory(self):
        """Process trading memory and generate trading-specific summaries and insights"""
        if not self.llm or not self.candidate_chat_history.messages:
            return
            
        new_lines = await self._get_new_lines_text()
        current_memory = await self._get_current_memory_text()
        
        prompt = dedent(f"""Analyze the trading events and extract trading summaries and insights.

        <intro>
        For TRADING SUMMARIES, focus on:
        1. Trading actions taken (BUY/SELL/HOLD) and their timing
        2. Market conditions during trades (price trends, volume, news)
        3. Trade outcomes: profit/loss, win rate, holding periods
        4. Overall trading performance in this period
        
        For TRADING INSIGHTS, extract:
        1. WINNING PATTERNS: What strategies worked well? (e.g., "Buying on dips with volume confirmation yields +3% average")
        2. LOSING PATTERNS: What mistakes were made? (e.g., "Panic selling in volatile markets led to -5% losses")
        3. RISK LESSONS: What risk management lessons emerged? (e.g., "Holding losing positions too long amplifies losses")
        4. MARKET CONDITIONS: What market patterns were observed? (e.g., "Strong downtrends require patience, not catching falling knives")
        
        Insight types: "winning_pattern", "losing_pattern", "risk_lesson", "market_condition"
        
        Avoid repeating information already in memory.
        Focus on ACTIONABLE insights that can improve future trading decisions.
        </intro>

        <output_format>
        Respond with JSON containing:
        - "summaries": array of trading summaries with:
            - "id": unique identifier
            - "importance": "high", "medium", or "low"
            - "content": summary of trading actions and outcomes
            - "trade_count": number of trades
            - "profit_loss": cumulative profit/loss percentage
        - "insights": array of trading insights with:
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
            structured_llm = self.llm.with_structured_output(TradingMemoryOutput)
            response = await structured_llm.ainvoke(prompt)
            
            new_summaries = response.summaries
            new_insights = response.insights
            
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
    
    async def get_summary(self, n: Optional[int] = None) -> List[TradingSummary]:
        if n is None:
            return self.summaries
        return self.summaries[-n:] if len(self.summaries) > n else self.summaries
    
    async def get_insight(self, n: Optional[int] = None) -> List[TradingInsight]:
        if n is None:
            return self.insights
        return self.insights[-n:] if len(self.insights) > n else self.insights


@MEMORY_SYSTEM.register_module(name="trading_memory_system", force=True)
class TradingMemorySystem:
    """Trading-specific memory system for comprehensive trading performance tracking and learning"""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1",
                 max_summaries: int = 15,
                 max_insights: int = 50):
        
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
    
        self.session_memory: Dict[str, TradingCombinedMemory] = {}
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
        """Start new trading session"""
        session_info = SessionInfo(
            session_id=session_id,
            agent_name=agent_name,
            task_id=task_id,
            description=description
        )
        self.session_info[session_id] = session_info
        self.current_session_id = session_id
        
        # Initialize TradingCombinedMemory for this session
        self.session_memory[session_id] = TradingCombinedMemory(
            model_name=self.model_name, 
            max_summaries=self.max_summaries,
            max_insights=self.max_insights
        )
        
        logger.info(f"| Started trading memory session: {session_id}")
        return session_id
    
    async def end_session(self, session_id: Optional[str] = None):
        """End trading session"""
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
                        task_id: Optional[str] = None):
        """Add trading event to memory"""
        event_id = "trade_event_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
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
            
    async def get_event(self, n: Optional[int] = None) -> List[ChatEvent]:
        session_id = self.current_session_id
        if session_id and session_id in self.session_memory:
            return await self.session_memory[session_id].get_event(n=n)
        return []
    
    async def get_summary(self, n: Optional[int] = None) -> List[TradingSummary]:
        session_id = self.current_session_id
        if session_id and session_id in self.session_memory:
            return await self.session_memory[session_id].get_summary(n=n)
        return []
    
    async def get_insight(self, n: Optional[int] = None) -> List[TradingInsight]:
        session_id = self.current_session_id
        if session_id and session_id in self.session_memory:
            return await self.session_memory[session_id].get_insight(n=n)
        return []

