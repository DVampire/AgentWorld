"""
Memory system for recording optimizer optimization history and experiences.
Used for agent self-evolution by learning from past optimization experiences.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import json
import os
import uuid

from src.logger import logger
from src.utils import file_lock
from src.memory.types import Memory
from src.registry import MEMORY_SYSTEM


class OptimizationRecord(BaseModel):
    """Record of a single optimization step"""
    id: str = Field(default_factory=lambda: f"opt_{uuid.uuid4().hex[:8]}", description="Unique identifier for the optimization record")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the optimization")
    
    # Optimization context
    optimizer_type: str = Field(description="Type of optimizer (e.g., 'ReflectionOptimizer', 'GRPOTextualOptimizer')")
    agent_name: str = Field(description="Name of the agent being optimized")
    task: str = Field(description="Task description")
    optimization_step: int = Field(description="Step number in the optimization process")
    
    # Prompt variables
    variable_name: str = Field(description="Name of the prompt variable being optimized")
    variable_description: Optional[str] = Field(default=None, description="Description of the prompt variable")
    before_value: str = Field(description="Prompt value before optimization")
    after_value: str = Field(description="Prompt value after optimization")
    
    # Execution and reflection
    execution_result: Optional[str] = Field(default=None, description="Agent execution result")
    reflection_analysis: Optional[str] = Field(default=None, description="Reflection analysis (if applicable)")
    
    # Performance metrics
    reward: Optional[float] = Field(default=None, description="Reward value (if applicable)")
    loss: Optional[float] = Field(default=None, description="Loss value (if applicable)")
    advantage: Optional[float] = Field(default=None, description="Advantage value (if applicable)")
    kl_divergence: Optional[float] = Field(default=None, description="KL divergence (if applicable)")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def __str__(self):
        return f"OptimizationRecord(id={self.id}, optimizer={self.optimizer_type}, step={self.optimization_step}, variable={self.variable_name})"
    
    def __repr__(self):
        return self.__str__()


class OptimizationSession(BaseModel):
    """Session tracking a complete optimization run"""
    session_id: str = Field(description="Unique session identifier")
    optimizer_type: str = Field(description="Type of optimizer")
    agent_name: str = Field(description="Name of the agent")
    task: str = Field(description="Task description")
    start_time: datetime = Field(default_factory=datetime.now, description="Session start time")
    end_time: Optional[datetime] = Field(default=None, description="Session end time")
    total_steps: int = Field(default=0, description="Total optimization steps")
    records: List[OptimizationRecord] = Field(default_factory=list, description="Optimization records")
    
    def __str__(self):
        return f"OptimizationSession(id={self.session_id}, optimizer={self.optimizer_type}, steps={self.total_steps})"
    
    def __repr__(self):
        return self.__str__()


@MEMORY_SYSTEM.register_module(force=True)
class OptimizerMemorySystem(Memory):
    """Memory system for recording optimizer optimization history and experiences."""
    
    require_grad: bool = Field(default=False, description="Whether the memory system requires gradients")
    
    def __init__(self, 
                 base_dir: Optional[str] = None,
                 max_records_per_session: int = 1000,
                 require_grad: bool = False,
                 **kwargs):
        super().__init__(require_grad=require_grad, **kwargs)
        
        if base_dir is not None:
            self.base_dir = base_dir
        
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| Optimizer memory system base directory: {self.base_dir}")
        self.save_path = os.path.join(self.base_dir, "optimizer_memory.json")
        
        self.max_records_per_session = max_records_per_session
        
        # Storage
        self.sessions: Dict[str, OptimizationSession] = {}
        self.current_session_id: Optional[str] = None
        
        # Auto-load from JSON if file exists (will be done async if needed)
        # Note: We can't do async in __init__, so loading will happen on first async call
        self._load_pending = self.save_path and os.path.exists(self.save_path)
    
    async def _ensure_loaded(self):
        """Ensure memory is loaded from JSON if pending."""
        if hasattr(self, '_load_pending') and self._load_pending:
            await self.load_from_json(self.save_path)
            self._load_pending = False
    
    async def start_session(self,
                            session_id: str,
                            agent_name: Optional[str] = None,
                            task_id: Optional[str] = None,
                            description: Optional[str] = None,
                            optimizer_type: Optional[str] = None,
                            **kwargs) -> str:
        """Start a new optimization session.
        
        Args:
            session_id: Session ID
            agent_name: Name of the agent being optimized
            task_id: Optional task ID (can be used as task description)
            description: Optional description (used as task description if provided)
            optimizer_type: Type of optimizer (e.g., 'ReflectionOptimizer')
            **kwargs: Additional arguments
            
        Returns:
            Session ID
        """
        await self._ensure_loaded()
        
        # Use description or task_id as task
        task = description or task_id or ""
        
        # Get optimizer_type from kwargs if not provided
        if optimizer_type is None:
            optimizer_type = kwargs.get("optimizer_type", "UnknownOptimizer")
        
        session = OptimizationSession(
            session_id=session_id,
            optimizer_type=optimizer_type,
            agent_name=agent_name or "",
            task=task,
            start_time=datetime.now()
        )
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        logger.info(f"| 🚀 Started optimization session: {session_id} ({optimizer_type})")
        return session_id
    
    async def start_optimization_session(self,
                                   optimizer_type: str,
                                   agent_name: str,
                                   task: str,
                                   session_id: Optional[str] = None) -> str:
        """Start a new optimization session (deprecated, use start_session instead).
        
        Args:
            optimizer_type: Type of optimizer (e.g., 'ReflectionOptimizer')
            agent_name: Name of the agent being optimized
            task: Task description
            session_id: Optional session ID. If None, generates a new one.
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"opt_session_{uuid.uuid4().hex[:8]}"
        
        return await self.start_session(
            session_id=session_id,
            agent_name=agent_name,
            description=task,
            optimizer_type=optimizer_type
        )
    
    async def end_session(self, session_id: Optional[str] = None):
        """End an optimization session.
        
        Args:
            session_id: Optional session ID. If None, uses current session.
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.sessions:
            self.sessions[session_id].end_time = datetime.now()
            
            if session_id == self.current_session_id:
                self.current_session_id = None
            
            # Auto-save to JSON
            if self.save_path:
                await self.save_to_json(self.save_path)
            
            logger.info(f"| ✅ Ended optimization session: {session_id}")
    
    async def end_optimization_session(self, session_id: Optional[str] = None):
        """End an optimization session (deprecated, use end_session instead).
        
        Args:
            session_id: Optional session ID. If None, uses current session.
        """
        await self.end_session(session_id)
    
    async def add_event(self,
                        step_number: int,
                        event_type,
                        data: Any,
                        agent_name: str,
                        task_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        **kwargs) -> OptimizationRecord:
        """Add event to optimizer memory system.
        
        Args:
            step_number: Step number (optimization step)
            event_type: Event type (not used, kept for compatibility)
            data: Event data (dict containing optimization record fields)
            agent_name: Agent name
            task_id: Optional task ID
            session_id: Optional session ID. If None, uses current session.
            **kwargs: Additional arguments (can include optimization-specific fields)
            
        Returns:
            OptimizationRecord
        """
        # Extract optimization-specific fields from data or kwargs
        if isinstance(data, dict):
            variable_name = data.get("variable_name", kwargs.get("variable_name", ""))
            before_value = data.get("before_value", kwargs.get("before_value", ""))
            after_value = data.get("after_value", kwargs.get("after_value", ""))
            execution_result = data.get("execution_result", kwargs.get("execution_result"))
            reflection_analysis = data.get("reflection_analysis", kwargs.get("reflection_analysis"))
            variable_description = data.get("variable_description", kwargs.get("variable_description"))
            reward = data.get("reward", kwargs.get("reward"))
            loss = data.get("loss", kwargs.get("loss"))
            advantage = data.get("advantage", kwargs.get("advantage"))
            kl_divergence = data.get("kl_divergence", kwargs.get("kl_divergence"))
            metadata = data.get("metadata", kwargs.get("metadata"))
        else:
            # Fallback to kwargs if data is not a dict
            variable_name = kwargs.get("variable_name", "")
            before_value = kwargs.get("before_value", "")
            after_value = kwargs.get("after_value", "")
            execution_result = kwargs.get("execution_result")
            reflection_analysis = kwargs.get("reflection_analysis")
            variable_description = kwargs.get("variable_description")
            reward = kwargs.get("reward")
            loss = kwargs.get("loss")
            advantage = kwargs.get("advantage")
            kl_divergence = kwargs.get("kl_divergence")
            metadata = kwargs.get("metadata")
        
        optimization_step = step_number
        
        return await self.record_optimization(
            variable_name=variable_name,
            before_value=before_value,
            after_value=after_value,
            execution_result=execution_result,
            reflection_analysis=reflection_analysis,
            variable_description=variable_description,
            reward=reward,
            loss=loss,
            advantage=advantage,
            kl_divergence=kl_divergence,
            optimization_step=optimization_step,
            session_id=session_id,
            metadata=metadata
        )
    
    async def record_optimization(self,
                           variable_name: str,
                           before_value: str,
                           after_value: str,
                           execution_result: Optional[str] = None,
                           reflection_analysis: Optional[str] = None,
                           variable_description: Optional[str] = None,
                           reward: Optional[float] = None,
                           loss: Optional[float] = None,
                           advantage: Optional[float] = None,
                           kl_divergence: Optional[float] = None,
                           optimization_step: Optional[int] = None,
                           session_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> OptimizationRecord:
        """Record a single optimization step.
        
        Args:
            variable_name: Name of the prompt variable being optimized
            before_value: Prompt value before optimization
            after_value: Prompt value after optimization
            execution_result: Optional agent execution result
            reflection_analysis: Optional reflection analysis
            variable_description: Optional description of the variable
            reward: Optional reward value
            loss: Optional loss value
            advantage: Optional advantage value
            kl_divergence: Optional KL divergence value
            optimization_step: Optional step number. If None, auto-increments.
            session_id: Optional session ID. If None, uses current session.
            metadata: Optional additional metadata
            
        Returns:
            OptimizationRecord
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No active optimization session. Call start_session() first.")
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found.")
        
        session = self.sessions[session_id]
        
        # Auto-increment step if not provided
        if optimization_step is None:
            optimization_step = session.total_steps + 1
        
        # Create optimization record
        record = OptimizationRecord(
            optimizer_type=session.optimizer_type,
            agent_name=session.agent_name,
            task=session.task,
            optimization_step=optimization_step,
            variable_name=variable_name,
            variable_description=variable_description,
            before_value=before_value,
            after_value=after_value,
            execution_result=execution_result,
            reflection_analysis=reflection_analysis,
            reward=reward,
            loss=loss,
            advantage=advantage,
            kl_divergence=kl_divergence,
            metadata=metadata or {}
        )
        
        # Add to session
        session.records.append(record)
        session.total_steps = max(session.total_steps, optimization_step)
        
        # Limit records per session
        if len(session.records) > self.max_records_per_session:
            session.records = session.records[-self.max_records_per_session:]
        
        # Auto-save to JSON
        if self.save_path:
            await self.save_to_json(self.save_path)
        
        logger.debug(f"| 📝 Recorded optimization: {record.id} (step {optimization_step}, variable: {variable_name})")
        return record
    
    async def get_event(self, n: Optional[int] = None, session_id: Optional[str] = None) -> List[OptimizationRecord]:
        """Get events (optimization records) from memory system.
        
        Args:
            n: Number of events to retrieve. If None, returns all events.
            session_id: Optional session ID. If None, uses current session.
            
        Returns:
            List of OptimizationRecord
        """
        await self._ensure_loaded()
        
        records = []
        
        # Collect records from matching sessions
        target_sessions = []
        if session_id:
            if session_id in self.sessions:
                target_sessions = [session_id]
        else:
            if self.current_session_id:
                target_sessions = [self.current_session_id]
            else:
                target_sessions = list(self.sessions.keys())
        
        for sid in target_sessions:
            if sid in self.sessions:
                records.extend(self.sessions[sid].records)
        
        # Sort by timestamp (most recent first)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        
        # Apply limit
        if n is not None:
            records = records[:n]
        
        return records
    
    async def get_optimization_history(self,
                                agent_name: Optional[str] = None,
                                optimizer_type: Optional[str] = None,
                                task_keyword: Optional[str] = None,
                                variable_name: Optional[str] = None,
                                session_id: Optional[str] = None,
                                limit: Optional[int] = None) -> List[OptimizationRecord]:
        """Query optimization history (deprecated, use get_event instead).
        
        Args:
            agent_name: Filter by agent name
            optimizer_type: Filter by optimizer type
            task_keyword: Filter by task keyword (substring match)
            variable_name: Filter by variable name
            session_id: Filter by session ID
            limit: Maximum number of records to return
            
        Returns:
            List of OptimizationRecord matching the filters
        """
        await self._ensure_loaded()
        
        records = []
        
        # Collect records from matching sessions
        for sid, session in self.sessions.items():
            if session_id and sid != session_id:
                continue
            if agent_name and session.agent_name != agent_name:
                continue
            if optimizer_type and session.optimizer_type != optimizer_type:
                continue
            if task_keyword and task_keyword.lower() not in session.task.lower():
                continue
            
            # Filter records by variable name
            for record in session.records:
                if variable_name and record.variable_name != variable_name:
                    continue
                records.append(record)
        
        # Sort by timestamp (most recent first)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            records = records[:limit]
        
        return records
    
    async def get_best_practices(self,
                          agent_name: Optional[str] = None,
                          optimizer_type: Optional[str] = None,
                          variable_name: Optional[str] = None,
                          metric: str = "reward",
                          top_k: int = 10) -> List[OptimizationRecord]:
        """Get best practices based on performance metrics.
        
        Args:
            agent_name: Filter by agent name
            optimizer_type: Filter by optimizer type
            variable_name: Filter by variable name
            metric: Metric to use for ranking ('reward', 'loss', 'advantage')
            top_k: Number of top records to return
            
        Returns:
            List of top OptimizationRecord sorted by metric
        """
        records = await self.get_optimization_history(
            agent_name=agent_name,
            optimizer_type=optimizer_type,
            variable_name=variable_name
        )
        
        # Filter records with the metric
        valid_records = []
        for record in records:
            value = None
            if metric == "reward" and record.reward is not None:
                value = record.reward
            elif metric == "loss" and record.loss is not None:
                value = record.loss
            elif metric == "advantage" and record.advantage is not None:
                value = record.advantage
            
            if value is not None:
                valid_records.append((record, value))
        
        # Sort by metric value
        if metric == "loss":
            # For loss, lower is better
            valid_records.sort(key=lambda x: x[1])
        else:
            # For reward/advantage, higher is better
            valid_records.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return [record for record, _ in valid_records[:top_k]]
    
    async def get_session_info(self, session_id: Optional[str] = None) -> Optional[OptimizationSession]:
        """Get session information.
        
        Args:
            session_id: Optional session ID. If None, uses current session.
            
        Returns:
            OptimizationSession or None
        """
        await self._ensure_loaded()
        
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id:
            return self.sessions.get(session_id)
        return None
    
    async def clear_session(self, session_id: Optional[str] = None):
        """Clear a specific session.
        
        Args:
            session_id: Optional session ID. If None, uses current session.
        """
        await self._ensure_loaded()
        
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.sessions:
            del self.sessions[session_id]
            
            if session_id == self.current_session_id:
                self.current_session_id = None
            
            # Auto-save to JSON
            if self.save_path:
                await self.save_to_json(self.save_path)
            
            logger.info(f"| 🗑️ Cleared optimization session: {session_id}")
    
    async def clear(self):
        """Clear all sessions."""
        await self._ensure_loaded()
        
        self.sessions.clear()
        self.current_session_id = None
        
        # Auto-save to JSON
        if self.save_path:
            await self.save_to_json(self.save_path)
        
        logger.info("| 🗑️ Cleared all optimization sessions")
    
    async def save_to_json(self, file_path: str) -> str:
        """Save optimizer memory to JSON file.
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to the saved file
        """
        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare metadata
            metadata = {
                "memory_system_type": "optimizer_memory_system",
                "current_session_id": self.current_session_id,
                "session_ids": list(self.sessions.keys()),
                "total_sessions": len(self.sessions),
                "total_records": sum(len(session.records) for session in self.sessions.values())
            }
            
            # Prepare sessions data
            sessions_data = {}
            for session_id, session in self.sessions.items():
                sessions_data[session_id] = {
                    "session_info": session.model_dump(mode="json", exclude={"records"}),
                    "records": [record.model_dump(mode="json") for record in session.records]
                }
            
            # Prepare save data
            save_data = {
                "metadata": metadata,
                "sessions": sessions_data
            }
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"| 💾 Optimizer memory saved to {file_path}")
            return str(file_path)
    
    async def load_from_json(self, file_path: str) -> bool:
        """Load optimizer memory from JSON file.
        
        Args:
            file_path: File path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Optimizer memory file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                # Validate format
                if "metadata" not in load_data or "sessions" not in load_data:
                    raise ValueError(
                        f"Invalid optimizer memory format. Expected {{'metadata': {{...}}, 'sessions': {{...}}}}, "
                        f"got keys: {list(load_data.keys())}"
                    )
                
                # Restore metadata
                metadata = load_data.get("metadata", {})
                self.current_session_id = metadata.get("current_session_id")
                
                # Restore sessions
                sessions_data = load_data.get("sessions", {})
                logger.debug(f"| 📊 Restoring {len(sessions_data)} optimization sessions from JSON")
                
                for session_id, session_data in sessions_data.items():
                    # Restore session info
                    session_info_data = session_data.get("session_info", {})
                    if session_info_data.get("start_time"):
                        session_info_data["start_time"] = datetime.fromisoformat(session_info_data["start_time"])
                    if session_info_data.get("end_time"):
                        session_info_data["end_time"] = datetime.fromisoformat(session_info_data["end_time"])
                    
                    session = OptimizationSession(**session_info_data)
                    
                    # Restore records
                    records_data = session_data.get("records", [])
                    for record_data in records_data:
                        if record_data.get("timestamp"):
                            record_data["timestamp"] = datetime.fromisoformat(record_data["timestamp"])
                        session.records.append(OptimizationRecord(**record_data))
                    
                    self.sessions[session_id] = session
                
                logger.info(f"| 📂 Optimizer memory loaded from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load optimizer memory from {file_path}: {e}", exc_info=True)
                return False

