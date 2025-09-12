"""Memory manager for session-based memory operations."""
from typing import Optional, Any

from src.memory.memory_system import MemorySystem
from src.logger import logger

class MemoryManager:
    """Session-based memory manager for agent execution history."""
    def __init__(self, **memory_config):
        # MemorySystem can work as basic memory store or with LangChain enhancements
        self.memory_system = MemorySystem(**memory_config)
    
    async def start_session(self, 
                            session_id: str, 
                            agent_name: Optional[str] = None, 
                            task_id: Optional[str] = None, 
                            description: Optional[str] = None
                            ):
        await self.memory_system.start_session(session_id, agent_name, task_id, description)
    
    async def end_session(self,
                          session_id: Optional[str] = None
                          ):
        await self.memory_system.end_session(session_id)
    
    async def add_event(self, 
                  step_number: int,
                  event_type,
                  data: Any,
                  agent_name: str,
                  task_id: Optional[str] = None,
                  **kwargs):
        
        await self.memory_system.add_event(step_number,
                                           event_type, 
                                           data, 
                                           agent_name,
                                           task_id,
                                           **kwargs)
        logger.info(f"| Added event successfully.")

    
    async def get_event(self, n: Optional[int] = None):
        return await self.memory_system.get_event(n=n)
    
    async def get_state(self, n: Optional[int] = None):
        
        state = dict()
        events = await self.memory_system.get_event(n=n)
        summaries = await self.memory_system.get_summary(n=n)
        insights = await self.memory_system.get_insight(n=n)
        
        state["events"] = events
        state["summaries"] = summaries
        state["insights"] = insights
        
        logger.info(f"| Get memory state successfully.")
        
        return state