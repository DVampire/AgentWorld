"""ACP Context Manager

Manages agent contexts, their states, and lifecycle operations.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import uuid

from src.agents.protocol.types import (
    AgentContext, ContextState, ContextAction, ACPContextInfo,
    ACPError, ACPErrorCode
)
from src.logger import logger


class ACPContextManager:
    """Manages agent contexts and their lifecycle"""
    
    def __init__(self, max_contexts: int = 100, context_lifetime: Optional[float] = None):
        self.max_contexts = max_contexts
        self.context_lifetime = context_lifetime  # seconds
        self._contexts: Dict[str, AgentContext] = {}
        self._context_states: Dict[str, ContextState] = {}
        self._active_sessions: Set[str] = set()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Start cleanup task if lifetime is set
        if self.context_lifetime:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_contexts())
    
    async def create_context(
        self, 
        agent_id: str,
        agent_name: str,
        agent_type: str,
        session_id: Optional[str] = None,
        available_environments: Optional[List[str]] = None,
        available_tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """Create a new agent context"""
        
        if len(self._contexts) >= self.max_contexts:
            raise ACPError(
                code=ACPErrorCode.INTERNAL_ERROR,
                message=f"Maximum number of contexts ({self.max_contexts}) reached"
            )
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id in self._active_sessions:
            raise ACPError(
                code=ACPErrorCode.INTERNAL_ERROR,
                message=f"Session {session_id} already has an active context"
            )
        
        # Create context state
        state = ContextState(
            agent_id=agent_id,
            session_id=session_id,
            status="initializing",
            metadata=metadata
        )
        
        # Create agent context
        context = AgentContext(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            session_id=session_id,
            state=state,
            available_environments=available_environments or [],
            available_tools=available_tools or [],
            metadata=metadata
        )
        
        # Store context
        self._contexts[agent_id] = context
        self._context_states[agent_id] = state
        self._active_sessions.add(session_id)
        
        logger.info(f"Created ACP context for agent {agent_name} (ID: {agent_id})")
        return context
    
    async def get_context(self, agent_id: str) -> Optional[AgentContext]:
        """Get agent context by ID"""
        return self._contexts.get(agent_id)
    
    async def get_context_by_session(self, session_id: str) -> Optional[AgentContext]:
        """Get agent context by session ID"""
        for context in self._contexts.values():
            if context.session_id == session_id:
                return context
        return None
    
    async def update_context_state(
        self, 
        agent_id: str, 
        status: Optional[str] = None,
        current_task: Optional[str] = None,
        step_number: Optional[int] = None,
        last_action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update context state"""
        
        context = self._contexts.get(agent_id)
        if not context:
            return False
        
        state = self._context_states.get(agent_id)
        if not state:
            return False
        
        # Update state fields
        if status is not None:
            state.status = status
        if current_task is not None:
            state.current_task = current_task
        if step_number is not None:
            state.step_number = step_number
        if last_action is not None:
            state.last_action = last_action
            state.last_action_time = datetime.now()
            state.action_count += 1
        if metadata is not None:
            state.metadata = metadata or {}
        
        # Update context
        context.state = state
        context.updated_at = datetime.now()
        
        return True
    
    async def execute_context_action(
        self, 
        agent_id: str, 
        action: ContextAction
    ) -> Dict[str, Any]:
        """Execute an action within a context"""
        
        context = self._contexts.get(agent_id)
        if not context:
            raise ACPError(
                code=ACPErrorCode.CONTEXT_NOT_FOUND,
                message=f"Context not found for agent {agent_id}"
            )
        
        # Update context state
        await self.update_context_state(
            agent_id=agent_id,
            status="running",
            last_action=action.operation
        )
        
        try:
            # Execute action based on type
            if action.action_type == "ecp":
                result = await self._execute_ecp_action(context, action)
            elif action.action_type == "mcp":
                result = await self._execute_mcp_action(context, action)
            elif action.action_type == "internal":
                result = await self._execute_internal_action(context, action)
            elif action.action_type == "bridge":
                result = await self._execute_bridge_action(context, action)
            else:
                raise ACPError(
                    code=ACPErrorCode.INVALID_PARAMS,
                    message=f"Unknown action type: {action.action_type}"
                )
            
            # Update context state on success
            await self.update_context_state(
                agent_id=agent_id,
                status="ready"
            )
            
            return result
            
        except Exception as e:
            # Update context state on error
            await self.update_context_state(
                agent_id=agent_id,
                status="error"
            )
            
            # Increment error count
            state = self._context_states.get(agent_id)
            if state:
                state.error_count += 1
            
            raise ACPError(
                code=ACPErrorCode.ACTION_EXECUTION_ERROR,
                message=f"Action execution failed: {str(e)}"
            )
    
    async def _execute_ecp_action(self, context: AgentContext, action: ContextAction) -> Dict[str, Any]:
        """Execute ECP action"""
        # TODO: Implement ECP action execution
        return {"action_type": "ecp", "operation": action.operation, "status": "not_implemented"}
    
    async def _execute_mcp_action(self, context: AgentContext, action: ContextAction) -> Dict[str, Any]:
        """Execute MCP action"""
        # TODO: Implement MCP action execution
        return {"action_type": "mcp", "operation": action.operation, "status": "not_implemented"}
    
    async def _execute_internal_action(self, context: AgentContext, action: ContextAction) -> Dict[str, Any]:
        """Execute internal ACP action"""
        if action.operation == "get_state":
            return {
                "context_state": context.state.dict(),
                "available_environments": context.available_environments,
                "available_tools": context.available_tools
            }
        elif action.operation == "update_metadata":
            if action.args.get("metadata"):
                context.metadata = action.args["metadata"]
                context.updated_at = datetime.now()
            return {"status": "updated", "metadata": context.metadata}
        else:
            raise ACPError(
                code=ACPErrorCode.METHOD_NOT_FOUND,
                message=f"Unknown internal action: {action.operation}"
            )
    
    async def _execute_bridge_action(self, context: AgentContext, action: ContextAction) -> Dict[str, Any]:
        """Execute bridge action"""
        # TODO: Implement bridge action execution
        return {"action_type": "bridge", "operation": action.operation, "status": "not_implemented"}
    
    async def destroy_context(self, agent_id: str) -> bool:
        """Destroy an agent context"""
        
        context = self._contexts.get(agent_id)
        if not context:
            return False
        
        # Remove from active sessions
        self._active_sessions.discard(context.session_id)
        
        # Remove context and state
        del self._contexts[agent_id]
        if agent_id in self._context_states:
            del self._context_states[agent_id]
        
        logger.info(f"Destroyed ACP context for agent {agent_id}")
        return True
    
    async def list_contexts(self) -> List[ACPContextInfo]:
        """List all active contexts"""
        contexts_info = []
        
        for context in self._contexts.values():
            state = self._context_states.get(context.agent_id)
            info = ACPContextInfo(
                context_id=context.agent_id,
                agent_id=context.agent_id,
                agent_name=context.agent_name,
                status=state.status if state else "unknown",
                available_actions=["get_state", "update_metadata", "execute_action"],
                environment_count=len(context.available_environments),
                tool_count=len(context.available_tools),
                last_activity=state.last_action_time if state else context.updated_at,
                metadata=context.metadata
            )
            contexts_info.append(info)
        
        return contexts_info
    
    async def _cleanup_expired_contexts(self):
        """Cleanup expired contexts"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.context_lifetime:
                    continue
                
                current_time = datetime.now()
                expired_agents = []
                
                for agent_id, context in self._contexts.items():
                    if context.updated_at + timedelta(seconds=self.context_lifetime) < current_time:
                        expired_agents.append(agent_id)
                
                for agent_id in expired_agents:
                    await self.destroy_context(agent_id)
                    logger.info(f"Cleaned up expired context for agent {agent_id}")
                    
            except Exception as e:
                logger.error(f"Error in context cleanup: {e}")
    
    async def shutdown(self):
        """Shutdown the context manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all contexts
        self._contexts.clear()
        self._context_states.clear()
        self._active_sessions.clear()
        
        logger.info("ACP Context Manager shutdown complete")
