"""Agent Context Protocol (ACP) Types

Core type definitions for the Agent Context Protocol.
"""

import json
from typing import Any, Dict, List, Optional, Union, Literal, Type, Callable, Set
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime

from src.environments.protocol.types import EnvironmentInfo, ActionInfo, ActionResult
from src.tools.protocol.types import ToolInfo


class ACPErrorCode(Enum):
    """ACP error codes"""
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AGENT_NOT_FOUND = -32001
    CONTEXT_NOT_FOUND = -32002
    ACTION_EXECUTION_ERROR = -32003
    BRIDGE_ERROR = -32004
    TOOL_ORCHESTRATION_ERROR = -32005


class ACPError(BaseModel):
    """ACP error structure"""
    code: ACPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None


class ACPRequest(BaseModel):
    """ACP request structure"""
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None


class ACPResponse(BaseModel):
    """ACP response structure"""
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ACPError] = None


class ACPNotification(BaseModel):
    """ACP notification structure"""
    method: str
    params: Optional[Dict[str, Any]] = None


class ContextState(BaseModel):
    """Agent context state"""
    agent_id: str
    session_id: str
    status: Literal["initializing", "ready", "running", "paused", "error", "shutdown"]
    current_task: Optional[str] = None
    step_number: int = 0
    max_steps: int = 20
    last_action: Optional[str] = None
    last_action_time: Optional[datetime] = None
    action_count: int = 0
    error_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ContextAction(BaseModel):
    """Context action definition"""
    action_type: Literal["ecp", "mcp", "internal", "bridge"]
    operation: str
    args: Dict[str, Any] = Field(default_factory=dict)
    target_agent: Optional[str] = None
    target_environment: Optional[str] = None
    target_tool: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentContext(BaseModel):
    """Agent context information"""
    agent_id: str
    agent_name: str
    agent_type: str
    session_id: str
    state: ContextState
    available_environments: List[str] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    memory_context: Optional[Dict[str, Any]] = None
    environment_context: Optional[Dict[str, Any]] = None
    tool_context: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None




class ACPCapabilities(BaseModel):
    """ACP server capabilities"""
    protocol_version: str = "1.0.0"
    supported_protocols: List[str] = Field(default_factory=lambda: ["ecp", "mcp"])
    supported_actions: List[str] = Field(default_factory=lambda: [
        "create_context", "destroy_context", "execute_action", 
        "bridge_protocol", "orchestrate_tools", "get_state"
    ])
    max_concurrent_contexts: int = 100
    max_context_lifetime: Optional[float] = None
    features: List[str] = Field(default_factory=lambda: [
        "context_persistence", "protocol_bridging", "tool_orchestration",
        "environment_integration", "mcp_wrapper"
    ])


class ACPContextInfo(BaseModel):
    """ACP context information for external access"""
    context_id: str
    agent_id: str
    agent_name: str
    status: str
    available_actions: List[str] = Field(default_factory=list)
    environment_count: int = 0
    tool_count: int = 0
    last_activity: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ACPToolInfo(BaseModel):
    """ACP tool information (for MCP wrapper)"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    category: str = "acp"
    protocol: Literal["ecp", "mcp", "acp"]
    requires_context: bool = True
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    """Agent information for registration"""
    name: str
    type: str
    description: str
    cls: Optional[Any] = None
    instance: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    def __str__(self):
        return f"AgentInfo(name={self.name}, type={self.type}, description={self.description})"
    
    def __repr__(self):
        return self.__str__()


# Update forward references
AgentContext.model_rebuild()
ContextState.model_rebuild()
ContextAction.model_rebuild()
AgentInfo.model_rebuild()
