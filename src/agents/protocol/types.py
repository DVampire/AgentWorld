"""Agent Context Protocol (ACP) Types

Core type definitions for the Agent Context Protocol.
"""

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class ACPErrorCode(Enum):
    """ACP error codes"""
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AGENT_NOT_FOUND = -32001

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
