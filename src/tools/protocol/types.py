"""Tool Context Protocol (TCP) Types

Core type definitions for the Tool Context Protocol.
"""

import json
from typing import Any, Dict, List, Optional, Union, Literal, Type, Callable
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime

class ToolResponse(BaseModel):
    content: str = Field(description="The content of the tool response.")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra data of the tool response.")
    
    def __str__(self) -> str:
        return f"ToolResponse(content={self.content}, extra={self.extra})"
    
    def __repr__(self) -> str:
        return self.__str__()

class TCPErrorCode(Enum):
    """TCP error codes"""
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TOOL_NOT_FOUND = -32001
    CAPABILITY_NOT_FOUND = -32002
    EXECUTION_ERROR = -32003


class TCPError(BaseModel):
    """TCP error structure"""
    code: TCPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None


class TCPRequest(BaseModel):
    """TCP request structure"""
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None


class TCPResponse(BaseModel):
    """TCP response structure"""
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[TCPError] = None


class TCPNotification(BaseModel):
    """TCP notification structure"""
    method: str
    params: Optional[Dict[str, Any]] = None


class ToolInfo(BaseModel):
    """Tool information for registration"""
    name: str
    type: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    cls: Optional[Type[Any]] = None  # For lazy instantiation
    instance: Optional[Any] = None
    
    def __str__(self):
        schema = self.args_schema.model_json_schema()
        return json.dumps(schema, indent=4)
    
    def __repr__(self):
        return self.__str__()




class ToolExecutionResult(BaseModel):
    """Tool execution result"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class TCPCapabilities(BaseModel):
    """TCP server capabilities"""
    protocol_version: str = "1.0.0"
    supported_capabilities: List[str] = Field(default_factory=lambda: [
        "function", "action", "query", "transform"
    ])
    supported_operations: List[str] = Field(default_factory=lambda: [
        "register_tool", "list_tools", "execute_tool", "get_tool_info"
    ])
    max_concurrent_executions: int = 100
    features: List[str] = Field(default_factory=lambda: [
        "tool_registration", "capability_management", "execution_tracking"
    ])


# Update forward references
ToolInfo.model_rebuild()
