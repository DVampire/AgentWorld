"""Environment Context Protocol (ECP) Types

Core type definitions for the Environment Context Protocol.
"""
from typing import Any, Dict, Optional, Union, Literal, Type, Callable
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime

from src.environments.protocol.environment import BaseEnvironment

class ECPErrorCode(Enum):
    """ECP error codes"""
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    ENVIRONMENT_NOT_FOUND = -32001
    ACTION_NOT_FOUND = -32002
    ACTION_EXECUTION_ERROR = -32003


class ECPError(BaseModel):
    """ECP error structure"""
    code: ECPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None


class ECPRequest(BaseModel):
    """ECP request structure"""
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None


class ECPResponse(BaseModel):
    """ECP response structure"""
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ECPError] = None


class ECPNotification(BaseModel):
    """ECP notification structure"""
    method: str
    params: Optional[Dict[str, Any]] = None

class EnvironmentInfo(BaseModel):
    """Environment information"""
    name: str
    type: str
    rules: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    actions: Dict[str, "ActionInfo"]
    cls: Optional[Type[BaseEnvironment]] = None
    instance: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    def __str__(self):
        return f"EnvironmentInfo(name={self.name}, type={self.type}, description={self.description})"
    
    def __repr__(self):
        return self.__str__()


class ActionInfo(BaseModel):
    """Action information (equivalent to MCP tool)"""
    env_name: str
    name: str
    type: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    function: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    def __str__(self):
        return f"ActionInfo(env_name={self.env_name}, name={self.name}, type={self.type}, description={self.description})"
    
    def __repr__(self):
        return self.__str__()
    
class ScreenshotInfo(BaseModel):
    """Screenshot information"""
    transformed: bool = Field(default=False, description="Whether the screenshot has been transformed")
    screenshot: str = Field(default="Screenshot base64")
    screenshot_path: str = Field(default="Screenshot path")
    screenshot_description: str = Field(default="Screenshot description")
    transform_info: Optional[Dict[str, Any]] = Field(default=None, description="Transform information")

class ActionResult(BaseModel):
    """Action result"""
    success: bool = Field(default=True)
    message: str = Field(default="Action result")
    screenshot: str = Field(default="Screenshot base64")
    screenshot_path: str = Field(default="Screenshot path")
    screenshot_description: str = Field(default="Screenshot description")

# Update forward references
EnvironmentInfo.model_rebuild()
ActionInfo.model_rebuild()
ScreenshotInfo.model_rebuild()
ActionResult.model_rebuild()
