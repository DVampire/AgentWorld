"""Environment Context Protocol (ECP) Types

Core type definitions for the Environment Context Protocol.
"""
from typing import Any, Dict, Optional, Union, Type, Callable
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from pydantic import BaseModel, Field, ConfigDict


class Environment(BaseModel):
    """Base abstract class for ECP environments"""
    
    name: str = Field(description="The name of the environment.")
    type: str = Field(description="The type of the environment.")
    args_schema: Type[BaseModel] = Field(description="The args schema of the environment.")
    description: str = Field(description="The description of the environment.")
    metadata: Dict[str, Any] = Field(description="The metadata of the environment.")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        extra="allow"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the state of the environment"""
        raise NotImplementedError("Get state method not implemented")

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

class ActionConfig(BaseModel):
    """Action configuration (equivalent to MCP tool)"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    env_name: str = Field(description="The name of the environment this action belongs to")
    name: str = Field(description="The name of the action")
    type: str = Field(description="The type of the action")
    description: str = Field(description="The description of the action")
    args_schema: Optional[Type[BaseModel]] = Field(default=None, description="The args schema of the action")
    function: Optional[Callable] = Field(default=None, description="The function implementing the action")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the action")
    
    def __str__(self):
        return f"ActionConfig(env_name={self.env_name}, name={self.name}, type={self.type}, description={self.description})"
    
    def __repr__(self):
        return self.__str__()


class EnvironmentConfig(BaseModel):
    """Environment configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the environment")
    type: str = Field(description="The type of the environment")
    rules: str = Field(description="The rules of the environment")
    description: str = Field(description="The description of the environment")
    args_schema: Optional[Type[BaseModel]] = Field(default=None, description="The args schema of the environment")
    actions: Dict[str, ActionConfig] = Field(default_factory=dict, description="Dictionary of actions available in this environment")
    cls: Optional[Type[Environment]] = Field(default=None, description="The class of the environment")
    config: Optional[Dict[str, Any]] = Field(default={}, description="The initialization configuration of the environment")
    instance: Optional[Any] = Field(default=None, description="The instance of the environment")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the environment")
    
    def __str__(self):
        return f"EnvironmentConfig(name={self.name}, type={self.type}, description={self.description}, actions={len(self.actions)})"
    
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
    success: bool = Field(default=True, description="Whether the action was successful")
    message: str = Field(default="Action result", description="The message of the action result")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra information of the action result")
    
class EnvironmentState(BaseModel):
    """Environment state"""
    state: str = Field(default="State", description="The state of the environment")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra information of the state")

# Update forward references
EnvironmentConfig.model_rebuild()
ActionConfig.model_rebuild()
ScreenshotInfo.model_rebuild()
ActionResult.model_rebuild()
EnvironmentState.model_rebuild()
