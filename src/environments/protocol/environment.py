"""
Abstract base class for environments in the Environment Context Protocol.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Type

class BaseEnvironment(BaseModel):
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
