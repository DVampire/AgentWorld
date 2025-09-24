"""Type definitions for transformation protocols."""

from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class ProtocolType(str, Enum):
    """Supported protocol types."""
    ECP = "ecp"  # Environment Context Protocol
    TCP = "tcp"  # Tool Context Protocol  
    ACP = "acp"  # Agent Context Protocol

class TransformationType(str, Enum):
    """Types of transformations."""
    T2E = "t2e"  # TCP to ECP - Convert TCP tools to ECP environment
    T2A = "t2a"  # TCP to ACP - Provide TCP tools to ACP agent
    E2T = "e2t"  # ECP to TCP - Convert ECP environment to TCP tools
    E2A = "e2a"  # ECP to ACP - Convert ECP environment to ACP agent
    A2T = "a2t"  # ACP to TCP - Convert ACP agent to TCP tools
    A2E = "a2e"  # ACP to ECP - Convert ACP agent to ECP environment
    
class E2TRequest(BaseModel):
    """ECP to TCP transformation request."""
    type: str = Field(default="e2t", description="The type of transformation.")
    env_names: List[str] = Field(default=[], description="The names of the environments to transform.")
    
class E2TResponse(BaseModel):
    """ECP to TCP transformation response."""
    success: bool = Field(default=True, description="Whether the transformation was successful.")
    message: str = Field(default="", description="The message of the transformation.")