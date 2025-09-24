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


class TransformationRequest(BaseModel):
    """Request for protocol transformation."""
    
    transformation_type: TransformationType = Field(description="Type of transformation to perform")
    source_protocol: ProtocolType = Field(description="Source protocol type")
    target_protocol: ProtocolType = Field(description="Target protocol type")
    source_identifiers: List[str] = Field(description="Identifiers in source protocol (tool names, env names, agent names)")
    target_name: Optional[str] = Field(default=None, description="Name for the target resource")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Transformation configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TransformationResponse(BaseModel):
    """Response from protocol transformation."""
    
    success: bool = Field(description="Whether the transformation was successful")
    target_identifiers: List[str] = Field(description="Identifiers in target protocol")
    transformation_id: Optional[str] = Field(default=None, description="Unique ID for this transformation")
    error_message: Optional[str] = Field(default=None, description="Error message if transformation failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Transformation metadata")


class TransformationError(Exception):
    """Exception raised during transformation."""
    
    def __init__(self, message: str, transformation_type: Optional[TransformationType] = None):
        self.message = message
        self.transformation_type = transformation_type
        super().__init__(self.message)


class ProtocolMapping(BaseModel):
    """Mapping between protocols."""
    
    source_protocol: ProtocolType = Field(description="Source protocol")
    target_protocol: ProtocolType = Field(description="Target protocol")
    source_identifier: str = Field(description="Source identifier")
    target_identifier: str = Field(description="Target identifier")
    mapping_type: str = Field(description="Type of mapping relationship")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Mapping metadata")


class TransformationConfig(BaseModel):
    """Configuration for transformations."""
    
    enable_async: bool = Field(default=True, description="Enable async execution")
    timeout: Optional[float] = Field(default=None, description="Timeout for operations")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    error_handling: str = Field(default="strict", description="Error handling strategy")
    auto_cleanup: bool = Field(default=True, description="Automatically cleanup resources")
    logging_level: str = Field(default="INFO", description="Logging level for transformations")
