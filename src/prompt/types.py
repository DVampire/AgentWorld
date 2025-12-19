"""Prompt Context Protocol (PCP) Types

Core type definitions for the Prompt Context Protocol.
"""

from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field, ConfigDict

class Prompt(BaseModel):
    """Base class for all prompt templates"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the prompt")
    description: str = Field(description="The description of the prompt")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the prompt")

    def __str__(self):
        return f"Prompt(name={self.name}, description={self.description}, metadata={self.metadata})"

    def __repr__(self):
        return self.__str__()


class PromptConfig(BaseModel):
    """Prompt configuration for registration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the prompt")
    type: str = Field(description="The type of the prompt")
    description: str = Field(description="The description of the prompt")
    version: str = Field(default="1.0.0", description="Version of the prompt")
    template: str = Field(description="The template string for the prompt")
    variables: Optional[list] = Field(default=None, description="The variables used in the template")
    cls: Optional[Type[Prompt]] = Field(default=None, description="The class of the prompt")
    instance: Optional[Any] = Field(default=None, description="The instance of the prompt")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The initialization configuration of the prompt")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the prompt")
    
    def __str__(self):
        return f"PromptConfig(name={self.name}, type={self.type}, description={self.description}, version={self.version})"
    
    def __repr__(self):
        return self.__str__()
