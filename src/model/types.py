from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, ConfigDict, Field

class ModelConfig(BaseModel):
    """Configuration container describing a single LLM/provider pairing."""

    model_name: str = Field(description="Human-readable name used across the codebase.")
    model_type: str = Field(description="Model type, e.g. 'chat/completions', 'responses', 'embeddings'.")
    model_id: str = Field(description="Provider-specific identifier passed to the API.")
    provider: str = Field(description="Provider slug, e.g. 'openai', 'anthropic'.")
    api_base: Optional[str] = Field(default=None, description="Override API base URL.")
    api_key: Optional[str] = Field(default=None, description="Override API key.")
    temperature: Optional[float] = Field(default=None, description="Temperature parameter for the model.")
    reasoning: Optional[Dict[str, Any]] = Field(default={
        "reasoning_effort": "high"
    }, description="Reasoning configuration.")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum completion tokens for chat/completions models.")
    max_output_tokens: Optional[int] = Field(default=None, description="Maximum output tokens for responses API models.")
    supports_streaming: bool = Field(default=True, description="Whether streaming is supported.")
    supports_functions: bool = Field(default=False, description="Whether tool/function calling is supported.")
    supports_vision: bool = Field(default=False, description="Whether multimodal inputs are supported.")
    output_version: Optional[str] = Field(
        default=None,
        description="Optional output schema version when required by provider.",
    )
    fallback_model: Optional[str] = Field(
        default=None,
        description="Fallback model name to use if the primary model fails due to policy/content filter errors.",
    )

class LLMResponse(BaseModel):
    """
    Wrapper for LLM responses that normalizes output from different APIs.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    success: bool = Field(description="Whether the model call was successful")
    message: Union[str, BaseModel] = Field(description="The message from the model call")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="The extra data from the model call")

__all__ = ["ModelConfig", "LLMResponse"]

