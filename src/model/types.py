from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, ConfigDict, Field

class ModelConfig(BaseModel):
    """Configuration container describing a single LLM/provider pairing."""

    model_name: str = Field(description="Human-readable name used across the codebase.")
    model_id: str = Field(description="Provider-specific identifier passed to the API.")
    provider: str = Field(description="Provider slug, e.g. 'openai', 'anthropic'.")
    model_type: Literal["completion", "embedding"] = Field(
        default="completion",
        description="Type of model: 'completion' for chat/completion models, 'embedding' for embedding models."
    )
    api_base: Optional[str] = Field(default=None, description="Override API base URL.")
    api_key: Optional[str] = Field(default=None, description="Override API key.")
    default_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default kwargs merged into every invocation.",
    )
    supports_streaming: bool = Field(default=True, description="Whether streaming is supported.")
    supports_functions: bool = Field(default=False, description="Whether tool/function calling is supported.")
    supports_vision: bool = Field(default=False, description="Whether multimodal inputs are supported.")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens allowed per completion.")
    # Indicates whether the provider requires the responses API (e.g., GPT-5)
    use_responses_api: bool = Field(
        default=False,
        description="Flag to force responses API usage (e.g., GPT-5/o-series).",
    )
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

