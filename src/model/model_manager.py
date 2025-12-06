"""
Lightweight model manager inspired by LiteLLM.
Provides a unified interface for registering, identifying, and invoking models.
Supports a single API surface (model, messages, tools, response_format).
"""

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Union, Type

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.model.types import ModelConfig, LLMResponse
from src.model.completion_handler import CompletionHandler
from src.model.api_request import APIRequest
from src.logger import logger


class ModelManager:
    """
    Central registry and invoker for heterogeneous LLM providers.

    Responsibilities:
    1. Register and store model configurations.
    2. Provide a unified chat-style invocation interface.
    """
    
    def __init__(self):
        """
        Initialize the manager.
        """
        self.models: Dict[str, ModelConfig] = {}
        self.model_aliases: Dict[str, str] = {}
        
        # Supported providers
        self.providers: List[str] = [
            "openai",
            "anthropic",
            "google", 
            "deepseek",
            "openrouter",
        ]
        
        # Initialize handlers (will be set after models are initialized)
        self.completion_handler: Optional[CompletionHandler] = None
        self.max_tokens: int = 16384
        self.default_temperature: float = 0.7
    
    async def initialize(self):
        self._initialize_default_models()
        # Initialize handlers with models
        self.completion_handler = CompletionHandler(self.models)
        logger.info(f"| Model manager initialized successfully.")
    
    def _initialize_default_models(self):
        self._initialize_openrouter_models()
        self._initialize_embedding_models()
            
    def _initialize_openrouter_models(self):     
        # General models
        models = [
            # OpenAI models for chat, vision.
            {
                "model_name": "openrouter/gpt-4o",
                "model_id": "openai/gpt-4o",
            },
            {
                "model_name": "openrouter/gpt-4.1",
                "model_id": "openai/gpt-4.1",
            },
            {
                "model_name": "openrouter/gpt-5",
                "model_id": "openai/gpt-5",
            },
            {
                "model_name": "openrouter/gpt-5.1",
                "model_id": "openai/gpt-5.1",
            },
            {
                "model_name": "openrouter/o3",
                "model_id": "openai/o3",
            },
            # Gemini models for chat, vision, audio, video, pdf, etc.
            {
                "model_name": "openrouter/gemini-2.5-flash",
                "model_id": "google/gemini-2.5-flash",
            },
            {
                "model_name": "openrouter/gemini-2.5-pro",
                "model_id": "google/gemini-2.5-pro",
            },
            # Anthropic models for chat, vision.
            {
                "model_name": "openrouter/claude-4.5-sonnet",
                "model_id": "anthropic/claude-4.5-sonnet",
            }
        ]
        for model in models:
            # O-series models only support temperature=1
            model_name = model["model_name"]
            temperature = 1.0 if model_name.startswith("o") else self.default_temperature
            
            # Set fallback models
            fallback_model = "openrouter/o3"
            
            self.models[model_name] = ModelConfig(
                model_name=model_name,
                model_id=model["model_id"],
                provider="openrouter",
                api_base=os.getenv("OPENROUTER_API_BASE"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                default_params={
                    "temperature": temperature,
                },
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                max_tokens=self.max_tokens,
                use_responses_api=False,
                output_version=None,
                fallback_model=fallback_model,
            )
            
        # Deep research models
        models = [
            {
                "model_name": "openrouter/o4-mini-deep-research",
                "model_id": "openai/o4-mini-deep-research",
            },
            {
                "model_name": "openrouter/o3-deep-research",
                "model_id": "openai/o3-deep-research",
            },
            {
                "model_name": "openrouter/sonar-deep-research",
                "model_id": "perplexity/sonar-deep-research",
            }
        ]
        for model in models:
            # O-series models only support temperature=1
            model_name = model["model_name"]
            temperature = self.default_temperature
            
            # Set fallback models
            fallback_model = "o3-deep-research"
            
            self.models[model_name] = ModelConfig(
                model_name=model_name,
                model_id=model["model_id"],
                provider="openrouter",
                api_base=os.getenv("OPENROUTER_API_BASE"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                default_params={
                    "temperature": temperature,
                },
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                max_tokens=self.max_tokens,
                use_responses_api=False,
                output_version=None,
                fallback_model=fallback_model,
            )
            
        # Transcribe models
        models = [
            {
                "model_name": "openrouter/gpt-4o-transcribe",
                "model_id": "openai/gpt-4o-audio-preview",
            }
        ]
        for model in models:
            model_name = model["model_name"]
            
            temperature = 1.0 if model_name.startswith("o") else self.default_temperature
            
            # Set fallback models
            fallback_model = "openrouter/gpt-4o-transcribe"
            
            self.models[model_name] = ModelConfig(
                model_name=model_name,
                model_id=model["model_id"],
                provider="openrouter",
                api_base=os.getenv("OPENROUTER_API_BASE"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                default_params={
                    "temperature": temperature,
                },
                supports_streaming=True,
                supports_functions=False,
                supports_vision=False,
                max_tokens=self.max_tokens,
                use_responses_api=False,
                output_version=None,
                fallback_model=fallback_model,
            )
        
        # Code models
        models = [
            {
                "model_name": "openrouter/gpt-5-codex",
                "model_id": "openai/gpt-5-codex",
            },
            {
                "model_name": "openrouter/gpt-5.1-codex",
                "model_id": "openai/gpt-5.1-codex",
            }
        ]
        for model in models:
            model_name = model["model_name"]
            temperature = self.default_temperature
            
            # Set fallback models
            fallback_model = "openrouter/gpt-5-codex"
            
            self.models[model_name] = ModelConfig(
                model_name=model_name,
                model_id=model["model_id"],
                provider="openrouter",
                api_base=os.getenv("OPENROUTER_API_BASE"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                default_params={
                    "temperature": temperature,
                },
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                max_tokens=self.max_tokens,
                use_responses_api=False,
                output_version=None,
                fallback_model=fallback_model,
            )
    
    def _initialize_embedding_models(self):
        """Initialize embedding models."""
        # Embedding models via OpenRouter
        models = [
            {
                "model_name": "openrouter/text-embedding-3-large",
                "model_id": "openai/text-embedding-3-large",
            }
        ]
        
        for model in models:
            model_name = model["model_name"]
            
            # Set fallback models
            fallback_model = "openrouter/text-embedding-3-large"
            
            self.models[model_name] = ModelConfig(
                model_name=model_name,
                model_id=model["model_id"],
                provider="openrouter",
                model_type="embedding",
                api_base=os.getenv("OPENROUTER_API_BASE"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                default_params={},
                supports_streaming=False,
                supports_functions=False,
                supports_vision=False,
                max_tokens=None,
                use_responses_api=False,
                output_version=None,
                fallback_model=fallback_model,
            )

    def completion(
        self,
        model: str,
        messages: List[Any],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream: bool = False,
        use_responses_api: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Invoke models synchronously via litellm.completion.
        
        Args:
            model: Model name
            messages: List of messages
            tools: Optional list of tools (function definitions). If provided, uses function calling.
            response_format: Optional response format (BaseModel class or dict). If provided, uses structured output.
            stream: Whether to stream the response
            use_responses_api: Whether to use responses API
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with:
            - success: bool indicating if the call was successful
            - message: 
              - If tools: List of function calls with name and arguments
              - If response_format: Parsed BaseModel instance
              - Otherwise: String content
            - extra: Additional response metadata
        """
        # Validate that tools and response_format are not both provided
        if tools and response_format:
            raise ValueError("tools and response_format cannot be used together")
        
        if not self.completion_handler:
            raise RuntimeError("Model manager not initialized. Call initialize() first.")
        
        current_model = model
        try:
            return self.completion_handler.completion(
                model=current_model,
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                use_responses_api=use_responses_api,
                **kwargs,
            )
        except Exception as e:
            # Check if we have a fallback model
            model_config = self.models.get(current_model)
            if model_config and model_config.fallback_model:
                fallback_model = model_config.fallback_model
                logger.warning(f"| Primary model {current_model} failed, falling back to {fallback_model}: {e}")
                try:
                    # Retry with fallback model
                    result = self.completion_handler.completion(
                        model=fallback_model,
                        messages=messages,
                        tools=tools,
                        response_format=response_format,
                        stream=stream,
                        use_responses_api=use_responses_api,
                        **kwargs,
                    )
                    logger.info(f"| Fallback model {fallback_model} succeeded")
                    return result
                except Exception as fallback_error:
                    logger.error(f"| Fallback model {fallback_model} also failed: {fallback_error}")
                    return LLMResponse(
                        success=False,
                        message=f"Both primary model ({current_model}) and fallback model ({fallback_model}) failed. Primary error: {e}, Fallback error: {fallback_error}",
                        extra=None
                    )
            
            logger.error(f"Model completion failed: {e}")
            return LLMResponse(
                success=False,
                message=str(e),
                extra=None
            )
    
    async def acompletion(
        self,
        model: str,
        messages: List[Any],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream: bool = False,
        use_responses_api: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Invoke models asynchronously via litellm.acompletion.
        
        Args:
            model: Model name
            messages: List of messages
            tools: Optional list of tools (function definitions). If provided, uses function calling.
            response_format: Optional response format (BaseModel class or dict). If provided, uses structured output.
            stream: Whether to stream the response
            use_responses_api: Whether to use responses API
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with:
            - success: bool indicating if the call was successful
            - message: 
              - If tools: List of function calls with name and arguments
              - If response_format: Parsed BaseModel instance
              - Otherwise: String content
            - extra: Additional response metadata
        """
        # Validate that tools and response_format are not both provided
        if tools and response_format:
            raise ValueError("tools and response_format cannot be used together")
        
        if not self.completion_handler:
            raise RuntimeError("Model manager not initialized. Call initialize() first.")
        
        current_model = model
        try:
            return await self.completion_handler.acompletion(
                model=current_model,
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                use_responses_api=use_responses_api,
                **kwargs,
            )
        except Exception as e:
            # Check if we have a fallback model
            model_config = self.models.get(current_model)
            if model_config and model_config.fallback_model:
                fallback_model = model_config.fallback_model
                logger.warning(f"| Primary model {current_model} failed, falling back to {fallback_model}: {e}")
                try:
                    # Retry with fallback model
                    result = await self.completion_handler.acompletion(
                        model=fallback_model,
                        messages=messages,
                        tools=tools,
                        response_format=response_format,
                        stream=stream,
                        use_responses_api=use_responses_api,
                        **kwargs,
                    )
                    logger.info(f"| Fallback model {fallback_model} succeeded")
                    return result
                except Exception as fallback_error:
                    logger.error(f"| Fallback model {fallback_model} also failed: {fallback_error}")
                    return LLMResponse(
                        success=False,
                        message=f"Both primary model ({current_model}) and fallback model ({fallback_model}) failed. Primary error: {e}, Fallback error: {fallback_error}",
                        extra=None
                    )
            
            logger.error(f"Model acompletion failed: {e}")
            return LLMResponse(
                success=False,
                message=str(e),
                extra=None
            )
    
    def embedding(
        self,
        model: str,
        messages: List[Any],
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Get embeddings synchronously.
        
        Args:
            model: Model name (must be an embedding model)
            messages: List of messages to embed (same format as completion)
            encoding_format: Format of the embeddings ("float" or "base64")
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with:
            - success: bool indicating if the call was successful
            - message: Status message
            - extra: Contains embeddings list and additional response metadata
        """
        if not self.completion_handler:
            raise RuntimeError("Model manager not initialized. Call initialize() first.")
        
        current_model = model
        try:
            return self.completion_handler.embedding(
                model=current_model,
                messages=messages,
                encoding_format=encoding_format,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            return LLMResponse(
                success=False,
                message=f"Embedding request failed: {e}",
                extra={"error": str(e)}
            )
    
    async def aembedding(
        self,
        model: str,
        messages: List[Any],
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Get embeddings asynchronously.
        
        Args:
            model: Model name (must be an embedding model)
            messages: List of messages to embed (same format as completion)
            encoding_format: Format of the embeddings ("float" or "base64")
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with:
            - success: bool indicating if the call was successful
            - message: Status message
            - extra: Contains embeddings list and additional response metadata
        """
        if not self.completion_handler:
            raise RuntimeError("Model manager not initialized. Call initialize() first.")
        
        current_model = model
        try:
            return await self.completion_handler.aembedding(
                model=current_model,
                messages=messages,
                encoding_format=encoding_format,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Async embedding request failed: {e}")
            return LLMResponse(
                success=False,
                message=f"Embedding request failed: {e}",
                extra={"error": str(e)}
            )


    # Global singleton instance
model_manager = ModelManager()