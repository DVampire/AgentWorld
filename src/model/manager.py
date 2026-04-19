"""
Model manager for OpenAI models.
Provides a unified interface for registering and invoking OpenAI models.
"""

import asyncio
from typing import Optional, Dict, List, Any, Union, TYPE_CHECKING
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.model.types import ModelConfig, LLMResponse, LLMExtra
from src.model.openai.chat import ChatOpenAI
from src.model.openai.response import ResponseOpenAI
from src.model.openai.transcribe import TranscribeOpenAI
from src.model.openai.embedding import EmbeddingOpenAI
from src.model.openrouter.chat import ChatOpenRouter
from src.model.anthropic.chat import ChatAnthropic
from src.model.google.chat import ChatGoogle
from src.model.newapi.chat import ChatNewAPI
from src.model.newapi.response import ResponseNewAPI
from src.message.types import Message
from src.logger import logger
from src.utils import hvac_client

if TYPE_CHECKING:
    from src.tool.types import Tool


class ApiKeyPool:
    """
    Thread-safe round-robin API key pool.

    Each provider registers its key env-var (which may be a single key or a
    comma-separated list) and an optional base-URL env-var.  Callers obtain the
    next key via `next_key(provider)`.
    """

    def __init__(self):
        self._keys: Dict[str, List[str]] = {}
        self._bases: Dict[str, Optional[str]] = {}
        self._indices: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _parse_keys(env_var: str) -> List[str]:
        """Read env var and split by comma; works for both single key and CSV list."""
        raw = hvac_client.get(env_var)
        return [k.strip() for k in raw.split(",") if k.strip()]

    def register(
        self,
        provider: str,
        key_env: str,
        base_env: Optional[str] = None,
        default_base: Optional[str] = None,
    ) -> "ApiKeyPool":
        """Register a provider. Returns self for chaining."""
        self._keys[provider] = self._parse_keys(key_env)
        self._bases[provider] = (hvac_client.get(base_env) or default_base) if base_env else default_base
        self._indices[provider] = 0
        return self

    async def get_base(self, provider: str) -> Optional[str]:
        return self._bases.get(provider)

    async def get_key(self, provider: str) -> Optional[str]:
        """Return next key in round-robin order (asyncio-safe)."""
        async with self._lock:
            keys = self._keys.get(provider, [])
            if not keys:
                return None
            idx = self._indices.get(provider, 0)
            key = keys[idx]
            self._indices[provider] = (idx + 1) % len(keys)
            return key


class ModelManager:
    """
    Central registry and invoker for OpenAI models.

    Responsibilities:
    1. Register and store model configurations.
    2. Provide a unified invocation interface using ChatOpenAI or ResponseOpenAI.
    """
    
    def __init__(self):
        """Initialize the manager."""
        self.models: Dict[str, ModelConfig] = {}
        self.model_clients: Dict[str, Union[ChatOpenAI, ResponseOpenAI, TranscribeOpenAI, EmbeddingOpenAI, ChatOpenRouter, ChatAnthropic]] = {}
        # Key pool: provider -> round-robin keys / base URLs
        self._key_pool = ApiKeyPool()
        
        # Default parameters
        self.max_tokens: int = 32768
        self.default_temperature: float = 0.7
        self.default_timeout: float = 600.0
        self.default_reasoning: Dict[str, Any] = {
            "reasoning_effort": "high"
        }
        self.default_plugins: Optional[List[Dict[str, Any]]] = [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": "mistral-ocr"
                }
            },
            {
                "id": "web", 
                "max_results": 10,
                # "engine": "exa"
            },
            { 
                "id": "response-healing"
            }
        ]
    

    async def initialize(self):
        """Initialize the manager and register default models."""
        # Register all providers into the key pool (single env var, auto-detects CSV list)
        (
            self._key_pool
            .register("openai",      "OPENAI_API_KEY",      "OPENAI_API_BASE",    "")
            .register("openrouter",  "OPENROUTER_API_KEY",  "OPENROUTER_API_BASE",  "")
            .register("anthropic",   "ANTHROPIC_API_KEY",   "ANTHROPIC_API_BASE", "")
            .register("google",      "GOOGLE_API_KEY",     "GOOGLE_API_BASE",    "")
            .register("newapi",      "NEWAPI_API_KEY",       "NEWAPI_API_BASE",     "")
            .register("int_openrouter", "INT_OPENROUTER_API_KEY", "INT_OPENROUTER_API_BASE", "")
        )
        await self._initialize_openai_models()
        await self._initialize_openrouter_models()
        await self._initialize_internal_openrouter_models()
        await self._initialize_anthropic_models()
        await self._initialize_google_models()
        await self._initialize_newapi_models()
        logger.info(f"| Model manager initialized successfully with {len(self.models)} models.")
    
    async def _initialize_openai_models(self):
        """Initialize OpenAI models."""
        # General chat/completions models
        chat_models = [
            {
                "model_name": "openai/gpt-4o",
                "model_id": "gpt-4o",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-4.1"
            },
            {
                "model_name": "openai/gpt-4.1",
                "model_type": "chat/completions",
                "model_id": "gpt-4.1",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-4o",
            },
        ]
        
        # Responses API models
        response_models = [
            {
                "model_name": "openai/gpt-5",
                "model_type": "responses",
                "model_id": "gpt-5",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openai/gpt-5.1",
                "model_type": "responses",
                "model_id": "gpt-5.1",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/o3",
                "model_type": "responses",
                "model_id": "o3",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5.1",
            },
            {
                "model_name": "openai/o3-mini",
                "model_type": "responses",
                "model_id": "o3-mini",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5.1",
            },
            {
                "model_name": "openai/gpt-5.2",
                "model_type": "responses",
                "model_id": "gpt-5.1",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/gpt-5.3",
                "model_type": "responses",
                "model_id": "gpt-5.3",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/gpt-5.4",
                "model_type": "responses",
                "model_id": "gpt-5.4",
                "reasoning": {
                    "reasoning": {"effort": "high"}
                },
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/gpt-5.4-pro",
                "model_type": "responses",
                "model_id": "gpt-5.4-pro",
                "reasoning": {
                    "reasoning": {"effort": "high"}
                },
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5.4",
            }
        ]
        
        # Transcription models
        transcribe_models = [
            {
                "model_name": "openai/gpt-4o-transcribe",
                "model_type": "transcriptions",
                "model_id": "gpt-4o-transcribe",
                "fallback_model": "openai/gpt-4o-transcribe",
            }
        ]
        
        # Embedding models
        embedding_models = [
            {
                "model_name": "openai/text-embedding-3-small",
                "model_type": "embeddings",
                "model_id": "text-embedding-3-small",
                "fallback_model": "openai/text-embedding-3-large",
            },
            {
                "model_name": "openai/text-embedding-3-large",
                "model_type": "embeddings",
                "model_id": "text-embedding-3-large",
                "fallback_model": "openai/text-embedding-3-large",
            },
            {
                "model_name": "openai/text-embedding-ada-002",
                "model_type": "embeddings",
                "model_id": "text-embedding-ada-002",
                "fallback_model": "openai/text-embedding-3-large",
            },
        ]

        api_base=await self._key_pool.get_base("openai")
        api_key=await self._key_pool.get_key("openai")
        
        # Register chat models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",
                key_pool_name="openai",
                api_base=api_base,
                api_key=api_key,
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                timeout=model.get("timeout", self.default_timeout),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
        
        # Register response models
        for model in response_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",
                key_pool_name="openai",
                api_base=api_base,
                api_key=api_key,
                reasoning=model.get("reasoning"),
                max_output_tokens=model.get("max_output_tokens"),
                timeout=model.get("timeout", self.default_timeout),
                supports_streaming=False,  # Responses API may not support streaming
                supports_functions=False,  # Responses API may not support tools
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
        
        # Register transcription models
        for model in transcribe_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",
                key_pool_name="openai",
                api_base=api_base,
                api_key=api_key,
                timeout=model.get("timeout", self.default_timeout),
                supports_streaming=False,
                supports_functions=False,
                supports_vision=False,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)

        # Register embedding models
        for model in embedding_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",
                key_pool_name="openai",
                api_base=api_base,
                api_key=api_key,
                timeout=model.get("timeout", self.default_timeout),
                supports_streaming=False,
                supports_functions=False,
                supports_vision=False,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _initialize_openrouter_models(self):
        """Initialize OpenRouter models (OpenAI models via OpenRouter)."""
        chat_models = [
            # OpenAI models
            {
                "model_name": "openrouter/gpt-4o",
                "model_id": "openai/gpt-4o",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-4.1",
                "model_id": "openai/gpt-4.1",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-5",
                "model_id": "openai/gpt-5",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-5.1",
                "model_id": "openai/gpt-5.1",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-5.2",
                "model_id": "openai/gpt-5.2",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-5.3",
                "model_id": "openai/gpt-5.3",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-5.4",
                "model_id": "openai/gpt-5.4",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-5.4-pro",
                "model_id": "openai/gpt-5.4-pro",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/o3",
                "model_id": "openai/o3",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": 1.0,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/o3-mini",
                "model_id": "openai/o3-mini",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": 1.0,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/gpt-5.3-codex",
                "model_id": "openai/gpt-5.3-codex",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gpt-5.3",
            },
            # Anthropic models
            {
                "model_name": "openrouter/claude-sonnet-3.5",
                "model_id": "anthropic/claude-3.5-sonnet",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-3.7",
                "model_id": "anthropic/claude-3.7-sonnet",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-4",
                "model_id": "anthropic/claude-sonnet-4",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/claude-opus-4",
                "model_id": "anthropic/claude-opus-4",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-4.5",
                "model_id": "anthropic/claude-sonnet-4.5",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/claude-opus-4.5",
                "model_id": "anthropic/claude-opus-4.5",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-4.6",
                "model_id": "anthropic/claude-sonnet-4.6",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/claude-opus-4.6",
                "model_id": "anthropic/claude-opus-4.6",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            # Gemini models for chat, vision, audio, video, pdf, etc.
            {
                "model_name": "openrouter/gemini-2.5-flash",
                "model_type": "chat/completions",
                "model_id": "google/gemini-2.5-flash",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "openrouter/gemini-2.5-pro",
                "model_type": "chat/completions",
                "model_id": "google/gemini-2.5-pro",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "openrouter/gemini-3-pro-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3-pro-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "openrouter/gemini-3.1-pro-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-pro-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "gemini/gemini-3.1-pro-preview",
            },
            {
                "model_name": "openrouter/gemini-3-flash-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3-flash-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "gemini/gemini-3-flash-preview",
            },
            {
                "model_name": "openrouter/gemini-2.5-flash-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-2.5-flash",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview-plugins",
            },
            {
                "model_name": "openrouter/gemini-3-flash-preview-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3-flash-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview-plugins",
            },
            {
                "model_name": "openrouter/gemini-3.1-flash-lite-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-flash-lite-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview-plugins",
            },
            {
                "model_name": "openrouter/gemini-3.1-flash-lite-preview-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-flash-lite-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview-plugins",
            },
            {
                "model_name": "openrouter/gemini-3.1-pro-preview-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-pro-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview-plugins",
            },
            # Qwen models
            {
                "model_name": "openrouter/qwen3-coder",
                "model_id": "qwen/qwen3-coder",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openrouter/qwen3-max",
                "model_id": "qwen/qwen3-max",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            #deepseek models
            {
                "model_name": "openrouter/deepseek-v3.2",
                "model_id": "deepseek/deepseek-v3.2",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            # X-ai models
            {
                "model_name": "openrouter/grok-4.1-fast",
                "model_id": "x-ai/grok-4.1-fast",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            }
        ]

        api_base=await self._key_pool.get_base("openrouter")
        api_key=await self._key_pool.get_key("openrouter")
        
        # Register OpenRouter models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openrouter",
                key_pool_name="openrouter",
                api_base=api_base,
                api_key=api_key,
                reasoning=model.get("reasoning") if model.get("reasoning") else None,
                plugins=model.get("plugins") if model.get("plugins") else None,
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                timeout=model.get("timeout", self.default_timeout),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)

    async def _initialize_internal_openrouter_models(self):
        """Initialize OpenRouter models (OpenAI models via OpenRouter)."""
        chat_models = [
            # OpenAI models
            {
                "model_name": "int_openrouter/gpt-5.4",
                "model_id": "openai/gpt-5.4",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gpt-5.4",
            },
            {
                "model_name": "int_openrouter/gpt-5.4-pro",
                "model_id": "openai/gpt-5.4-pro",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gpt-5.4",
            },
            # Anthropic models
            {
                "model_name": "int_openrouter/claude-sonnet-4.6",
                "model_id": "anthropic/claude-sonnet-4.6",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gpt-5.4",
            },
            {
                "model_name": "int_openrouter/claude-opus-4.6",
                "model_id": "anthropic/claude-opus-4.6",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gpt-5.4",
            },
            # Gemini models for chat, vision, audio, video, pdf, etc.
            {
                "model_name": "int_openrouter/gemini-3.1-pro-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-pro-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gemini-3.1-pro-preview",
            },
            {
                "model_name": "int_openrouter/gemini-3-flash-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3-flash-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "int_openrouter/gemini-3.1-flash-lite-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-flash-lite-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gemini-3-flash-preview-plugins",
            },
            {
                "model_name": "int_openrouter/gemini-3.1-flash-lite-preview-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-flash-lite-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gemini-3-flash-preview-plugins",
            },
            {
                "model_name": "int_openrouter/gemini-3.1-pro-preview-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-pro-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/gemini-3-flash-preview-plugins",
            },
            # X-ai models
            {
                "model_name": "int_openrouter/grok-4.1-fast",
                "model_id": "x-ai/grok-4.1-fast",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "int_openrouter/grok-4.1-fast",
            }
        ]

        api_base=await self._key_pool.get_base("int_openrouter")
        api_key=await self._key_pool.get_key("int_openrouter")

        # Register Internal OpenRouter models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openrouter",
                key_pool_name="int_openrouter",
                api_base=api_base,
                api_key=api_key,
                reasoning=model.get("reasoning") if model.get("reasoning") else None,
                plugins=model.get("plugins") if model.get("plugins") else None,
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                timeout=model.get("timeout", self.default_timeout),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _initialize_anthropic_models(self):
        """Initialize Anthropic models."""
        chat_models = [
            # Note: Claude 3.5 Sonnet is deprecated, use Claude 3.7 Sonnet or Claude Sonnet 4.5 instead
            # {
            #     "model_name": "anthropic/claude-sonnet-3.5",
            #     "model_id": "claude-3-5-sonnet-20240620",
            #     "model_type": "chat/completions",
            #     "temperature": self.default_temperature,
            #     "max_completion_tokens": self.max_tokens,
            #     "fallback_model": "anthropic/claude-sonnet-4.5",
            # },
            {
                "model_name": "anthropic/claude-sonnet-3.7",
                "model_id": "claude-3-7-sonnet-20250219",  # Anthropic model ID with version
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "anthropic/claude-sonnet-4.5",
            },
            {
                "model_name": "anthropic/claude-sonnet-4",
                "model_id": "claude-sonnet-4-20250514",  # Anthropic model ID with version
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "anthropic/claude-sonnet-4.5",
            },
            # {
            #     "model_name": "anthropic/claude-opus-4",
            #     "model_id": "claude-opus-4-20250514",  # Anthropic model ID with version
            #     "model_type": "chat/completions",
            #     "temperature": self.default_temperature,
            #     "max_completion_tokens": self.max_tokens,
            #     "fallback_model": "anthropic/claude-sonnet-4.5",
            # },
            {
                "model_name": "anthropic/claude-sonnet-4.5",
                "model_id": "claude-sonnet-4-5-20250929",  # Anthropic model ID with version
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "anthropic/claude-sonnet-4.5",
            },
            # {
            #     "model_name": "anthropic/claude-opus-4.5",
            #     "model_id": "claude-opus-4-1-20250805",  # Correct Anthropic model ID (Opus 4.1)
            #     "model_type": "chat/completions",
            #     "temperature": self.default_temperature,
            #     "max_completion_tokens": self.max_tokens,
            #    "fallback_model": "anthropic/claude-sonnet-4.5",
            # },
        ]

        api_base=await self._key_pool.get_base("anthropic")
        api_key=await self._key_pool.get_key("anthropic")
        
        # Register Anthropic models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="anthropic",
                key_pool_name="anthropic",
                api_base=api_base,
                api_key=api_key,
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                timeout=model.get("timeout", self.default_timeout),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _initialize_newapi_models(self):
        """Initialize New-API models (OpenAI-compatible proxy)."""
        chat_models = [
            # Gemini models for chat, vision, audio, video, pdf, etc.
            {
                "model_name": "newapi/gemini-3.1-pro-preview",
                "model_id": "gemini-3.1-pro-preview",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/grok-4.1-fast",
            },
            {
                "model_name": "newapi/gemini-3.1-flash-lite-preview",
                "model_id": "gemini-3.1-flash-lite-preview",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/grok-4.1-fast",
            },
            # OpenAI-compatible models
            {
                "model_name": "newapi/gpt-5.4",
                "model_id": "gpt-5.4",
                "model_type": "responses",
                "reasoning": self.default_reasoning,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/grok-4.1-fast",
            },
            {
                "model_name": "newapi/gpt-5.4-pro",
                "model_id": "gpt-5.4-pro",
                "model_type": "responses",
                "timeout": 3600.0,  # 1 hour timeout for long-running tasks
                "reasoning": self.default_reasoning,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/grok-4.1-fast",
            },
            {
                "model_name": "newapi/gpt-5-nano",
                "model_id": "gpt-5-nano",
                "model_type": "responses",
                "reasoning": self.default_reasoning,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/grok-4.1-fast",
            },
            # Anthropic-compatible models
            {
                "model_name": "newapi/claude-opus-4.6",
                "model_id": "claude-opus-4-6",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/grok-4.1-fast",
            },
            # X-ai-compatible models could be added here as well if needed
            {
                "model_name": "newapi/grok-4.1-fast",
                "model_id": "grok-4.1-fast",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/grok-4.1-fast",
            }
        ]

        api_base=await self._key_pool.get_base("newapi")
        api_key=await self._key_pool.get_key("newapi")

        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="newapi",
                key_pool_name="newapi",
                api_base=api_base,
                api_key=api_key,
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                reasoning=model.get("reasoning"),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)

    async def _initialize_google_models(self):
        """Initialize Google Gemini models."""
        chat_models = [
            {
                "model_name": "google/gemini-2.5-flash",
                "model_id": "gemini-2.5-flash",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "google/gemini-2.5-pro",
                "model_id": "gemini-2.5-pro",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "google/gemini-3-pro-preview",
                "model_id": "gemini-3-pro-preview",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "google/gemini-3.1-pro-preview",
                "model_id": "gemini-3.1-pro-preview",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "google/gemini-3-flash-preview",
                "model_id": "gemini-3-flash-preview",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            {
                "model_name": "google/gemini-3.1-flash-lite-preview",
                "model_id": "gemini-3.1-flash-lite-preview",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            }
        ]
        
        api_base=await self._key_pool.get_base("google")
        api_key=await self._key_pool.get_key("google")

        # Register Google models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="google",
                key_pool_name="google",
                api_base=api_base,
                api_key=api_key,
                reasoning=model.get("reasoning"),
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=None,
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _create_client(self, config: ModelConfig) -> None:
        """Create and cache a single client for the given model config."""
        self.model_clients[config.model_name] = await self._build_client(config)
        logger.info(f"| Created client for {config.model_name}")

    async def _build_client(self, config: ModelConfig):
        """Build a single client instance from a ModelConfig (sync, no side effects)."""
        if config.provider == "newapi":
            if config.model_type == "chat/completions":
                return ChatNewAPI(
                    model=config.model_id,
                    api_key=config.api_key,
                    base_url=config.api_base,
                    temperature=config.temperature or self.default_temperature,
                    max_completion_tokens=config.max_completion_tokens or self.max_tokens,
                )
            elif config.model_type == "responses":
                return ResponseNewAPI(
                    model=config.model_id,
                    api_key=config.api_key,
                    base_url=config.api_base,
                    reasoning=config.reasoning if config.reasoning else None,
                    max_output_tokens=config.max_output_tokens or self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported model type {config.model_type} for New-API provider")
        elif config.provider == "openrouter":
            if config.model_type == "chat/completions":
                return ChatOpenRouter(
                    model=config.model_id,
                    api_key=config.api_key,
                    base_url=config.api_base,
                    reasoning=config.reasoning if config.reasoning else None,
                    plugins=config.plugins if config.plugins else None,
                    temperature=config.temperature or self.default_temperature,
                    max_completion_tokens=config.max_completion_tokens or self.max_tokens
                )
            else:
                raise ValueError(f"Unsupported model type {config.model_type} for OpenRouter provider")
        elif config.provider == "anthropic":
            if config.model_type == "chat/completions":
                return ChatAnthropic(
                    model=config.model_id,
                    api_key=config.api_key,
                    base_url=config.api_base,
                    reasoning=config.reasoning if config.reasoning else None,
                    temperature=config.temperature or self.default_temperature,
                    max_tokens=config.max_completion_tokens or self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported model type {config.model_type} for Anthropic provider")
        elif config.provider == "google":
            if config.model_type == "chat/completions":
                return ChatGoogle(
                    model=config.model_id,
                    api_key=config.api_key,
                    reasoning=config.reasoning if config.reasoning else None,
                    temperature=config.temperature or self.default_temperature,
                    max_output_tokens=config.max_completion_tokens or self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported model type {config.model_type} for Google provider")
        elif config.model_type == "responses":
            return ResponseOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
                reasoning=config.reasoning if config.reasoning else None,
                max_output_tokens=config.max_output_tokens or self.max_tokens,
            )
        elif config.model_type == "transcriptions":
            return TranscribeOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
            )
        elif config.model_type == "embeddings":
            return EmbeddingOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
            )
        else:
            return ChatOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
                temperature=config.temperature or self.default_temperature,
                reasoning=config.reasoning if config.reasoning else None,
                max_completion_tokens=config.max_completion_tokens or self.max_tokens,
            )

    async def _get_client(self, model: str):
        """Return the cached client with the next round-robin key already set."""
        client = self.model_clients.get(model)
        if client:
            model_config = self.models.get(model)
            pool_name = (model_config.key_pool_name or model_config.provider) if model_config else None
            key = await self._key_pool.get_key(pool_name) if pool_name else None
            if key:
                client.set_api_key(key)
        return client

    async def register_model(self, config: ModelConfig) -> None:
        """Register a new model configuration."""
        if config.provider not in ["openai", "openrouter", "anthropic", "google", "newapi"]:
            raise ValueError(f"Only OpenAI, OpenRouter, Anthropic, Google, NewAPI, and Internal OpenRouter models are supported. Got provider: {config.provider}")
        
        self.models[config.model_name] = config
        await self._create_client(config)
        logger.info(f"Registered model: {config.model_name}")
    
    def _log_usage(self, model_name: str, result: LLMResponse) -> None:
        """Log token usage and estimated cost after a model call."""
        if not result.success or not result.extra or not result.extra.data:
            return
        usage = result.extra.data.get("usage")
        if not usage:
            return

        # Normalize token field names across providers
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_token_count") or 0
        total_tokens = usage.get("total_tokens") or usage.get("total_token_count") or (input_tokens + output_tokens)

        # Some providers (e.g. OpenRouter) may include cost directly
        cost = usage.get("cost")
        # Anthropic may include cache-related fields
        cache_creation = usage.get("cache_creation_input_tokens") or 0
        cache_read = usage.get("cache_read_input_tokens") or 0

        parts = [
            f"model={model_name}",
            f"input={input_tokens}",
            f"output={output_tokens}",
            f"total={total_tokens}",
        ]
        if cache_creation or cache_read:
            parts.append(f"cache_create={cache_creation}")
            parts.append(f"cache_read={cache_read}")
        if cost is not None:
            parts.append(f"cost=${cost:.6f}")
        caller = getattr(self, '_current_caller', None)
        if caller:
            parts.append(f"caller={caller}")

        logger.info(f"| 💰 Usage: {', '.join(parts)}")

    async def _call_client(
        self,
        client,
        model_config,
        messages: List[Message],
        tools: Optional[List["Tool"]],
        response_format: Optional[Union[BaseModel, Dict]],
        stream: bool,
        plugins: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> LLMResponse:
        """Invoke a single client according to its model type."""
        if model_config and model_config.model_type == "transcriptions":
            return await client(messages=messages, **kwargs)
        elif model_config and model_config.model_type == "embeddings":
            embedding_kwargs = {k: v for k, v in kwargs.items() if k not in ["tools", "response_format", "stream"]}
            return await client(messages=messages, **embedding_kwargs)
        else:
            # plugins is only supported by OpenRouter, not OpenAI/Anthropic
            is_openrouter = model_config and model_config.provider == "openrouter"
            call_kwargs = dict(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )
            if is_openrouter:
                call_kwargs["plugins"] = plugins
            return await client(**call_kwargs)

    async def __call__(
        self,
        model: str,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        plugins: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        caller: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Invoke a model asynchronously with retry (default 3 attempts).

        Args:
            model: Model name
            messages: List of Message objects
            tools: Optional list of Tool instances
            response_format: Optional response format (Pydantic model or dict)
            stream: Whether to stream the response
            max_retries: Number of attempts before falling back / giving up (default 3)
            caller: Optional identifier of who is calling (e.g. 'v2/query_gen', 'analyzer/merge')
            **kwargs: Additional parameters

        Returns:
            LLMResponse with formatted message
        """
        self._current_caller = caller
        if tools and response_format:
            raise ValueError("tools and response_format cannot be used together")

        if model not in self.model_clients:
            return LLMResponse(
                success=False,
                message=f"Model {model} not found. Available models: {list(self.models.keys())}",
            )

        model_config = self.models.get(model)

        # --- Primary model with retries ---
        import time as _t
        import httpx
        last_exc: Exception = None
        for attempt in range(max_retries):
            _call_start = _t.time()
            try:
                client = await self._get_client(model)
                result = await self._call_client(
                    client, model_config, messages, tools, response_format, stream, plugins, kwargs
                )
                self._log_usage(model, result)
                if not result.success:
                    raise Exception(result.message or "Model returned success=False")
                is_chat = not model_config or model_config.model_type not in ("transcriptions", "embeddings")
                if is_chat and not result.message:
                    raise Exception("Model returned empty message")
                return result
            except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_exc = e
                _elapsed = _t.time() - _call_start
                caller_tag = f", caller={self._current_caller}" if getattr(self, '_current_caller', None) else ""
                logger.error(
                    f"| ❌ Model {model} timed out ({_elapsed:.0f}s{caller_tag}): {e}"
                )
                break  # timeout不重试，直接走fallback
            except Exception as e:
                last_exc = e
                _elapsed = _t.time() - _call_start
                caller_tag = f", caller={self._current_caller}" if getattr(self, '_current_caller', None) else ""
                if attempt < max_retries - 1:
                    logger.warning(
                        f"| ⚠️ Model {model} attempt {attempt + 1}/{max_retries} failed ({_elapsed:.0f}s{caller_tag}): {e}, retrying..."
                    )
                else:
                    logger.error(
                        f"| ❌ Model {model} failed after {max_retries} attempts ({_elapsed:.0f}s{caller_tag}): {e}"
                    )

        # --- Fallback model (single attempt) ---
        if model_config and model_config.fallback_model:
            fallback_model = model_config.fallback_model
            logger.warning(f"| Primary model {model} exhausted retries, falling back to {fallback_model}")
            if fallback_model not in self.model_clients:
                return LLMResponse(
                    success=False,
                    message=f"Primary model {model} failed and fallback {fallback_model} not found. Error: {last_exc}",
                )
            fallback_config = self.models.get(fallback_model)
            try:
                fallback_client = await self._get_client(fallback_model)
                result = await self._call_client(
                    fallback_client, fallback_config, messages, tools, response_format, stream, plugins, kwargs
                )
                self._log_usage(fallback_model, result)
                if not result.success:
                    raise Exception(result.message or "Fallback model returned success=False")
                is_chat = not fallback_config or fallback_config.model_type not in ("transcriptions", "embeddings")
                if is_chat and not result.message:
                    raise Exception("Fallback model returned empty message")
                logger.info(f"| Fallback model {fallback_model} succeeded")
                return result
            except Exception as fallback_error:
                logger.error(f"| Fallback model {fallback_model} also failed: {fallback_error}")
                return LLMResponse(
                    success=False,
                    message=(
                        f"Both primary model ({model}) and fallback model ({fallback_model}) failed. "
                        f"Primary error: {last_exc}, Fallback error: {fallback_error}"
                    ),
                    extra=None,
                )

        return LLMResponse(success=False, message=str(last_exc))
    
    async def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get the configuration for a model."""
        return await self.models.get(model)
    
    async def list(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())


    # Global singleton instance
model_manager = ModelManager()