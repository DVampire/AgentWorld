from typing import Optional, Dict, Any, Union, List
import requests
import aiohttp
import os
import json

from src.model.types import ModelConfig
from src.message.types import Message
from src.logger import logger

class APIRequest:
    """Handles HTTP requests to LLM APIs using requests/aiohttp."""
    
    # Default API endpoints for each provider
    DEFAULT_ENDPOINTS = {
        "openrouter": "/chat/completions",
        "openai": "/chat/completions",
        "anthropic": "/messages",
        "google": "/models",
    }
    
    # Default embedding endpoints for each provider
    DEFAULT_EMBEDDING_ENDPOINTS = {
        "openrouter": "/embeddings",
        "openai": "/embeddings",
        "anthropic": "/embeddings",
        "google": "/models",
    }
    
    # Default base URLs for each provider
    DEFAULT_BASE_URLS = {
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta",
    }
    
    @staticmethod
    def _get_api_url(provider: str, api_base: Optional[str] = None, model_type: str = "completion") -> str:
        """Get the API URL for the provider."""
        if model_type == "embedding":
            endpoint = APIRequest.DEFAULT_EMBEDDING_ENDPOINTS.get(provider, "/embeddings")
        else:
            endpoint = APIRequest.DEFAULT_ENDPOINTS.get(provider, "/chat/completions")
        
        if api_base:
            return f"{api_base.rstrip('/')}{endpoint}"
        else:
            base_url = APIRequest.DEFAULT_BASE_URLS.get(provider, APIRequest.DEFAULT_BASE_URLS["openrouter"])
            return f"{base_url}{endpoint}"
    
    @staticmethod
    def _get_headers(provider: str, api_key: Optional[str] = None) -> Dict[str, str]:
        """Get headers for the API request."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if not api_key:
            return headers
        
        # Authorization header format for each provider
        auth_headers = {
            "openrouter": ("Authorization", f"Bearer {api_key}"),
            "openai": ("Authorization", f"Bearer {api_key}"),
            "anthropic": ("x-api-key", api_key),
            "google": ("x-goog-api-key", api_key),
        }
        
        # Get auth header for provider or default to Bearer
        auth_key, auth_value = auth_headers.get(provider, ("Authorization", f"Bearer {api_key}"))
        headers[auth_key] = auth_value

        # Optional headers for OpenRouter
        if provider == "openrouter":
            if os.getenv("OPENROUTER_SITE_URL"):
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
            if os.getenv("OPENROUTER_SITE_NAME"):
                headers["X-Title"] = os.getenv("OPENROUTER_SITE_NAME")
        
        return headers
    
    @staticmethod
    def request(
        config: ModelConfig,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make a synchronous HTTP request."""
        # Get API URL and headers (use model_type to determine endpoint)
        api_url = APIRequest._get_api_url(config.provider, config.api_base, model_type=config.model_type)
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        headers = APIRequest._get_headers(config.provider, api_key)
        
        # Build request data based on model type
        if config.model_type == "embedding":
            # For embedding, extract input from messages
            messages = params.get("messages", [])
            input_data = APIRequest._extract_text_from_messages(messages)
            request_data = {
                "model": params.get("model", config.model_id),
                "input": input_data,
                "encoding_format": params.get("encoding_format", "float"),
            }
        else:
            # For completion
            request_data = {
                "model": params.get("model", config.model_id),
                "messages": params.get("messages", []),
            }
        
        if params.get("tools"):
            request_data["tools"] = params["tools"]
        
        if params.get("response_format"):
            request_data["response_format"] = params["response_format"]
        
        if params.get("stream"):
            request_data["stream"] = True
            
        if params.get('plugins'):
            request_data["plugins"] = params["plugins"]
        
        # Add any additional parameters from params (excluding internal ones)
        exclude_keys = {"model", "messages", "tools", "response_format", "stream", "api_key", "api_base", "plugins", "encoding_format", "input"}
        for key, value in params.items():
            if key not in exclude_keys:
                request_data[key] = value
        
        # Make the request
        try:
            response = requests.post(
                url=api_url,
                headers=headers,
                data=json.dumps(request_data),
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    raise Exception(f"API request failed: {error_detail}")
                except:
                    raise Exception(f"API request failed: {e.response.text}")
            raise Exception(f"API request failed: {str(e)}")
    
    @staticmethod
    async def arequest(
        config: ModelConfig,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make an asynchronous HTTP request."""
        # Get API URL and headers (use model_type to determine endpoint)
        api_url = APIRequest._get_api_url(config.provider, config.api_base, model_type=config.model_type)
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        headers = APIRequest._get_headers(config.provider, api_key)
        
        # Build request data based on model type
        if config.model_type == "embedding":
            # For embedding, extract input from messages
            messages = params.get("messages", [])
            input_data = APIRequest._extract_text_from_messages(messages)
            request_data = {
                "model": params.get("model", config.model_id),
                "input": input_data,
                "encoding_format": params.get("encoding_format", "float"),
            }
        else:
            # For completion
            request_data = {
                "model": params.get("model", config.model_id),
                "messages": params.get("messages", []),
            }
        
        if params.get("tools"):
            request_data["tools"] = params["tools"]
        
        if params.get("response_format"):
            request_data["response_format"] = params["response_format"]
        
        if params.get("stream"):
            request_data["stream"] = True
            
        if params.get('plugins'):
            request_data["plugins"] = params["plugins"]
        
        # Add any additional parameters from params (excluding internal ones)
        exclude_keys = {"model", "messages", "tools", "response_format", "stream", "api_key", "api_base", "plugins", "encoding_format", "input"}
        for key, value in params.items():
            if key not in exclude_keys:
                request_data[key] = value
        
        # Make the async request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=api_url,
                    headers=headers,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Async HTTP request failed: {e}")
            raise Exception(f"API request failed: {str(e)}")
    
    @staticmethod
    def _extract_text_from_messages(messages: Union[List[Message], List[Dict[str, Any]]]) -> Union[str, List[str]]:
        """Extract text content from messages for embedding.
        
        Args:
            messages: List of Message objects or formatted message dicts
            
        Returns:
            If single message: str
            If multiple messages: List[str] (one per message)
        """
        texts = []
        for message in messages:
            # Handle both Message objects and dict formats
            if isinstance(message, dict):
                # Formatted message dict: {"role": "user", "content": [...]}
                content = message.get("content", "")
            else:
                # Message object
                content = message.content
            
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            text_parts.append(text)
                if text_parts:
                    texts.append(" ".join(text_parts))
                elif content:
                    # Fallback: convert to string
                    texts.append(str(content))
            else:
                texts.append(str(content))
        
        # Return single string if only one message, otherwise return list
        if len(texts) == 1:
            return texts[0]
        return texts