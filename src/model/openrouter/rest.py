import httpx
import json
from typing import Optional, Dict, Any, List

from src.logger import logger

class OpenRouterREST:
    """Handles HTTP requests to OpenRouter API using httpx."""
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_ENDPOINT = "/chat/completions"
    
    @staticmethod
    def _get_api_url(base_url: Optional[str] = None) -> str:
        """Get the API URL for OpenRouter."""
        base = base_url or OpenRouterREST.DEFAULT_BASE_URL
        base = base.rstrip('/')
        endpoint = OpenRouterREST.DEFAULT_ENDPOINT
        return f"{base}{endpoint}"
    
    @staticmethod
    def _get_headers(
        api_key: Optional[str] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Get headers for the OpenRouter API request."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # OpenRouter-specific headers
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title
        
        # Merge with default headers if any
        if default_headers:
            headers.update(default_headers)
        
        return headers
    
    @staticmethod
    async def request(
        model: str,
        messages: List[Dict[str, Any]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        plugins: Optional[List[Dict[str, Any]]] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 300.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Make an asynchronous HTTP request to OpenRouter API.
        
        Args:
            model: Model identifier (e.g., "google/gemini-2.5-flash")
            messages: List of message dictionaries
            api_key: OpenRouter API key
            base_url: Base URL for OpenRouter API (defaults to https://openrouter.ai/api/v1)
            plugins: Optional list of plugins (e.g., for PDF parsing)
            http_referer: Optional HTTP-Referer header
            x_title: Optional X-Title header
            default_headers: Optional default headers to merge
            timeout: Request timeout in seconds (default: 300)
            **kwargs: Additional parameters (temperature, max_completion_tokens, tools, etc.)
        
        Returns:
            Response dictionary from OpenRouter API
        """
        # Get API URL and headers
        api_url = OpenRouterREST._get_api_url(base_url)
        headers = OpenRouterREST._get_headers(
            api_key=api_key,
            http_referer=http_referer,
            x_title=x_title,
            default_headers=default_headers,
        )
        
        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        
        # Add plugins if provided
        if plugins:
            payload["plugins"] = plugins
        
        # Add other parameters from kwargs
        for key, value in kwargs.items():
            if key == "max_completion_tokens": # OpenRouter uses max_tokens instead of max_completion_tokens
                payload["max_tokens"] = value
            else:
                payload[key] = value
        
        # Make the async request
        try:
            timeout_obj = httpx.Timeout(timeout) if timeout else None
            async with httpx.AsyncClient(timeout=timeout_obj) as client:
                response = await client.post(
                    url=api_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API HTTP error: {e}")
            try:
                error_detail = e.response.json()
                raise Exception(f"OpenRouter API request failed: {error_detail}")
            except:
                raise Exception(f"OpenRouter API request failed: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"OpenRouter API request error: {e}")
            raise Exception(f"OpenRouter API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouter API request: {e}")
            raise

