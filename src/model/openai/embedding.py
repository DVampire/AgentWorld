from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional, Union, List
import httpx

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
    from openai.types.shared.chat_model import ChatModel
except ImportError:
    # Fallback if openai package is not available
    AsyncOpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    RateLimitError = Exception
    ChatModel = str

from pydantic import BaseModel, Field, ConfigDict

from src.message.types import Message, HumanMessage, SystemMessage, ContentPartText
from src.model.types import LLMResponse
from src.logger import logger


class EmbeddingOpenAI(BaseModel):
    """
    A wrapper around AsyncOpenAI that provides a unified interface for OpenAI embeddings.
    
    This class is specifically designed for embedding models like text-embedding-3-small, text-embedding-3-large, etc.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Model configuration
    model: Union[ChatModel, str] = "text-embedding-3-small"

    # Client initialization parameters
    api_key: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    websocket_base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[Union[float, httpx.Timeout]] = None
    max_retries: int = 5
    default_headers: Optional[Mapping[str, str]] = None
    default_query: Optional[Mapping[str, object]] = None
    http_client: Optional[httpx.AsyncClient] = None
    _strict_response_validation: bool = False

    # Embedding parameters
    dimensions: Optional[int] = None  # Optional dimensions for text-embedding-3 models
    encoding_format: Optional[str] = None  # "float" or "base64"

    @property
    def provider(self) -> str:
        return 'openai'

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters dictionary."""
        base_params = {
            'api_key': self.api_key,
            'organization': self.organization,
            'project': self.project,
            'base_url': self.base_url,
            'websocket_base_url': self.websocket_base_url,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'default_headers': self.default_headers,
            'default_query': self.default_query,
            '_strict_response_validation': self._strict_response_validation,
        }

        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # Add http_client if provided
        if self.http_client is not None:
            client_params['http_client'] = self.http_client

        return client_params

    def get_client(self) -> AsyncOpenAI:
        """
        Returns an AsyncOpenAI client.

        Returns:
            AsyncOpenAI: An instance of the AsyncOpenAI client.
        """
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install it with: pip install openai")
        
        client_params = self._get_client_params()
        return AsyncOpenAI(**client_params)

    @property
    def name(self) -> str:
        return str(self.model)

    def _extract_text_from_messages(self, messages: List[Message]) -> List[str]:
        """Extract text content from messages.
        
        Returns:
            List of text strings extracted from messages
        """
        texts = []
        
        for message in messages:
            if isinstance(message, (HumanMessage, SystemMessage)):
                if isinstance(message.content, str):
                    texts.append(message.content)
                elif isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, ContentPartText):
                            texts.append(part.text)
        
        return texts

    async def __call__(
        self,
        messages: List[Message],
        dimensions: Optional[int] = None,
        encoding_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous embedding call via OpenAI API.

        Args:
            messages: List of Message objects (will extract text content from messages)
            dimensions: Optional dimensions for text-embedding-3 models
            encoding_format: Optional encoding format ("float" or "base64")
            **kwargs: Additional parameters

        Returns:
            LLMResponse with embedding vectors
        """
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install it with: pip install openai")

        try:
            client = self.get_client()
            
            # Extract text from messages
            texts = self._extract_text_from_messages(messages)
            if not texts:
                return LLMResponse(
                    success=False,
                    message="No text content found in messages",
                    extra={"error": "No text content found in messages"}
                )
            
            # Build parameters for embedding
            params: dict[str, Any] = {
                "model": self.model,
                "input": texts if len(texts) > 1 else texts[0],  # Single string or list of strings
            }

            # Add embedding parameters
            if dimensions is not None:
                params["dimensions"] = dimensions
            elif self.dimensions is not None:
                params["dimensions"] = self.dimensions

            if encoding_format is not None:
                params["encoding_format"] = encoding_format
            elif self.encoding_format is not None:
                params["encoding_format"] = self.encoding_format

            # Merge additional kwargs
            params.update(kwargs)

            # Make the API call
            response = await client.embeddings.create(**params)

            # Extract embeddings from response
            embeddings = []
            if hasattr(response, 'data'):
                for item in response.data:
                    if hasattr(item, 'embedding'):
                        embeddings.append(item.embedding)
                    elif isinstance(item, dict):
                        embeddings.append(item.get('embedding'))
            elif isinstance(response, dict):
                data = response.get('data', [])
                for item in data:
                    if isinstance(item, dict):
                        embeddings.append(item.get('embedding'))

            # Format response message
            if len(embeddings) == 1:
                message = f"Embedding vector with {len(embeddings[0])} dimensions"
            else:
                message = f"{len(embeddings)} embedding vectors"

            # Format response
            extra = {
                "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                "embeddings": embeddings,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else None,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else None,
                } if hasattr(response, 'usage') else None,
            }

            return LLMResponse(
                success=True,
                message=message,
                extra=extra
            )

        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            return LLMResponse(
                success=False,
                message=f"Rate limit error: {e.message}",
                extra={"error": str(e), "model": self.name}
            )
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return LLMResponse(
                success=False,
                message=f"API connection error: {str(e)}",
                extra={"error": str(e), "model": self.name}
            )
        except APIStatusError as e:
            logger.error(f"API status error: {e}")
            return LLMResponse(
                success=False,
                message=f"API status error: {e.message}",
                extra={"error": str(e), "status_code": e.status_code, "model": self.name}
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                extra={"error": str(e), "model": self.name}
            )

