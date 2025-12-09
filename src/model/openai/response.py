from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional, Union, List, Dict, Type
import httpx

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
    from openai.types.shared.chat_model import ChatModel
    from openai.types.shared_params.reasoning_effort import ReasoningEffort
except ImportError:
    # Fallback if openai package is not available
    AsyncOpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    RateLimitError = Exception
    ChatModel = str
    ReasoningEffort = str

from pydantic import BaseModel, Field, ConfigDict

from src.message.types import Message
from src.model.openai.serializer import OpenAIResponseSerializer
from src.model.types import LLMResponse
from src.logger import logger


class ResponseOpenAI(BaseModel):
    """
    A wrapper around AsyncOpenAI that provides a unified interface for OpenAI responses API.
    
    This class is specifically designed for GPT-5 and o-series models that use the responses API endpoint.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Model configuration
    model: Union[ChatModel, str]

    # Model params for responses API
    reasoning_effort: ReasoningEffort = 'high'
    max_output_tokens: Optional[int] = 16384
    temperature: Optional[float] = None  # Not used for reasoning models, but kept for compatibility

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

    def _get_usage(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract usage information from responses API response."""
        usage = None
        try:
            if hasattr(response, 'usage') and response.usage is not None:
                usage_obj = response.usage
                
                # Responses API uses input_tokens and output_tokens instead of prompt_tokens and completion_tokens
                input_tokens = getattr(usage_obj, 'input_tokens', None)
                output_tokens = getattr(usage_obj, 'output_tokens', None)
                total_tokens = getattr(usage_obj, 'total_tokens', None)
                
                # Fallback to prompt_tokens/completion_tokens if input_tokens/output_tokens not available
                if input_tokens is None:
                    input_tokens = getattr(usage_obj, 'prompt_tokens', 0)
                if output_tokens is None:
                    output_tokens = getattr(usage_obj, 'completion_tokens', 0)
                if total_tokens is None:
                    total_tokens = getattr(usage_obj, 'total_tokens', 0)
                
                usage = {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': total_tokens,
                }
                
                # Handle reasoning tokens if available
                if hasattr(usage_obj, 'output_tokens_details'):
                    details = usage_obj.output_tokens_details
                    if details and hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens
                        if reasoning_tokens is not None:
                            usage['reasoning_tokens'] = reasoning_tokens
                elif hasattr(usage_obj, 'completion_tokens_details'):
                    # Fallback to completion_tokens_details
                    details = usage_obj.completion_tokens_details
                    if details and hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens
                        if reasoning_tokens is not None:
                            usage['reasoning_tokens'] = reasoning_tokens
                
                # Handle cached tokens if available
                if hasattr(usage_obj, 'input_tokens_details'):
                    prompt_details = usage_obj.input_tokens_details
                    if prompt_details and hasattr(prompt_details, 'cached_tokens'):
                        usage['prompt_cached_tokens'] = prompt_details.cached_tokens
                elif hasattr(usage_obj, 'prompt_tokens_details'):
                    # Fallback to prompt_tokens_details
                    prompt_details = usage_obj.prompt_tokens_details
                    if prompt_details and hasattr(prompt_details, 'cached_tokens'):
                        usage['prompt_cached_tokens'] = prompt_details.cached_tokens
        except (AttributeError, TypeError) as e:
            logger.debug(f"Error extracting usage: {e}")
            pass
        
        return usage

    def _get_reasoning(self, response: Any) -> Optional[str]:
        """Extract reasoning information from responses API response."""
        reasoning = None
        try:
            # Responses API may have reasoning in different places
            if hasattr(response, 'output') and response.output is not None:
                # Check output for reasoning
                output = response.output
                if isinstance(output, list):
                    for item in output:
                        # Handle both object attributes and dict access
                        item_type = None
                        if hasattr(item, 'type'):
                            item_type = item.type
                        elif isinstance(item, dict):
                            item_type = item.get('type')
                        
                        if item_type == 'reasoning':
                            # Try to get content from various possible fields
                            if hasattr(item, 'content') and item.content:
                                reasoning = item.content
                                break
                            elif isinstance(item, dict) and item.get('content'):
                                reasoning = item.get('content')
                                break
                            elif hasattr(item, 'text') and item.text:
                                reasoning = item.text
                                break
                            elif isinstance(item, dict) and item.get('text'):
                                reasoning = item.get('text')
                                break
                            elif hasattr(item, 'summary') and item.summary:
                                reasoning = item.summary
                                break
                            elif isinstance(item, dict) and item.get('summary'):
                                reasoning = item.get('summary')
                                break
                            
                            # If reasoning type found but no content, set to None
                            if reasoning is None:
                                reasoning = None
                                break
                elif isinstance(output, dict):
                    if output.get('type') == 'reasoning':
                        reasoning = output.get('content') or output.get('text') or output.get('summary') or None
            
            # Also check for reasoning_details
            if reasoning is None and hasattr(response, 'reasoning_details'):
                reasoning_details = response.reasoning_details
                if reasoning_details:
                    for detail in reasoning_details:
                        if hasattr(detail, 'type'):
                            detail_type = detail.type
                            if detail_type == "reasoning.text" and hasattr(detail, 'text'):
                                reasoning = detail.text
                                break
                            elif detail_type == "reasoning.summary" and hasattr(detail, 'summary'):
                                reasoning = detail.summary
                                break
                        elif isinstance(detail, dict):
                            detail_type = detail.get("type")
                            if detail_type == "reasoning.text":
                                reasoning = detail.get("text")
                                break
                            elif detail_type == "reasoning.summary":
                                reasoning = detail.get("summary")
                                break
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            logger.debug(f"Error extracting reasoning: {e}")
            pass
        
        return reasoning

    def _extract_output_text(self, response: Any) -> str:
        """Extract output text from responses API response."""
        text = ""
        try:
            # Try output_text attribute first
            if hasattr(response, 'output_text') and response.output_text is not None:
                return response.output_text
            
            # Try output attribute
            if hasattr(response, 'output') and response.output is not None:
                output = response.output
                if isinstance(output, list):
                    for item in output:
                        if hasattr(item, 'type'):
                            if item.type == 'message':
                                if hasattr(item, 'content'):
                                    content = item.content
                                    if isinstance(content, list):
                                        for content_item in content:
                                            if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                                if hasattr(content_item, 'text'):
                                                    return content_item.text
                                            elif isinstance(content_item, dict) and content_item.get('type') == 'output_text':
                                                return content_item.get('text', '')
                        elif isinstance(item, dict):
                            if item.get('type') == 'message':
                                content = item.get('content', [])
                                for content_item in content:
                                    if content_item.get('type') == 'output_text':
                                        return content_item.get('text', '')
                elif isinstance(output, dict):
                    if output.get('type') == 'message':
                        content = output.get('content', [])
                        for content_item in content:
                            if content_item.get('type') == 'output_text':
                                return content_item.get('text', '')
        except (AttributeError, KeyError, TypeError, IndexError):
            pass
        
        return text

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous completion call via OpenAI responses API.

        Args:
            messages: List of Message objects (HumanMessage, SystemMessage, AssistantMessage)
            tools: Optional list of tools for function calling (may not be supported in responses API)
            response_format: Optional response format (Pydantic model or dict)
            stream: Whether to stream the response (may not be supported in responses API)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with formatted message
        """
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install it with: pip install openai")

        # Serialize messages to responses API format
        input_messages = OpenAIResponseSerializer.serialize_messages(messages)

        try:
            client = self.get_client()
            
            # Build parameters for responses API
            params: dict[str, Any] = {
                "model": self.model,
                "input": input_messages,
            }

            # Add reasoning_effort (required for responses API)
            params["reasoning"] = {
                "effort": self.reasoning_effort,
            }

            # Add max_output_tokens if specified
            if self.max_output_tokens is not None:
                params["max_output_tokens"] = self.max_output_tokens

            # Handle response_format
            if response_format:
                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    # Pydantic model - convert to JSON schema format
                    json_schema = response_format.model_json_schema()
                    params["response_format"] = {
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'response',
                            'strict': True,
                            'schema': json_schema,
                        }
                    }
                elif isinstance(response_format, dict):
                    # Dict format - use directly
                    params["response_format"] = response_format
                else:
                    logger.warning(f"Unsupported response_format type: {type(response_format)}")

            # Note: tools and stream may not be supported in responses API
            if tools:
                logger.warning("Tools may not be supported in responses API")
                # params["tools"] = tools  # Uncomment if supported

            if stream:
                logger.warning("Streaming may not be supported in responses API")
                # params["stream"] = True  # Uncomment if supported

            # Merge additional kwargs
            params.update(kwargs)

            # Make the API call using responses API
            response = await client.responses.create(**params)

            # Format response
            return self._format_response(response, tools=tools, response_format=response_format)

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

    def _format_response(
        self,
        response: Any,
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """Format OpenAI responses API response into LLMResponse."""
        try:
            usage = self._get_usage(response)
            reasoning = self._get_reasoning(response)
            output_text = self._extract_output_text(response)

            # Handle structured output
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                if not output_text:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra={"raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)}
                    )
                
                # Parse JSON content
                import json
                try:
                    data = json.loads(output_text)
                    parsed_model = response_format.model_validate(data)
                    
                    # Format as string
                    model_name = response_format.__name__
                    model_dict = parsed_model.model_dump()
                    
                    field_lines = []
                    for field_name, field_value in model_dict.items():
                        field_lines.append(f"{field_name}={field_value!r}")
                    
                    formatted_message = f"Response result:\n\n{model_name}(\n"
                    formatted_message += ",\n".join(f"    {line}" for line in field_lines)
                    formatted_message += "\n)"
                    
                    extra = {
                        "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                        "parsed_model": parsed_model,
                        "usage": usage,
                        "reasoning": reasoning,
                    }
                    
                    return LLMResponse(
                        success=True,
                        message=formatted_message,
                        extra=extra
                    )
                except json.JSONDecodeError as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse JSON from response: {e}",
                        extra={"error": str(e), "content": output_text}
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra={"error": str(e), "content": output_text}
                    )

            # Default: return content as string
            else:
                extra = {
                    "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    "usage": usage,
                    "reasoning": reasoning,
                }
                
                return LLMResponse(
                    success=True,
                    message=output_text,
                    extra=extra
                )

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra={"error": str(e)}
            )

