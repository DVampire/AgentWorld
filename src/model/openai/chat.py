from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional, Union, List, Dict, Type, overload
import httpx

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
    from openai.types.chat import ChatCompletionContentPartTextParam
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.shared.chat_model import ChatModel
    from openai.types.shared_params.reasoning_effort import ReasoningEffort
    from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
except ImportError:
    # Fallback if openai package is not available
    AsyncOpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    RateLimitError = Exception
    ChatCompletion = dict
    ChatModel = str
    ReasoningEffort = str
    JSONSchema = dict
    ResponseFormatJSONSchema = dict
    ChatCompletionContentPartTextParam = dict

from pydantic import BaseModel, Field, ConfigDict

from src.message.types import Message, HumanMessage, SystemMessage, AssistantMessage
from src.model.openai.serializer import OpenAIChatSerializer
from src.model.types import LLMResponse
from src.logger import logger


class ChatOpenAI(BaseModel):
    """
    A wrapper around AsyncOpenAI that provides a unified interface for OpenAI chat completions.
    
    This class accepts AsyncOpenAI parameters and provides methods for chat completions
    with support for tools, response_format, and streaming.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Model configuration
    model: Union[ChatModel, str]

    # Model params
    temperature: Optional[float] = 0.7
    frequency_penalty: Optional[float] = 0.3
    reasoning_effort: ReasoningEffort = 'low'
    seed: Optional[int] = None
    service_tier: Optional[Literal['auto', 'default', 'flex', 'priority', 'scale']] = None
    top_p: Optional[float] = None
    max_completion_tokens: Optional[int] = 16384

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

    reasoning_models: Optional[List[Union[ChatModel, str]]] = Field(
        default_factory=lambda: [
            'o3',
            'gpt-5',
            'gpt-5.1',
        ]
    )

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

    def _get_usage(self, response: ChatCompletion) -> Optional[Dict[str, Any]]:
        """Extract usage information from response."""
        if response.usage is not None:
            completion_tokens = response.usage.completion_tokens
            completion_token_details = response.usage.completion_tokens_details
            
            if completion_token_details is not None:
                reasoning_tokens = completion_token_details.reasoning_tokens
                if reasoning_tokens is not None:
                    completion_tokens += reasoning_tokens
            
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
            
            if response.usage.prompt_tokens_details is not None:
                usage['prompt_cached_tokens'] = response.usage.prompt_tokens_details.cached_tokens
        else:
            usage = None
        
        return usage

    def _get_reasoning(self, message) -> Optional[str]:
        """Extract reasoning information from message."""
        reasoning = None
        try:
            # Try to get reasoning directly from message
            if hasattr(message, 'reasoning') and message.reasoning is not None:
                reasoning = message.reasoning
            elif hasattr(message, 'reasoning_details') and message.reasoning_details is not None:
                # Try to extract from reasoning_details
                reasoning_details = message.reasoning_details
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
        except (AttributeError, KeyError, TypeError, IndexError):
            pass
        
        return reasoning

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous completion call via OpenAI API.

        Args:
            messages: List of Message objects (HumanMessage, SystemMessage, AssistantMessage)
            tools: Optional list of tools for function calling
            response_format: Optional response format (Pydantic model or dict)
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            LLMResponse with formatted message
        """
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install it with: pip install openai")

        # Serialize messages to OpenAI format
        openai_messages = OpenAIChatSerializer.serialize_messages(messages)

        try:
            client = self.get_client()
            
            # Build model parameters
            model_params: dict[str, Any] = {}

            if self.temperature is not None:
                model_params['temperature'] = self.temperature
            if self.frequency_penalty is not None:
                model_params['frequency_penalty'] = self.frequency_penalty
            if self.max_completion_tokens is not None:
                model_params['max_completion_tokens'] = self.max_completion_tokens
            if self.top_p is not None:
                model_params['top_p'] = self.top_p
            if self.seed is not None:
                model_params['seed'] = self.seed
            if self.service_tier is not None:
                model_params['service_tier'] = self.service_tier

            # Handle reasoning models
            if self.reasoning_models and any(str(m).lower() in str(self.model).lower() for m in self.reasoning_models):
                model_params['reasoning_effort'] = self.reasoning_effort
                # Remove temperature and frequency_penalty for reasoning models
                model_params.pop('temperature', None)
                model_params.pop('frequency_penalty', None)

            # Add tools if provided
            if tools:
                model_params['tools'] = tools

            # Handle response_format
            if response_format:
                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    # Pydantic model - convert to JSON schema format
                    json_schema = response_format.model_json_schema()
                    model_params['response_format'] = {
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'response',
                            'strict': True,
                            'schema': json_schema,
                        }
                    }
                elif isinstance(response_format, dict):
                    # Dict format - use directly
                    model_params['response_format'] = response_format
                else:
                    logger.warning(f"Unsupported response_format type: {type(response_format)}")

            # Handle streaming
            if stream:
                model_params['stream'] = True

            # Merge additional kwargs
            model_params.update(kwargs)

            # Make the API call
            response = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **model_params,
            )

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
        response: ChatCompletion,
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """Format OpenAI response into LLMResponse."""
        try:
            if not response.choices:
                return LLMResponse(
                    success=False,
                    message="No choices in response",
                    extra={"raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)}
                )

            message = response.choices[0].message
            usage = self._get_usage(response)
            finish_reason = response.choices[0].finish_reason
            reasoning = self._get_reasoning(message)

            # Handle function calling
            if tools and message.tool_calls:
                # Format tool_calls as string
                formatted_lines = []
                functions = []
                
                for tool_call in message.tool_calls:
                    function_info = tool_call.function
                    name = function_info.name
                    arguments_str = function_info.arguments
                    
                    # Parse arguments if it's a string
                    import json
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    # Format arguments as keyword arguments
                    if arguments:
                        args_str = ", ".join([f"{k}={v!r}" for k, v in arguments.items()])
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")
                    
                    functions.append({
                        "name": name,
                        "args": arguments
                    })
                
                formatted_message = "\n".join(formatted_lines)
                
                extra = {
                    "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    "functions": functions,
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "reasoning": reasoning,
                }
                
                return LLMResponse(
                    success=True,
                    message=formatted_message,
                    extra=extra
                )

            # Handle structured output
            elif response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                content = message.content or ""
                if not content:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra={"raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)}
                    )
                
                # Parse JSON content
                import json
                try:
                    data = json.loads(content)
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
                        "finish_reason": finish_reason,
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
                        extra={"error": str(e), "content": content}
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra={"error": str(e), "content": content}
                    )

            # Default: return content as string
            else:
                content = message.content or ""
                
                extra = {
                    "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "reasoning": reasoning,
                }
                
                return LLMResponse(
                    success=True,
                    message=content,
                    extra=extra
                )

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra={"error": str(e)}
            )

