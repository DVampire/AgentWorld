from collections.abc import Mapping
from typing import Any, Optional, Union, List, Dict, Type
import httpx
import dirtyjson

try:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.shared.chat_model import ChatModel
except ImportError:
    ChatCompletion = dict
    ChatModel = str

from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.model.types import LLMResponse, LLMExtra
from src.message.types import Message
from src.model.newapi.serializer import NewAPIChatSerializer
from src.model.newapi.rest import NewAPIClient
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool


class ChatNewAPI(BaseModel):
    """
    A wrapper around NewAPIClient that provides a unified interface for New-API chat completions.

    New-API is an OpenAI-compatible API, so this class accepts standard OpenAI-like parameters
    and provides methods for chat completions with support for tools, response_format, and streaming.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Model configuration
    model: Union[ChatModel, str]

    # Model params
    temperature: Optional[float] = 0.7
    frequency_penalty: Optional[float] = 0.3
    seed: Optional[int] = None
    top_p: Optional[float] = None
    max_completion_tokens: Optional[int] = 16384

    # Client initialization parameters
    api_key: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[Union[float, httpx.Timeout]] = httpx.Timeout(600.0, connect=30.0)
    default_headers: Optional[Mapping[str, str]] = None
    http_client: Optional[httpx.AsyncClient] = None

    reasoning_models: Optional[List[Union[ChatModel, str]]] = Field(
        default_factory=lambda: []
    )

    @property
    def provider(self) -> str:
        return 'newapi'

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    @property
    def name(self) -> str:
        return str(self.model)

    def get_client(self) -> NewAPIClient:
        return NewAPIClient(
            api_key=self.api_key,
            base_url=str(self.base_url) if self.base_url else None,
            default_headers=self.default_headers,
            timeout=self.timeout,
            http_client=self.http_client,
        )

    def _get_usage(self, response: ChatCompletion) -> Optional[Dict[str, Any]]:
        if response.usage is not None:
            return response.usage.model_dump()
        return None

    def _get_reasoning(self, message) -> Optional[str]:
        reasoning = None
        try:
            if hasattr(message, 'reasoning') and message.reasoning is not None:
                reasoning = message.reasoning
            elif hasattr(message, 'reasoning_details') and message.reasoning_details is not None:
                for detail in message.reasoning_details:
                    if hasattr(detail, 'type'):
                        if detail.type == "reasoning.text" and hasattr(detail, 'text'):
                            reasoning = detail.text
                            break
                        elif detail.type == "reasoning.summary" and hasattr(detail, 'summary'):
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

    async def _build_params(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Serialize messages and build API parameters."""
        serialized_messages = NewAPIChatSerializer.serialize_messages(messages)

        params: Dict[str, Any] = {}

        if self.temperature is not None:
            params['temperature'] = self.temperature
        if self.frequency_penalty is not None:
            params['frequency_penalty'] = self.frequency_penalty
        if self.max_completion_tokens is not None:
            params['max_completion_tokens'] = self.max_completion_tokens
        if self.top_p is not None:
            params['top_p'] = self.top_p
        if self.seed is not None:
            params['seed'] = self.seed

        # Remove unsupported params for reasoning models
        if self.reasoning_models and any(str(m).lower() in str(self.model).lower() for m in self.reasoning_models):
            params.pop('temperature', None)
            params.pop('frequency_penalty', None)

        if tools:
            formatted_tools = NewAPIChatSerializer.serialize_tools(tools)
            if formatted_tools:
                params['tools'] = formatted_tools

        if response_format:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                params['response_format'] = NewAPIChatSerializer.serialize_response_format(response_format)
            elif isinstance(response_format, BaseModel):
                params['response_format'] = NewAPIChatSerializer.serialize_response_format(response_format)
            elif isinstance(response_format, dict):
                params['response_format'] = response_format
            else:
                logger.warning(f"Unsupported response_format type: {type(response_format)}")

        if stream:
            params['stream'] = True

        params.update(kwargs)

        return {
            "messages": serialized_messages,
            "params": params,
        }

    async def _call_model(
        self,
        messages: List[Dict[str, Any]],
        **params: Any,
    ) -> ChatCompletion:
        """Call the New-API model."""
        client = self.get_client()
        return await client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )

    async def _format_response(
        self,
        response: ChatCompletion,
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """Format New-API response into LLMResponse."""
        try:
            if not response.choices:
                return LLMResponse(
                    success=False,
                    message="No choices in response",
                    extra=LLMExtra(
                        data={"raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)}
                    )
                )

            message = response.choices[0].message
            usage = self._get_usage(response)
            finish_reason = response.choices[0].finish_reason
            reasoning = self._get_reasoning(message)

            # Handle function/tool calling
            if tools and message.tool_calls:
                formatted_lines = []
                functions = []

                for tool_call in message.tool_calls:
                    function_info = tool_call.function
                    name = function_info.name
                    arguments_str = function_info.arguments

                    try:
                        arguments = dirtyjson.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except (dirtyjson.Error, ValueError, TypeError):
                        arguments = {}

                    if arguments:
                        args_str = ", ".join([f"{k}={v!r}" for k, v in arguments.items()])
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")

                    functions.append({"name": name, "arguments": arguments})

                formatted_message = "\n".join(formatted_lines)

                return LLMResponse(
                    success=True,
                    message=formatted_message,
                    extra=LLMExtra(
                        data={
                            "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                            "functions": functions,
                            "usage": usage,
                            "finish_reason": finish_reason,
                            "reasoning": reasoning,
                        }
                    )
                )

            # Handle structured output
            elif response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                content = message.content or ""
                if not content:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra=LLMExtra(
                            data={"raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)}
                        )
                    )

                try:
                    data = dirtyjson.loads(content)
                    parsed_model = response_format.model_validate(data)

                    model_name = response_format.__name__
                    model_dict = parsed_model.model_dump()
                    field_lines = [f"{k}={v!r}" for k, v in model_dict.items()]
                    formatted_message = f"Response result:\n\n{model_name}(\n"
                    formatted_message += ",\n".join(f"    {line}" for line in field_lines)
                    formatted_message += "\n)"

                    return LLMResponse(
                        success=True,
                        message=formatted_message,
                        extra=LLMExtra(
                            data={
                                "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                                "usage": usage,
                                "finish_reason": finish_reason,
                                "reasoning": reasoning,
                            },
                            parsed_model=parsed_model
                        )
                    )
                except (dirtyjson.Error, ValueError, TypeError) as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse JSON from response: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": content})
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": content})
                    )

            # Default: return content as string
            else:
                content = message.content or ""
                return LLMResponse(
                    success=True,
                    message=content,
                    extra=LLMExtra(
                        data={
                            "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                            "usage": usage,
                            "finish_reason": finish_reason,
                            "reasoning": reasoning,
                        }
                    )
                )

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra=LLMExtra(data={"error": str(e)})
            )

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous completion call via New-API.

        Args:
            messages: List of Message objects (HumanMessage, SystemMessage, AssistantMessage)
            tools: Optional list of Tool instances
            response_format: Optional response format (Pydantic model or dict)
            stream: Whether to stream the response
            **kwargs: Additional parameters passed to the API

        Returns:
            LLMResponse with formatted message
        """
        try:
            built = await self._build_params(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )

            response = await self._call_model(
                messages=built["messages"],
                **built["params"],
            )

            return await self._format_response(
                response=response,
                tools=tools,
                response_format=response_format,
            )

        except httpx.TimeoutException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ChatNewAPI: {e}")
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )
