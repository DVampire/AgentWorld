from collections.abc import Mapping
from typing import Any, Optional, Union, List, Dict, Type
import httpx

try:
    from openai.types.shared.chat_model import ChatModel
except ImportError:
    ChatModel = str

from pydantic import BaseModel, Field, ConfigDict

from src.message.types import Message
from src.model.openai.serializer import OpenAIResponseSerializer, OpenAIChatSerializer
from src.model.types import LLMResponse, LLMExtra
from src.model.newapi.rest import NewAPIClient
from src.logger import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool


class ResponseNewAPI(BaseModel):
    """
    A wrapper around NewAPIClient that provides a unified interface for the New-API responses endpoint.

    This class is for models that use the /responses API endpoint (e.g., reasoning models)
    via an OpenAI-compatible New-API deployment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Model configuration
    model: Union[ChatModel, str]

    # Model params for responses API
    reasoning: Optional[Dict[str, Any]] = None
    max_output_tokens: Optional[int] = 16384
    temperature: Optional[float] = None

    # Client initialization parameters
    api_key: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[Union[float, httpx.Timeout]] = httpx.Timeout(600.0, connect=30.0)
    default_headers: Optional[Mapping[str, str]] = None
    http_client: Optional[httpx.AsyncClient] = None

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

    def _get_usage(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract usage information from responses API response dict."""
        usage = None
        try:
            usage_data = None
            if isinstance(response, dict):
                usage_data = response.get('usage')
            elif hasattr(response, 'usage'):
                usage_data = response.usage

            if usage_data is None:
                return None

            if isinstance(usage_data, dict):
                input_tokens = usage_data.get('input_tokens') or usage_data.get('prompt_tokens', 0)
                output_tokens = usage_data.get('output_tokens') or usage_data.get('completion_tokens', 0)
                total_tokens = usage_data.get('total_tokens', 0)
                usage = {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': total_tokens,
                }
                # Reasoning tokens
                output_details = usage_data.get('output_tokens_details') or usage_data.get('completion_tokens_details')
                if isinstance(output_details, dict):
                    reasoning_tokens = output_details.get('reasoning_tokens')
                    if reasoning_tokens is not None:
                        usage['reasoning_tokens'] = reasoning_tokens
                # Cached tokens
                input_details = usage_data.get('input_tokens_details') or usage_data.get('prompt_tokens_details')
                if isinstance(input_details, dict):
                    cached = input_details.get('cached_tokens')
                    if cached is not None:
                        usage['prompt_cached_tokens'] = cached
            else:
                # Object with attributes
                input_tokens = getattr(usage_data, 'input_tokens', None) or getattr(usage_data, 'prompt_tokens', 0)
                output_tokens = getattr(usage_data, 'output_tokens', None) or getattr(usage_data, 'completion_tokens', 0)
                total_tokens = getattr(usage_data, 'total_tokens', 0)
                usage = {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': total_tokens,
                }
        except (AttributeError, TypeError) as e:
            logger.debug(f"Error extracting usage: {e}")
        return usage

    def _get_reasoning(self, response: Any) -> Optional[str]:
        """Extract reasoning text from responses API response dict."""
        reasoning = None
        try:
            output = None
            if isinstance(response, dict):
                output = response.get('output')
            elif hasattr(response, 'output'):
                output = response.output

            if isinstance(output, list):
                for item in output:
                    item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                    if item_type == 'reasoning':
                        if isinstance(item, dict):
                            reasoning = item.get('content') or item.get('text') or item.get('summary')
                        else:
                            reasoning = (
                                getattr(item, 'content', None)
                                or getattr(item, 'text', None)
                                or getattr(item, 'summary', None)
                            )
                        break
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            logger.debug(f"Error extracting reasoning: {e}")
        return reasoning

    def _extract_output_text(self, response: Any) -> str:
        """Extract output text from responses API response dict."""
        try:
            # Try output_text first
            if isinstance(response, dict):
                if response.get('output_text') is not None:
                    return response['output_text']
                output = response.get('output', [])
            else:
                if hasattr(response, 'output_text') and response.output_text is not None:
                    return response.output_text
                output = getattr(response, 'output', [])

            if isinstance(output, list):
                for item in output:
                    item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                    if item_type == 'message':
                        content = item.get('content', []) if isinstance(item, dict) else getattr(item, 'content', [])
                        if isinstance(content, list):
                            for part in content:
                                part_type = part.get('type') if isinstance(part, dict) else getattr(part, 'type', None)
                                if part_type == 'output_text':
                                    return part.get('text', '') if isinstance(part, dict) else getattr(part, 'text', '')
        except (AttributeError, KeyError, TypeError, IndexError):
            pass
        return ""

    async def _build_params(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build parameters for the responses API call."""
        input_messages = OpenAIResponseSerializer.serialize_messages(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
        }

        # Reasoning: convert {"reasoning_effort": "high"} → {"reasoning": {"effort": "high"}}
        if self.reasoning is not None:
            if "reasoning_effort" in self.reasoning:
                params["reasoning"] = {"effort": self.reasoning["reasoning_effort"]}
            else:
                params.update(self.reasoning)

        if self.max_output_tokens is not None:
            params["max_output_tokens"] = self.max_output_tokens

        # Response format (responses API uses text.format with flat structure)
        if response_format:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                optimized = OpenAIChatSerializer.serialize_response_format(response_format)
                schema = optimized['json_schema']['schema']
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema,
                    }
                }
            elif isinstance(response_format, BaseModel):
                model_class = type(response_format)
                optimized = OpenAIChatSerializer.serialize_response_format(model_class)
                schema = optimized['json_schema']['schema']
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": model_class.__name__,
                        "strict": True,
                        "schema": schema,
                    }
                }
            elif isinstance(response_format, dict):
                if "text" in response_format:
                    params["text"] = response_format["text"]
                elif "type" in response_format and "name" in response_format and "schema" in response_format:
                    params["text"] = {"format": response_format}
                elif "type" in response_format and "json_schema" in response_format:
                    json_schema_obj = response_format["json_schema"]
                    params["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": json_schema_obj.get("name", "response"),
                            "strict": json_schema_obj.get("strict", True),
                            "schema": json_schema_obj.get("schema", {}),
                        }
                    }
                else:
                    params["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": "response",
                            "strict": True,
                            "schema": response_format,
                        }
                    }
            else:
                logger.warning(f"Unsupported response_format type: {type(response_format)}")

        if tools:
            logger.warning("Tools may not be supported in responses API")

        if stream:
            logger.warning("Streaming may not be supported in responses API")

        params.update(kwargs)

        return {
            "input": input_messages,
            "params": params,
        }

    async def _call_model(
        self,
        input_messages: List[Dict[str, Any]],
        **params: Any,
    ) -> Any:
        """Call the /responses endpoint via NewAPIClient."""
        client = self.get_client()
        return await client.responses.create(**params)

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous completion call via New-API responses endpoint.

        Args:
            messages: List of Message objects (HumanMessage, SystemMessage, AssistantMessage)
            tools: Optional list of Tool instances (may not be supported in responses API)
            response_format: Optional response format (Pydantic model class, instance or dict)
            stream: Whether to stream the response (may not be supported in responses API)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with formatted message
        """
        try:
            params = await self._build_params(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )

            response = await self._call_model(
                input_messages=params["input"],
                **params["params"],
            )

            return await self._format_response(
                response=response,
                tools=tools,
                response_format=response_format,
            )

        except httpx.TimeoutException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ResponseNewAPI: {e}")
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )

    async def _format_response(
        self,
        response: Any,
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """Format responses API response into LLMResponse."""
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
                        extra=LLMExtra(data={"raw_response": response if isinstance(response, dict) else str(response)})
                    )

                import json
                try:
                    data = json.loads(output_text)
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
                            parsed_model=parsed_model,
                            data={
                                "raw_response": response if isinstance(response, dict) else str(response),
                                "usage": usage,
                                "reasoning": reasoning,
                            }
                        )
                    )
                except json.JSONDecodeError as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse JSON from response: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": output_text})
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": output_text})
                    )

            # Default: return content as string
            else:
                return LLMResponse(
                    success=True,
                    message=output_text,
                    extra=LLMExtra(
                        data={
                            "raw_response": response if isinstance(response, dict) else str(response),
                            "usage": usage,
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
