import httpx
from typing import Optional, Dict, Any, List, Union
from collections.abc import Mapping

try:
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_function_tool_call_param import Function as ChatCompletionFunction
    from openai.types.completion_usage import CompletionUsage
except ImportError:
    ChatCompletion = dict
    Choice = dict
    ChatCompletionMessage = dict
    ChatCompletionMessageToolCall = dict
    ChatCompletionFunction = dict
    CompletionUsage = dict

from src.logger import logger


class NewAPICompletions:
    """New-API completions client (OpenAI-compatible REST API)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = 300.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else None
        self.default_headers = default_headers
        self.timeout = timeout
        self._http_client = http_client
        self._endpoint = "/chat/completions"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.default_headers:
            headers.update(self.default_headers)
        return headers

    def _get_api_url(self) -> str:
        return f"{self.base_url}{self._endpoint}"

    def _dict_to_chat_completion(self, data: Dict[str, Any]) -> ChatCompletion:
        """Convert dict response to ChatCompletion object (compatible with OpenAI SDK)."""
        if ChatCompletion == dict:
            return data

        choices_data = data.get("choices", [])
        choices = []
        for choice_data in choices_data:
            message_data = choice_data.get("message", {})

            tool_calls = None
            if message_data.get("tool_calls"):
                tool_calls = []
                for tc_data in message_data["tool_calls"]:
                    function_data = tc_data.get("function", {})
                    tool_call = ChatCompletionMessageToolCall(
                        id=tc_data.get("id", ""),
                        type="function",
                        function=ChatCompletionFunction(
                            name=function_data.get("name", ""),
                            arguments=function_data.get("arguments", "{}")
                        )
                    )
                    tool_calls.append(tool_call)

            message = ChatCompletionMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                tool_calls=tool_calls,
            )

            _VALID_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call"}
            raw_finish = choice_data.get("finish_reason") or "stop"
            finish_reason = raw_finish if raw_finish in _VALID_FINISH_REASONS else "stop"
            choice = Choice(
                finish_reason=finish_reason,
                index=choice_data.get("index", 0),
                message=message,
            )
            choices.append(choice)

        usage_data = data.get("usage", {})
        usage = None
        if usage_data:
            usage = CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return ChatCompletion(
            id=data.get("id", ""),
            choices=choices,
            created=data.get("created", 0),
            model=data.get("model", ""),
            object="chat.completion",
            usage=usage,
        )

    async def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        Create a chat completion (OpenAI-compatible).

        Args:
            model: Model identifier (e.g., "gpt-4o")
            messages: List of message dictionaries
            **kwargs: Additional parameters (temperature, max_tokens, tools, etc.)

        Returns:
            ChatCompletion object (compatible with OpenAI SDK format)
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        for key, value in kwargs.items():
            if value is None:
                continue
            if key == "max_completion_tokens":
                payload["max_tokens"] = value
            else:
                payload[key] = value

        headers = self._get_headers()
        api_url = self._get_api_url()

        timeout_obj = self.timeout
        if isinstance(timeout_obj, (int, float)):
            timeout_obj = httpx.Timeout(timeout_obj)

        try:
            if self._http_client:
                client = self._http_client
                should_close = False
            else:
                client = httpx.AsyncClient(timeout=timeout_obj)
                should_close = True

            try:
                response = await client.post(
                    url=api_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                response_dict = response.json()
                return self._dict_to_chat_completion(response_dict)
            finally:
                if should_close:
                    await client.aclose()
        except httpx.HTTPStatusError as e:
            logger.error(f"New-API HTTP error: {e}")
            try:
                error_detail = e.response.json()
                raise Exception(f"New-API request failed: {error_detail}")
            except:
                raise Exception(f"New-API request failed: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"New-API request error: {e}")
            raise Exception(f"New-API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in New-API request: {e}")
            raise


class NewAPIChatNamespace:
    """Chat namespace for NewAPIClient (mirrors OpenAI's chat namespace)."""

    def __init__(self, completions: NewAPICompletions):
        self.completions = completions


class NewAPIResponses:
    """New-API responses client for the /responses endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = 300.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else None
        self.default_headers = default_headers
        self.timeout = timeout
        self._http_client = http_client
        self._endpoint = "/responses"

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.default_headers:
            headers.update(self.default_headers)
        return headers

    def _get_api_url(self) -> str:
        return f"{self.base_url}{self._endpoint}"

    async def create(self, **params: Any) -> Dict[str, Any]:
        """
        Create a response via the /responses endpoint.

        Args:
            **params: API parameters (model, input, reasoning, max_output_tokens, etc.)

        Returns:
            Response dict from the API
        """
        headers = self._get_headers()
        api_url = self._get_api_url()

        timeout_obj = self.timeout
        if isinstance(timeout_obj, (int, float)):
            timeout_obj = httpx.Timeout(timeout_obj)

        try:
            if self._http_client:
                client = self._http_client
                should_close = False
            else:
                client = httpx.AsyncClient(timeout=timeout_obj)
                should_close = True

            try:
                response = await client.post(
                    url=api_url,
                    headers=headers,
                    json=params,
                )
                response.raise_for_status()
                return response.json()
            finally:
                if should_close:
                    await client.aclose()
        except httpx.HTTPStatusError as e:
            logger.error(f"New-API responses HTTP error: {e}")
            try:
                error_detail = e.response.json()
                raise Exception(f"New-API responses request failed: {error_detail}")
            except Exception:
                raise Exception(f"New-API responses request failed: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"New-API responses request error: {e}")
            raise Exception(f"New-API responses request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in New-API responses request: {e}")
            raise


class NewAPIClient:
    """New-API client (OpenAI-compatible, similar to AsyncOpenAI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = 300.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers
        self.timeout = timeout
        self._http_client = http_client

        completions = NewAPICompletions(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            timeout=timeout,
            http_client=http_client,
        )
        self.chat = NewAPIChatNamespace(completions=completions)
        self.responses = NewAPIResponses(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            timeout=timeout,
            http_client=http_client,
        )
