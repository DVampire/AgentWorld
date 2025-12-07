"""
Completion handler for LLM chat completions.
Handles building parameters and formatting responses for completion API calls.
"""

import json
from typing import Optional, Dict, List, Any, Union, Type
from pydantic import BaseModel
import jsonlines
from io import StringIO
import numpy as np

from src.message.message_manager import message_manager
from src.model.types import ModelConfig, LLMResponse
from src.logger import logger
from src.model.api_request import APIRequest


class CompletionHandler:
    """Handler for LLM completion operations."""
    
    def __init__(self, models: Dict[str, ModelConfig]):
        """
        Initialize the completion handler.
        
        Args:
            models: Dictionary of model configurations
        """
        self.models = models
    
    def build_params(
        self,
        model: str,
        messages: List[Any],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        use_responses_api: bool = False,
        plugins: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare a parameters dictionary consumable by litellm for completion.
        
        Args:
            model: Model name
            messages: List of messages
            tools: Optional list of tools
            response_format: Optional response format
            stream: Whether to stream the response
            use_responses_api: Whether to use responses API
            plugins: Optional list of plugins
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of parameters for litellm completion
        """
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openai")
        
        # Handle embedding models
        if config.model_type == "embedding":
            return self._build_embedding_params(model, messages, **kwargs)
        
        # Handle completion models
        return self._build_completion_params(model, messages, tools, response_format, stream, use_responses_api, plugins, **kwargs)
    
    def _build_completion_params(self,
                                 model: str,
                                 messages: List[Any],
                                 tools: Optional[Union[List[Dict], List[Any]]] = None,
                                 response_format: Optional[Union[BaseModel, Dict]] = None,
                                 stream: bool = False,
                                 use_responses_api: bool = False,
                                 plugins: Optional[List[Dict[str, Any]]] = None,
                                 **kwargs: Any) -> Dict[str, Any]:
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openai")
        
        provider = config.provider
        model_id = config.model_id
        
        param_type = "general" # "general", "perplexity"
        
        if provider == "openai":
            param_type = "general"
        elif provider == "openrouter":
            model_source = model_id.split("/")[0]
            if model_source == "perplexity":
                param_type = "perplexity"
            else:
                param_type = "general"
                
        if param_type == "general":
            params = self._build_general_params(model, messages, tools, response_format, stream, use_responses_api, plugins, **kwargs)
        elif param_type == "perplexity":
            params = self._build_perplexity_params(model, messages, tools, response_format, stream, use_responses_api, plugins, **kwargs)

        return params
    
    def _build_embedding_params(self,
                                 model: str,
                                 messages: List[Any],
                                 encoding_format: str = "float",
                                 **kwargs: Any
                                 ) -> Dict[str, Any]:
        """Build parameters for embedding requests."""
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openrouter", model_type="embedding")
        
        # Get formatted payload from message_manager (for messages)
        payload = message_manager(
            messages=messages, 
            model_config=config,
        )
        
        params: Dict[str, Any] = {
            "model": config.model_id,
            "encoding_format": encoding_format,
            **config.default_params,
            **kwargs,
        }
        
        # Add messages from payload
        if "messages" in payload:
            params["messages"] = payload["messages"]
        if "input" in payload:
            params["messages"] = payload["input"]  # For responses API compatibility
        
        if config.api_key:
            params["api_key"] = config.api_key
        if config.api_base:
            params["api_base"] = config.api_base

        return params
        
    def _build_general_params(self, 
                              model: str,
                              messages: List[Any],
                              tools: Optional[Union[List[Dict], List[Any]]] = None,
                              response_format: Optional[Union[BaseModel, Dict]] = None,
                              stream: bool = False,
                              use_responses_api: bool = False,
                              plugins: Optional[List[Dict[str, Any]]] = None,
                              **kwargs: Any
                              ) -> Dict[str, Any]:
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openai")

        # Get formatted payload from message_manager (includes messages, tools, response_format)
        payload = message_manager(
            messages=messages, 
            model_config=config,
            tools=tools,
            response_format=response_format
        )

        params: Dict[str, Any] = {
            "model": config.model_id,
            "stream": stream,
            **config.default_params,
            **kwargs,
        }

        # Add messages or input from payload
        if "messages" in payload:
            params["messages"] = payload["messages"]
        if "input" in payload:
            params["input"] = payload["input"]

        # Add tools from payload if present
        if "tools" in payload:
            params["tools"] = payload["tools"]

        # Add response_format from payload if present
        if "response_format" in payload:
            params["response_format"] = payload["response_format"]

        if config.max_tokens and "max_tokens" not in params:
            params["max_tokens"] = config.max_tokens

        if config.api_key:
            params["api_key"] = config.api_key
        if config.api_base:
            params["api_base"] = config.api_base

        return params
    
    def _build_perplexity_params(self, 
                                 model: str,
                                 messages: List[Any],
                                 tools: Optional[Union[List[Dict], List[Any]]] = None,
                                 response_format: Optional[Union[BaseModel, Dict]] = None,
                                 stream: bool = False,
                                 use_responses_api: bool = False,
                                 plugins: Optional[List[Dict[str, Any]]] = None,
                                 **kwargs: Any
                                 ) -> Dict[str, Any]:
        
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openai")
        
        # Get formatted payload from message_manager (includes messages, tools, response_format)
        payload = message_manager(
            messages=messages, 
            model_config=config,
            tools=tools,
            response_format=response_format,
            plugins=plugins
        )
        
        params: Dict[str, Any] = {
            "model": f"{config.provider}/{config.model_id}",
        }
        
        if "messages" in payload:
            params["messages"] = payload["messages"]
            
        if config.api_key:
            params["api_key"] = config.api_key
        if config.api_base:
            params["api_base"] = config.api_base
        
        return params
    
    def format_response(
        self,
        response: Dict[str, Any],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        model_type: str = "completion",
        num_messages: Optional[int] = None,
        is_single_input: bool = False,
    ) -> LLMResponse:
        """
        Format response into LLMResponse based on model type.
        
        Args:
            response: Raw response from API
            tools: Optional list of tools (if function calling was used)
            response_format: Optional response format (if structured output was used)
            model_type: Type of model ("completion" or "embedding")
            num_messages: Number of messages (for embedding response message)
            is_single_input: Whether the input was a single text (for embedding)
            
        Returns:
            LLMResponse with formatted message
        """
        # Handle embedding responses
        if model_type == "embedding":
            return self._format_embedding_response(response, num_messages, is_single_input)
        
        # Handle completion responses
        return self._format_completion_response(response, tools, response_format)
    
    def _format_completion_response(
        self,
        response: Dict[str, Any],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> LLMResponse:
        """
        Format completion response into LLMResponse.
        
        Args:
            response: Raw response from API
            tools: Optional list of tools (if function calling was used)
            response_format: Optional response format (if structured output was used)
            
        Returns:
            LLMResponse with formatted message
        """
        try:
            # Function calling path
            if tools:
                # Parse tool_calls from response
                tool_calls = []
                try:
                    if "choices" in response:
                        message = response["choices"][0]["message"]
                        tool_calls = message.get("tool_calls", [])
                    elif "output" in response:
                        for item in response["output"]:
                            if item.get("type") == "message":
                                tool_calls = item.get("tool_calls", [])
                                if tool_calls:
                                    break
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Failed to parse function calling response: {e}")
                
                functions = []
                if tool_calls:
                    # Format tool_calls as string: "Calling function add(a=1, b=2)"
                    formatted_lines = []
                    for tool_call in tool_calls:
                        function_info = tool_call.get("function", {})
                        name = function_info.get("name", "")
                        arguments_str = function_info.get("arguments", "")
                        
                        # Parse arguments if it's a string
                        if isinstance(arguments_str, str):
                            try:
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                arguments = {}
                        else:
                            arguments = arguments_str
                        
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
                    # Extract usage and reasoning
                    usage = response.get("usage")
                    reasoning = None
                    try:
                        if "choices" in response and len(response["choices"]) > 0:
                            message = response["choices"][0].get("message", {})
                            reasoning = message.get("reasoning")
                            if reasoning is None and "reasoning_details" in message:
                                reasoning_details = message.get("reasoning_details", [])
                                if reasoning_details:
                                    for detail in reasoning_details:
                                        if detail.get("type") == "reasoning.text":
                                            reasoning = detail.get("text")
                                            break
                                        elif detail.get("type") == "reasoning.summary":
                                            reasoning = detail.get("summary")
                                            break
                    except (KeyError, IndexError, TypeError):
                        pass
                    
                    extra = {
                        "raw_response": response,
                        "functions": functions,
                        "usage": usage,
                        "reasoning": reasoning
                    }
                    return LLMResponse(
                        success=True,
                        message=formatted_message,
                        extra=extra
                    )
                else:
                    # No tool_calls found, extract content as string
                    content = ""
                    try:
                        if "choices" in response:
                            content = response["choices"][0]["message"].get("content") or ""
                        elif "output" in response:
                            for item in response["output"]:
                                if item.get("type") == "message":
                                    for content_item in item.get("content", []):
                                        if content_item.get("type") == "output_text":
                                            content = content_item.get("text", "")
                                            break
                    except (KeyError, IndexError, TypeError):
                        pass
                    
                    # Extract usage and reasoning
                    usage = response.get("usage")
                    reasoning = None
                    try:
                        if "choices" in response and len(response["choices"]) > 0:
                            message = response["choices"][0].get("message", {})
                            reasoning = message.get("reasoning")
                            if reasoning is None and "reasoning_details" in message:
                                reasoning_details = message.get("reasoning_details", [])
                                if reasoning_details:
                                    for detail in reasoning_details:
                                        if detail.get("type") == "reasoning.text":
                                            reasoning = detail.get("text")
                                            break
                                        elif detail.get("type") == "reasoning.summary":
                                            reasoning = detail.get("summary")
                                            break
                    except (KeyError, IndexError, TypeError):
                        pass
                    
                    extra = {
                        "raw_response": response,
                        "usage": usage,
                        "reasoning": reasoning
                    }
                    return LLMResponse(
                        success=True,
                        message=content,
                        extra=extra
                    )
            
            # Structured output path
            elif response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Extract content
                content = ""
                try:
                    if "choices" in response:
                        content = response["choices"][0]["message"].get("content") or ""
                    elif "output" in response:
                        for item in response["output"]:
                            if item.get("type") == "message":
                                for content_item in item.get("content", []):
                                    if content_item.get("type") == "output_text":
                                        content = content_item.get("text", "")
                                        break
                except (KeyError, IndexError, TypeError):
                    pass
                
                if not content:
                    raise ValueError("Empty response content from model")
                
                # Parse JSON: try standard JSON first, then JSONL format if needed
                data = None
                try:
                    # First, try parsing as a single JSON object (handles formatted JSON with newlines)
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    # If that fails with "Extra data" error, try JSONL format (one JSON per line)
                    if "Extra data" in str(e):
                        try:
                            # JSONL format: each line is a complete JSON object
                            parsed_objects = list(jsonlines.Reader(StringIO(content)))
                            if parsed_objects:
                                data = parsed_objects[0]
                                if len(parsed_objects) > 1:
                                    logger.warning(f"Multiple JSON objects found in response, using the first one")
                            else:
                                raise ValueError("No valid JSON object found in JSONL format")
                        except Exception as jsonl_error:
                            raise ValueError(f"Could not parse JSON from content. JSON error: {e}, JSONL error: {jsonl_error}. First 500 chars: {content[:500]}...") from e
                    else:
                        # Other JSON decode errors, re-raise
                        raise ValueError(f"Could not parse JSON from content. Error: {e}. First 500 chars: {content[:500]}...") from e
                
                if data is None:
                    raise ValueError(f"Could not extract valid JSON from content. First 500 chars: {content[:500]}...")
                
                try:
                    parsed_model = response_format.model_validate(data)
                    
                    # Format as string: "Response result:\n\nResponse(\nfield1=value1,\nfield2=value2\n)"
                    model_name = response_format.__name__
                    model_dict = parsed_model.model_dump()
                    
                    # Format fields as keyword arguments
                    field_lines = []
                    for field_name, field_value in model_dict.items():
                        field_lines.append(f"{field_name}={field_value!r}")
                    
                    formatted_message = f"Response result:\n\n{model_name}(\n"
                    formatted_message += ",\n".join(f"    {line}" for line in field_lines)
                    formatted_message += "\n)"
                    
                    # Extract usage and reasoning
                    usage = response.get("usage")
                    reasoning = None
                    try:
                        if "choices" in response and len(response["choices"]) > 0:
                            message = response["choices"][0].get("message", {})
                            reasoning = message.get("reasoning")
                            if reasoning is None and "reasoning_details" in message:
                                reasoning_details = message.get("reasoning_details", [])
                                if reasoning_details:
                                    for detail in reasoning_details:
                                        if detail.get("type") == "reasoning.text":
                                            reasoning = detail.get("text")
                                            break
                                        elif detail.get("type") == "reasoning.summary":
                                            reasoning = detail.get("summary")
                                            break
                    except (KeyError, IndexError, TypeError):
                        pass
                    
                    extra = {
                        "raw_response": response,
                        "parsed_model": parsed_model,
                        "usage": usage,
                        "reasoning": reasoning
                    }
                    return LLMResponse(
                        success=True,
                        message=formatted_message,
                        extra=extra
                    )
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON from response. Error: {e}") from e
                except Exception as e:
                    raise ValueError(f"Failed to validate response against schema. Error: {e}") from e
            
            # Default: return content as string
            else:
                content = ""
                try:
                    if "choices" in response:
                        content = response["choices"][0]["message"].get("content") or ""
                    elif "output" in response:
                        for item in response["output"]:
                            if item.get("type") == "message":
                                for content_item in item.get("content", []):
                                    if content_item.get("type") == "output_text":
                                        content = content_item.get("text", "")
                                        break
                except (KeyError, IndexError, TypeError):
                    pass
                
                # Extract usage and reasoning
                usage = response.get("usage")
                reasoning = None
                try:
                    if "choices" in response and len(response["choices"]) > 0:
                        message = response["choices"][0].get("message", {})
                        reasoning = message.get("reasoning")
                        if reasoning is None and "reasoning_details" in message:
                            reasoning_details = message.get("reasoning_details", [])
                            if reasoning_details:
                                for detail in reasoning_details:
                                    if detail.get("type") == "reasoning.text":
                                        reasoning = detail.get("text")
                                        break
                                    elif detail.get("type") == "reasoning.summary":
                                        reasoning = detail.get("summary")
                                        break
                except (KeyError, IndexError, TypeError):
                    pass
                
                extra = {
                    "raw_response": response,
                    "usage": usage,
                    "reasoning": reasoning
                }
                return LLMResponse(
                    success=True,
                    message=content,
                    extra=extra
                )
        except Exception as e:
            logger.error(f"Failed to format completion response. Error: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format completion response. Error: {e}",
                extra=None
            )
    
    def _format_embedding_response(
        self,
        response: Dict[str, Any],
        num_messages: Optional[int] = None,
        is_single_input: bool = False,
    ) -> LLMResponse:
        """
        Format embedding response into LLMResponse.
        
        Args:
            response: Raw response from embedding API
            num_messages: Number of messages that were embedded
            is_single_input: Whether the input was a single text (not a list)
            
        Returns:
            LLMResponse with embeddings in extra field
            - If single input: embeddings is a single np.array
            - If multiple inputs: embeddings is a list of np.arrays
        """
        try:
            # Extract embeddings from response
            embeddings_raw = []
            if "data" in response:
                for item in response["data"]:
                    if "embedding" in item:
                        embeddings_raw.append(item["embedding"])
            
            # Convert to numpy arrays
            embeddings = [np.array(emb) for emb in embeddings_raw]
            
            # Determine message based on number of messages
            if num_messages is None:
                num_messages = len(embeddings)
            
            # If single input, return single embedding (not list)
            if is_single_input and len(embeddings) == 1:
                embeddings_result = embeddings[0]
                message = f"Successfully generated embedding for message (dimension: {len(embeddings[0])})"
            else:
                embeddings_result = embeddings
                if num_messages == 1:
                    message = f"Successfully generated embedding for message (dimension: {len(embeddings[0]) if embeddings else 0})"
                else:
                    message = f"Successfully generated embeddings for {num_messages} messages (dimension: {len(embeddings[0]) if embeddings else 0})"
            
            extra = {
                "embeddings": embeddings_result,
                "raw_response": response,
                "usage": response.get("usage"),
            }
            
            return LLMResponse(
                success=True,
                message=message,
                extra=extra
            )
        except Exception as e:
            logger.error(f"Failed to format embedding response. Error: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format embedding response. Error: {e}",
                extra={"error": str(e)}
            )
    
    def completion(
        self,
        model: str,
        messages: List[Any],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        use_responses_api: bool = False,
        plugins: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute synchronous completion call via HTTP requests.
        
        Args:
            model: Model name
            messages: List of messages
            tools: Optional list of tools
            response_format: Optional response format
            stream: Whether to stream the response
            use_responses_api: Whether to use responses API
            plugins: Optional list of plugins
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with formatted message
        """
        # Build params first
        params = self.build_params(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            stream=stream,
            use_responses_api=use_responses_api,
            plugins=plugins,
            **kwargs,
        )
        
        # Get config
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openrouter")
        
        # Make request
        response = APIRequest.request(config, params)
        return self.format_response(response, tools=tools, response_format=response_format)
    
    async def acompletion(
        self,
        model: str,
        messages: List[Any],
        tools: Optional[Union[List[Dict], List[Any]]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        use_responses_api: bool = False,
        plugins: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous completion call via HTTP requests.
        
        Args:
            model: Model name
            messages: List of messages
            tools: Optional list of tools
            response_format: Optional response format
            stream: Whether to stream the response
            use_responses_api: Whether to use responses API
            plugins: Optional list of plugins
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with formatted message
        """
        # Build params first
        params = self.build_params(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            stream=stream,
            use_responses_api=use_responses_api,
            plugins=plugins,
            **kwargs,
        )
        
        # Get config
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openrouter")
        
        # Make async request
        response = await APIRequest.arequest(config, params)
        return self.format_response(response, tools=tools, response_format=response_format)
    
    def embedding(
        self,
        model: str,
        messages: List[Any],
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute synchronous embedding call via HTTP requests.
        
        Args:
            model: Model name (must be an embedding model)
            messages: List of messages to embed
            encoding_format: Format of the embeddings ("float" or "base64")
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with embeddings in extra field
        """
        # Build params first
        params = self.build_params(
            model=model,
            messages=messages,
            encoding_format=encoding_format,
            **kwargs,
        )
        
        # Get config
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openrouter", model_type="embedding")
        
        # Make request
        try:
            response = APIRequest.request(config, params)
            # Determine if single input (1 message = single input)
            is_single_input = len(messages) == 1
            return self.format_response(
                response,
                model_type="embedding",
                num_messages=len(messages),
                is_single_input=is_single_input
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
        Execute asynchronous embedding call via HTTP requests.
        
        Args:
            model: Model name (must be an embedding model)
            messages: List of messages to embed
            encoding_format: Format of the embeddings ("float" or "base64")
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with embeddings in extra field
        """
        # Build params first
        params = self.build_params(
            model=model,
            messages=messages,
            encoding_format=encoding_format,
            **kwargs,
        )
        
        # Get config
        config = self.models.get(model) or ModelConfig(model_name=model, provider="openrouter", model_type="embedding")
        
        # Make async request
        try:
            response = await APIRequest.arequest(config, params)
            # Determine if single input (1 message = single input)
            is_single_input = len(messages) == 1
            return self.format_response(
                response,
                model_type="embedding",
                num_messages=len(messages),
                is_single_input=is_single_input
            )
        except Exception as e:
            logger.error(f"Async embedding request failed: {e}")
            return LLMResponse(
                success=False,
                message=f"Embedding request failed: {e}",
                extra={"error": str(e)}
            )

