from tkinter import N
from typing import Optional, Any, List, Union
import json
import requests
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


from src.logger import logger
from src.infrastructures.models.message_manager import MessageManager


class RestfulClient():
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 api_type: str = "chat/completions",
                 model: str = "o3"):
        self.base_url = base_url
        self.api_key = api_key
        self.api_type = api_type
        self.model = model
        
    def completion(self,
                   model,
                   messages,
                   **kwargs):

        headers = {
            "app_key": self.api_key,
            "Content-Type": "application/json"
        }

        model = model.split("/")[-1]
        data = {
            "model": model,
            "messages": messages,
        }

        # Add any additional kwargs to the data
        if kwargs:
            data.update(kwargs)
            
        try:
            response = requests.post(
                f"{self.base_url}/{self.api_type}",
                json=data,
                headers=headers,
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error calling {self.api_type} API: {e}")
            raise e
        return None

    async def acompletion(self,
                          model,
                          messages,
                          **kwargs):

        headers = {
            "app_key": self.api_key,
            "Content-Type": "application/json"
        }

        model = model.split("/")[-1]
        data = {
            "model": model,
            "messages": messages,
        }

        # Add any additional kwargs to the data
        if kwargs:
            data.update(kwargs)

        try:
            response = await self.http_client.post(
                f"{self.base_url}/{self.api_type}",
                json=data,
                headers=headers,
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error calling {self.api_type} API: {e}")
            raise e
        return None
    

class RestfulSearchClient():
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 api_type: str = "responses",
                 model: str = "o3",
                 verbose: bool = False):
        self.base_url = base_url
        self.api_key = api_key
        self.api_type = api_type
        self.model = model
        self.verbose = verbose

    def completion(self,
                   model,
                   input,
                   tools,
                   **kwargs):

        headers = {
            "app_key": self.api_key,
            "Content-Type": "application/json"
        }

        model = model.split("/")[-1]
        data = {
            "model": model,
            "input": input,
            "tools": tools,
            "stream": False,
        }

        # Add any additional kwargs to the data
        if kwargs:
            data.update(kwargs)

        response = requests.post(
            f"{self.base_url}/{self.api_type}",
            json=data,
            headers=headers,
        )
        
        print(response.text)

        response_text = response.text
        for line in response_text.split('\n'):
            if line.strip():
                try:
                    json_line = line.strip()
                    if json_line.startswith("data: ") and "response.completed" in json_line:
                        json_line = json_line.replace("data: ", "").strip()
                        res = json.loads(json_line)['response']
                        return res
                except Exception as e:
                    logger.error(f"Error parsing line: {line}, error: {e}")
        return None
                    
    async def acompletion(self,
                          model,
                          input,
                          tools,
                          **kwargs):
        
        headers = {
            "app_key": self.api_key,
            "Content-Type": "application/json"
        }
        
        
        model = model.split("/")[-1]
        data = {
            "model": model,
            "input": input,
            "tools": tools,
            "stream": False,
        }
        
        # Add any additional kwargs to the data
        if kwargs:
            data.update(kwargs)

        response = await requests.post(
            f"{self.base_url}/{self.api_type}",
            json=data,
            headers=headers,
        )
        
        response_text = response.text
        for line in response_text.split('\n'):
            if line.strip():
                try:
                    json_line = line.strip()
                    if json_line.startswith("data: ") and "response.completed" in json_line:
                        json_line = json_line.replace("data: ", "").strip()
                        res = json.loads(json_line)['response']
                        return res
                except Exception as e:
                    logger.error(f"Error parsing line: {line}, error: {e}")
        return None


class ChatRestful():
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_type: str = "chat/completions",
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_type = api_type
        
        self.client = self.create_client()
        self.message_manager = MessageManager(api_type=api_type, model=model)

        super(ChatRestful, self).__init__(**kwargs)

    def create_client(self):
        return RestfulClient(base_url=self.base_url,
                             api_key=self.api_key,
                             api_type=self.api_type,
                             model=self.model)

    def _prepare_completion_kwargs(
            self,
            messages: List[Union[HumanMessage, AIMessage]],
            **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, response_format, etc.)
        3. Default values in self.kwargs
        """
        messages = self.message_manager(messages)
        
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
        }

        return completion_kwargs


    async def generate(
        self,
        messages: List[Union[HumanMessage, AIMessage]],
        **kwargs,
    ) -> AIMessage:

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            **kwargs,
        )

        # Async call to the LiteLLM client for completion
        response = self.client.completion(**completion_kwargs)

        content = response['output']
        return AIMessage(content=content)

    async def __call__(self, *args, **kwargs) -> AIMessage:
        """
        Call the model with the given arguments.
        This is a convenience method that calls `generate` with the same arguments.
        """
        return await self.generate(*args, **kwargs)
    
    def invoke(self, *args, **kwargs) -> AIMessage:
        return self.generate(*args, **kwargs)
    
    async def ainvoke(self, *args, **kwargs) -> AIMessage:
        return await self.generate(*args, **kwargs)

class ChatRestfulSearch(): 
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        api_type: str = "responses",
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_type = api_type
        self.verbose = verbose
        
        self.client = self.create_client()
        self.message_manager = MessageManager(api_type=api_type, model=model)
        
        super(ChatRestfulSearch, self).__init__(**kwargs)


    def create_client(self):
        return RestfulSearchClient(base_url=self.base_url,
                                      api_key=self.api_key,
                                      api_type=self.api_type,
                                      model=self.model,
                                      verbose=self.verbose)

    def _prepare_completion_kwargs(
            self,
            model: str,
            messages: List[Union[HumanMessage, AIMessage]],
            **kwargs,
    ) -> dict[str, Any]:
        
        messages = self.message_manager(messages)

        completion_kwargs = {
            "model": model,
            "input": messages,
        }

        completion_kwargs['tools'] = [
            {"type": "web_search_preview"},
            {
                "type": "code_interpreter",
                "container": {"type": "auto"}
            }
        ]

        return completion_kwargs

    async def generate(
        self,
        messages: List[Union[HumanMessage, AIMessage]],
        **kwargs,
    ) -> AIMessage:

        completion_kwargs = self._prepare_completion_kwargs(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        # Async call to the LiteLLM client for completion
        response = self.client.completion(**completion_kwargs)
        
        content = response['output'][-1]['content'][-1]['text']

        return AIMessage(content=content)

    async def __call__(self, *args, **kwargs) -> AIMessage:
        """
        Call the model with the given arguments.
        This is a convenience method that calls `generate` with the same arguments.
        """
        return await self.generate(*args, **kwargs)
    
    def invoke(self, *args, **kwargs) -> AIMessage:
        return self.generate(*args, **kwargs)
    
    async def ainvoke(self, *args, **kwargs) -> AIMessage:
        return await self.generate(*args, **kwargs)