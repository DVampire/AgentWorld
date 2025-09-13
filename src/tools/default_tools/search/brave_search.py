from __future__ import annotations
from typing import Any, Optional, Dict, Type, List
import asyncio
import json
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, BaseModel
import requests

from langchain_core.documents import Document
from src.utils import get_env

from src.tools.default_tools.search.base import SearchItem, SearchToolArgs
from src.tools.base import ToolResponse

class BraveSearchWrapper(BaseModel):
    """Wrapper around the Brave search engine."""

    """The API key to use for the Brave search engine."""
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    """The base URL for the Brave search engine."""
    api_key: SecretStr = Field(default=get_env("BRAVE_SEARCH_API_KEY"))

    def run(self, query: str, 
            num_results: Optional[int] = 5,
            country: Optional[str] = "us",
            lang: Optional[str] = "en",
            filter_year: Optional[int] = None) -> str:
        """Query the Brave search engine and return the results as a JSON string.

        Args:
            query: The query to search for.

        Returns: The results as a JSON string.

        """
        web_search_results = self._search_request(query=query, 
                                                  num_results=num_results,
                                                  country=country,
                                                  lang=lang,
                                                  filter_year=filter_year)
        final_results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "description": " ".join(
                    filter(
                        None, [item.get("description"), *item.get("extra_snippets", [])]
                    )
                ),
            }
            for item in web_search_results
        ]
        return json.dumps(final_results, ensure_ascii=False, indent=4)

    def download_documents(self, query: str) -> List[Document]:
        """Query the Brave search engine and return the results as a list of Documents.

        Args:
            query: The query to search for.

        Returns: The results as a list of Documents.

        """
        results = self._search_request(query)
        return [
            Document(
                page_content=" ".join(
                    filter(
                        None, [item.get("description"), *item.get("extra_snippets", [])]
                    )
                ),
                metadata={"title": item.get("title"), "link": item.get("url")},
            )
            for item in results
        ]

    def _search_request(self, query: str, 
                        num_results: Optional[int] = 5,
                        country: Optional[str] = "us",
                        lang: Optional[str] = "en",
                        filter_year: Optional[int] = None) -> List[dict]:
        headers = {
            "X-Subscription-Token": self.api_key.get_secret_value(),
            "Accept": "application/json",
        }
        req = requests.PreparedRequest()
        params = {**self.search_kwargs, **{"q": query, 
                                           "count": num_results,
                                           "country": country.upper(),
                                           "search_lang": lang.lower(),
                                           "extra_snippets": True
                                           }}
        req.prepare_url(self.base_url, params)
        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")

        response = requests.get(req.url, headers=headers)
        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        return response.json().get("web", {}).get("results", [])


class BraveSearch(BaseTool):
    """Tool that queries the BraveSearch.

    Api key can be provided as an environment variable BRAVE_SEARCH_API_KEY
    or as a parameter.


    Example usages:
    .. code-block:: python
        # uses BRAVE_SEARCH_API_KEY from environment
        tool = BraveSearch()

    .. code-block:: python
        # uses the provided api key
        tool = BraveSearch.from_api_key("your-api-key")

    .. code-block:: python
        # uses the provided api key and search kwargs
        tool = BraveSearch.from_api_key(
                                api_key = "your-api-key",
                                search_kwargs={"max_results": 5}
                                )

    .. code-block:: python
        # uses BRAVE_SEARCH_API_KEY from environment
        tool = BraveSearch.from_search_kwargs({"max_results": 5})
    """

    name: str = "brave_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    args_schema: Type[SearchToolArgs] = SearchToolArgs
    search_wrapper: BraveSearchWrapper = Field(default_factory=BraveSearchWrapper)
    
    def __init__(self, **kwargs):
        """Initialize the BraveSearch tool."""
        super().__init__(**kwargs)

    @classmethod
    def from_api_key(
        cls, api_key: str, search_kwargs: Optional[dict] = None, **kwargs: Any
    ) -> BraveSearch:
        """Create a tool from an api key.

        Args:
            api_key: The api key to use.
            search_kwargs: Any additional kwargs to pass to the search wrapper.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        wrapper = BraveSearchWrapper(
            api_key=SecretStr(api_key), search_kwargs=search_kwargs or {}
        )
        return cls(search_wrapper=wrapper, **kwargs)

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> BraveSearch:
        """Create a tool from search kwargs.

        Uses the environment variable BRAVE_SEARCH_API_KEY for api key.

        Args:
            search_kwargs: Any additional kwargs to pass to the search wrapper.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        # we can not provide api key because it's calculated in the wrapper,
        # so the ignore is needed for linter
        # not ideal but needed to keep the tool code changes non-breaking
        wrapper = BraveSearchWrapper(search_kwargs=search_kwargs)
        return cls(search_wrapper=wrapper, **kwargs)
    
    async def _arun(
        self,
        query: str,
        num_results: Optional[int] = 5,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = None,
    ) -> ToolResponse:
        """Use the tool asynchronously."""
        
        try:
            results = self.search_wrapper.run(query, 
                                             num_results,
                                             country, 
                                             lang,
                                             filter_year)
            json_data = json.loads(results)
            data = []
            for item in json_data:
                data.append(SearchItem(title=item["title"], 
                                       url=item["url"], 
                                       description=item["description"])
                            )
            return ToolResponse(content=results, extra={"data": data})
            
        except Exception as e:
            return ToolResponse(content=f"Error in asynchronous execution: {str(e)}")

    def _run(self, 
             query: str, 
             num_results: Optional[int] = 5,
             country: Optional[str] = "us",
             lang: Optional[str] = "en",
             filter_year: Optional[int] = None) -> ToolResponse:
        """Convert a file to markdown synchronously (fallback)."""
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(query, num_results, country, lang, filter_year))
            finally:
                loop.close()
        except Exception as e:
            return ToolResponse(content=f"Error in synchronous execution: {str(e)}")
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "type": "brave_search"
        }