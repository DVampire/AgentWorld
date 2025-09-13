from __future__ import annotations
from typing import Any, Optional, Dict, List, Type
import asyncio
import json
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr
from firecrawl import AsyncFirecrawlApp
from src.utils import get_env

from src.tools.default_tools.search.base import SearchItem, SearchToolArgs
from src.tools.base import ToolResponse

class FirecrawlSearch(BaseTool):
    """Tool that queries the Firecrawl search engine.

    Example usages:
    .. code-block:: python
        # basic usage
        tool = FirecrawlSearch()

    .. code-block:: python
        # with custom search kwargs
        tool = FirecrawlSearch.from_search_kwargs({"limit": 5})
    """

    name: str = "firecrawl_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    args_schema: Type[SearchToolArgs] = SearchToolArgs
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    api_key: Optional[SecretStr] = Field(default=get_env("FIRECRAWL_API_KEY"))

    def __init__(self, **data):
        """Initialize the FirecrawlSearch tool."""
        super().__init__(**data)

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> FirecrawlSearch:
        """Create a tool from search kwargs.

        Args:
            search_kwargs: Any additional kwargs to pass to the search function.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        return cls(search_kwargs=search_kwargs, **kwargs)

    async def _search_firecrawl(self, 
                                query: str, 
                                num_results: int = 10, 
                                filter_year: Optional[int] = None) -> List[SearchItem]:
        """
        Perform a Firecrawl search using the provided parameters.
        Returns a list of SearchItem objects.
        """
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable is required")
        
        app = AsyncFirecrawlApp(api_key=self.api_key.get_secret_value())
        search_kwargs = {
            "query": query,
            "limit": num_results,
        }

        # Only include tbs if it's a valid value
        if filter_year is not None:
            search_kwargs["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
        
        response = await app.search(**search_kwargs)

        results = []
        for item in response.web:
            title = item.title
            url = item.url
            description = item.description
            results.append(SearchItem(
                title=title,
                url=url,
                description=description
            ))

        return results
    
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
            
            # Perform search
            search_items = await self._search_firecrawl(query, num_results=num_results, filter_year=filter_year)
            
            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=2)
            
            return ToolResponse(content=results_json, extra={"data": search_items})
            
        except Exception as e:
            return ToolResponse(content=f"Error in asynchronous execution: {str(e)}")

    def _run(self, query: str, num_results: Optional[int] = 5, country: Optional[str] = "us", lang: Optional[str] = "en", filter_year: Optional[int] = None) -> ToolResponse:
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
            "type": "firecrawl_search"
        }