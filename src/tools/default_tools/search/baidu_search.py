from __future__ import annotations
from typing import Any, Optional, Dict, Type
import asyncio
import json
from pydantic import Field
from baidusearch.baidusearch import search
from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.tools.default_tools.search.base import SearchItem, SearchToolArgs
from src.tools.protocol.tool import BaseTool
from src.tools.protocol.types import ToolResponse

class BaiduSearch(BaseTool):
    """Tool that queries the Baidu search engine.

    Supported parameters:
    - query: Query to search for (required)
    - num_results: Number of search results to return (default: 10)
    - country: Country to search in (default: us)
    - lang: Language to search in (default: en)
    - filter_year: Year to filter results by (default: None)

    Example usages:
    .. code-block:: python
        # basic usage
        tool = BaiduSearch()

    .. code-block:: python
        # with custom search kwargs
        tool = BaiduSearch.from_search_kwargs({"num_results": 5, "country": "us", "lang": "en", "filter_year": 2025})
    """

    name: str = "baidu_search"
    type: str = "Search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events in Chinese or search Chinese content."
        " input should be a search query."
    )
    args_schema: Type[SearchToolArgs] = SearchToolArgs
    metadata: Dict[str, Any] = Field(default_factory=dict)
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **kwargs):
        """Initialize the BaiduSearch tool."""
        super().__init__(**kwargs)

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> BaiduSearch:
        """Create a tool from search kwargs.

        Args:
            search_kwargs: Any additional kwargs to pass to the search function.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        return cls(search_kwargs=search_kwargs, **kwargs)
    
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
            raw_results = search(query, num_results=num_results)
            
            # Convert raw results to SearchItem format
            search_items = []
            for i, item in enumerate(raw_results):
                if isinstance(item, str):
                    # If it's just a URL
                    search_items.append(
                        SearchItem(title=f"Baidu Result {i+1}", url=item, description=None)
                    )
                elif isinstance(item, dict):
                    # If it's a dictionary with details
                    search_items.append(
                        SearchItem(
                            title=item.get("title", f"Baidu Result {i+1}"),
                            url=item.get("url", ""),
                            description=item.get("abstract", None),
                        )
                    )
                else:
                    # Try to get attributes directly
                    try:
                        search_items.append(
                            SearchItem(
                                title=getattr(item, "title", f"Baidu Result {i+1}"),
                                url=getattr(item, "url", ""),
                                description=getattr(item, "abstract", None),
                            )
                        )
                    except Exception:
                        # Fallback to a basic result
                        search_items.append(
                            SearchItem(
                                title=f"Baidu Result {i+1}", url=str(item), description=None
                            )
                        )
            
            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=2)
            
            return ToolResponse(content=results_json, extra={"data": search_items})
            
        except Exception as e:
            return ToolResponse(content=f"Error in asynchronous execution: {str(e)}")

    def _run(self, query: str, 
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
            "type": "baidu_search"
        }