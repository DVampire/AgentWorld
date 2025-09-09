from __future__ import annotations
from typing import Any, Optional, Dict, Type, List
import asyncio
import json
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, model_validator, Field
from dotenv import load_dotenv
load_dotenv(verbose=True)
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from src.tools.default_tools.search.base import SearchItem, SearchToolArgs
from src.tools.base import ToolResponse

class DuckDuckGoSearchAPIWrapper(BaseModel):
    """Wrapper for DuckDuckGo Search API.

    Free and does not require any setup.
    """

    region: Optional[str] = "wt-wt"
    """
    See https://pypi.org/project/duckduckgo-search/#regions
    """
    safesearch: str = "moderate"
    """
    Options: strict, moderate, off
    """
    time: Optional[str] = "y"
    """
    Options: d, w, m, y
    """
    max_results: int = 5
    backend: str = "auto"
    """
    Options: auto, html, lite
    """
    source: str = "text"
    """
    Options: text, news, images
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that python package exists in environment."""
        try:
            from ddgs import DDGS  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import ddgs python package. "
                "Please install it with `pip install -U ddgs`."
            )
        return values

    def _ddgs_text(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo text search and return results."""
        from ddgs import DDGS

        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=max_results or self.max_results,
                backend=self.backend,
            )
            if ddgs_gen:
                return [r for r in ddgs_gen]
        return []

    def _ddgs_news(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo news search and return results."""
        from ddgs import DDGS

        with DDGS() as ddgs:
            ddgs_gen = ddgs.news(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
                max_results=max_results or self.max_results,
            )
            if ddgs_gen:
                return [r for r in ddgs_gen]
        return []

    def _ddgs_images(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo image search and return results."""
        from ddgs import DDGS

        with DDGS() as ddgs:
            ddgs_gen = ddgs.images(
                query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max_results or self.max_results,
            )
            if ddgs_gen:
                return [r for r in ddgs_gen]
        return []

    def run(self, query: str) -> str:
        """Run query through DuckDuckGo and return concatenated results."""
        results = self.results(query, max_results=self.max_results, source=self.source)
        return json.dumps(results, ensure_ascii=False, indent=4)

    def results(
        self, query: str, max_results: int, source: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Run query through DuckDuckGo and return metadata.

        Args:
            query: The query to search for.
            max_results: The number of results to return.
            source: The source to look from.

        Returns:
            A list of dictionaries with the following keys:
                description - The description of the result.
                title - The title of the result.
                url - The link to the result.
        """
        source = source or self.source
        if source == "text":
            results = [
                {
                    "description": r["body"], 
                    "title": r["title"], 
                    "url": r["href"]
                }
                for r in self._ddgs_text(query, max_results=max_results)
            ]
        elif source == "news":
            results = [
                {
                    "description": r["body"],
                    "title": r["title"],
                    "url": r["url"],
                    "date": r["date"],
                    "source": r["source"],
                }
                for r in self._ddgs_news(query, max_results=max_results)
            ]
        elif source == "images":
            results = [
                {
                    "title": r["title"],
                    "thumbnail": r["thumbnail"],
                    "image": r["image"],
                    "url": r["url"],
                    "height": r["height"],
                    "width": r["width"],
                    "source": r["source"],
                }
                for r in self._ddgs_images(query, max_results=max_results)
            ]
        else:
            results = []

        if results is None:
            results = [{"Result": "No good DuckDuckGo Search Result was found"}]

        return results


class DuckDuckGoSearch(BaseTool):
    """Tool that queries the DuckDuckGo search engine.

    Example usages:
    .. code-block:: python
        # basic usage
        tool = DuckDuckGoSearch()

    .. code-block:: python
        # with custom search kwargs
        tool = DuckDuckGoSearch.from_search_kwargs({"max_results": 5})
    """

    name: str = "duckduckgo_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    args_schema: Type[SearchToolArgs] = SearchToolArgs
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    search_wrapper: DuckDuckGoSearchAPIWrapper = Field(default_factory=DuckDuckGoSearchAPIWrapper)

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> DuckDuckGoSearch:
        """Create a tool from search kwargs.

        Args:
            search_kwargs: Any additional kwargs to pass to the search wrapper.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        wrapper = DuckDuckGoSearchAPIWrapper(search_kwargs=search_kwargs)
        return cls(search_wrapper=wrapper, search_kwargs=search_kwargs, **kwargs)
    
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
            results = self.search_wrapper.run(query)
            
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
            "type": "duckduckgo_search"
        }